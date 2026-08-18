"""S7 — task isolation on the **post-injection** index (plan §8).

Order matters and is not negotiable: isolation runs after injection, never
before. A distractor built for task A lives in the shared index and is a
candidate in the results for B, C, D — and if it happens to be a valid
alternative path for B, that is exactly the labelling hole isolation exists to
close. A check on v0 cannot see it.

This does *not* reduce to the §7.6 argument. Unreachability is common-mode and
cancels under group-normalised advantage; an unlabelled alternative path does
not — the rollouts that found the shortcut are precisely the ones penalised,
which is a differential signal pointing the wrong way.

Two different things are checked, and the distinction matters:

**Duplicate labelling.** Two tasks with the *same* gold set are one task
labelled twice; keeping both double-counts it in train and can straddle the
train/holdout boundary. One survives, chosen deterministically by ``task_id``.

**An unlabelled path.** Everything in the result list that is not this task's
gold is a candidate shortcut, and only a judge can say whether it actually
yields the answer — including another task's gold, which the retriever has no
reason to keep out of the top-k. Rejecting on the mere *presence* of another
task's gold is not that test: it fires on co-retrieval, which is the normal
state of affairs once S1 mines within a folder (the better the scoping, the
more neighbouring tasks retrieve each other), and it fires symmetrically, so
both members of a pair die rather than one. It also could never see the case
it was named for: a gold chunk shared with another task is *this* task's gold
too, so it was filtered out before the check ever ran.
"""
from __future__ import annotations

import asyncio
import json
from collections import Counter

from ..llm import BaseLLM
from ..utils import append_jsonl, ensure_parent, load_done_keys, log, read_jsonl
from .config import SidConfig
from .env import Env
from .prompts import ISOLATION_SYS, isolation_user


def duplicate_gold_sets(cfg: SidConfig, todo: list[dict]) -> dict[str, str]:
    """``task_id -> the task_id whose gold set it repeats``.

    Seeded from the tasks a previous run already kept, so a resume cannot admit
    a twin of one. Ownership goes to the lowest ``task_id`` so the survivor does
    not depend on the order the pool happens to arrive in.
    """
    owner: dict[frozenset[str], str] = {}
    for rec in read_jsonl(cfg.paths.isolated):
        owner.setdefault(frozenset(rec["gold_chunk_ids"]), rec["task_id"])
    dupes: dict[str, str] = {}
    for rec in sorted(todo, key=lambda r: r["task_id"]):
        first = owner.setdefault(frozenset(rec["gold_chunk_ids"]), rec["task_id"])
        if first != rec["task_id"]:
            dupes[rec["task_id"]] = first
    return dupes


def _gold_overlap(records: list[dict]) -> dict[str, float | int]:
    """How much the pool shares gold at all — the thing the old rule was
    reaching for, now measured instead of acted on. Sharing a chunk between
    two questions is normal; sharing the whole gold set is not."""
    owners: dict[str, set[str]] = {}
    for r in records:
        for cid in r["gold_chunk_ids"]:
            owners.setdefault(cid, set()).add(r["task_id"])
    shared = {t for ids in owners.values() if len(ids) > 1 for t in ids}
    n = max(1, len(records))
    return {"tasks_sharing_a_gold_chunk": len(shared),
            "share_sharing_a_gold_chunk": round(len(shared) / n, 4)}


async def run_isolation(cfg: SidConfig, env: Env, judge: BaseLLM,
                        records: list[dict]) -> list[dict]:
    if not cfg.isolation.enabled:
        log.info("S7: isolation disabled")
        return records

    seen = load_done_keys(cfg.paths.isolation_decisions, "task_id")
    todo = [r for r in records if r["task_id"] not in seen]
    dupes = duplicate_gold_sets(cfg, todo)
    checked = [r for r in todo if r["task_id"] not in dupes]
    log.info("S7: isolating %d tasks on %s (%d already decided, %d duplicate "
             "gold sets)", len(checked), env.corpus.version, len(seen), len(dupes))

    gold_owner: dict[str, str] = {}
    for r in records:
        for cid in r["gold_chunk_ids"]:
            gold_owner.setdefault(cid, r["task_id"])

    reasons = Counter()
    kept: list[dict] = []

    # one probe per task with the same retriever the agent will use
    probes = await env.searcher.probe_many([r["question"] for r in checked],
                                           cfg.isolation.top_k)

    async def one(rec: dict, probe) -> tuple[dict | None, str]:
        gold = set(rec["gold_chunk_ids"])
        # Everything that is not this task's gold is a candidate shortcut,
        # another task's gold included — the retriever has no reason to keep it
        # out of the top-k, and only the judge can say whether it actually
        # yields this answer. Presence alone is co-retrieval, not a leak.
        unlabelled = [cid for cid in probe.hit_ids if cid not in gold]
        if not unlabelled or not cfg.isolation.judge_alternative_paths:
            return rec, "kept"
        head = unlabelled[: cfg.isolation.judge_top_n]
        try:
            v = await judge.complete_json(
                ISOLATION_SYS,
                isolation_user(rec["question"], rec["answer"],
                               [(cid, env.corpus.text(cid)) for cid in head]))
        except Exception:                                      # noqa: BLE001
            return rec, "kept"
        alt = [c for c in v.get("alternative_path_chunk_ids", []) if c in set(head)]
        if not alt:
            return rec, "kept"
        if any(env.corpus.is_synthetic(c) for c in alt):
            return None, "injected_distractor_became_path"
        if any(gold_owner.get(c, rec["task_id"]) != rec["task_id"] for c in alt):
            return None, "other_task_gold"
        return None, "alternative_document"

    results = await asyncio.gather(*(one(r, p) for r, p in zip(checked, probes)))
    for task_id, twin in sorted(dupes.items()):
        append_jsonl(cfg.paths.isolation_decisions,
                     {"task_id": task_id, "outcome": "duplicate_gold_set",
                      "same_gold_as": twin})
        reasons["duplicate_gold_set"] += 1
    for (rec, outcome), src in zip(results, checked):
        append_jsonl(cfg.paths.isolation_decisions,
                     {"task_id": src["task_id"], "outcome": outcome})
        if rec is None:
            reasons[outcome] += 1
            continue
        rec = {**rec, "provenance": {
            **rec["provenance"],
            "gates_passed": rec["provenance"]["gates_passed"] + ["S7_isolation"]}}
        append_jsonl(cfg.paths.isolated, rec)
        kept.append(rec)

    report = {
        "corpus": cfg.corpus_name,
        "index_version": env.corpus.version,
        "checked": len(todo),
        "kept": len(kept),
        "excluded": {
            "duplicate_gold_set": reasons["duplicate_gold_set"],
            "other_task_gold": reasons["other_task_gold"],
            "alternative_document": reasons["alternative_document"],
            "injected_distractor_became_path": reasons["injected_distractor_became_path"],
        },
        "gold_overlap": _gold_overlap(records),
        "note": "a rising injected_distractor_became_path means §7.5 verification "
                "needs tightening; other_task_gold counts only tasks the judge "
                "found another task's gold to actually answer, not co-retrieval",
    }
    ensure_parent(cfg.paths.isolation_report)
    with open(cfg.paths.isolation_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info("S7: kept %d/%d — excluded %s", len(kept), len(todo), report["excluded"])
    return list(read_jsonl(cfg.paths.isolated))
