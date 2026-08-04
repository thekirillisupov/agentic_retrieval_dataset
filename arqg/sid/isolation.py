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

A candidate is dropped when its result list contains:
  1. gold chunks of another task,
  2. unlabelled documents yielding the same answer,
  3. injected distractors that accidentally became a valid path.
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


async def run_isolation(cfg: SidConfig, env: Env, judge: BaseLLM,
                        records: list[dict]) -> list[dict]:
    if not cfg.isolation.enabled:
        log.info("S7: isolation disabled")
        return records

    seen = load_done_keys(cfg.paths.isolation_decisions, "task_id")
    todo = [r for r in records if r["task_id"] not in seen]
    log.info("S7: isolating %d tasks on %s (%d already decided)",
             len(todo), env.corpus.version, len(seen))

    gold_owner: dict[str, str] = {}
    for r in records:
        for cid in r["gold_chunk_ids"]:
            gold_owner.setdefault(cid, r["task_id"])

    reasons = Counter()
    kept: list[dict] = []

    # one probe per task with the same retriever the agent will use
    probes = await env.searcher.probe_many([r["question"] for r in todo],
                                           cfg.isolation.top_k)

    async def one(rec: dict, probe) -> tuple[dict | None, str]:
        gold = set(rec["gold_chunk_ids"])
        others = [cid for cid in probe.hit_ids
                  if cid not in gold and gold_owner.get(cid, rec["task_id"]) != rec["task_id"]]
        if others:
            return None, "other_task_gold"

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
        return None, ("injected_distractor_became_path"
                      if any(env.corpus.is_synthetic(c) for c in alt)
                      else "alternative_document")

    results = await asyncio.gather(*(one(r, p) for r, p in zip(todo, probes)))
    for (rec, outcome), src in zip(results, todo):
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
            "other_task_gold": reasons["other_task_gold"],
            "alternative_document": reasons["alternative_document"],
            "injected_distractor_became_path": reasons["injected_distractor_became_path"],
        },
        "note": "a rising third category means §7.5 verification needs tightening",
    }
    ensure_parent(cfg.paths.isolation_report)
    with open(cfg.paths.isolation_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info("S7: kept %d/%d — excluded %s", len(kept), len(todo), report["excluded"])
    return list(read_jsonl(cfg.paths.isolated))
