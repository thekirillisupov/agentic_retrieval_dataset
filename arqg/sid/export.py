"""S8 (partial) — final task pool, splits and datamix statistics.

Teacher-trajectory collection (plan §9.1) needs the RL harness — four tools, the
`<state>` format, the per-episode doc_id remapping — and is deliberately out of
this pipeline. What is here is everything the harness would consume: the frozen
task pool on a pinned index version, a train/holdout split with MinHash
de-duplication, and the coverage/difficulty statistics the datamix is balanced
on (§9.3).

The plan's SFT/RL split (§9.4) is **not** made here. It divides the pool by what
each half will be *used for*, and nothing in this pipeline collects
trajectories, so the boundary would be drawn on no evidence — and drawing it
early costs tasks: enforcing disjoint gold sets across the boundary discards
whichever side loses the overlap. The pool stays whole; split it when there is a
trainer to split it for.

Two things the plan is emphatic about and that are implemented here:

* **holdout distractors are injected into the same index** — otherwise holdout
  sits in a systematically sparser neighbourhood than train and is artificially
  easy. Since injection happens before the split, this holds by construction.
* **`share_singleton_groups`** is a required output: below 95%, NDCG has to be
  computed per fact group, not per chunk, or the metric penalises a rollout that
  returned an equally valid member of a redundant group.
"""
from __future__ import annotations

import json
import random
import re
import statistics as st
from collections import Counter
from typing import Any

from ..utils import ensure_parent, log, write_jsonl
from .config import SidConfig
from .retrieval import gap_bin
from .sections import shared_depth

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


# --------------------------------------------------------------------------- #
# MinHash de-duplication (§9.4, §9.5) — no external dependency
# --------------------------------------------------------------------------- #
def shingles(text: str, k: int = 4) -> set[str]:
    toks = [t.lower() for t in _TOKEN.findall(text or "")]
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def minhash(text: str, n_perm: int = 64) -> tuple[int, ...]:
    import hashlib
    sh = shingles(text)
    if not sh:
        return tuple([0] * n_perm)
    sig = []
    for p in range(n_perm):
        best = None
        for s in sh:
            h = int(hashlib.md5(f"{p}:{s}".encode("utf-8")).hexdigest()[:8], 16)
            best = h if best is None or h < best else best
        sig.append(best or 0)
    return tuple(sig)


def jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if not a or not b:
        return 0.0
    return sum(x == y for x, y in zip(a, b)) / len(a)


def dedup(records: list[dict], threshold: float) -> tuple[list[dict], int]:
    """Drop near-duplicate questions; gold-set overlap is handled at split time."""
    kept: list[dict] = []
    sigs: list[tuple[int, ...]] = []
    dropped = 0
    for rec in records:
        sig = minhash(rec["question"])
        if any(jaccard(sig, s) >= threshold for s in sigs):
            dropped += 1
            continue
        sigs.append(sig)
        kept.append(rec)
    return kept, dropped


# --------------------------------------------------------------------------- #
# Finalisation
# --------------------------------------------------------------------------- #
def finalize_record(cfg: SidConfig, rec: dict) -> dict:
    bins = cfg.export.fused_gap_bins
    cx = dict(rec.get("complexity", {}))
    fused = float(cx.get("fused_gap", 1.0))
    cx["fused_gap_bin"] = gap_bin(fused, bins)
    # the `lexicon` arm needs lexical difficulty that the dense branch does NOT
    # resolve — high lex_gap alone would dilute the arm with tasks bge-m3 solves
    cx["in_lexicon_arm"] = (gap_bin(float(cx.get("lex_gap", 1.0)), bins) == "high"
                            and cx["fused_gap_bin"] == "high")
    groups = rec.get("fact_groups", [[c] for c in rec["gold_chunk_ids"]])
    sections = [f.get("section", "") for f in rec.get("facts", []) if f.get("section")]
    return {
        "task_id": rec["task_id"],
        "corpus": rec["corpus"],
        "language": rec["language"],
        "index_version": rec.get("index_version", "v0"),
        "question": rec["question"],
        "answer": rec["answer"],
        "gold_chunk_ids": rec["gold_chunk_ids"],
        "fact_groups": groups,
        "coverage": rec["coverage"],
        "complexity": cx,
        "distractors": rec.get("distractors", {"injected": False, "n_injected": 0}),
        "metrics": rec.get("metrics", {}),
        "provenance": {**rec.get("provenance", {}),
                       "facts": [f["fact_id"] for f in rec.get("facts", [])],
                       "fact_group_sizes": [len(g) for g in groups],
                       "sections": sorted(set(sections)),
                       "section_shared_depth": shared_depth(sections)},
    }


def split_pool(cfg: SidConfig, tasks: list[dict]) -> dict[str, list[dict]]:
    """Train / holdout only.

    Holdout is stratified over mechanic × difficulty with a deliberate tilt into
    the hard tail, and its distractors are already in the same index — injection
    happens before the split, so holdout cannot end up in a systematically
    sparser neighbourhood than train and thereby artificially easy (§9.5).
    """
    rng = random.Random(cfg.export.seed)
    pool = list(tasks)
    rng.shuffle(pool)

    strata: dict[tuple[str, str], list[dict]] = {}
    for t in pool:
        key = (t["coverage"]["A1_mechanic"], t["complexity"]["fused_gap_bin"])
        strata.setdefault(key, []).append(t)
    # hard tail carries double weight in the holdout
    weight = {"low": 1.0, "mid": 1.5, "high": 2.5}
    scored = sorted(strata.items(), key=lambda kv: -weight.get(kv[0][1], 1.0) * len(kv[1]))

    holdout: list[dict] = []
    target = min(cfg.export.holdout_size, max(0, len(pool) // 5))
    i = 0
    while len(holdout) < target and any(v for _, v in scored):
        _, bucket = scored[i % len(scored)]
        if bucket:
            holdout.append(bucket.pop())
        i += 1
        if i > target * 10 + len(scored):
            break

    holdout_ids = {t["task_id"] for t in holdout}
    train = [t for t in pool if t["task_id"] not in holdout_ids]
    # a train task sharing gold with a holdout task would leak the evaluation
    # targets, which is the one boundary worth enforcing at this stage
    holdout_gold = {c for t in holdout for c in t["gold_chunk_ids"]}
    leaked = [t for t in train if set(t["gold_chunk_ids"]) & holdout_gold]
    if leaked:
        log.info("S8: %d train tasks share gold with holdout — dropped", len(leaked))
        train = [t for t in train if not (set(t["gold_chunk_ids"]) & holdout_gold)]
    return {"train": train, "holdout": holdout}


DEFAULT_FUSED_GAP_TARGET = {"low": 0.30, "mid": 0.40, "high": 0.30}


def datamix_stats(tasks: list[dict],
                  target_fused_gap_share: dict[str, float] | None = None) -> dict[str, Any]:
    def tally(fn) -> dict[str, int]:
        return dict(Counter(fn(t) for t in tasks))

    n = max(1, len(tasks))
    group_sizes = [len(g) for t in tasks for g in t["fact_groups"]]
    singleton = sum(1 for s in group_sizes if s == 1) / max(1, len(group_sizes))
    mech = tally(lambda t: t["coverage"]["A1_mechanic"])
    mean_mech = st.mean(mech.values()) if mech else 0
    fused_bins = tally(lambda t: t["complexity"]["fused_gap_bin"])
    n_inj = sum(1 for t in tasks if t["distractors"].get("injected"))
    prov = [t.get("provenance", {}) for t in tasks]
    depths = [p.get("section_shared_depth", 0) for p in prov]
    root_only = sum(1 for d in depths if d <= 1)
    multi_section = sum(1 for p in prov if len(p.get("sections", [])) > 1)
    levels: Counter = Counter()
    types: Counter = Counter()
    for t in tasks:
        levels.update(t["distractors"].get("levels", {}))
        types.update(t["distractors"].get("types", {}))
    total_levels = max(1, sum(levels.values()))

    return {
        "n_tasks": len(tasks),
        "coverage": {
            "by_mechanic": mech,
            "mechanic_balance_ok": all(abs(v - mean_mech) <= 0.3 * mean_mech
                                       for v in mech.values()) if mech else False,
            "by_submechanic": tally(lambda t: t["coverage"]["A1_submechanic"]),
            "share_negation": round(sum(1 for t in tasks
                                        if t["coverage"]["has_negation"]) / n, 4),
        },
        "complexity": {
            "hop_depth": tally(lambda t: str(t["complexity"]["hop_depth"])),
            "fused_gap_bins": fused_bins,
            "fused_gap_share": {k: round(v / n, 4) for k, v in fused_bins.items()},
            # the same target S4's 1-of-N selection steers towards, so this
            # grades the pool against the goal it was built for rather than a
            # second one written down here
            "target_fused_gap_share": dict(
                target_fused_gap_share or DEFAULT_FUSED_GAP_TARGET),
            "lexicon_arm_size": sum(1 for t in tasks
                                    if t["complexity"].get("in_lexicon_arm")),
            "sparse_origin": sum(1 for t in tasks
                                 if t["complexity"].get("sparse_origin")),
        },
        "gold": {
            "mean_gold_chunks": round(st.mean([len(t["gold_chunk_ids"]) for t in tasks]), 3)
            if tasks else 0,
            "mean_fact_groups": round(st.mean([len(t["fact_groups"]) for t in tasks]), 3)
            if tasks else 0,
            "share_singleton_groups": round(singleton, 4),
            "ndcg_granularity": "chunk" if singleton >= 0.95 else "fact_group",
            "ndcg_note": ("groups are singletons — chunk granularity is identical"
                          if singleton >= 0.95 else
                          "non-singleton groups present: NDCG MUST be computed over "
                          "fact groups (first-ranked member scores the group) and "
                          "B_i counted in groups, or the metric penalises a rollout "
                          "that returned an equally valid group member"),
        },
        "sections": {
            # How topically coherent the gold sets are, read off the `title`
            # breadcrumb. `share_root_only` is the pathology S1's section
            # scoping exists to remove: a gold set whose fragments share nothing
            # but the corpus root was bridged by a coincidence, and the composer
            # has to invent the link. Mined globally this corpus produced 51%;
            # a non-trivial share here means scoping is not reaching the pool.
            "share_root_only": round(root_only / n, 4),
            "share_multi_section": round(multi_section / n, 4),
            "shared_depth_median": round(st.median(depths), 1) if depths else 0,
            "shared_depth": dict(Counter(str(d) for d in depths)),
        },
        # Which S1 channel the surviving tasks came from. The doc2doc channel
        # exists to reach folders no repeated surface form bridges, so its share
        # here against its share of the mined pool is what says whether those
        # folders produce tasks or only candidates.
        "bridge_kind": dict(Counter(t.get("provenance", {}).get("bridge_kind", "entity")
                                    for t in tasks)),
        "distractors": {
            "tasks_with_injection": n_inj,
            "share_tasks_with_injection": round(n_inj / n, 4),
            "levels": dict(levels),
            "share_L3": round(levels.get("L3_generated", 0) / total_levels, 4),
            "share_L3_alarm": levels.get("L3_generated", 0) / total_levels > 0.10,
            "types": dict(types),
            "share_tasks_with_empty_L2_band": round(
                sum(1 for t in tasks
                    if t["distractors"].get("l2_band_size", 0) < 2) / n, 4),
        },
    }


def run_export(cfg: SidConfig, records: list[dict],
               extra_stats: dict[str, Any] | None = None) -> dict[str, Any]:
    tasks = [finalize_record(cfg, r) for r in records]
    tasks, dropped = dedup(tasks, cfg.export.dedup_threshold)
    if dropped:
        log.info("S8: MinHash dropped %d near-duplicate questions", dropped)

    write_jsonl(cfg.paths.tasks, tasks)
    splits = split_pool(cfg, tasks)
    for name, rows in splits.items():
        write_jsonl(cfg.paths.split(name), rows)
        log.info("S8: split %-8s %d tasks -> %s", name, len(rows), cfg.paths.split(name))

    stats = datamix_stats(tasks, cfg.export.target_fused_gap_share)
    stats["splits"] = {k: len(v) for k, v in splits.items()}
    stats["deduplicated"] = dropped
    if extra_stats:
        stats.update(extra_stats)
    ensure_parent(cfg.paths.stats)
    with open(cfg.paths.stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    log.info("S8: %d tasks -> %s", len(tasks), cfg.paths.tasks)
    return stats


def print_stats(stats: dict[str, Any]) -> None:
    print(json.dumps(stats, ensure_ascii=False, indent=2))
