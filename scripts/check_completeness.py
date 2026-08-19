#!/usr/bin/env python3
"""Audit an *existing* SID run for set-completeness — no LLM, no network.

Tasks produced before S3c existed carry no declared filter, so this auditor
reconstructs the weakest defensible one from metadata alone: for every facet in
``completeness.filter_fields`` on which ALL gold documents agree, an ``eq``
constraint. That filter is by construction no stricter than the question's own
wording (the question may carry constraints the shared facets do not), so the
reported excess is an UPPER BOUND — the same estimate as the manual audit that
motivated S3c (32/35 aggregation tasks with excess, median 19 vs 2).

For each aggregation-type task it reports:

* ``n_truth`` / ``n_excess``     — documents matching the shared-facet filter;
* ``verdict``                    — ``exact`` (no excess), ``augmentable``
  (truth small enough for the safe-augmentation branch), ``repairable``
  (every excess document is separable from the gold by some facet), or
  ``unrepairable`` (some excess document coincides with the gold on every
  available facet — no rephrasing can exclude it);
* the separating fields, so a human can sanity-check what the repair loop
  would have asked the composer to add.

Usage (from the repo root, where the run's data actually lives):

    python scripts/check_completeness.py --config config_sid_zakupki.yaml \
        [--tasks out_sid_zakupki_tiny/tasks.jsonl] \
        [--report out_sid_zakupki_tiny/completeness_audit.jsonl]

Without ``--tasks`` the most downstream task file of the config's out_dir is
used (tasks.jsonl → minimized.jsonl → gated.jsonl).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arqg.sid.completeness import CompletenessChecker
from arqg.sid.config import SidConfig
from arqg.sid.corpus import load_corpus
from arqg.utils import read_jsonl, write_jsonl


def _default_filter_fields(cfg: SidConfig) -> list[str]:
    """When the config predates the completeness block, fall back to the
    surfaced facets plus the raw numeric/date companions they were derived
    from."""
    if cfg.completeness.filter_fields:
        return list(cfg.completeness.filter_fields)
    out = list(cfg.facets.fields)
    for extra in ("price_start", "published"):
        if extra not in out:
            out.append(extra)
    return [f for f in out if f not in ("section", "winner")]


def shared_facet_filter(checker: CompletenessChecker, gold_keys: list[str]) -> list[dict]:
    """eq-constraints on every field all gold documents agree on (non-empty)."""
    docs = [checker.docs[k] for k in gold_keys if k in checker.docs]
    if not docs:
        return []
    out: list[dict] = []
    for fld in checker.cc.filter_fields:
        vals = [str(checker.raw(d, fld) or "") for d in docs]
        if all(vals) and len({v.strip().lower() for v in vals}) == 1:
            out.append({"field": fld, "op": "eq", "value": vals[0]})
    return out


def audit_task(checker: CompletenessChecker, rec: dict) -> dict:
    gold_keys = sorted({checker.doc_of(c) for c in rec["gold_chunk_ids"]})
    filt = shared_facet_filter(checker, gold_keys)
    row = {
        "task_id": rec.get("task_id"),
        "mechanic": rec.get("coverage", {}).get("A1_mechanic"),
        "question": rec.get("question", "")[:200],
        "gold_docs": gold_keys,
        "filter": filt,
    }
    if not filt:
        row.update(verdict="no_filter", n_truth=None, n_excess=None)
        return row
    gold_docs = [checker.docs[k] for k in gold_keys if k in checker.docs]
    truth = [d for d in checker.docs.values() if checker.matches_all(d, filt)]
    excess = [d for d in truth if d.key not in set(gold_keys)]
    row.update(n_truth=len(truth), n_excess=len(excess),
               excess_docs=[d.key for d in excess][:30])
    if not excess:
        row["verdict"] = "exact"
    elif len(truth) <= checker.cc.augment_max_docs:
        row["verdict"] = "augmentable"
    elif all(checker.separable(d, gold_docs) for d in excess):
        row["verdict"] = "repairable"
    else:
        row["verdict"] = "unrepairable"
        row["inseparable_docs"] = [d.key for d in excess
                                   if not checker.separable(d, gold_docs)][:10]
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--tasks", default="",
                    help="task file to audit (default: most downstream of out_dir)")
    ap.add_argument("--report", default="",
                    help="write per-task rows here (default: <out_dir>/completeness_audit.jsonl)")
    ap.add_argument("--mechanics", default="set_aggregation,constraint_intersection")
    args = ap.parse_args()

    cfg = SidConfig.load(args.config)
    cfg.completeness.filter_fields = _default_filter_fields(cfg)
    mechanics = {m.strip() for m in args.mechanics.split(",") if m.strip()}

    tasks_path = args.tasks
    if not tasks_path:
        for p in (cfg.paths.tasks, cfg.paths.minimized, cfg.paths.gated):
            if os.path.exists(p):
                tasks_path = p
                break
    if not tasks_path or not os.path.exists(tasks_path):
        raise SystemExit("no task file found — pass --tasks explicitly")

    corpus = load_corpus(cfg)
    checker = CompletenessChecker(cfg, corpus)
    print(f"corpus: {len(corpus)} chunks, {len(checker.docs)} documents "
          f"by {cfg.completeness.doc_field!r}; auditing {tasks_path}")
    print(f"filter fields: {', '.join(cfg.completeness.filter_fields)}")

    rows = [audit_task(checker, rec) for rec in read_jsonl(tasks_path)
            if rec.get("coverage", {}).get("A1_mechanic") in mechanics]
    report = args.report or cfg.paths._p("completeness_audit.jsonl")
    write_jsonl(report, rows)

    verdicts = Counter(r["verdict"] for r in rows)
    excesses = [r["n_excess"] for r in rows if r.get("n_excess") is not None]
    with_excess = [e for e in excesses if e > 0]
    print(f"\n{len(rows)} aggregation-type tasks audited -> {report}")
    for v in ("exact", "augmentable", "repairable", "unrepairable", "no_filter"):
        if verdicts.get(v):
            print(f"  {v:14s} {verdicts[v]:4d}  ({verdicts[v] / len(rows):.0%})")
    if excesses:
        print(f"  excess > 0 in {len(with_excess)}/{len(excesses)} "
              f"({len(with_excess) / len(excesses):.0%}); "
              f"median n_truth = {statistics.median([r['n_truth'] for r in rows if r.get('n_truth') is not None])}, "
              f"median excess (where > 0) = "
              f"{statistics.median(with_excess) if with_excess else 0}")
    print("\nNOTE: the reconstructed filter is weaker than the question's own "
          "wording, so every excess figure is an upper bound; 'exact' verdicts "
          "are safe, 'unrepairable' ones deserve a manual look before deletion.")
    rescuable = verdicts.get("exact", 0) + verdicts.get("augmentable", 0) + \
        verdicts.get("repairable", 0)
    if rows:
        print(f"upper-bound rescue rate (exact + augmentable + repairable): "
              f"{rescuable}/{len(rows)} ({rescuable / len(rows):.0%})")
    # machine-readable one-liner for run reports
    print(json.dumps({"n_tasks": len(rows), **verdicts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
