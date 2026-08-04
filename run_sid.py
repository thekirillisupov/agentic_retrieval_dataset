#!/usr/bin/env python3
"""SID — synthetic dataset factory for agentic retrieval (plan v0.8, v1).

Stages (each reads/writes JSONL and resumes independently):

    compat     S0  index/unit compatibility, available fields, v0 manifest
    mine       S1  entity ↔ chunk subgraphs (rare bridging entities)
    facts      S3  atomic facts with verbatim spans
    compose    S3  1-of-N question composition per coverage cell
    gates      S4  G_BROAD + G_REACH (retrieval-only) then G_SOLVE
    minimize   S5  G_MIN (leave-one-fact-out) + G_REP (fact groups)
    density    §7.1 τ_sim, corpus density norm, per-task injection budget
    distract   S6  distractor cascade + injection, v0 -> v1
    isolate    S7  cross-task isolation on the post-injection index
    export     S8  final pool, train/holdout split, datamix stats
    all            everything, in order
    stats          re-print the last stats.json

Examples:
    python run_sid.py all --config config_sid.yaml
    python run_sid.py gates --config config_sid.yaml
    # offline dry run, no model / GPU / API key:
    python run_sid.py all --config config_sid.yaml --backend mock \
        --corpus tests/sample_corpus_sid.jsonl --out-dir out_sid_demo
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os

from arqg.sid import pipeline
from arqg.sid.compat import run_compat
from arqg.sid.config import SidConfig
from arqg.sid.export import print_stats
from arqg.sid.subgraphs import run_mining
from arqg.utils import setup_logging

STAGES = ["compat", "mine", "facts", "compose", "gates", "minimize",
          "density", "distract", "isolate", "export", "all", "stats"]


def main() -> None:
    p = argparse.ArgumentParser(description="SID synthetic retrieval dataset pipeline")
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--config", default=None, help="path to YAML config")
    p.add_argument("--backend", default=None,
                   help="override LLM backend for generator AND judge "
                        "(openai|gateway|anthropic|mock); 'mock' also mocks embeddings")
    p.add_argument("--corpus", default=None, help="override the v0 corpus path")
    p.add_argument("--out-dir", default=None, help="override the output directory")
    args = p.parse_args()

    cfg = SidConfig.load(args.config)
    if args.backend:
        cfg.llm.backend = cfg.judge.backend = args.backend
        if args.backend == "mock":
            cfg.embed.backend = "mock"
    if args.corpus:
        cfg.paths.corpus = args.corpus
    if args.out_dir:
        cfg.paths.out_dir = args.out_dir
    setup_logging(cfg.log_level)

    if args.stage == "compat":
        run_compat(cfg)
    elif args.stage == "mine":
        run_mining(cfg)
    elif args.stage == "facts":
        asyncio.run(pipeline.stage_facts(cfg))
    elif args.stage == "compose":
        asyncio.run(pipeline.stage_compose(cfg))
    elif args.stage == "gates":
        asyncio.run(pipeline.stage_gates(cfg))
    elif args.stage == "minimize":
        asyncio.run(pipeline.stage_minimize(cfg))
    elif args.stage == "density":
        asyncio.run(pipeline.stage_density(cfg))
    elif args.stage == "distract":
        asyncio.run(pipeline.stage_distract(cfg))
    elif args.stage == "isolate":
        asyncio.run(pipeline.stage_isolate(cfg))
    elif args.stage == "export":
        print_stats(pipeline.stage_export(cfg))
    elif args.stage == "all":
        print_stats(asyncio.run(pipeline.run_all(cfg)))
    elif args.stage == "stats":
        if not os.path.exists(cfg.paths.stats):
            raise SystemExit(f"no stats at {cfg.paths.stats} — run `export` first")
        with open(cfg.paths.stats, "r", encoding="utf-8") as f:
            print_stats(json.load(f))


if __name__ == "__main__":
    main()
