#!/usr/bin/env python3
"""Build a dialogue retrieval training set from MuSiQue multi-hop decompositions.

Turns each MuSiQue ``question_decomposition`` into a ``<client>/<bot>`` transcript
where a hop's ``#k`` back-reference becomes conversational anaphora, and emits one
retrieval item per turn. Gold for turn ``t`` is *only* hop ``t``'s supporting
paragraph (the bot has already said the earlier answers in the transcript).

Two files are produced under ``--out-dir``:

    musique_corpus.jsonl     every example's paragraphs, as chunks (retrieval pool)
    musique_dialogues.jsonl  the dialogue items (same schema as dataset.jsonl)

The anaphora rewrite uses the same OpenAI-compatible / vLLM / Anthropic LLM client
as the rest of the pipeline (configured via ``--config``); pass ``--anaphora
heuristic`` for a deterministic, model-free rewrite, or ``--backend mock`` for an
offline dry run.

Examples:
    # LLM-rewritten anaphora, pulling MuSiQue from the HF hub
    export OPENAI_API_KEY=sk-...
    python scripts/build_musique_dialogues.py --config config.yaml

    # offline dry run, no network / no GPU, on a local dump
    python scripts/build_musique_dialogues.py --backend mock \
        --input data/musique_ans_v1.0_train.jsonl --limit 50 --out-dir out_demo

    # model-free deterministic build
    python scripts/build_musique_dialogues.py --anaphora heuristic --input musique.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arqg.config import Config
from arqg.llm import make_client
from arqg.musique import MusiqueOptions, build
from arqg.utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to YAML config (LLM settings)")
    p.add_argument("--backend", default=None,
                   help="override LLM backend (openai|gateway|anthropic|mock)")
    p.add_argument("--out-dir", default=None, help="override output directory")

    # source selection
    p.add_argument("--hf-name", default="bdsaglam/musique",
                   help="HF dataset id (default: bdsaglam/musique)")
    p.add_argument("--config-name", default="answerable",
                   help="HF dataset config: 'answerable' or 'default' (default: answerable)")
    p.add_argument("--split", default="train", help="dataset split (default: train)")
    p.add_argument("--input", default="",
                   help="local musique .jsonl/.json (dir ok) instead of the HF hub")
    p.add_argument("--limit", type=int, default=0, help="cap number of examples (0 = all)")

    # dialogue / turn options
    p.add_argument("--min-turn", type=int, default=1,
                   help="lowest turn index to emit (2 = skip the standalone first hop)")
    p.add_argument("--only-anaphora-turns", action="store_true",
                   help="emit only turns whose hop actually references an earlier answer")
    p.add_argument("--anaphora", choices=["llm", "heuristic"], default="llm",
                   help="how to rewrite #k references (default: llm)")

    # output paths (default under --out-dir)
    p.add_argument("--corpus-path", default="", help="override corpus output path")
    p.add_argument("--dataset-path", default="", help="override dialogue output path")
    return p.parse_args()


async def _run(cfg: Config, opts: MusiqueOptions) -> None:
    llm = make_client(cfg.llm) if opts.anaphora == "llm" else None
    try:
        await build(cfg, llm, opts)
    finally:
        if llm is not None:
            await llm.aclose()


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config)
    if args.backend:
        cfg.llm.backend = args.backend
    if args.out_dir:
        cfg.paths.out_dir = args.out_dir
    setup_logging(cfg.log_level)

    opts = MusiqueOptions(
        hf_name=args.hf_name,
        config_name=args.config_name,
        split=args.split,
        input_path=args.input,
        limit=args.limit,
        min_turn=args.min_turn,
        only_anaphora_turns=args.only_anaphora_turns,
        anaphora=args.anaphora,
        corpus_path=args.corpus_path,
        dataset_path=args.dataset_path,
    )
    asyncio.run(_run(cfg, opts))


if __name__ == "__main__":
    main()
