#!/usr/bin/env python3
"""Orchestrator for the agentic-retrieval dataset pipeline.

Stages (each reads/writes JSONL and is independently resumable):

    windows    build contiguous neighbour windows from chunks
    generate   LLM: generate multi-chunk (neighbour) candidate questions
    docunits   build whole-document units for the simple/hard generator
    gen-docs   LLM: generate simple (1 passage) & hard (many passages) questions
    verify     LLM judges: groundedness + minimality -> final gold set
    clues      decompose each question into atomic clues + retrieval requests
    retrieve   embed corpus, index it, retrieve top-k passages per clue
    collect-positives  validate retrieved passages -> expand positive set
    negatives  (optional) mine hard negatives with an embedder
    finalize   assemble out/dataset.jsonl
    all        run the whole thing end to end
    stats      print dataset statistics

Examples:
    python run_pipeline.py all --config config.yaml
    python run_pipeline.py generate --config config.yaml
    python run_pipeline.py all --config config.example.yaml --backend mock   # dry run
"""
from __future__ import annotations

import argparse
import asyncio

from arqg.clues import make_clues as run_make_clues
from arqg.collect import collect_positives as run_collect
from arqg.config import Config
from arqg.data import ChunkStore, load_chunks
from arqg.docunits import build_doc_units
from arqg.generate import generate as run_generate
from arqg.generate_docs import generate_docs as run_generate_docs
from arqg.llm import make_client
from arqg.negatives import mine_negatives
from arqg.retrieve import retrieve as run_retrieve
from arqg.utils import log, read_jsonl, setup_logging, write_jsonl
from arqg.verify import verify as run_verify
from arqg.windows import build_windows


def _store(cfg: Config) -> ChunkStore:
    return ChunkStore(load_chunks(cfg.paths.corpus))


def cmd_windows(cfg: Config) -> None:
    store = _store(cfg)
    windows = build_windows(store, cfg.windows, cfg.filters)
    n = write_jsonl(cfg.paths.windows, (w.to_dict() for w in windows))
    log.info("wrote %d windows -> %s", n, cfg.paths.windows)


async def cmd_generate(cfg: Config) -> None:
    llm = make_client(cfg.llm)
    try:
        await run_generate(cfg, llm)
    finally:
        await llm.aclose()


def cmd_docunits(cfg: Config) -> None:
    store = _store(cfg)
    units = build_doc_units(store, cfg.docgen, cfg.filters)
    n = write_jsonl(cfg.paths.docunits, (u.to_dict() for u in units))
    log.info("wrote %d document units -> %s", n, cfg.paths.docunits)


async def cmd_gen_docs(cfg: Config) -> None:
    llm = make_client(cfg.llm)
    try:
        await run_generate_docs(cfg, llm)
    finally:
        await llm.aclose()


async def cmd_verify(cfg: Config) -> None:
    store = _store(cfg)
    judge = make_client(cfg.verify.judge)
    try:
        await run_verify(cfg, judge, store)
    finally:
        await judge.aclose()


async def cmd_clues(cfg: Config) -> None:
    llm = make_client(cfg.llm)
    try:
        await run_make_clues(cfg, llm, _store(cfg))
    finally:
        await llm.aclose()


async def cmd_retrieve(cfg: Config) -> None:
    await run_retrieve(cfg, _store(cfg))


async def cmd_collect(cfg: Config) -> None:
    judge = make_client(cfg.collect.judge)
    try:
        await run_collect(cfg, judge, _store(cfg))
    finally:
        await judge.aclose()


def cmd_negatives(cfg: Config) -> None:
    mine_negatives(cfg, _store(cfg))


def cmd_finalize(cfg: Config) -> None:
    """Assemble dataset.jsonl from the most-downstream item file
    (collected > verified). When negatives are enabled, that stage already
    wrote dataset.jsonl, so we leave it."""
    if cfg.negatives.enabled:
        log.info("finalize: dataset.jsonl is produced by the negatives stage")
        return
    src = cfg.paths.items_source()
    items = list(read_jsonl(src))
    n = write_jsonl(cfg.paths.dataset, items)
    log.info("finalize: wrote %d items from %s -> %s", n, src, cfg.paths.dataset)


def cmd_stats(cfg: Config) -> None:
    import statistics as st
    items = list(read_jsonl(cfg.paths.dataset)) or list(read_jsonl(cfg.paths.verified))
    if not items:
        log.info("no dataset found yet")
        return
    golds = [len(i.get("gold_chunk_ids", [])) for i in items]

    def tally(key: str) -> str:
        counts: dict[str, int] = {}
        for i in items:
            counts[i.get(key, "?")] = counts.get(i.get(key, "?"), 0) + 1
        return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))

    print(f"items:           {len(items)}")
    print(f"gold chunks/item mean={st.mean(golds):.2f} median={st.median(golds)} "
          f"min={min(golds)} max={max(golds)}")
    print(f"single-chunk (1 gold):  {sum(g == 1 for g in golds)} "
          f"({100*sum(g==1 for g in golds)/len(golds):.1f}%)")
    print(f"multi-chunk (>=2 gold): {sum(g >= 2 for g in golds)} "
          f"({100*sum(g>=2 for g in golds)/len(golds):.1f}%)")
    print(f"with hard negatives:    {sum(1 for i in items if i.get('hard_negative_ids'))}")
    pos = [len(i["positive_chunk_ids"]) for i in items if i.get("positive_chunk_ids")]
    if pos:
        expanded = sum(1 for i in items
                       if len(i.get("positive_chunk_ids", [])) > len(i.get("gold_chunk_ids", [])))
        print(f"positives/item mean={st.mean(pos):.2f} max={max(pos)}; "
              f"items expanded beyond gold: {expanded} ({100*expanded/len(items):.1f}%)")
    print("profiles:        " + tally("profile"))
    print("difficulty:      " + tally("difficulty"))
    print("question types:  " + tally("question_type"))
    print("question styles: " + tally("question_style"))


async def cmd_all(cfg: Config) -> None:
    # neighbour-window multi-hop generator
    if cfg.generate.enabled:
        cmd_windows(cfg)
        await cmd_generate(cfg)
    # document-level simple/hard generator (separate config block)
    if cfg.docgen.enabled:
        cmd_docunits(cfg)
        await cmd_gen_docs(cfg)
    await cmd_verify(cfg)
    if cfg.collect.enabled:
        await cmd_clues(cfg)
        if cfg.retrieve.enabled:
            # automated retrieval: build the index and fill the collect round
            await cmd_retrieve(cfg)
            await cmd_collect(cfg)
        else:
            # manual round: emit requests and stop with resume instructions
            log.warning(
                "collect enabled, retrieve disabled. Next:\n"
                "  1. retrieve top-%d passages per clue in %s\n"
                "  2. save them as %s (see README format)\n"
                "  3. run: collect-positives, then finalize",
                cfg.collect.top_k, cfg.paths.retrieval_requests, cfg.paths.retrieval_results)
            cmd_stats(cfg)
            return
    if cfg.negatives.enabled:
        cmd_negatives(cfg)
    cmd_finalize(cfg)
    cmd_stats(cfg)


def main() -> None:
    p = argparse.ArgumentParser(description="Agentic-retrieval dataset pipeline")
    p.add_argument("stage", choices=[
        "windows", "generate", "docunits", "gen-docs", "verify",
        "clues", "retrieve", "collect-positives", "negatives",
        "finalize", "stats", "all"])
    p.add_argument("--config", default=None, help="path to YAML config")
    p.add_argument("--backend", default=None,
                   help="override LLM backend for generator AND judge (openai|gateway|anthropic|mock)")
    p.add_argument("--chunks", default=None, help="override path to corpus chunks file/dir")
    p.add_argument("--index", default=None, help="override corpus index file/dir (preferred over --chunks)")
    p.add_argument("--out-dir", default=None, help="override output directory")
    args = p.parse_args()

    cfg = Config.load(args.config)
    if args.backend:
        cfg.llm.backend = args.backend
        cfg.verify.judge.backend = args.backend
        cfg.collect.judge.backend = args.backend
        if args.backend == "mock":   # offline dry run: embeddings too
            cfg.retrieve.backend = "mock"
    if args.chunks:
        cfg.paths.chunks = args.chunks
    if args.index:
        cfg.paths.index = args.index
    if args.out_dir:
        cfg.paths.out_dir = args.out_dir
    setup_logging(cfg.log_level)

    if args.stage == "windows":
        cmd_windows(cfg)
    elif args.stage == "generate":
        asyncio.run(cmd_generate(cfg))
    elif args.stage == "docunits":
        cmd_docunits(cfg)
    elif args.stage == "gen-docs":
        asyncio.run(cmd_gen_docs(cfg))
    elif args.stage == "clues":
        asyncio.run(cmd_clues(cfg))
    elif args.stage == "retrieve":
        asyncio.run(cmd_retrieve(cfg))
    elif args.stage == "collect-positives":
        asyncio.run(cmd_collect(cfg))
    elif args.stage == "verify":
        asyncio.run(cmd_verify(cfg))
    elif args.stage == "negatives":
        cmd_negatives(cfg)
    elif args.stage == "finalize":
        cmd_finalize(cfg)
    elif args.stage == "stats":
        cmd_stats(cfg)
    elif args.stage == "all":
        asyncio.run(cmd_all(cfg))


if __name__ == "__main__":
    main()
