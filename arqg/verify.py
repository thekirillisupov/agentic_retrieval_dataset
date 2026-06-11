"""Stage 3 — verify candidates and emit the final dataset (async, resumable).

This is where dataset quality is enforced. Two independent judge passes:

1. Groundedness: is the answer fully supported, is the question standalone &
   specific, and is it NOT trivially answerable from world knowledge?
2. Minimality: shrink the candidate's required chunks to the strictly-necessary
   set, and confirm no single chunk alone suffices (the multi-hop guarantee).

A candidate becomes a dataset item only if it passes every enabled gate and the
final (minimal) gold set still has >= min_gold_chunks chunks.
"""
from __future__ import annotations

import asyncio

from .config import Config
from .data import ChunkStore
from .llm import BaseLLM, LLMError
from .prompts import (JUDGE_SYSTEM, groundedness_user, minimality_user)
from .schema import Candidate, Chunk, DatasetItem
from .utils import append_jsonl, load_done_keys, log, read_jsonl


async def verify(cfg: Config, judge: BaseLLM, store: ChunkStore) -> None:
    candidates = [Candidate(**c) for c in read_jsonl(cfg.paths.candidates)]
    done = load_done_keys(cfg.paths.verified, "id")
    todo = [c for c in candidates if c.candidate_id not in done]
    log.info("verify: %d candidates, %d done, %d to do", len(candidates), len(done), len(todo))

    lock = asyncio.Lock()
    kept = 0

    async def worker(c: Candidate) -> None:
        nonlocal kept
        try:
            item = await _verify_one(cfg, judge, store, c)
        except LLMError as e:
            log.error("verify failed for %s: %s", c.candidate_id, e)
            return
        if item is None:
            return
        async with lock:
            append_jsonl(cfg.paths.verified, item.to_dict())
            kept += 1

    sem = asyncio.Semaphore(cfg.verify.judge.max_concurrency)

    async def run(c):
        async with sem:
            await worker(c)

    await asyncio.gather(*(run(c) for c in todo))
    log.info("verify: kept %d / %d -> %s", kept, len(todo), cfg.paths.verified)


def _chunks_for(store: ChunkStore, ids: list[str]) -> list[Chunk]:
    out = []
    for cid in ids:
        c = store.get_by_id(cid)
        if c is not None:
            out.append(c)
    return out


async def _verify_one(cfg: Config, judge: BaseLLM, store: ChunkStore,
                      c: Candidate) -> DatasetItem | None:
    vc = cfg.verify
    verdict: dict = {}

    # --- Judge 2 first: establish the minimal necessary gold set --------- #
    gold_ids = list(c.required_chunk_ids)
    if vc.run_minimality:
        cand_chunks = _chunks_for(store, c.required_chunk_ids)
        if len(cand_chunks) < cfg.generate.min_gold_chunks:
            return None
        m = await judge.complete_json(JUDGE_SYSTEM, minimality_user(c.question, cand_chunks))
        verdict["minimality"] = m
        necessary = [i for i in m.get("necessary_chunk_ids", []) if i in set(c.required_chunk_ids)]
        if necessary:
            gold_ids = necessary
        if not m.get("answerable", True):
            return None
        if vc.drop_if_single_chunk_sufficient and m.get("single_chunk_sufficient", False):
            return None
        if len(gold_ids) < cfg.generate.min_gold_chunks:
            return None

    # --- Judge 1: groundedness / standalone / specificity ---------------- #
    if vc.run_groundedness:
        gold_chunks = _chunks_for(store, gold_ids)
        g = await judge.complete_json(
            JUDGE_SYSTEM, groundedness_user(c.question, c.answer, gold_chunks))
        verdict["groundedness"] = g
        if not g.get("supported", False) or not g.get("answer_correct", False):
            return None
        if vc.require_standalone and not g.get("standalone", False):
            return None
        if vc.require_specific and not g.get("specific", False):
            return None
        if g.get("answerable_from_world_knowledge", False):
            return None

    return DatasetItem(
        id=c.candidate_id,
        question=c.question,
        answer=c.answer,
        gold_chunk_ids=gold_ids,
        file_name=c.file_name,
        question_type=c.question_type,
        question_style=c.question_style,
        num_gold=len(gold_ids),
        window_chunk_ids=c.window_chunk_ids,
        verification=verdict,
        generation_model=c.generation_model,
        judge_model=cfg.verify.judge.model,
    )
