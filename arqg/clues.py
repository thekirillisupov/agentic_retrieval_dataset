"""Stage 4a — decompose each verified question into atomic clues.

Produces two files:
* ``clues.jsonl``              — internal clue records (kept for the collect step)
* ``retrieval_requests.jsonl`` — WHAT YOU RETRIEVE FOR: one query per clue.

You then run retrieval over the whole corpus for each clue and return
``retrieval_results.jsonl`` (see README format); the ``collect-positives`` stage
validates those passages and expands the positive set.
"""
from __future__ import annotations

import asyncio

from .config import Config
from .data import ChunkStore
from .generate import _gather
from .llm import BaseLLM, LLMError
from .prompts import CLUE_SYSTEM, clue_user
from .schema import Chunk, Clue, DatasetItem
from .utils import append_jsonl, load_done_keys, log, read_jsonl, write_jsonl


async def make_clues(cfg: Config, llm: BaseLLM, store: ChunkStore) -> None:
    items = [DatasetItem(**d) for d in read_jsonl(cfg.paths.verified)]
    done = load_done_keys(cfg.paths.clues, "item_id")
    todo = [it for it in items if it.id not in done]
    log.info("clues: %d items, %d done, %d to do", len(items), len(done), len(todo))

    lock = asyncio.Lock()
    written = 0

    async def worker(it: DatasetItem) -> None:
        nonlocal written
        try:
            clues = await _clues_for_item(cfg, llm, store, it)
        except LLMError as e:
            log.error("clue generation failed for %s: %s", it.id, e)
            return
        async with lock:
            for c in clues:
                append_jsonl(cfg.paths.clues, c.to_dict())
                written += 1

    await _gather(todo, worker, cfg.llm.max_concurrency)
    log.info("clues: wrote %d clues -> %s", written, cfg.paths.clues)
    _write_requests(cfg)


async def _clues_for_item(cfg: Config, llm: BaseLLM, store: ChunkStore,
                          it: DatasetItem) -> list[Clue]:
    gold_chunks: list[Chunk] = [c for c in (store.get_by_id(i) for i in it.gold_chunk_ids) if c]
    if not gold_chunks:
        return []
    valid_gold = set(it.gold_chunk_ids)
    obj = await llm.complete_json(CLUE_SYSTEM, clue_user(it.question, it.answer, gold_chunks))
    raw_clues = obj.get("clues", [])
    out: list[Clue] = []
    for n, rc in enumerate(raw_clues if isinstance(raw_clues, list) else []):
        text = (rc.get("clue") if isinstance(rc, dict) else str(rc)) or ""
        text = text.strip()
        if not text:
            continue
        src = [s for s in (rc.get("source_gold_ids", []) if isinstance(rc, dict) else [])
               if s in valid_gold]
        if not src:
            src = list(it.gold_chunk_ids)   # fall back to attributing to all gold
        out.append(Clue(
            clue_id=Clue.make_id(it.id, n),
            item_id=it.id,
            question=it.question,
            answer=it.answer,
            clue=text,
            source_gold_ids=src,
        ))
    if not out:  # degenerate fallback: one clue = the question itself
        out.append(Clue(Clue.make_id(it.id, 0), it.id, it.question, it.answer,
                        it.question, list(it.gold_chunk_ids)))
    return out


def _write_requests(cfg: Config) -> None:
    """Rewrite the retrieval-request file from all clues (idempotent)."""
    reqs = [
        {"clue_id": c["clue_id"], "item_id": c["item_id"],
         "query": c["clue"], "top_k": cfg.collect.top_k}
        for c in read_jsonl(cfg.paths.clues)
    ]
    n = write_jsonl(cfg.paths.retrieval_requests, reqs)
    log.info("clues: wrote %d retrieval requests -> %s", n, cfg.paths.retrieval_requests)
