"""Stage 2 — generate candidate questions from windows (async, resumable)."""
from __future__ import annotations

import asyncio

from .config import Config
from .llm import BaseLLM, LLMError
from .prompts import GEN_SYSTEM, gen_user
from .schema import Candidate, Chunk, Window
from .utils import append_jsonl, load_done_keys, log, read_jsonl


async def generate(cfg: Config, llm: BaseLLM) -> None:
    windows = [Window(**w) for w in read_jsonl(cfg.paths.windows)]
    done = load_done_keys(cfg.paths.candidates, "window_id")
    todo = [w for w in windows if w.window_id not in done]
    log.info("generate: %d windows total, %d already done, %d to do",
             len(windows), len(done), len(todo))

    lock = asyncio.Lock()
    written = 0

    async def worker(w: Window) -> None:
        nonlocal written
        try:
            cands = await _generate_for_window(cfg, llm, w)
        except LLMError as e:
            log.error("generation failed for %s: %s", w.window_id, e)
            return
        async with lock:
            for c in cands:
                append_jsonl(cfg.paths.candidates, c.to_dict())
                written += 1

    await _gather(todo, worker, cfg.llm.max_concurrency)
    log.info("generate: wrote %d candidates -> %s", written, cfg.paths.candidates)


async def _generate_for_window(cfg: Config, llm: BaseLLM, w: Window) -> list[Candidate]:
    chunks = [Chunk(w.file_name, idx, txt) for idx, txt in zip(w.indices, w.texts)]
    valid_ids = set(w.chunk_ids)
    out: list[Candidate] = []
    for n in range(cfg.generate.questions_per_window):
        obj = await llm.complete_json(GEN_SYSTEM, gen_user(chunks))
        req = _clean_ids(obj.get("required_chunk_ids", []), valid_ids)
        question = (obj.get("question") or "").strip()
        answer = (obj.get("answer") or "").strip()
        if not question or not answer:
            continue
        if cfg.generate.require_multi_chunk and len(req) < cfg.generate.min_gold_chunks:
            # generator failed the core requirement; skip (verification can't fix this)
            log.debug("skipping single-chunk candidate for %s", w.window_id)
            continue
        out.append(Candidate(
            candidate_id=Candidate.make_id(w.window_id, n),
            window_id=w.window_id,
            file_name=w.file_name,
            window_chunk_ids=w.chunk_ids,
            question=question,
            answer=answer,
            required_chunk_ids=req,
            question_type=str(obj.get("question_type", "multi_hop")),
            reasoning=str(obj.get("reasoning", "")),
            generation_model=cfg.llm.model,
            raw=obj,
        ))
    return out


def _clean_ids(ids, valid: set[str]) -> list[str]:
    seen: list[str] = []
    for i in ids if isinstance(ids, list) else []:
        i = str(i).strip()
        if i in valid and i not in seen:
            seen.append(i)
    return seen


async def _gather(items, worker, concurrency: int) -> None:
    """Bounded concurrency without holding all tasks resident at once."""
    sem = asyncio.Semaphore(concurrency)

    async def run(x):
        async with sem:
            await worker(x)

    await asyncio.gather(*(run(x) for x in items))
