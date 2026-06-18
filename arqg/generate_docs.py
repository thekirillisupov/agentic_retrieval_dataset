"""Stage 2b — generate simple/hard questions over document units.

A SECOND, separately-configured generation process (config block ``docgen``).
For each document unit it samples a difficulty:

* simple → one passage holds the whole answer (gold = 1)
* hard   → answer requires many passages across the document (gold >= 2, no cap)

Candidates are appended to the SAME ``candidates.jsonl`` as the neighbour
generator and carry a per-item verification policy, so the unchanged ``verify``
stage handles both and the final output keeps the identical ``gold_chunk_ids``
format.
"""
from __future__ import annotations

import asyncio

from .config import Config
from .generate import _clean_ids, _gather, sample_weighted
from .llm import BaseLLM, LLMError
from .prompts import DOC_GEN_SYSTEM, STYLES, doc_gen_user
from .schema import Candidate, Chunk, Window
from .utils import append_jsonl, load_done_keys, log, read_jsonl

PROFILE = "doc_simple_hard"


async def generate_docs(cfg: Config, llm: BaseLLM) -> None:
    units = [Window(**w) for w in read_jsonl(cfg.paths.docunits)]
    done = load_done_keys(cfg.paths.candidates, "window_id")
    todo = [u for u in units if u.window_id not in done]
    log.info("gen-docs: %d units total, %d already done, %d to do",
             len(units), len(done), len(todo))

    lock = asyncio.Lock()
    written = 0

    async def worker(u: Window) -> None:
        nonlocal written
        try:
            cands = await _generate_for_unit(cfg, llm, u)
        except LLMError as e:
            log.error("gen-docs failed for %s: %s", u.window_id, e)
            return
        async with lock:
            for c in cands:
                append_jsonl(cfg.paths.candidates, c.to_dict())
                written += 1

    await _gather(todo, worker, cfg.llm.max_concurrency)
    log.info("gen-docs: wrote %d candidates -> %s", written, cfg.paths.candidates)


def _sample_difficulty(cfg: Config, unit_id: str, n: int, single_chunk: bool) -> str:
    if single_chunk:
        return "simple"   # a one-chunk unit can only support a simple question
    dg = cfg.docgen
    return sample_weighted(dg.difficulty_weights, f"diff|{unit_id}|{n}", "hard")


def _sample_style(cfg: Config, unit_id: str, n: int) -> str:
    dg = cfg.docgen
    styles = {k: v for k, v in dg.styles.items() if k in STYLES}
    return sample_weighted(styles, f"{dg.style_seed}|{unit_id}|{n}", "simple_user")


async def _generate_for_unit(cfg: Config, llm: BaseLLM, u: Window) -> list[Candidate]:
    dg = cfg.docgen
    chunks = [Chunk(u.file_name, idx, txt) for idx, txt in zip(u.indices, u.texts)]
    valid_ids = set(u.chunk_ids)
    single_chunk_unit = len(u.chunk_ids) < 2
    out: list[Candidate] = []

    for n in range(dg.questions_per_unit):
        difficulty = _sample_difficulty(cfg, u.window_id, n, single_chunk_unit)
        style = _sample_style(cfg, u.window_id, n)
        obj = await llm.complete_json(DOC_GEN_SYSTEM, doc_gen_user(chunks, difficulty, style))
        req = _clean_ids(obj.get("required_chunk_ids", []), valid_ids)
        question = (obj.get("question") or "").strip()
        answer = (obj.get("answer") or "").strip()
        if not question or not answer:
            continue

        if difficulty == "simple":
            if not req:
                continue
            req = req[: max(1, dg.simple_max_gold)]   # simple rests on few passages
            min_gold, enforce_multi, run_min = 1, False, False
        else:  # hard
            if len(req) < dg.hard_min_gold:
                log.debug("gen-docs: hard question under-anchored for %s", u.window_id)
                continue
            if dg.hard_max_gold and len(req) > dg.hard_max_gold:
                req = req[: dg.hard_max_gold]
            min_gold, enforce_multi, run_min = dg.hard_min_gold, True, True

        out.append(Candidate(
            candidate_id=Candidate.make_id(u.window_id, n),
            window_id=u.window_id,
            file_name=u.file_name,
            window_chunk_ids=u.chunk_ids,
            question=question,
            answer=answer,
            required_chunk_ids=req,
            question_type=str(obj.get("question_type", "factoid" if difficulty == "simple" else "multi_hop")),
            question_style=style,
            profile=PROFILE,
            difficulty=difficulty,
            min_gold=min_gold,
            enforce_multi_chunk=enforce_multi,
            run_minimality=run_min,
            reasoning=str(obj.get("reasoning", "")),
            generation_model=cfg.llm.model,
            raw=obj,
        ))
    return out
