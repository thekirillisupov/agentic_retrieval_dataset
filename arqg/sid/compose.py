"""S3b — question composition with 1-of-N local diversity (plan §4.5, §5.2).

For each coverage cell we instantiate ``N`` candidates from **different
subgraphs** and with **different submechanics**, and only the best 1–2 survive
the cheap gates. Without that, one mechanic degenerates into the same template
with names swapped.

The plan builds each question incrementally (base → +hop → +constraint) with a
solvability check per step. v1 composes in one call and repairs against gate
feedback up to ``max_compose_iters`` — same corrective signal, a third of the
calls. The repair loop lives in ``gates.py`` where the feedback is produced.
"""
from __future__ import annotations

import asyncio
from typing import Iterable

from ..llm import BaseLLM
from ..utils import append_jsonl, load_done_keys, log, read_jsonl
from .config import SidConfig
from .prompts import COMPOSE_SYS, compose_user
from .schema import Candidate, sid_hash
from .taxonomy import Cell, CellSampler


def _facts_for_subgraph(subgraph: dict, facts_by_chunk: dict[str, list[dict]],
                        cap: int) -> list[dict]:
    """Facts from the subgraph's chunks, round-robin so no single chunk floods
    the prompt and every chunk gets a chance to contribute."""
    pools = [list(facts_by_chunk.get(cid, ())) for cid in subgraph["chunks"]]
    out: list[dict] = []
    i = 0
    while len(out) < cap and any(pools):
        pool = pools[i % len(pools)]
        if pool:
            out.append(pool.pop(0))
        i += 1
        if i > cap * len(pools) + len(pools):
            break
    return out


def plan_batches(cfg: SidConfig, subgraphs: list[dict],
                 facts_by_chunk: dict[str, list[dict]]) -> list[tuple[str, list[tuple[Cell, dict]]]]:
    """Group subgraphs into 1-of-N batches, one coverage cell per batch."""
    usable = [s for s in subgraphs
              if sum(1 for c in s["chunks"] if facts_by_chunk.get(c)) >= 2]
    log.info("S3b: %d/%d subgraphs have facts in >= 2 chunks",
             len(usable), len(subgraphs))
    sampler = CellSampler(cfg)
    n = max(1, cfg.taxonomy.candidates_per_cell)
    batches: list[tuple[str, list[tuple[Cell, dict]]]] = []
    for start in range(0, len(usable), n):
        group = usable[start:start + n]
        mechanic = sampler.next_mechanic()
        cells = sampler.batch(mechanic, len(group))
        batch_id = f"b_{sid_hash(mechanic, [g['subgraph_id'] for g in group])}"
        batches.append((batch_id, list(zip(cells, group))))
    return batches


async def compose_candidates(cfg: SidConfig, llm: BaseLLM, subgraphs: list[dict],
                             facts_by_chunk: dict[str, list[dict]]) -> list[Candidate]:
    done = load_done_keys(cfg.paths.candidates, "candidate_id")
    batches = plan_batches(cfg, subgraphs, facts_by_chunk)
    jobs: list[tuple[str, int, Cell, dict, list[dict]]] = []
    for batch_id, members in batches:
        for rank, (cell, subgraph) in enumerate(members):
            cid = f"c_{sid_hash(batch_id, subgraph['subgraph_id'], cell.submechanic)}"
            if cid in done:
                continue
            facts = _facts_for_subgraph(subgraph, facts_by_chunk,
                                        cfg.compose.max_facts_per_question + 3)
            if len({f["chunk_id"] for f in facts}) < 2:
                continue
            jobs.append((batch_id, rank, cell, subgraph, facts))

    log.info("S3b: composing %d candidates in %d batches", len(jobs), len(batches))

    async def one(job) -> None:
        batch_id, rank, cell, subgraph, facts = job
        cand = await compose_one(cfg, llm, batch_id, rank, cell, subgraph, facts)
        if cand is not None:
            append_jsonl(cfg.paths.candidates, cand.to_dict())

    await asyncio.gather(*(one(j) for j in jobs))
    out = [r for r in read_jsonl(cfg.paths.candidates)]
    log.info("S3b: %d candidates on disk -> %s", len(out), cfg.paths.candidates)
    return out


async def compose_one(cfg: SidConfig, llm: BaseLLM, batch_id: str, rank: int,
                      cell: Cell, subgraph: dict, facts: list[dict],
                      feedback: str = "", iters: int = 1) -> Candidate | None:
    try:
        obj = await llm.complete_json(COMPOSE_SYS, compose_user(cell, facts, feedback))
    except Exception as e:                                   # noqa: BLE001
        log.warning("S3b: compose failed for %s: %s", subgraph["subgraph_id"], e)
        return None

    question = str(obj.get("question", "")).strip()
    answer = str(obj.get("answer", "")).strip()
    used_ids = [str(i) for i in obj.get("used_fact_ids", [])]
    by_id = {f["fact_id"]: f for f in facts}
    used = [by_id[i] for i in used_ids if i in by_id][: cfg.compose.max_facts_per_question]

    if not question or not answer:
        return None
    if len(used) < cfg.compose.min_facts_per_question:
        log.debug("S3b: candidate rejected — %d facts referenced", len(used))
        return None
    if len({f["chunk_id"] for f in used}) < 2:
        log.debug("S3b: candidate rejected — single-chunk question")
        return None

    return Candidate(
        candidate_id=f"c_{sid_hash(batch_id, subgraph['subgraph_id'], cell.submechanic)}",
        batch_id=batch_id,
        instantiation_rank=rank,
        subgraph_id=subgraph["subgraph_id"],
        corpus=cfg.corpus_name,
        language=cfg.language,
        question=question,
        answer=answer,
        facts=used,
        mechanic=cell.mechanic,
        submechanic=cell.submechanic,
        has_negation=cell.has_negation,
        hop_depth=len(used),          # by construction; recomputed after G_MIN
        compose_iters=iters,
        generator_model=cfg.llm.model,
        reasoning=str(obj.get("reasoning", ""))[:500],
    )


def candidates_from_dicts(rows: Iterable[dict]) -> list[Candidate]:
    out = []
    for r in rows:
        r = dict(r)
        r.pop("chunk_ids", None)
        out.append(Candidate(**r))
    return out
