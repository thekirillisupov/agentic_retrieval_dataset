"""Stage 4c — collect-all-positives validation.

Inputs:
* ``verified.jsonl``          — the questions (with minimal gold)
* ``clues.jsonl``             — atomic clues per question (from the clues stage)
* ``retrieval_results.jsonl`` — YOUR top-k passages per clue

For every returned passage an entailment judge decides whether it actually
states the clue's fact. Passages that pass (plus the original gold) become the
question's positives, grouped per clue. Output ``collected.jsonl`` augments each
item with ``positive_chunk_ids`` / ``positive_groups`` and keeps ``gold_chunk_ids``.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict

from .config import Config
from .data import ChunkStore
from .llm import BaseLLM, LLMError
from .prompts import ENTAIL_SYSTEM, entailment_user
from .schema import Clue, DatasetItem
from .utils import append_jsonl, load_done_keys, log, read_jsonl


async def collect_positives(cfg: Config, judge: BaseLLM, store: ChunkStore) -> None:
    items = {d["id"]: DatasetItem(**d) for d in read_jsonl(cfg.paths.verified)}
    clues_by_item: dict[str, list[Clue]] = defaultdict(list)
    for c in read_jsonl(cfg.paths.clues):
        clues_by_item[c["item_id"]].append(Clue(**c))
    results = _load_results(cfg)
    log.info("collect: %d items, %d clues, results for %d clues",
             len(items), sum(len(v) for v in clues_by_item.values()), len(results))

    done = load_done_keys(cfg.paths.collected, "id")
    todo = [it for it in items.values() if it.id not in done]

    lock = asyncio.Lock()
    added_total = 0

    async def worker(it: DatasetItem) -> None:
        nonlocal added_total
        try:
            n_added = await _collect_for_item(cfg, judge, store, it,
                                              clues_by_item.get(it.id, []), results)
        except LLMError as e:
            log.error("collect failed for %s: %s", it.id, e)
            # still emit the item unchanged so the dataset stays complete
            async with lock:
                append_jsonl(cfg.paths.collected, it.to_dict())
            return
        async with lock:
            append_jsonl(cfg.paths.collected, it.to_dict())
            added_total += n_added

    sem = asyncio.Semaphore(cfg.collect.judge.max_concurrency)

    async def run(it):
        async with sem:
            await worker(it)

    await asyncio.gather(*(run(it) for it in todo))
    log.info("collect: added %d extra positive chunks across %d items -> %s",
             added_total, len(todo), cfg.paths.collected)


def _load_results(cfg: Config) -> dict[str, list[dict]]:
    """clue_id -> list of returned passages."""
    out: dict[str, list[dict]] = {}
    for rec in read_jsonl(cfg.paths.retrieval_results):
        cid = rec.get("clue_id")
        if cid:
            out[cid] = rec.get("passages", []) or []
    return out


def _passage_text(store: ChunkStore, p: dict) -> tuple[str | None, str]:
    """Resolve (chunk_id, raw_text) for a returned passage, preferring the
    corpus lookup and falling back to any raw_text the caller supplied."""
    cid = p.get("chunk_id")
    if cid is None and p.get("file_name") is not None and p.get("index") is not None:
        cid = f"{p['file_name']}::{p['index']}"
    text = ""
    if cid:
        c = store.get_by_id(cid)
        if c is not None:
            text = c.raw_text
    if not text:
        text = p.get("raw_text") or ""
    return cid, text


async def _collect_for_item(cfg: Config, judge: BaseLLM, store: ChunkStore,
                            it: DatasetItem, clues: list[Clue],
                            results: dict[str, list[dict]]) -> int:
    cc = cfg.collect
    groups: list[dict] = []
    all_positive: list[str] = []

    for clue in clues:
        positives: list[str] = []
        if cc.require_original_gold:
            positives.extend(clue.source_gold_ids)

        passages = results.get(clue.clue_id, [])
        # judge each candidate passage concurrently
        checks = []
        cand_ids: list[str] = []
        for p in passages:
            cid, text = _passage_text(store, p)
            if not cid or not text:
                continue
            if cid in positives:
                continue  # already a known positive (the gold source)
            cand_ids.append(cid)
            checks.append(_supports(judge, clue.clue, text))
        verdicts = await asyncio.gather(*checks) if checks else []

        seen = set(positives)
        for cid, ok in zip(cand_ids, verdicts):
            if ok and cid not in seen:
                positives.append(cid)
                seen.add(cid)
                if cc.max_positives_per_clue and len(positives) >= cc.max_positives_per_clue:
                    break

        groups.append({
            "clue_id": clue.clue_id,
            "clue": clue.clue,
            "source_gold_ids": clue.source_gold_ids,
            "chunk_ids": positives,
        })
        for cid in positives:
            if cid not in all_positive:
                all_positive.append(cid)

    # union with original gold (covers items that had no clues/results)
    for cid in it.gold_chunk_ids:
        if cid not in all_positive:
            all_positive.append(cid)

    extra = len(all_positive) - len(set(it.gold_chunk_ids))
    it.positive_chunk_ids = all_positive
    it.positive_groups = groups
    it.num_positives = len(all_positive)
    it.judge_model = cfg.collect.judge.model or it.judge_model
    return max(0, extra)


async def _supports(judge: BaseLLM, clue: str, passage: str) -> bool:
    obj = await judge.complete_json(ENTAIL_SYSTEM, entailment_user(clue, passage))
    return bool(obj.get("supports", False))
