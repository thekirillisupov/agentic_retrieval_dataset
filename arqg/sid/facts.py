"""S3a — atomic fact extraction with verbatim spans (plan §5.1).

Facts are extracted per *chunk*, not per subgraph, and cached: a chunk that
appears in several subgraphs is paid for once. The harness check
``verbatim_span in chunk_text`` (after whitespace normalisation) is what keeps
`fact_normalized` from drifting into invention — a fact whose span cannot be
located in its source chunk is dropped, not repaired.
"""
from __future__ import annotations

import asyncio
import re

from ..llm import BaseLLM
from ..utils import append_jsonl, load_done_keys, log, read_jsonl
from .config import SidConfig
from .corpus import SidCorpus
from .prompts import FACTS_SYS, facts_user
from .schema import Fact, sid_hash

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", (text or "")).strip().lower()


def span_is_verbatim(span: str, chunk_text: str) -> bool:
    return bool(span) and _norm(span) in _norm(chunk_text)


def load_facts(path: str) -> dict[str, list[dict]]:
    """chunk_id -> facts, as written by previous runs."""
    out: dict[str, list[dict]] = {}
    for rec in read_jsonl(path):
        out.setdefault(rec["chunk_id"], []).append(rec)
    return out


async def extract_facts(cfg: SidConfig, llm: BaseLLM, corpus: SidCorpus,
                        chunk_ids: list[str]) -> dict[str, list[dict]]:
    done = load_done_keys(cfg.paths.facts, "chunk_id")
    todo = [c for c in dict.fromkeys(chunk_ids) if c not in done]
    log.info("S3a: facts for %d chunks (%d cached)", len(todo), len(done))

    async def one(cid: str) -> None:
        text = corpus.text(cid)
        if not text:
            return
        try:
            obj = await llm.complete_json(
                FACTS_SYS, facts_user(cid, text, cfg.compose.max_facts_per_chunk))
        except Exception as e:                              # noqa: BLE001
            log.warning("S3a: fact extraction failed for %s: %s", cid, e)
            return
        kept = 0
        for i, raw in enumerate(obj.get("facts", [])[: cfg.compose.max_facts_per_chunk]):
            span = str(raw.get("verbatim_span", ""))
            if cfg.compose.verbatim_match and not span_is_verbatim(span, text):
                log.debug("S3a: dropping non-verbatim span in %s", cid)
                continue
            fact = Fact(
                fact_id=f"f_{sid_hash(cid, i, span)}",
                chunk_id=cid,
                verbatim_span=span,
                fact_normalized=str(raw.get("fact_normalized", "")).strip(),
                entities=[str(e) for e in raw.get("entities", [])][:8],
                discriminating_attributes=[
                    str(a) for a in raw.get("discriminating_attributes", [])][:8],
            )
            if not fact.fact_normalized:
                continue
            append_jsonl(cfg.paths.facts, fact.to_dict())
            kept += 1
        if kept == 0:
            # mark the chunk as processed so a resume does not retry it forever
            append_jsonl(cfg.paths.facts, {"chunk_id": cid, "fact_id": "", "empty": True})

    await asyncio.gather(*(one(c) for c in todo))
    facts = {cid: [f for f in rows if f.get("fact_id")]
             for cid, rows in load_facts(cfg.paths.facts).items()}
    n = sum(len(v) for v in facts.values())
    log.info("S3a: %d facts over %d chunks", n, len([k for k, v in facts.items() if v]))
    return facts
