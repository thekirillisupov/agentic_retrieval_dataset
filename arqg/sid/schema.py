"""Records that flow between SID stages (all JSONL-serialised)."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


def sid_hash(*parts: Any, n: int = 10) -> str:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()[:n]


@dataclass
class BridgeEntity:
    surface: str
    type: str
    source: str          # "ner" (heuristic extractor) | "index_tag"
    idf: float
    chunks: list[str]


@dataclass
class Subgraph:
    """S1 output — 2–5 chunks tied together by rare shared entities."""
    subgraph_id: str
    corpus: str
    index_version: str
    chunks: list[str]
    bridge_entities: list[dict[str, Any]]
    index_tags: dict[str, Any] = field(default_factory=dict)
    hop_depth_potential: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Fact:
    """S3 atomic fact with a verbatim span into its source chunk (plan §5.1)."""
    fact_id: str
    chunk_id: str
    verbatim_span: str
    fact_normalized: str
    entities: list[str] = field(default_factory=list)
    discriminating_attributes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Candidate:
    """S3 output — a composed question before any gate has run."""
    candidate_id: str
    batch_id: str                       # 1-of-N group (plan §4.5)
    instantiation_rank: int
    subgraph_id: str
    corpus: str
    language: str
    question: str
    answer: str
    facts: list[dict[str, Any]]         # the Facts the question rests on
    mechanic: str
    submechanic: str
    has_negation: bool
    hop_depth: int
    compose_iters: int = 1
    generator_model: str = ""
    reasoning: str = ""

    @property
    def chunk_ids(self) -> list[str]:
        seen: list[str] = []
        for f in self.facts:
            if f["chunk_id"] not in seen:
                seen.append(f["chunk_id"])
        return seen

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["chunk_ids"] = self.chunk_ids
        return d


def new_task_id(corpus: str, language: str, n: int) -> str:
    return f"syn_{language}_{corpus}_{n:06d}"
