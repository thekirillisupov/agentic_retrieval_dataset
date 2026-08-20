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
    # the `title` folder the chunks were mined within ("" = mined globally),
    # and how many leading path segments they actually share. `path_shared_depth
    # == 1` means "nothing but the corpus root" — the pathology scoping exists
    # to remove, kept as a measurable rather than an assumption.
    path_scope: str = ""
    path_shared_depth: int = 0
    # which channel found this subgraph: "entity" (a shared rare surface form)
    # or "similarity" (doc2doc, see simbridge.py). A similarity subgraph may
    # still list `bridge_entities` — anything its chunks happen to share — but
    # nothing guaranteed it, which is why S3 re-checks the anchor.
    bridge_kind: str = "entity"
    pair_similarity: float = 0.0

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
    # breadcrumb of the source chunk. Carried on the fact rather than looked up
    # per stage: the fact is what actually travels (facts.jsonl → candidate →
    # gate repair → task), so the composer and the repair loop get the section
    # without every one of them having to hold the corpus.
    section: str = ""
    # ... and its facet header, for the same reason. Refreshed from the corpus
    # on every load (see facts.attach_context), because unlike the fact itself
    # it is a *view* of the chunk: turning `facets.fields` on must not require
    # re-extracting a cache that is already correct.
    facets: str = ""

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
    bridge_kind: str = "entity"         # which S1 channel produced the subgraph
    # S3c (completeness.py): the structured filter the question's constraints
    # declare over the facet metadata, the field its answer projects (when the
    # answer enumerates one facet), and the completeness verdict. Empty on
    # mechanics outside `completeness.mechanics` and on corpora without facets.
    filter: list[dict[str, Any]] = field(default_factory=list)
    answer_field: str = ""
    completeness: dict[str, Any] = field(default_factory=dict)
    # disambiguation_first only: the ambiguous description the question opens
    # with, exactly as it is phrased there. G_AMBIG probes the index with it
    # and demands >= 2 plausible referents — a descriptor matching one document
    # is a paraphrase, not an ambiguity (gates.py: ambiguity_check).
    descriptor: str = ""
    # verbatim_lookup only: the exact identifier the question carries. G_VERBATIM
    # demands it verbatim in the question AND in a gold passage, with the entry
    # chunk lexically easy and dense-hard (gates.py: verbatim_check).
    identifier: str = ""

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
