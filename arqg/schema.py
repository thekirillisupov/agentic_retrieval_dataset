"""Dataclasses describing the records that flow between pipeline stages.

Everything is serialised to JSONL between stages so each stage is independently
runnable, resumable and debuggable.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any


def chunk_id(file_name: str, index: int) -> str:
    """Stable, human-readable id for a chunk.

    We keep ``file_name`` and ``index`` because the caller's knowledge base is
    addressed that way; the composite id is only a convenience handle.
    """
    return f"{file_name}::{index}"


@dataclass
class Chunk:
    file_name: str
    index: int
    raw_text: str
    document_id: str = ""   # logical document the passage belongs to
    title: str = ""         # document/passage title (used to enrich embeddings)
    # Everything else a corpus knows about this chunk but the five fields above
    # do not carry — a categorical facet (region, customer, ОКПД2 code, ...),
    # loaded either inline from the corpus record or merged in from a metadata
    # sidecar keyed by chunk id (see arqg/sid/corpus.py). Never written to the
    # agent-visible export; SID's S1 reads it to group chunks for mining (see
    # arqg/sid/scoping.py) and S0 reports what it found (see compat.py).
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return chunk_id(self.file_name, self.index)

    @property
    def n_chars(self) -> int:
        return len(self.raw_text)


@dataclass
class Window:
    """A contiguous run of chunks from a single document.

    The window is the *generation context*; only a subset of it (the verified
    necessary chunks) ends up as gold.
    """
    window_id: str
    file_name: str
    indices: list[int]           # contiguous, ascending
    chunk_ids: list[str]
    texts: list[str]
    n_chars: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def make_id(file_name: str, indices: list[int]) -> str:
        h = hashlib.sha1(f"{file_name}|{indices}".encode("utf-8")).hexdigest()[:12]
        return f"w_{h}"


@dataclass
class Candidate:
    """A raw generated question before verification."""
    candidate_id: str
    window_id: str
    file_name: str
    window_chunk_ids: list[str]
    question: str
    answer: str
    required_chunk_ids: list[str]      # model's claim of needed chunks
    question_type: str
    question_style: str = "simple_user"
    # --- generation provenance + per-candidate verification policy --------- #
    # These let one verify stage handle candidates from different generation
    # processes (neighbour-window multi-hop vs document simple/hard).
    profile: str = "neighbor_multihop"   # which generator produced this
    difficulty: str = "hard"             # "simple" | "hard"
    min_gold: int = 2                    # required minimum gold chunks after verify
    enforce_multi_chunk: bool = True     # drop if a single chunk alone suffices
    run_minimality: bool = True          # run the gold-set minimisation judge
    reasoning: str = ""
    generation_model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def make_id(window_id: str, n: int) -> str:
        return f"{window_id}__c{n}"


@dataclass
class DatasetItem:
    """A verified, final dataset record."""
    id: str
    question: str
    answer: str
    gold_chunk_ids: list[str]          # minimal, verified-necessary set
    file_name: str
    question_type: str
    question_style: str
    num_gold: int
    window_chunk_ids: list[str]
    profile: str = "neighbor_multihop"
    difficulty: str = "hard"
    verification: dict[str, Any] = field(default_factory=dict)
    hard_negative_ids: list[str] = field(default_factory=list)
    # collect-all-positives: every corpus chunk validated to satisfy a clue,
    # so near-duplicate sources of the same fact count as relevant, not as
    # false positives. positive_chunk_ids ⊇ gold_chunk_ids once collected.
    positive_chunk_ids: list[str] = field(default_factory=list)
    positive_groups: list[dict[str, Any]] = field(default_factory=list)
    num_positives: int = 0
    generation_model: str = ""
    judge_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Clue:
    """An atomic, self-contained fact that the question requires. Used both as a
    retrieval query (you return top-k passages for it) and as the entailment
    target when validating which passages count as positives."""
    clue_id: str
    item_id: str
    question: str
    answer: str
    clue: str
    source_gold_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def make_id(item_id: str, n: int) -> str:
        return f"{item_id}__k{n}"
