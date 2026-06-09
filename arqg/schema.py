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
    num_gold: int
    window_chunk_ids: list[str]
    verification: dict[str, Any] = field(default_factory=dict)
    hard_negative_ids: list[str] = field(default_factory=list)
    generation_model: str = ""
    judge_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
