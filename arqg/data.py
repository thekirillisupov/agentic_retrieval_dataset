"""Chunk loading, neighbour-indexed store, and seed-eligibility filtering."""
from __future__ import annotations

import json
import os
import re
from typing import Iterator

from .config import FilterConfig
from .schema import Chunk, chunk_id
from .utils import log

_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")

#: The pipeline's canonical five fields; anything else on a record is a
#: metadata facet and lands in ``Chunk.meta`` instead (see ``schema.Chunk``).
_CORE_FIELDS = {"file_name", "index", "raw_text", "document_id", "title"}


def load_chunks(path: str) -> list[Chunk]:
    """Load chunks from a .jsonl file, a .json array, or a directory of either.

    Expected record shape: {"file_name", "index", "raw_text"}. Any additional
    keys on a record (a corpus that inlines its facets rather than keeping
    them in a separate sidecar) are kept on ``Chunk.meta``.
    """
    chunks: list[Chunk] = []
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            chunks.extend(load_chunks(os.path.join(path, name)))
        return chunks

    for rec in _iter_records(path):
        try:
            chunks.append(
                Chunk(
                    file_name=str(rec["file_name"]),
                    index=int(rec["index"]),
                    raw_text=rec["raw_text"] if rec["raw_text"] is not None else "",
                    document_id=str(rec.get("document_id", "") or ""),
                    title=str(rec.get("title", "") or ""),
                    meta={k: v for k, v in rec.items() if k not in _CORE_FIELDS},
                )
            )
        except (KeyError, TypeError, ValueError) as e:
            log.warning("skipping malformed chunk record: %s (%s)", rec, e)
    log.info("loaded %d chunks from %s", len(chunks), path)
    return chunks


def _iter_records(path: str) -> Iterator[dict]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        yield from (data if isinstance(data, list) else [data])
    else:
        # Unknown extension: sniff first non-space char ('[' => json array, else jsonl)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        stripped = content.lstrip()
        if stripped.startswith("["):
            data = json.loads(stripped)
            yield from (data if isinstance(data, list) else [data])
        else:
            for line in content.splitlines():
                line = line.strip()
                if line:
                    yield json.loads(line)


def alpha_ratio(text: str) -> float:
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return 0.0
    letters = sum(1 for c in non_space if _LETTER_RE.match(c))
    return letters / len(non_space)


def cyrillic_ratio(text: str) -> float:
    letters = _LETTER_RE.findall(text)
    if not letters:
        return 0.0
    cyr = len(_CYRILLIC_RE.findall(text))
    return cyr / len(letters)


class ChunkStore:
    """Indexes chunks by composite id and by (file_name -> ordered indices),
    so neighbours and contiguous windows are O(1)/O(k) to fetch."""

    def __init__(self, chunks: list[Chunk]):
        self._by_id: dict[str, Chunk] = {}
        self._by_file: dict[str, dict[int, Chunk]] = {}
        for c in chunks:
            cid = c.id
            if cid in self._by_id:
                log.warning("duplicate chunk id %s — keeping first", cid)
                continue
            self._by_id[cid] = c
            self._by_file.setdefault(c.file_name, {})[c.index] = c
        # cache sorted index lists per file
        self._sorted_indices: dict[str, list[int]] = {
            fn: sorted(idx_map) for fn, idx_map in self._by_file.items()
        }

    def __len__(self) -> int:
        return len(self._by_id)

    @property
    def files(self) -> list[str]:
        return list(self._sorted_indices.keys())

    def get(self, file_name: str, index: int) -> Chunk | None:
        return self._by_file.get(file_name, {}).get(index)

    def get_by_id(self, cid: str) -> Chunk | None:
        return self._by_id.get(cid)

    def all_chunks(self) -> list[Chunk]:
        return list(self._by_id.values())

    def file_indices(self, file_name: str) -> list[int]:
        return self._sorted_indices.get(file_name, [])

    def contiguous_window(self, file_name: str, start_index: int, size: int) -> list[Chunk] | None:
        """Return ``size`` chunks at consecutive indices starting at
        ``start_index``, or None if any are missing (a true gap in the doc)."""
        out: list[Chunk] = []
        for i in range(start_index, start_index + size):
            c = self.get(file_name, i)
            if c is None:
                return None
            out.append(c)
        return out


def is_eligible_seed(chunk: Chunk, cfg: FilterConfig) -> bool:
    """Whether a chunk is suitable as the *seed* of a generation window.

    Rejects boilerplate (too short/long, mostly digits or symbols, not enough
    Russian text). Rejected chunks remain usable as neighbours for context.
    """
    n = chunk.n_chars
    if n < cfg.min_chars or n > cfg.max_chars:
        return False
    if len(chunk.raw_text.split()) < cfg.min_words:
        return False
    if alpha_ratio(chunk.raw_text) < cfg.min_alpha_ratio:
        return False
    if cyrillic_ratio(chunk.raw_text) < cfg.min_cyrillic_ratio:
        return False
    return True
