"""Build document-level generation units for the simple/hard generator.

Unlike neighbour windows (a tight run of 2-4 chunks), a *doc unit* spans a whole
document so the generator can either pick one passage (simple) or combine many
across the document (hard). Very large documents are split into a few big
consecutive spans so a unit stays within token/cost limits.

A unit reuses the ``Window`` schema (it's just "a set of chunks with ids"); its
id is prefixed ``d_`` to avoid colliding with neighbour-window ids in the shared
candidates file.
"""
from __future__ import annotations

import hashlib
import random

from .config import DocGenConfig, FilterConfig
from .data import ChunkStore, is_eligible_seed
from .schema import Chunk, Window
from .utils import log


def _unit_id(file_name: str, indices: list[int]) -> str:
    h = hashlib.sha1(f"{file_name}|{indices}".encode("utf-8")).hexdigest()[:12]
    return f"d_{h}"


def build_doc_units(store: ChunkStore, dgcfg: DocGenConfig, fcfg: FilterConfig) -> list[Window]:
    ucfg = dgcfg.units
    rng = random.Random(ucfg.seed)
    files = store.files
    rng.shuffle(files)

    units: list[Window] = []
    for file_name in files:
        indices = store.file_indices(file_name)
        chunks = [store.get(file_name, i) for i in indices]
        spans = _split_into_spans(chunks, ucfg.max_doc_chars, ucfg.max_doc_chunks)
        kept = 0
        for span in spans:
            if kept >= ucfg.max_units_per_file:
                break
            if len(span) < ucfg.min_unit_chunks:
                continue
            # require at least one eligible (non-junk) chunk to anchor a question
            if not any(is_eligible_seed(c, fcfg) for c in span):
                continue
            units.append(_to_unit(file_name, span))
            kept += 1

    rng.shuffle(units)
    if ucfg.target_units and len(units) > ucfg.target_units:
        units = units[: ucfg.target_units]
    log.info("built %d document units from %d files", len(units), len(files))
    return units


def _split_into_spans(chunks: list[Chunk], max_chars: int, max_chunks: int) -> list[list[Chunk]]:
    """Greedily pack consecutive chunks into spans bounded by chars and count.
    A document that fits the caps yields a single span (the whole document)."""
    spans: list[list[Chunk]] = []
    cur: list[Chunk] = []
    cur_chars = 0
    for c in chunks:
        if cur and (cur_chars + c.n_chars > max_chars or len(cur) >= max_chunks):
            spans.append(cur)
            cur, cur_chars = [], 0
        cur.append(c)
        cur_chars += c.n_chars
    if cur:
        spans.append(cur)
    return spans


def _to_unit(file_name: str, chunks: list[Chunk]) -> Window:
    indices = [c.index for c in chunks]
    return Window(
        window_id=_unit_id(file_name, indices),
        file_name=file_name,
        indices=indices,
        chunk_ids=[c.id for c in chunks],
        texts=[c.raw_text for c in chunks],
        n_chars=sum(c.n_chars for c in chunks),
    )
