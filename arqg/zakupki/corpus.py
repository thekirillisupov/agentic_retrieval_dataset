"""Sections -> pipeline chunks, plus the near-duplicate report this source is for.

Chunking follows the rule the rest of the pipeline assumes (``arqg.windows``):
neighbours are ``index ± 1`` within one ``file_name``, so a document's sections
must be emitted in document order and never interleaved. Short adjacent sections
are merged (otherwise a notification becomes twenty one-line chunks and no window
carries a real fact), long ones are split on line boundaries with the section
header repeated so a mid-document chunk still says what it is about.

:func:`duplicate_report` measures the property that makes ЕИС worth ingesting.
Two notions of duplicate:

* **exact** — identical text after whitespace/case normalisation. Boilerplate
  that repeats verbatim (стандартные требования к участникам, условия
  обеспечения). Genuinely indistinguishable passages, and a retriever cannot be
  blamed for confusing them.
* **structural** — identical once every digit, date and amount is masked. This is
  the interesting bucket: same template, *different* customer, deadline and
  price. Discriminating between those is precisely the skill the dataset trains,
  and the share of chunks living in a structural group of size > 1 is the number
  to look at when deciding whether a slice of ЕИС is worth generating over.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Any, Iterable, Iterator

from ..utils import ensure_parent, log, write_jsonl
from .parse import ProcurementDoc, Section

_DIGITS_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass
class ChunkOptions:
    max_chars: int = 1400        # hard cap on a chunk
    merge_below: int = 600       # sections shorter than this may absorb a neighbour
    min_chars: int = 120         # drop chunks with less signal than this
    max_chunks_per_doc: int = 40
    min_chunks: int = 2          # never merge a document below this, if it can be helped


def _normalise(text: str) -> str:
    return _WS_RE.sub(" ", text.strip().lower())


def _content_hash(text: str) -> str:
    return hashlib.sha1(_normalise(text).encode("utf-8")).hexdigest()


def _template_hash(text: str) -> str:
    """Hash of the text with every number masked: the document's *shape*."""
    return hashlib.sha1(_DIGITS_RE.sub("#", _normalise(text)).encode("utf-8")).hexdigest()


def _shingles(text: str, n: int = 5) -> set[str]:
    tokens = _TOKEN_RE.findall(_normalise(text))
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: str, b: str) -> float:
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #
def _split_section(sec: Section, opts: ChunkOptions) -> list[str]:
    """One section -> one or more chunk texts, split on line boundaries."""
    full = sec.text()
    if len(full) <= opts.max_chars:
        return [full]
    parts: list[str] = []
    buf: list[str] = []
    size = len(sec.title)
    part_no = 1
    for line in sec.lines:
        if buf and size + len(line) + 1 > opts.max_chars:
            head = sec.title if part_no == 1 else f"{sec.title} (продолжение {part_no})"
            parts.append("\n".join([head] + buf))
            part_no += 1
            buf, size = [], len(sec.title) + 20
        buf.append(line)
        size += len(line) + 1
    if buf:
        head = sec.title if part_no == 1 else f"{sec.title} (продолжение {part_no})"
        parts.append("\n".join([head] + buf))
    return parts


def chunk_document(doc: ProcurementDoc, opts: ChunkOptions | None = None) -> list[str]:
    """Chunk texts for one document, in document order.

    Merging is capped by ``min_chunks``: a document that *can* yield a neighbour
    window must not lose that ability to tidier chunking. If the merged result
    falls below the floor, the unmerged one is used instead.
    """
    opts = opts or ChunkOptions()
    texts = _chunk(doc, opts)
    if len(texts) < opts.min_chunks and opts.merge_below > 0:
        unmerged = _chunk(doc, replace(opts, merge_below=0))
        if len(unmerged) > len(texts):
            return unmerged
    return texts


def _chunk(doc: ProcurementDoc, opts: ChunkOptions) -> list[str]:
    texts: list[str] = []
    pending: Section | None = None

    for sec in doc.sections:
        if pending is not None:
            if pending.n_chars < opts.merge_below and \
                    pending.n_chars + sec.n_chars <= opts.max_chars:
                pending = Section(pending.title,
                                  pending.lines + ["", sec.title] + sec.lines)
                continue
            texts.extend(_split_section(pending, opts))
        pending = sec
    if pending is not None:
        texts.extend(_split_section(pending, opts))

    texts = [t for t in texts if len(t) >= opts.min_chars]

    # Forward merging cannot help the *last* section: there is nothing after it
    # to absorb. Left alone, a one-line closing section («Этап определения
    # поставщика — признана несостоявшейся.») becomes a chunk thousands of
    # documents share verbatim, which can never serve as anyone's gold passage.
    # Fold it backwards instead — but never down to a single chunk, or the
    # document stops being able to form a neighbour window at all.
    if len(texts) > 2 and len(texts[-1]) < opts.merge_below \
            and len(texts[-2]) + len(texts[-1]) + 2 <= opts.max_chars:
        texts[-2:] = ["\n\n".join(texts[-2:])]

    if len(texts) > opts.max_chunks_per_doc:
        log.debug("zakupki: %s truncated %d -> %d chunks",
                  doc.file_name, len(texts), opts.max_chunks_per_doc)
        texts = texts[:opts.max_chunks_per_doc]
    return texts


def iter_chunk_records(docs: Iterable[ProcurementDoc],
                       opts: ChunkOptions | None = None) -> Iterator[dict[str, Any]]:
    """Chunk records in the pipeline's input format.

    Documents that collide on ``file_name`` (the same procurement fetched twice,
    e.g. by two overlapping date windows) are emitted once — a duplicated
    ``{file_name, index}`` pair would silently overwrite a chunk downstream.
    """
    seen_files: set[str] = set()
    dropped = 0
    for doc in docs:
        if doc.file_name in seen_files:
            dropped += 1
            log.debug("zakupki: %s already emitted, skipping duplicate", doc.file_name)
            continue
        seen_files.add(doc.file_name)
        for index, text in enumerate(chunk_document(doc, opts)):
            yield {
                "file_name": doc.file_name,
                "index": index,
                "raw_text": text,
                "document_id": doc.doc_id,
                "title": doc.title,
            }
    if dropped:
        log.warning("zakupki: %d document(s) dropped as duplicate file names — "
                    "a broken id column collapses distinct rows onto one document",
                    dropped)


def build_corpus(docs: Iterable[ProcurementDoc], corpus_path: str, *,
                 opts: ChunkOptions | None = None,
                 docs_path: str = "") -> dict[str, Any]:
    """Write the corpus JSONL (and optionally a document-level dump). Returns stats."""
    records: list[dict[str, Any]] = []
    doc_rows: list[dict[str, Any]] = []
    doc_types: Counter[str] = Counter()

    materialised = list(docs)
    for doc in materialised:
        doc_types[doc.doc_type] += 1
        if docs_path:
            doc_rows.append(doc.to_dict())
    records = list(iter_chunk_records(materialised, opts))

    n = write_jsonl(corpus_path, records)
    log.info("zakupki: wrote %d chunks over %d documents -> %s",
             n, len(materialised), corpus_path)
    if docs_path:
        write_jsonl(docs_path, doc_rows)
        log.info("zakupki: wrote %d documents -> %s", len(doc_rows), docs_path)

    return {
        "n_documents_parsed": len(materialised),
        "n_files": len({r["file_name"] for r in records}),
        "n_chunks": n,
        "n_chars": sum(len(r["raw_text"]) for r in records),
        "mean_chunk_chars": round(
            sum(len(r["raw_text"]) for r in records) / n, 1) if n else 0.0,
        "mean_chunks_per_doc": round(n / len(materialised), 2) if materialised else 0.0,
        "document_types": dict(doc_types.most_common()),
    }


# --------------------------------------------------------------------------- #
# near-duplicate report
# --------------------------------------------------------------------------- #
#: How many distinct hashes may keep a text sample. Counts are exact regardless;
#: this only bounds the memory spent on examples and the similarity estimate, and
#: a merged ЕИС corpus is large enough that holding a sample per group would cost
#: more than the corpus itself.
MAX_TRACKED_GROUPS = 50_000
SAMPLE_CHARS = 400


class _DuplicateCounter:
    """Streaming duplicate statistics: exact counts, sampled examples."""

    def __init__(self, hasher, *, max_tracked: int = MAX_TRACKED_GROUPS):
        self.hasher = hasher
        self.max_tracked = max_tracked
        self.counts: dict[str, int] = defaultdict(int)
        self._first: dict[str, tuple[str, str]] = {}     # hash -> (chunk_id, sample)
        self.groups: dict[str, dict[str, Any]] = {}      # hash -> example

    def add(self, chunk_id: str, text: str) -> None:
        h = self.hasher(text)
        self.counts[h] += 1
        n = self.counts[h]
        if n == 1:
            if len(self._first) < self.max_tracked:
                self._first[h] = (chunk_id, text[:SAMPLE_CHARS])
        elif n == 2:
            first = self._first.pop(h, None)
            if first is not None:
                self.groups[h] = {
                    "chunk_ids": [first[0], chunk_id],
                    "sample": first[1],
                    "similarity": round(jaccard(first[1], text[:SAMPLE_CHARS]), 4),
                }
        elif h in self.groups and len(self.groups[h]["chunk_ids"]) < 3:
            self.groups[h]["chunk_ids"].append(chunk_id)

    def summary(self, n_chunks: int, *, with_similarity: bool,
                examples: int) -> dict[str, Any]:
        sizes = [c for c in self.counts.values() if c > 1]
        out: dict[str, Any] = {
            "n_groups": len(sizes),
            "n_chunks_in_groups": sum(sizes),
            "share_of_chunks": round(sum(sizes) / n_chunks, 4) if n_chunks else 0.0,
            "largest_group": max(sizes, default=0),
        }
        if with_similarity:
            # How much text survives the digit mask — i.e. how confusable a
            # template-identical pair really is. Measured on the first two members
            # of each sampled group, over the first SAMPLE_CHARS characters.
            sims = [g["similarity"] for g in self.groups.values()]
            out["mean_jaccard_within_group"] = (
                round(sum(sims) / len(sims), 4) if sims else 0.0)
            out["groups_sampled"] = len(sims)
        ranked = sorted(self.groups.items(), key=lambda kv: -self.counts[kv[0]])
        out["examples"] = [{"group_size": self.counts[h], **g} for h, g in ranked[:examples]]
        return out


def duplicate_report(records: Iterable[dict[str, Any]], *,
                     examples: int = 5) -> dict[str, Any]:
    """Exact and structural duplicate statistics over chunk records.

    Streams: a merged corpus runs to hundreds of thousands of chunks and does not
    need to be resident to be counted. Group counts are exact; the examples and
    the similarity estimate come from a bounded sample of groups.
    """
    exact = _DuplicateCounter(_content_hash)
    template = _DuplicateCounter(_template_hash)
    n_chunks = 0
    files: set[str] = set()
    for rec in records:
        n_chunks += 1
        files.add(rec["file_name"])
        chunk_id = f"{rec['file_name']}::{rec['index']}"
        text = rec["raw_text"]
        exact.add(chunk_id, text)
        template.add(chunk_id, text)
    if not n_chunks:
        return {"n_chunks": 0}
    return {
        "n_chunks": n_chunks,
        "n_files": len(files),
        "exact_duplicates": exact.summary(n_chunks, with_similarity=False, examples=examples),
        "structural_duplicates": template.summary(n_chunks, with_similarity=True,
                                                  examples=examples),
    }


def write_report(report: dict[str, Any], path: str) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info("zakupki: report -> %s", path)
