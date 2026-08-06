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
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

from ..utils import ensure_parent, log, write_jsonl
from .parse import ProcurementDoc, Section

_DIGITS_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass
class ChunkOptions:
    max_chars: int = 1400        # hard cap on a chunk
    merge_below: int = 600       # sections shorter than this may absorb the next one
    min_chars: int = 120         # drop chunks with less signal than this
    max_chunks_per_doc: int = 40


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
    """Chunk texts for one document, in document order."""
    opts = opts or ChunkOptions()
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
    for doc in docs:
        if doc.file_name in seen_files:
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
        "n_documents": len(materialised),
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
def _groups(records: Sequence[dict[str, Any]], hasher) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        buckets[hasher(rec["raw_text"])].append(i)
    return {h: idx for h, idx in buckets.items() if len(idx) > 1}


def _summarise(records: Sequence[dict[str, Any]], buckets: dict[str, list[int]],
               *, with_similarity: bool, examples: int) -> dict[str, Any]:
    in_groups = sum(len(v) for v in buckets.values())
    ranked = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
    out: dict[str, Any] = {
        "n_groups": len(buckets),
        "n_chunks_in_groups": in_groups,
        "share_of_chunks": round(in_groups / len(records), 4) if records else 0.0,
        "largest_group": len(ranked[0][1]) if ranked else 0,
    }
    if with_similarity:
        # Mean pairwise similarity of a group's first two members: how much
        # actual text survives the mask, i.e. how confusable the pair really is.
        sims = [jaccard(records[idx[0]]["raw_text"], records[idx[1]]["raw_text"])
                for _, idx in ranked[:200]]
        out["mean_jaccard_within_group"] = round(sum(sims) / len(sims), 4) if sims else 0.0
    out["examples"] = [
        {
            "group_size": len(idx),
            "chunk_ids": [f"{records[i]['file_name']}::{records[i]['index']}" for i in idx[:3]],
            "sample": records[idx[0]]["raw_text"][:400],
        }
        for _, idx in ranked[:examples]
    ]
    return out


def duplicate_report(records: Sequence[dict[str, Any]], *, examples: int = 5) -> dict[str, Any]:
    """Exact and structural duplicate statistics over chunk records."""
    records = list(records)
    if not records:
        return {"n_chunks": 0}
    exact = _groups(records, _content_hash)
    template = _groups(records, _template_hash)
    return {
        "n_chunks": len(records),
        "n_documents": len({r["file_name"] for r in records}),
        "exact_duplicates": _summarise(records, exact, with_similarity=False, examples=examples),
        "structural_duplicates": _summarise(records, template, with_similarity=True,
                                            examples=examples),
    }


def write_report(report: dict[str, Any], path: str) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info("zakupki: report -> %s", path)
