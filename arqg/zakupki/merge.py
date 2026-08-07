"""Several ЕИС dumps -> one corpus, plus a per-chunk metadata sidecar.

The dumps are slices of the same registry with different columns: one carries the
winner and the region name, another the ОКПД2 code and the item list, a third the
contract date. Concatenating them would give the same procurement twice with half
the fields each time; merging on the registry number gives one document that
knows everything anyone published about it.

Merging is streamed, not buffered. Records whose registry number is a real
19-digit number are held in an index so a later source can complete them; records
with an anonymised or synthetic id — ``pn_lot_*`` and friends, which can never
match anything — are written out immediately. Without that split a 0.5M-row dump
would have to sit in memory in full just so a 4.5k-row dump could be merged into
it.

Two files come out:

``<name>.jsonl``       the corpus, in the pipeline's five-field format
``<name>_meta.jsonl``  one record per chunk: where the document came from
                       (dump file, row, portal URL) and every facet worth
                       filtering or searching on
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from ..utils import ensure_parent, log
from .corpus import ChunkOptions, chunk_document
from .facets import Facets
from .tabular import TableProfile, iter_facets

#: A registry number that two sources could plausibly agree on.
MERGE_KEY_MIN_DIGITS = 19


@dataclass
class SourceSpec:
    path: str
    profile: TableProfile
    limit: int = 0


def is_mergeable(facets: Facets) -> bool:
    """Only real registry numbers are worth holding open for a later source."""
    num = facets.purchase_number
    return num.isdigit() and len(num) >= MERGE_KEY_MIN_DIGITS


def merge_sources(specs: Iterable[SourceSpec]) -> Iterator[Facets]:
    """Yield merged facets across sources, streaming what cannot be merged."""
    index: dict[str, Facets] = {}
    n_in = n_merged = n_streamed = 0

    for spec in specs:
        for facets in iter_facets(spec.path, spec.profile, limit=spec.limit):
            n_in += 1
            if not is_mergeable(facets):
                n_streamed += 1
                yield facets
                continue
            existing = index.get(facets.purchase_number)
            if existing is None:
                index[facets.purchase_number] = facets
            else:
                existing.merge(facets)
                n_merged += 1

    log.info("zakupki: merged %d records — %d streamed, %d held, %d folded into a "
             "record from another source", n_in, n_streamed, len(index), n_merged)
    yield from index.values()


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def meta_records(facets: Facets, chunks: list[str], file_name: str) -> Iterator[dict[str, Any]]:
    """One record per chunk: where it sits, plus the facets a query filters on."""
    shared = facets.chunk_facets()
    for index, text in enumerate(chunks):
        yield {
            "chunk_id": f"{file_name}::{index}",
            "file_name": file_name,
            "index": index,
            "document_id": facets.doc_id,
            "section": text.split("\n", 1)[0],
            "n_chars": len(text),
            **shared,
        }


def document_record(facets: Facets, file_name: str, n_chunks: int) -> dict[str, Any]:
    """One record per document: full facets, keywords, and every path it came from."""
    return {
        "document_id": facets.doc_id,
        "file_name": file_name,
        "n_chunks": n_chunks,
        "chunk_ids": [f"{file_name}::{i}" for i in range(n_chunks)],
        "title": facets.title(),
        "keywords": facets.keywords(),
        **facets.summary(),
    }


def build_merged(specs: Iterable[SourceSpec], corpus_path: str, *,
                 meta_path: str = "", docs_path: str = "",
                 opts: ChunkOptions | None = None,
                 doc_type: str = "eisProcurement") -> dict[str, Any]:
    """Merge, chunk and write.

    Everything is written as it is produced — a merged corpus over a 0.5M-row
    dump does not fit in memory, and the duplicate report is computed afterwards
    by re-reading the corpus file.
    """
    opts = opts or ChunkOptions(merge_below=250, min_chars=40)
    for path in (corpus_path, meta_path, docs_path):
        if path:
            ensure_parent(path)

    n_docs = n_chunks = n_chars = n_thin = 0
    seen: set[str] = set()
    per_dataset: dict[str, int] = {}
    n_multi_source = 0
    corpus = open(corpus_path, "w", encoding="utf-8")
    meta = open(meta_path, "w", encoding="utf-8") if meta_path else None
    docs = open(docs_path, "w", encoding="utf-8") if docs_path else None
    try:
        for facets in merge_sources(specs):
            doc = facets.to_doc(doc_type)
            if doc is None:
                n_thin += 1
                continue
            if doc.file_name in seen:
                continue
            seen.add(doc.file_name)
            chunks = chunk_document(doc, opts)
            if len(chunks) < 2:
                n_thin += 1
                continue
            n_docs += 1
            datasets = {s.dataset for s in facets.sources}
            n_multi_source += len(datasets) > 1
            for name in datasets:
                per_dataset[name] = per_dataset.get(name, 0) + 1
            for index, text in enumerate(chunks):
                corpus.write(json.dumps({
                    "file_name": doc.file_name, "index": index, "raw_text": text,
                    "document_id": doc.doc_id, "title": doc.title},
                    ensure_ascii=False) + "\n")
                n_chunks += 1
                n_chars += len(text)
            if meta is not None:
                for record in meta_records(facets, chunks, doc.file_name):
                    meta.write(json.dumps(record, ensure_ascii=False) + "\n")
            if docs is not None:
                docs.write(json.dumps(document_record(facets, doc.file_name, len(chunks)),
                                      ensure_ascii=False) + "\n")
    finally:
        corpus.close()
        for handle in (meta, docs):
            if handle is not None:
                handle.close()

    log.info("zakupki: wrote %d chunks over %d documents -> %s",
             n_chunks, n_docs, corpus_path)
    for label, path in (("chunk metadata", meta_path), ("document metadata", docs_path)):
        if path:
            log.info("zakupki: %s -> %s", label, path)
    if n_thin:
        log.warning("zakupki: %d record(s) skipped — fewer than two sections, so they "
                    "could never form a neighbour window", n_thin)

    return {
        "n_documents_parsed": n_docs + n_thin,
        "n_files": n_docs,
        "n_chunks": n_chunks,
        "n_chars": n_chars,
        "n_skipped_thin": n_thin,
        "n_documents_from_several_sources": n_multi_source,
        "mean_chunk_chars": round(n_chars / n_chunks, 1) if n_chunks else 0.0,
        "mean_chunks_per_doc": round(n_chunks / n_docs, 2) if n_docs else 0.0,
        "documents_per_dataset": dict(sorted(per_dataset.items())),
    }


def write_manifest(path: str, specs: Iterable[SourceSpec], stats: dict[str, Any],
                   corpus_path: str, meta_path: str) -> None:
    """Provenance for the merged corpus: what went in, under which licence."""
    payload = {
        "corpus": corpus_path,
        "metadata": meta_path,
        "sources": [{"path": s.path, "profile": s.profile.name,
                     "origin": s.profile.source, "licence": s.profile.licence,
                     "limit": s.limit or None} for s in specs],
        **stats,
    }
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("zakupki: manifest -> %s", path)
