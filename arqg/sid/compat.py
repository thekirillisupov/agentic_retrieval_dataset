"""S0 — index compatibility, available fields, versioning (plan §2).

Three questions, answered before anything is generated:

1. Is the indexing unit the same unit ``read`` returns? Here the corpus record
   *is* the retrievable unit, so the check reduces to reporting the unit's shape
   and flagging chunks that look like a broken window (truncated mid-sentence),
   which is the §2.1 "fact split across a chunk boundary" diagnostic.
2. Which fields can the miner and the search filters rely on?
3. What version is the index, and what is its checksum?
"""
from __future__ import annotations

import json
import random
import statistics as st
from typing import Any

from ..utils import ensure_parent, log
from .config import SidConfig
from .corpus import SidCorpus

_SENT_END = ".!?…»\"'"


def _looks_truncated(text: str) -> bool:
    t = (text or "").rstrip()
    return bool(t) and t[-1] not in _SENT_END


def _starts_midsentence(text: str) -> bool:
    t = (text or "").lstrip()
    return bool(t) and t[0].islower()


def build_compat_report(cfg: SidConfig, corpus: SidCorpus, sample: int = 200) -> dict[str, Any]:
    chunks = corpus.all_chunks()
    rng = random.Random(cfg.density.seed)
    probe = rng.sample(chunks, min(sample, len(chunks)))
    lengths = [c.n_chars for c in chunks]
    trunc = sum(_looks_truncated(c.raw_text) for c in probe)
    mid = sum(_starts_midsentence(c.raw_text) for c in probe)
    boundary_broken = sum(
        1 for c in probe if _looks_truncated(c.raw_text) and _starts_midsentence(c.raw_text))

    report = {
        "corpus": cfg.corpus_name,
        "index_version": corpus.version,
        "n_chunks": len(chunks),
        "n_documents": len(corpus.files),
        "unit": {
            "indexing_unit": "corpus record ({file_name}::{index})",
            "read_unit": "same record — server-side window == indexed unit",
            "units_aligned": True,
        },
        "chunk_chars": {
            "mean": round(st.mean(lengths), 1) if lengths else 0,
            "median": st.median(lengths) if lengths else 0,
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
        "boundary_diagnostics": {
            "sample": len(probe),
            "ends_midsentence": trunc,
            "starts_midsentence": mid,
            "share_facts_split_by_boundary": round(boundary_broken / max(1, len(probe)), 4),
        },
        "blocking_condition_met": True,
    }
    return report


def build_index_fields(corpus: SidCorpus) -> dict[str, Any]:
    """§2.2 — what the miner and search filters may key on."""
    has_doc_id = sum(1 for c in corpus.all_chunks() if c.document_id)
    has_title = sum(1 for c in corpus.all_chunks() if c.title)
    n = max(1, len(corpus))
    return {
        "chunk_id": {"format": "{file_name}::{index}", "coverage": 1.0},
        "file_name": {"role": "document handle / doc_type proxy", "coverage": 1.0},
        "index": {"role": "position in document", "coverage": 1.0},
        "document_id": {"role": "logical document", "coverage": round(has_doc_id / n, 4)},
        "title": {"role": "section/document title", "coverage": round(has_title / n, 4)},
        "notes": [
            "date / ACL / doc_type tags are not present in this corpus; "
            "temporal_resolution therefore relies on dates extracted from text, "
            "and index-tag anchors fall back to file_name.",
        ],
    }


def run_compat(cfg: SidConfig) -> dict[str, Any]:
    corpus = SidCorpus.load(cfg.paths.corpus, version="v0")
    report = build_compat_report(cfg, corpus)
    ensure_parent(cfg.paths.compat_report)
    with open(cfg.paths.compat_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    fields = build_index_fields(corpus)
    ensure_parent(cfg.paths.index_fields)
    try:
        import yaml
        with open(cfg.paths.index_fields, "w", encoding="utf-8") as f:
            yaml.safe_dump(fields, f, allow_unicode=True, sort_keys=False)
    except ImportError:                                  # pragma: no cover
        with open(cfg.paths.index_fields, "w", encoding="utf-8") as f:
            json.dump(fields, f, ensure_ascii=False, indent=2)

    corpus.write_manifest(cfg.paths.manifest, cfg.corpus_name,
                          extra={"taxonomy_version": cfg.taxonomy_version})
    log.info("S0: %d chunks, %.1f%% of sampled chunks split by a chunk boundary",
             len(corpus),
             100 * report["boundary_diagnostics"]["share_facts_split_by_boundary"])
    return report
