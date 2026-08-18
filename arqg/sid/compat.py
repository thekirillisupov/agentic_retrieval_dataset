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
from collections import Counter, defaultdict
from typing import Any

from ..utils import ensure_parent, log
from .config import SidConfig
from .corpus import SidCorpus
from .scoping import BUILTIN_FIELDS, facet_header
from .sections import depth as section_depth, scope_of

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


def _meta_fields_report(corpus: SidCorpus) -> dict[str, Any]:
    """§2.2, generalised — every facet in ``Chunk.meta``, coverage and
    cardinality, so a config can point ``mining.scope_field`` at whichever one
    actually groups the corpus (see scoping.py). Populated either by
    `load_chunks` (facets inlined on the corpus record) or by a metadata
    sidecar merged in at load time (``paths.meta``, see corpus.py)."""
    n = max(1, len(corpus))
    coverage: Counter = Counter()
    cardinality: dict[str, set[str]] = defaultdict(set)
    for c in corpus.all_chunks():
        for k, v in c.meta.items():
            if v in (None, "", [], {}):
                continue
            coverage[k] += 1
            if isinstance(v, (str, int, float)):
                cardinality[k].add(str(v))
    return {
        k: {"coverage": round(coverage[k] / n, 4), "n_distinct": len(cardinality[k])}
        for k in sorted(coverage)
    }


def build_index_fields(corpus: SidCorpus, cfg: SidConfig | None = None) -> dict[str, Any]:
    """§2.2 — what the miner and search filters may key on."""
    has_doc_id = sum(1 for c in corpus.all_chunks() if c.document_id)
    titled = [c.title for c in corpus.all_chunks() if c.title]
    n = max(1, len(corpus))
    breadcrumbs = [t for t in titled if section_depth(t) > 1]
    depths = sorted(section_depth(t) for t in breadcrumbs)
    scopes = {scope_of(t, gap=1) for t in breadcrumbs}
    scopes.discard("")
    meta_fields = _meta_fields_report(corpus)

    fields: dict[str, Any] = {
        "chunk_id": {"format": "{file_name}::{index}", "coverage": 1.0},
        "file_name": {"role": "document handle / doc_type proxy", "coverage": 1.0},
        "index": {"role": "position in document; same-document distance",
                  "coverage": 1.0},
        "document_id": {"role": "logical document", "coverage": round(has_doc_id / n, 4)},
        "title": {
            "role": "breadcrumb path — S1 mines within a folder of it"
                    if breadcrumbs else "section/document title",
            "coverage": round(len(titled) / n, 4),
            "share_breadcrumb": round(len(breadcrumbs) / n, 4),
            "median_depth": depths[len(depths) // 2] if depths else 0,
            "n_scopes_at_gap_1": len(scopes),
        },
    }
    if meta_fields:
        # Everything else the index carries — a categorical facet from a
        # metadata sidecar, or extra keys inlined on the corpus record.
        fields["meta_fields"] = meta_fields
    if cfg is not None:
        fields["scope"] = {
            "field": cfg.mining.scope_field, "strategy": cfg.mining.scope_strategy,
            "source": "builtin chunk field" if cfg.mining.scope_field in BUILTIN_FIELDS
                      else "meta_fields" if cfg.mining.scope_field in meta_fields
                      else "not found on this corpus",
        }
        # Grouping is only one of the two things a facet can do. This says
        # which ones are *searchable* — rendered into the passage both index
        # branches hold — and which the LLM sees while writing facts and
        # questions (see scoping.facet_header, env.passage_text).
        surfaced = list(cfg.facets.fields)
        fields["surfaced_facets"] = {
            "fields": surfaced,
            "in_passage": bool(surfaced) and cfg.facets.in_passage,
            "in_prompts": bool(surfaced) and cfg.facets.in_prompts,
            "missing_on_this_corpus": [f for f in surfaced
                                       if f not in meta_fields
                                       and f not in BUILTIN_FIELDS],
            "example": facet_header(corpus.all_chunks()[0], surfaced,
                                    cfg.facets.labels, cfg.facets.max_value_chars)
            if surfaced and len(corpus) else "",
        }

    notes = [
        "date / ACL / doc_type tags are not present in this corpus; "
        "temporal_resolution therefore relies on dates extracted from text.",
        ("title is a path, not a headline: it is the section anchor S1 scopes "
         "on by default (mining.scope_field=title, scope_strategy=path — see "
         "scoping.py). It is NOT registered as an entity-graph tag — that "
         "would bypass tau_idf and make every folder sibling a bridge."
         if breadcrumbs else
         "title is a flat name, not a path: with the default scope_field/"
         "scope_strategy S1 falls back to the unscoped global entity search."),
    ]
    if meta_fields:
        notes.append(
            "additional per-chunk facets are available (see meta_fields above) — "
            "point mining.scope_field at one of them with scope_strategy=exact "
            "to mine within chunks that share its value verbatim (a categorical "
            "facet, not a path), instead of the title breadcrumb.")
        notes.append(
            "a facet only reaches the retriever if `facets.fields` lists it: "
            "grouping chunks by a field the index cannot see gives the composer "
            "an attribute to name and the probe nothing to match it against.")
    fields["notes"] = notes
    return fields


def run_compat(cfg: SidConfig) -> dict[str, Any]:
    corpus = SidCorpus.load(cfg.paths.corpus, version="v0", meta_path=cfg.paths.meta)
    report = build_compat_report(cfg, corpus)
    ensure_parent(cfg.paths.compat_report)
    with open(cfg.paths.compat_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    fields = build_index_fields(corpus, cfg)
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
