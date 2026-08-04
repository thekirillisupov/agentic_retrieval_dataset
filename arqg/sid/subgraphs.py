"""S1 — mining entity ↔ chunk subgraphs (plan §3).

No full knowledge graph: a bipartite entity↔chunk map is enough. A subgraph is
2–5 chunks tied by entities that are (a) niche — idf above τ_idf, (b) not
ubiquitous — df ≤ τ_df, and (c) shared by at least two chunks.

Only v0 chunks are eligible: a subgraph built on an injected distractor would
produce a task resting entirely on synthetic text (§3.2).

Neighbourhood density is *measured* later (§7.1) and never filters here.
"""
from __future__ import annotations

import random
from collections import defaultdict

from ..data import is_eligible_seed
from ..utils import log, write_jsonl
from .config import SidConfig
from .corpus import SidCorpus
from .entities import EntityGraph
from .schema import Subgraph, sid_hash


def _doc_of(chunk_id: str) -> str:
    return chunk_id.split("::")[0]


def build_entity_graph(cfg: SidConfig, corpus: SidCorpus) -> EntityGraph:
    graph = EntityGraph()
    for cid in corpus.v0_ids():
        c = corpus.get(cid)
        if c is None:
            continue
        # Index tags are trusted anchors (§3.1) and would be passed here — but
        # this corpus exposes no doc_type / date / section tag (see
        # index_fields.yaml). `document_id` is deliberately NOT used: it links
        # every chunk of one document to every other, which is not a bridge,
        # it is "same file", and it would fill the pool with same-document
        # subgraphs that need no second query.
        graph.add_chunk(cid, c.raw_text, index_tags=None)
    log.info("S1: %d distinct entity surfaces over %d chunks",
             len(graph.chunks_of), graph.n_chunks)
    return graph


def mine_subgraphs(cfg: SidConfig, corpus: SidCorpus) -> list[Subgraph]:
    graph = build_entity_graph(cfg, corpus)
    m = cfg.mining
    tau_idf = graph.idf_threshold(m.idf_percentile, min_df=m.min_co_occurrence,
                                  max_df=m.max_document_frequency)
    rng = random.Random(m.seed)

    eligible = {cid for cid in corpus.v0_ids()
                if (c := corpus.get(cid)) and is_eligible_seed(c, cfg.filters)}

    bridges: list[tuple[str, list[str]]] = []
    for key, chunk_set in graph.chunks_of.items():
        chunks = sorted(chunk_set & eligible)
        if len(chunks) < m.min_co_occurrence:
            continue
        if len(chunk_set) > m.max_document_frequency:
            continue
        if graph.idf(key) < tau_idf and not graph.is_tag(key):
            continue
        bridges.append((key, chunks))

    # rarest bridges first — they make the most discriminating questions
    bridges.sort(key=lambda kv: -graph.idf(kv[0]))
    log.info("S1: %d bridge entities pass τ_idf=%.2f / τ_df=%d",
             len(bridges), tau_idf, m.max_document_frequency)

    out: list[Subgraph] = []
    seen_keys: set[frozenset[str]] = set()
    per_file: dict[str, int] = defaultdict(int)

    for key, chunks in bridges:
        made = 0
        # prefer chunk pairs from different documents: a cross-document bridge
        # forces a real second query instead of reading on in the same file
        ordered = sorted(chunks, key=lambda c: (_doc_of(c), c))
        for i in range(len(ordered)):
            if made >= m.max_subgraphs_per_entity:
                break
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                if m.cross_document_bonus and _doc_of(a) == _doc_of(b) and len(ordered) > 2:
                    continue
                members = [a, b]
                # extend with chunks sharing a *second* rare entity
                for extra in _extend(graph, members, eligible, tau_idf, m.max_chunks, rng):
                    members.append(extra)
                if len(members) < m.min_chunks:
                    continue
                if m.require_cross_document and len({_doc_of(c) for c in members}) < 2:
                    continue
                fkey = frozenset(members)
                if fkey in seen_keys:
                    continue
                if any(per_file[_doc_of(c)] >= m.max_subgraphs_per_file for c in members):
                    continue
                seen_keys.add(fkey)
                for c in members:
                    per_file[_doc_of(c)] += 1

                ents = _bridge_entities(graph, members, tau_idf)
                out.append(Subgraph(
                    subgraph_id=f"sg_{sid_hash(cfg.corpus_name, sorted(members))}",
                    corpus=cfg.corpus_name,
                    index_version="v0",
                    chunks=sorted(members),
                    bridge_entities=ents,
                    index_tags={"files": sorted({_doc_of(c) for c in members})},
                    hop_depth_potential=len(members),
                ))
                made += 1
                break
        if m.target_subgraphs and len(out) >= m.target_subgraphs:
            break

    log.info("S1: mined %d subgraphs", len(out))
    return out


def _extend(graph: EntityGraph, members: list[str], eligible: set[str],
            tau_idf: float, max_chunks: int, rng: random.Random) -> list[str]:
    """Chunks reachable from the current members through another rare entity."""
    if len(members) >= max_chunks:
        return []
    pool: set[str] = set()
    for cid in members:
        for key in graph.entities_of.get(cid, ()):
            if graph.idf(key) < tau_idf:
                continue
            pool |= (graph.chunks_of[key] & eligible)
    pool -= set(members)
    if not pool:
        return []
    picks = sorted(pool)
    rng.shuffle(picks)
    return picks[: max_chunks - len(members)]


def _bridge_entities(graph: EntityGraph, members: list[str],
                     tau_idf: float) -> list[dict]:
    """Entities actually shared by >= 2 members, richest first."""
    counts: dict[str, list[str]] = defaultdict(list)
    for cid in members:
        for key in graph.entities_of.get(cid, ()):
            counts[key].append(cid)
    ents = []
    for key, chunks in counts.items():
        if len(chunks) < 2:
            continue
        men = graph.mention[key]
        ents.append({"surface": men.surface, "type": men.type,
                     "source": "index_tag" if graph.is_tag(key) else "ner",
                     "idf": round(graph.idf(key), 3), "chunks": sorted(chunks)})
    ents.sort(key=lambda e: -e["idf"])
    return ents[:6]


def run_mining(cfg: SidConfig) -> list[Subgraph]:
    corpus = SidCorpus.load(cfg.paths.corpus, version="v0")
    subgraphs = mine_subgraphs(cfg, corpus)
    write_jsonl(cfg.paths.subgraphs, (s.to_dict() for s in subgraphs))
    log.info("S1: wrote %d subgraphs -> %s", len(subgraphs), cfg.paths.subgraphs)
    return subgraphs
