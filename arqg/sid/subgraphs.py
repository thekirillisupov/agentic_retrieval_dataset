"""S1 — mining entity ↔ chunk subgraphs (plan §3).

No full knowledge graph: a bipartite entity↔chunk map is enough. A subgraph is
2–5 chunks tied by entities that are (a) niche, (b) not ubiquitous — df ≤ τ_df,
and (c) shared by at least two chunks.

Only v0 chunks are eligible: a subgraph built on an injected distractor would
produce a task resting entirely on synthetic text (§3.2).

Neighbourhood density is *measured* later (§7.1) and never filters here.

**Where the search happens matters as much as what holds a subgraph together.**
Run over the whole corpus, "rare entity shared by two chunks" is satisfied by
coincidence far more often than by subject matter: on `ckr` half the subgraphs
mined that way shared nothing but the corpus root, bridged by словоформы like
«Перестал» or «Сторону». So the search is confined to a *section scope* built
from the `title` breadcrumb (see `sections.py`), and the entity's job narrows to
what it is actually good at — picking which two documents of that folder belong
in one question.

That reframing forces a second change. Global τ_idf is the wrong floor inside a
scope: an entity rare enough to clear it (df ≈ 2) lands twice in the same folder
only by coincidence, so the strict rule collapsed a 1587-scope corpus onto 122
folders. Discrimination inside a folder is a *local* property — an entity in two
of a folder's twelve chunks separates them regardless of its global df, and one
in most of the folder's chunks is the folder's subject rather than a bridge.
Hence `scope_df_ratio` as the upper bound, τ_df as the ubiquity ceiling, and
global idf demoted to the ordering key.

Even so the entity channel reaches only part of the corpus: on `ckr` 283 of 800
scopes carry a repeated surface form, and in 451 of the remaining 517 nothing
repeats at all — those folders are not filtered out, they are invisible to
exact matching. `mining.sim_bridge` adds a second channel that bridges them by
doc2doc similarity instead (see `simbridge.py`); the two share the per-folder
budget, and the entity channel keeps priority because its bridge names what the
chunks have in common while a similarity bridge only asserts that something
does.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

from ..data import is_eligible_seed
from ..utils import log, read_jsonl, write_jsonl
from .config import MiningConfig, SidConfig
from .corpus import SidCorpus, load_corpus
from .dense import DenseIndex
from .entities import EntityGraph, corpus_lowercase_vocab
from .schema import Subgraph, sid_hash
from .sections import scope_of, shared_depth
from .simbridge import CoRetrievability, SimBand, fit_band, scope_pairs


def _doc_of(chunk_id: str) -> str:
    return chunk_id.split("::")[0]


def _index_of(chunk_id: str) -> int:
    try:
        return int(chunk_id.split("::")[-1])
    except ValueError:
        return 0


def build_entity_graph(cfg: SidConfig, corpus: SidCorpus) -> EntityGraph:
    ids = corpus.v0_ids()
    texts = {cid: c.raw_text for cid in ids if (c := corpus.get(cid)) is not None}
    # a single corpus-wide pass: a bare capitalised word that is also spelled
    # lowercase somewhere in the same corpus is an ordinary word, not a name —
    # this is what actually holds against markdown tables/lists/links, rather
    # than enumerating every layout that capitalises a cell/item by accident.
    common_lc = corpus_lowercase_vocab(texts.values())

    graph = EntityGraph()
    for cid, raw_text in texts.items():
        # No index tags are registered here. `title` is a section tag, but it
        # enters mining as a *scope* (sections.py explains why): as a tag it
        # would bypass τ_idf via `is_tag`, inflate the df of every folder
        # member and let `_extend` pull in an arbitrary sibling — which is
        # exactly why `document_id` is kept out too. This corpus exposes no
        # doc_type / date / ACL tag at all (see index_fields.yaml).
        graph.add_chunk(cid, raw_text, index_tags=None, common_lc=common_lc)
    log.info("S1: %d distinct entity surfaces over %d chunks",
             len(graph.chunks_of), graph.n_chunks)
    return graph


# --------------------------------------------------------------------------- #
# admissible chunk pairs
# --------------------------------------------------------------------------- #
class _Admissible:
    """Whether two chunks may sit in one subgraph.

    Different documents always may. The same document may only if reading it
    would not hand the agent both chunks for free, and *that* is a claim about
    the document, not about a fixed index distance: positions 2 and 5 of a
    six-chunk page are one read apart, while positions 40 and 300 of a
    900-chunk page are different sections that need a second query. So both
    conditions apply — the document must be long enough to be worth navigating
    (`same_doc_min_chunks`) and the gap wide enough (`min_index_gap`).
    """

    def __init__(self, m: MiningConfig, doc_sizes: dict[str, int]):
        self.m = m
        self.doc_sizes = doc_sizes

    def ok(self, a: str, b: str) -> bool:
        m = self.m
        doc = _doc_of(a)
        if doc != _doc_of(b):
            return True
        if m.require_cross_document:
            return False
        if self.doc_sizes.get(doc, 0) < m.same_doc_min_chunks:
            return False
        return abs(_index_of(a) - _index_of(b)) >= m.min_index_gap

    def pairs(self, chunks: list[str], limit: int) -> list[tuple[str, str]]:
        """Up to ``limit`` admissible pairs from one bridge's chunks.

        Cross-document pairs first (a real second query, not reading on), and
        disjoint pairs before overlapping ones so one entity does not hand back
        the same chunk in every subgraph it seeds.
        """
        cross: list[tuple[str, str]] = []
        same: list[tuple[str, str]] = []
        for i, a in enumerate(chunks):
            for b in chunks[i + 1:]:
                if not self.ok(a, b):
                    continue
                (cross if _doc_of(a) != _doc_of(b) else same).append((a, b))
        ordered = cross + same if self.m.cross_document_bonus else sorted(cross + same)

        picked: list[tuple[str, str]] = []
        used: set[str] = set()
        for p in ordered:
            if len(picked) >= limit:
                return picked
            if used & set(p):
                continue
            picked.append(p)
            used |= set(p)
        for p in ordered:                  # then allow overlap rather than starve
            if len(picked) >= limit:
                break
            if p not in picked:
                picked.append(p)
        return picked


def _extend(graph: EntityGraph, members: list[str], pool: set[str],
            tau_idf: float, adm: _Admissible, rng: random.Random) -> list[str]:
    """Chunks of ``pool`` reachable from the members through another entity."""
    m = adm.m
    if len(members) >= m.max_chunks:
        return []
    reachable: set[str] = set()
    for cid in members:
        for key in graph.entities_of.get(cid, ()):
            if graph.idf(key) < tau_idf:
                continue
            reachable |= (graph.chunks_of[key] & pool)
    reachable -= set(members)
    picks = sorted(reachable)
    rng.shuffle(picks)
    # A hop into a document already in the subgraph is the cheapest hop there
    # is — the agent is holding that document. Grow across documents first and
    # only fall back to the ones already present.
    seen_docs = {_doc_of(c) for c in members}
    picks.sort(key=lambda c: _doc_of(c) in seen_docs)
    out: list[str] = []
    for cid in picks:
        if len(members) + len(out) >= m.max_chunks:
            break
        if all(adm.ok(cid, other) for other in members + out):
            out.append(cid)
    return out


@dataclass(frozen=True)
class _Bridge:
    """One work item: a chunk pair and what put it there."""
    pair: tuple[str, str]
    etype: str                 # entity type, or "SIM" for the doc2doc channel
    kind: str = "entity"       # "entity" | "similarity"
    key: str = ""              # the bridging entity; empty for similarity
    sim: float = 0.0


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


# --------------------------------------------------------------------------- #
# accumulation under the global quotas
# --------------------------------------------------------------------------- #
@dataclass
class _Pool:
    cfg: SidConfig
    graph: EntityGraph
    titles: dict[str, str]
    out: list[Subgraph] = field(default_factory=list)
    seen: set[frozenset[str]] = field(default_factory=set)
    per_file: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def full(self) -> bool:
        target = self.cfg.mining.target_subgraphs
        return bool(target) and len(self.out) >= target

    def add(self, members: list[str], tau_idf: float, scope: str,
            kind: str = "entity", sim: float = 0.0) -> bool:
        m = self.cfg.mining
        key = frozenset(members)
        if key in self.seen:
            return False
        docs = {_doc_of(c) for c in members}
        if any(self.per_file[d] >= m.max_subgraphs_per_file for d in docs):
            return False
        self.seen.add(key)
        # per *subgraph*, not per member chunk: counting occurrences made a
        # 3-chunk subgraph spend three of its document's slots at once, so the
        # quota bit at a third of its nominal value.
        for d in docs:
            self.per_file[d] += 1
        ordered = sorted(members)
        self.out.append(Subgraph(
            subgraph_id=f"sg_{sid_hash(self.cfg.corpus_name, ordered)}",
            corpus=self.cfg.corpus_name,
            index_version="v0",
            chunks=ordered,
            bridge_entities=_bridge_entities(self.graph, ordered, tau_idf),
            index_tags={"files": sorted({_doc_of(c) for c in ordered}),
                        "section_scope": scope},
            hop_depth_potential=len(ordered),
            path_scope=scope,
            path_shared_depth=shared_depth([self.titles.get(c, "") for c in ordered]),
            bridge_kind=kind,
            pair_similarity=round(sim, 4),
        ))
        return True


# --------------------------------------------------------------------------- #
# scoped mining
# --------------------------------------------------------------------------- #
def _group_by_scope(corpus: SidCorpus, eligible: set[str],
                    m: MiningConfig) -> tuple[dict[str, list[str]], list[str]]:
    """Split eligible chunks into section scopes plus a residue.

    A chunk lands in the residue when its title is too shallow to name a folder
    (`scope_of` returns "") or when its folder holds too few eligible chunks to
    mine. The residue is mined globally, so a corpus with flat titles behaves
    exactly as it did before scoping existed.
    """
    if m.path_scope_gap is None:
        return {}, sorted(eligible)

    scoped: dict[str, list[str]] = defaultdict(list)
    residue: list[str] = []
    for cid in sorted(eligible):
        c = corpus.get(cid)
        scope = scope_of(c.title if c else "", gap=m.path_scope_gap,
                         min_depth=m.min_scope_depth)
        if scope:
            scoped[scope].append(cid)
        else:
            residue.append(cid)

    too_small = {s for s, cids in scoped.items() if len(cids) < m.min_scope_chunks}
    for s in too_small:
        residue.extend(scoped.pop(s))
    log.info("S1: %d section scopes over %d chunks (+%d unscoped)",
             len(scoped), sum(len(v) for v in scoped.values()), len(residue))
    return dict(scoped), sorted(residue)


def _scope_bridges(graph: EntityGraph, cids: list[str], adm: _Admissible,
                   tau_idf: float) -> list[_Bridge]:
    """Ordered entity-bridge work items for one scope."""
    m = adm.m
    local: dict[str, list[str]] = defaultdict(list)
    for cid in cids:
        for key in graph.entities_of.get(cid, ()):
            local[key].append(cid)

    # the folder's own subject is in most of its chunks; a bridge is not
    df_max = max(m.min_co_occurrence, int(len(cids) * m.scope_df_ratio))
    keys: list[str] = []
    pairs: dict[str, list[tuple[str, str]]] = {}
    for key, members in local.items():
        if not (m.min_co_occurrence <= len(members) <= df_max):
            continue
        if graph.df(key) > m.max_document_frequency:
            continue
        if graph.idf(key) < tau_idf and not graph.is_tag(key):
            continue
        picked = adm.pairs(sorted(members), m.max_subgraphs_per_entity)
        if not picked:
            continue
        keys.append(key)
        pairs[key] = picked
    keys.sort(key=lambda k: (-graph.idf(k), k))

    # rarest bridge of every entity before the second-best of any of them
    items: list[_Bridge] = []
    for rank in range(max(1, m.max_subgraphs_per_entity)):
        for key in keys:
            if rank < len(pairs[key]):
                items.append(_Bridge(pair=pairs[key][rank], key=key,
                                     etype=graph.mention[key].type))
    return items


def _scope_sim_bridges(sim: "_SimChannel | None", cids: list[str],
                       adm: _Admissible) -> list[_Bridge]:
    """Doc2doc work items for one scope, appended *after* the entity ones so a
    folder spends its budget on named bridges first."""
    if sim is None:
        return []
    out = []
    for pair, s in scope_pairs(sim.dense, cids, sim.band,
                               adm.m.sim_bridge_max_pairs_per_scope):
        if adm.ok(*pair):
            out.append(_Bridge(pair=pair, etype="SIM", kind="similarity", sim=s))
    return out


@dataclass
class _SimChannel:
    dense: DenseIndex
    band: SimBand
    coret: CoRetrievability
    skipped: int = 0


@dataclass
class _ScopeState:
    chunks: set[str]
    items: list[_Bridge]
    cursor: int = 0
    made: int = 0
    by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def _emit_one(pool: _Pool, scope: str, st: _ScopeState, tau_idf: float,
              adm: _Admissible, rng: random.Random,
              sim: "_SimChannel | None" = None) -> bool:
    """Advance one scope by at most one subgraph."""
    m = adm.m
    per_type = max(1, round(m.max_bridge_type_share * m.max_subgraphs_per_path))
    while st.cursor < len(st.items):
        b = st.items[st.cursor]
        st.cursor += 1
        if st.by_type[b.etype] >= per_type:
            continue
        # the doc2doc channel's upper bound, paid only for a pair the budget
        # actually reaches: a partner already inside the top-k neighbours is
        # returned by the same query, which is a task G_BROAD rejects by
        # definition
        if b.kind == "similarity" and sim is not None:
            if sim.coret.co_retrievable(*b.pair, b.sim):
                sim.skipped += 1
                continue
        members = list(b.pair)
        members += _extend(pool.graph, members, st.chunks, tau_idf, adm, rng)
        if len(members) < m.min_chunks:
            continue
        if m.require_cross_document and len({_doc_of(c) for c in members}) < 2:
            continue
        if pool.add(members, tau_idf, scope, kind=b.kind, sim=b.sim):
            st.made += 1
            st.by_type[b.etype] += 1
            return True
    return False


def _mine_scoped(pool: _Pool, scopes: dict[str, list[str]], tau_idf: float,
                 adm: _Admissible, rng: random.Random,
                 sim: "_SimChannel | None" = None) -> None:
    m = adm.m
    plans: dict[str, _ScopeState] = {}
    n_entity = 0
    for scope, cids in scopes.items():
        items = _scope_bridges(pool.graph, cids, adm, tau_idf)
        n_entity += bool(items)
        items = items + _scope_sim_bridges(sim, cids, adm)
        if items:
            plans[scope] = _ScopeState(chunks=set(cids), items=items)
    if sim is not None:
        log.info("S1: %d/%d scopes carry an entity bridge, %d once doc2doc is "
                 "allowed to bridge too", n_entity, len(scopes), len(plans))
    else:
        log.info("S1: %d/%d scopes carry a usable bridge", len(plans), len(scopes))

    # Round-robin, one subgraph per scope per pass. `target_subgraphs` is a cap,
    # and draining scopes one at a time would spend the whole budget inside the
    # first few folders — breadth over folders is the point of scoping.
    before = len(pool.out)
    while plans and not pool.full:
        progressed = False
        for scope in sorted(plans):
            st = plans[scope]
            if _emit_one(pool, scope, st, tau_idf, adm, rng, sim):
                progressed = True
            if st.cursor >= len(st.items) or st.made >= m.max_subgraphs_per_path:
                del plans[scope]
            if pool.full:
                break
        if not progressed:
            break
    log.info("S1: %d subgraphs from section scopes", len(pool.out) - before)


def _mine_global(pool: _Pool, cids: list[str], tau_idf: float,
                 adm: _Admissible, rng: random.Random) -> None:
    """The unscoped search: rarest bridging entity in the whole corpus first."""
    m = adm.m
    allowed = set(cids)
    bridges: list[tuple[str, list[str]]] = []
    for key, chunk_set in pool.graph.chunks_of.items():
        chunks = sorted(chunk_set & allowed)
        if len(chunks) < m.min_co_occurrence:
            continue
        if len(chunk_set) > m.max_document_frequency:
            continue
        if pool.graph.idf(key) < tau_idf and not pool.graph.is_tag(key):
            continue
        bridges.append((key, chunks))
    bridges.sort(key=lambda kv: (-pool.graph.idf(kv[0]), kv[0]))
    log.info("S1: %d bridge entities pass τ_idf=%.2f / τ_df=%d over %d chunks",
             len(bridges), tau_idf, m.max_document_frequency, len(allowed))

    before = len(pool.out)
    for key, chunks in bridges:
        if pool.full:
            break
        for pair in adm.pairs(chunks, m.max_subgraphs_per_entity):
            members = list(pair)
            members += _extend(pool.graph, members, allowed, tau_idf, adm, rng)
            if len(members) < m.min_chunks:
                continue
            if m.require_cross_document and len({_doc_of(c) for c in members}) < 2:
                continue
            pool.add(members, tau_idf, "")
            if pool.full:
                break
    log.info("S1: %d subgraphs from the unscoped residue", len(pool.out) - before)


def _sim_channel(cfg: SidConfig, dense: DenseIndex | None) -> _SimChannel | None:
    m = cfg.mining
    if not m.sim_bridge:
        return None
    if dense is None or not dense.ids:
        log.warning("S1: mining.sim_bridge is on but no dense index was given — "
                    "mining with the entity channel only")
        return None
    band = fit_band(dense, m.sim_bridge_low_percentile, m.sim_bridge_exclude_top_k,
                    cfg.density.sample_chunks, cfg.density.seed)
    return _SimChannel(dense=dense, band=band,
                       coret=CoRetrievability(dense, m.sim_bridge_exclude_top_k))


def mine_subgraphs(cfg: SidConfig, corpus: SidCorpus,
                   dense: DenseIndex | None = None) -> list[Subgraph]:
    graph = build_entity_graph(cfg, corpus)
    m = cfg.mining
    rng = random.Random(m.seed)
    sim = _sim_channel(cfg, dense)

    eligible = {cid for cid in corpus.v0_ids()
                if (c := corpus.get(cid)) and is_eligible_seed(c, cfg.filters)}
    titles = {cid: (c.title if (c := corpus.get(cid)) else "") for cid in eligible}
    doc_sizes: dict[str, int] = defaultdict(int)
    for cid in corpus.ids():               # size of the WHOLE document, not of
        doc_sizes[_doc_of(cid)] += 1       # its eligible-seed subset
    pool = _Pool(cfg=cfg, graph=graph, titles=titles)
    adm = _Admissible(m, doc_sizes)

    scopes, residue = _group_by_scope(corpus, eligible, m)
    if scopes:
        tau_scoped = graph.idf_threshold(m.scope_idf_percentile,
                                         min_df=m.min_co_occurrence,
                                         max_df=m.max_document_frequency)
        _mine_scoped(pool, scopes, tau_scoped, adm, rng, sim)
    # The unscoped residue keeps the entity channel only. Doc2doc is a
    # *scope-local* relation here: inside a folder it says "these two of the
    # twelve", corpus-wide it would reintroduce the coincidental pairing that
    # scoping was added to remove, without a folder to bound it.
    if residue and not pool.full:
        tau_global = graph.idf_threshold(m.idf_percentile,
                                         min_df=m.min_co_occurrence,
                                         max_df=m.max_document_frequency)
        _mine_global(pool, residue, tau_global, adm, rng)

    if sim is not None and sim.skipped:
        log.info("S1: %d doc2doc pairs skipped as co-retrievable (partner inside "
                 "the top-%d neighbours)", sim.skipped, m.sim_bridge_exclude_top_k)
    log.info("S1: mined %d subgraphs", len(pool.out))
    return pool.out


async def run_mining(cfg: SidConfig) -> list[Subgraph]:
    # the same loader `build_env` uses, so the dense signature below matches the
    # one the gates will ask for and the embeddings are paid for once
    corpus = load_corpus(cfg)
    dense, embedder = None, None
    if cfg.mining.sim_bridge:
        # the same cached v0 index the gates will measure against, so enabling
        # the channel moves the embedding bill earlier rather than adding one
        from ..embeddings import make_embedder
        from .env import build_dense_for
        embedder = make_embedder(cfg.embed)
        dense = await build_dense_for(cfg, corpus, "v0", embedder)
    try:
        subgraphs = mine_subgraphs(cfg, corpus, dense)
    finally:
        if embedder is not None:
            await embedder.aclose()
    _warn_if_downstream_is_stale(cfg, subgraphs)
    write_jsonl(cfg.paths.subgraphs, (s.to_dict() for s in subgraphs))
    log.info("S1: wrote %d subgraphs -> %s", len(subgraphs), cfg.paths.subgraphs)
    _log_scope_mix(subgraphs)
    return subgraphs


def _warn_if_downstream_is_stale(cfg: SidConfig, subgraphs: list[Subgraph]) -> None:
    """Every stage resumes by skipping keys it has already decided, which cannot
    notice that S1 itself produced a *different* pool. Re-mining with changed
    settings therefore appends the new candidates alongside the old ones and
    exports both — so say so, loudly, rather than let a mixed pool through."""
    import os

    path = cfg.paths.candidates
    if not os.path.exists(path):
        return
    mined = {s.subgraph_id for s in subgraphs}
    cached = {r.get("subgraph_id") for r in read_jsonl(path)}
    orphans = cached - mined
    if orphans:
        log.warning(
            "S1: %s holds %d candidates from %d subgraphs this run did not mine. "
            "Resume keys off candidate_id, so those would survive into the export "
            "next to the new pool — delete %s (and gate_decisions/gated/minimized) "
            "or point paths.out_dir somewhere fresh.",
            path, sum(1 for r in read_jsonl(path) if r.get("subgraph_id") in orphans),
            len(orphans), os.path.basename(path))


def _log_scope_mix(subgraphs: list[Subgraph]) -> None:
    """The diagnostic scoping exists for: how much of the pool is still held
    together by nothing but the corpus root."""
    if not subgraphs:
        return
    n = len(subgraphs)
    rootish = sum(1 for s in subgraphs if s.path_shared_depth <= 1)
    scoped = sum(1 for s in subgraphs if s.path_scope)
    single_doc = sum(1 for s in subgraphs
                     if len({_doc_of(c) for c in s.chunks}) == 1)
    by_sim = [s for s in subgraphs if s.bridge_kind == "similarity"]
    log.info("S1: scoped %d/%d (%.0f%%), shared-root-only %d (%.0f%%), "
             "single-document %d (%.0f%%), distinct scopes %d",
             scoped, n, 100 * scoped / n, rootish, 100 * rootish / n,
             single_doc, 100 * single_doc / n,
             len({s.path_scope for s in subgraphs if s.path_scope}))
    if by_sim:
        anchored = sum(1 for s in by_sim if s.bridge_entities)
        log.info("S1: %d/%d subgraphs came from the doc2doc channel (%.0f%%), "
                 "%d of them share an entity anyway; median pair sim %.3f",
                 len(by_sim), n, 100 * len(by_sim) / n, anchored,
                 sorted(s.pair_similarity for s in by_sim)[len(by_sim) // 2])
