"""Tests for the SID pipeline — all offline on the mock backend.

Covers the invariants the plan is built on, not just the plumbing:
gold-set minimality is per fact, fact groups are never collapsed, gaps aggregate
min-within-group / max-across-groups, distractor verification actually rejects,
and synthetic markers never reach the agent-visible corpus.
"""
import asyncio
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arqg.sid.compat import run_compat
from arqg.sid.compose import has_shared_anchor, plan_batches
from arqg.sid.config import SidConfig
from arqg.sid.corpus import SidCorpus, content_hash
from arqg.sid.dense import DenseIndex
from arqg.sid.density import fit_density_model, task_density
from arqg.sid.distractors import (_answer_leaks, _attribute_in_question,
                                  verify_candidates, DistractorCandidate,
                                  TaskInjection)
from arqg.sid.entities import EntityGraph, extract_entities
from arqg.sid.env import build_env
from arqg.sid.export import dedup, datamix_stats, minhash, split_pool
from arqg.sid.facts import span_is_verbatim
from arqg.sid.gates import cheap_gates, g_min, remeasure
from arqg.sid.lexical import BM25Index, tokenize
from arqg.sid.mockllm import make_sid_client, sid_mock_handler
from arqg.sid.prompts import _facts_block, facts_user
from arqg.sid.retrieval import Probe, aggregate_gaps_over_groups, gap_bin
from arqg.sid.schema import Candidate
from arqg.sid.sections import scope_of, shared_depth
from arqg.sid.simbridge import CoRetrievability, SimBand, scope_pairs
from arqg.sid.subgraphs import (_Admissible, _scope_bridges, build_entity_graph,
                                mine_subgraphs, run_mining)
from arqg.embeddings import MockEmbedder
from arqg.schema import Chunk
from arqg.utils import read_jsonl

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "sample_corpus_sid.jsonl")


def _cfg(tmp_path) -> SidConfig:
    cfg = SidConfig()
    cfg.paths.corpus = CORPUS
    cfg.paths.out_dir = str(tmp_path)
    cfg.llm.backend = cfg.judge.backend = "mock"
    cfg.embed.backend = "mock"
    cfg.corpus_name = "demo"
    return cfg


# --------------------------------------------------------------------------- #
# Lexical branch
# --------------------------------------------------------------------------- #
def test_bm25_ranks_and_scores_targets():
    idx = BM25Index()
    idx.add_many([
        ("a", "гидроакустический буй Тритон-3 измеряет солёность воды"),
        ("b", "национальный парк площадью двенадцать тысяч гектаров"),
        ("c", "буй передаёт данные по спутниковому каналу"),
    ])
    hits, targets = idx.search_with_targets("буй Тритон-3", 3, ["a", "b"])
    assert hits[0][0] == "a"
    assert targets["a"] > targets["b"] == 0.0      # one pass gives lex_gap for free
    assert idx.idf("буй") < idx.idf("тритон")      # rarer term carries more idf


def test_stemming_collides_inflections():
    # inflected forms of a long enough stem must land on the same token
    assert len(set(tokenize("компания компании компанию компаний"))) == 1
    assert len(set(tokenize("подразделение подразделения"))) == 1
    # short words are left alone rather than mangled
    assert tokenize("буй") == ["буй"]


# --------------------------------------------------------------------------- #
# S1 — entities and subgraphs
# --------------------------------------------------------------------------- #
def test_entity_extraction_finds_names_dates_codes():
    text = ('Компания «Север» в 2015 году выпустила буй «Тритон-3» '
            'по условиям ТУ-4137 на сумму 4 млрд рублей.')
    surfaces = {e.surface for e in extract_entities(text)}
    assert "Север" in surfaces and "Тритон-3" in surfaces
    assert "2015" in surfaces
    assert any(s.startswith("ТУ") for s in surfaces)
    # a sentence-initial capitalised common noun is not an entity
    assert "Компания" not in surfaces


def test_idf_threshold_uses_the_bridge_eligible_population():
    """Estimated over every surface form, the percentile collapses onto the
    hapax maximum and admits nothing that can bridge two chunks."""
    g = EntityGraph()
    for i in range(20):
        g.add_chunk(f"c{i}", f"В отчёте упомянут объект «Альфа-{i}» отдельно.")
    g.add_chunk("c20", "В отчёте упомянут объект «Мостик» тут.")
    g.add_chunk("c21", "В отчёте упомянут объект «Мостик» и здесь.")
    naive = g.idf_threshold(75, min_df=1)
    scoped = g.idf_threshold(75, min_df=2)
    # over every surface the percentile sits on the hapax maximum, so nothing
    # with df >= 2 could ever clear it
    assert g.idf("мостик") < naive
    assert g.idf("мостик") >= scoped        # the only real bridge survives


def test_mining_produces_subgraphs_with_shared_entities(tmp_path):
    cfg = _cfg(tmp_path)
    corpus = SidCorpus.load(CORPUS)
    subgraphs = mine_subgraphs(cfg, corpus)
    assert len(subgraphs) >= 5
    for s in subgraphs:
        assert 2 <= len(s.chunks) <= cfg.mining.max_chunks
        assert s.index_version == "v0"
        assert s.bridge_entities, "a subgraph must be held together by something"
        for ent in s.bridge_entities:
            assert len(ent["chunks"]) >= 2


def test_mining_can_require_cross_document(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.mining.require_cross_document = True
    corpus = SidCorpus.load(CORPUS)
    for s in mine_subgraphs(cfg, corpus):
        assert len({c.split("::")[0] for c in s.chunks}) >= 2


# --------------------------------------------------------------------------- #
# S1 — section scoping
# --------------------------------------------------------------------------- #
def test_scope_of_drops_the_leaf_and_refuses_shallow_paths():
    t = "Пространство/Дефекты/СберБизнес/Платежи/Дефект 16471"
    assert scope_of(t, gap=0) == "Пространство/Дефекты/СберБизнес/Платежи"
    assert scope_of(t, gap=1) == "Пространство/Дефекты/СберБизнес"
    # a scope of one segment is the whole corpus — grouping on it is not grouping
    assert scope_of("Пространство/Дефекты/Дефект 16471", gap=1, min_depth=2) == ""
    assert scope_of("плоский заголовок", gap=1) == ""


def test_shared_depth_measures_topical_distance():
    assert shared_depth(["A/B/C/x", "A/B/C/y"]) == 3
    assert shared_depth(["A/B/C/x", "A/Z/Q/y"]) == 1     # only the root: unrelated
    assert shared_depth([]) == 0


def _filler(seed: str) -> str:
    """A chunk long enough to clear the seed-eligibility filters."""
    return (f"{seed} Настоящий раздел описывает порядок обработки обращения "
            "клиента и перечисляет действия оператора при возникновении "
            "нестандартной ситуации в работе сервиса, включая последовательность "
            "проверок и подготовку ответа заявителю по установленной форме.")


def test_mined_subgraphs_stay_inside_one_section(tmp_path):
    """The pathology scoping exists for: chunks bridged by a coincidence that
    share nothing but the corpus root."""
    cfg = _cfg(tmp_path)
    corpus = SidCorpus.load(CORPUS)
    subgraphs = mine_subgraphs(cfg, corpus)
    assert subgraphs
    titles = {c.id: c.title for c in corpus.all_chunks()}
    for s in subgraphs:
        assert s.path_scope, "every demo subgraph is scopeable"
        assert s.path_shared_depth >= cfg.mining.min_scope_depth
        assert shared_depth([titles[c] for c in s.chunks]) == s.path_shared_depth


def test_flat_titles_fall_back_to_the_unscoped_search(tmp_path):
    """A corpus without breadcrumbs must still be minable — the scope is an
    opportunity this index offers, not a requirement of the algorithm."""
    cfg = _cfg(tmp_path)
    flat = [Chunk(file_name=c.file_name, index=c.index, raw_text=c.raw_text,
                  document_id=c.document_id, title="")
            for c in SidCorpus.load(CORPUS).all_chunks()]
    subgraphs = mine_subgraphs(cfg, SidCorpus(flat))
    assert subgraphs, "flat titles must not starve mining"
    assert all(s.path_scope == "" for s in subgraphs)


def test_chunk_scope_of_dispatches_path_and_exact_strategies():
    """`scoping.py` is the layer above `sections.py`: it picks which field a
    chunk is grouped by (a builtin attribute or a `meta` key) and how a value
    becomes a scope key. `"path"` delegates to the breadcrumb mechanics
    unchanged; `"exact"` groups a flat categorical facet by its literal value."""
    from arqg.sid.scoping import scope_of as chunk_scope_of

    c = Chunk(file_name="d.txt", index=0, raw_text="x",
             title="A/B/C/Документ", meta={"region": "77", "empty": ""})
    assert chunk_scope_of(c, "title", "path", gap=0) == "A/B/C"
    assert chunk_scope_of(c, "region", "exact") == "77"
    assert chunk_scope_of(c, "empty", "exact") == ""
    assert chunk_scope_of(c, "missing_field", "exact") == ""
    with pytest.raises(ValueError):
        chunk_scope_of(c, "title", "no_such_strategy")


def test_group_by_scope_can_key_on_a_meta_field_instead_of_title(tmp_path):
    """The same grouping mechanism S1 uses for the title breadcrumb, pointed
    instead at an arbitrary `Chunk.meta` facet (e.g. zakupki's `region`)."""
    from arqg.sid.subgraphs import _group_by_scope

    cfg = _cfg(tmp_path)
    cfg.mining.scope_field = "region"
    cfg.mining.scope_strategy = "exact"
    chunks = [Chunk(file_name=f"d{i}.txt", index=0, raw_text=f"x{i}",
                    meta={"region": "77" if i < 3 else "78"}) for i in range(5)]
    corpus = SidCorpus(chunks)
    scoped, residue = _group_by_scope(corpus, {c.id for c in chunks}, cfg.mining)
    assert scoped == {"77": [c.id for c in chunks[:3]],
                      "78": [c.id for c in chunks[3:]]}
    assert residue == []


def test_group_by_scope_sends_chunks_missing_the_field_to_the_residue(tmp_path):
    from arqg.sid.subgraphs import _group_by_scope

    cfg = _cfg(tmp_path)
    cfg.mining.scope_field = "region"
    cfg.mining.scope_strategy = "exact"
    chunks = [Chunk(file_name=f"d{i}.txt", index=0, raw_text=f"x{i}",
                    meta={"region": "77"} if i < 2 else {}) for i in range(3)]
    corpus = SidCorpus(chunks)
    scoped, residue = _group_by_scope(corpus, {c.id for c in chunks}, cfg.mining)
    assert set(scoped.get("77", [])) == {chunks[0].id, chunks[1].id}
    assert residue == [chunks[2].id]


def test_mining_scopes_by_an_exact_meta_field_not_only_by_title(tmp_path):
    """A corpus with no breadcrumb `title` at all can still scope S1's search —
    on any meta facet, grouped by exact value rather than by folder. Two
    regions, each carrying its own bridging entity; a subgraph must not mix
    chunks from both."""
    cfg = _cfg(tmp_path)
    cfg.mining.scope_field = "region"
    cfg.mining.scope_strategy = "exact"
    chunks = []
    for i in range(4):
        region = "77" if i < 2 else "78"
        mark = "«Тритон-9»" if region == "77" else "«Альфа-1»"
        chunks.append(Chunk(file_name=f"d{i}.txt", index=0,
                            raw_text=_filler(f"Документ {i}.") + f" Изделие {mark} прошло приёмку.",
                            meta={"region": region}))
    corpus = SidCorpus(chunks)
    subgraphs = mine_subgraphs(cfg, corpus)
    assert subgraphs, "an exact-match facet must be usable as a scope"
    for s in subgraphs:
        regions = {corpus.get(cid).meta.get("region") for cid in s.chunks}
        assert len(regions) == 1, "an exact-strategy scope must not mix values"
        assert s.path_scope in ("77", "78")


def test_scope_bridge_rejects_the_folders_own_subject(tmp_path):
    """Inside a folder, discrimination is local: an entity in most of the
    folder's chunks is what the folder is *about*, not a bridge between two of
    its documents. Global rarity cannot express that."""
    cfg = _cfg(tmp_path)
    chunks = []
    for i in range(6):
        # «Гидрология» is in every chunk (the subject); «Тритон-9» in two
        extra = " Изделие «Тритон-9» прошло приёмку." if i in (1, 4) else ""
        chunks.append(Chunk(file_name=f"d{i}.txt", index=0,
                            raw_text=_filler("Проект «Гидрология».") + extra,
                            title=f"База/Море/Отчёты/Документ {i}"))
    corpus = SidCorpus(chunks)
    graph = build_entity_graph(cfg, corpus)
    adm = _Admissible(cfg.mining, {c.file_name: 1 for c in chunks})
    keys = {b.key for b in _scope_bridges(graph, [c.id for c in chunks], adm, 0.0)}
    assert "тритон-9" in keys
    assert "гидрология" not in keys, "the folder's subject is not a bridge"


def test_same_document_pairs_need_a_navigable_document(tmp_path):
    """`min_index_gap` alone is not the test: positions 0 and 9 of a ten-chunk
    page are one read apart, while the same gap in a long document is a
    different section."""
    cfg = _cfg(tmp_path)
    cfg.mining.same_doc_min_chunks = 20
    cfg.mining.min_index_gap = 8

    def corpus_of(n_chunks: int) -> SidCorpus:
        rows = []
        for i in range(n_chunks):
            mark = " Код ТУ-7731 указан в акте." if i in (0, 12) else ""
            rows.append(Chunk(file_name="doc.txt", index=i,
                              raw_text=_filler(f"Раздел {i}.") + mark,
                              title="База/Море/Отчёты/Единственный документ"))
        return SidCorpus(rows)

    long_doc = mine_subgraphs(cfg, corpus_of(30))
    short_doc = mine_subgraphs(cfg, corpus_of(14))
    assert long_doc, "a 30-chunk document may bridge positions 0 and 12"
    assert not short_doc, "a 14-chunk document is read, not navigated"


# --------------------------------------------------------------------------- #
# S1 — the doc2doc channel
# --------------------------------------------------------------------------- #
def _folder_without_a_repeated_entity() -> SidCorpus:
    """Four documents of one folder, no surface form shared by any two."""
    marks = ["«Альфа-1»", "«Бета-2»", "«Гамма-3»", "«Дельта-4»"]
    return SidCorpus([
        Chunk(file_name=f"d{i}.txt", index=0,
              raw_text=_filler(f"Изделие {mark} прошло приёмку."),
              title=f"База/Море/Отчёты/Документ {i}")
        for i, mark in enumerate(marks)])


def _dense_at_angles(ids: list[str], degrees: list[float]) -> DenseIndex:
    """Unit vectors on a circle: cos(i, j) is exactly cos(θi − θj), so a fixture
    states the similarities it means instead of approximating them."""
    rad = np.radians(np.asarray(degrees, dtype="float64"))
    mat = np.stack([np.cos(rad), np.sin(rad)], axis=1).astype("float32")
    return DenseIndex(list(ids), mat)


def test_sim_band_keeps_pairs_above_the_corpus_background():
    """Being in one folder is not a relation; the lower edge is where a pair
    stops being as far apart as two random chunks."""
    ids = ["a::0", "b::0", "c::0"]
    dense = _dense_at_angles(ids, [0, 45, 89])        # cos: .71, .02, .71
    band = SimBand(low=0.5, exclude_top_k=0, percentile=95)
    pairs = scope_pairs(dense, ids, band, limit=10)
    assert {frozenset(p) for p, _ in pairs} == {frozenset(("a::0", "b::0")),
                                                frozenset(("b::0", "c::0"))}
    # least co-retrievable first: inside a folder that is the axis left to spend
    assert [round(s, 2) for _, s in pairs] == sorted(round(s, 2) for _, s in pairs)


def test_co_retrievability_is_a_rank_not_a_cosine():
    """Measured on the first `ckr` run: with the partner inside the top-3
    neighbours G_BROAD passed 0.22 of the time and 0.13 became tasks, against
    0.55 / 0.30 beyond rank 50 — one query already returns both. The cosine
    cannot express it here (p96 = 0.54, p98 = 0.80), the rank can."""
    ids = [f"c{i}::0" for i in range(5)]
    dense = _dense_at_angles(ids, [0, 5, 10, 15, 60])
    far = float(dense.vec("c0::0") @ dense.vec("c4::0"))     # cos 60° = 0.5
    assert far == pytest.approx(0.5, abs=1e-3)
    # 0.5 is a high cosine, yet three chunks sit closer to c0 than c4 does
    assert not CoRetrievability(dense, 3).co_retrievable("c0::0", "c4::0", far)
    assert CoRetrievability(dense, 4).co_retrievable("c0::0", "c4::0", far)


def test_similarity_bridges_a_folder_no_entity_can(tmp_path):
    """The channel's whole reason to exist: on `ckr` 451 folders contain no
    surface form repeated in two chunks, so they are invisible to the entity
    miner rather than filtered out by one of its thresholds."""
    cfg = _cfg(tmp_path)
    corpus = _folder_without_a_repeated_entity()
    assert not mine_subgraphs(cfg, corpus), "no entity bridges this folder"

    cfg.mining.sim_bridge = True
    cfg.mining.sim_bridge_low_percentile = 50
    cfg.mining.sim_bridge_exclude_top_k = 0     # covered on its own above
    ids = [c.id for c in corpus.all_chunks()]
    dense = _dense_at_angles(ids, [0, 40, 88, 89])
    subgraphs = mine_subgraphs(cfg, corpus, dense)
    assert subgraphs, "doc2doc must reach a folder the entity channel cannot"
    assert all(s.bridge_kind == "similarity" for s in subgraphs)
    for s in subgraphs:
        a, b = s.chunks[0], s.chunks[1]
        assert s.pair_similarity == pytest.approx(float(dense.vec(a) @ dense.vec(b)),
                                                  abs=1e-3)


def test_the_rank_ceiling_is_what_rejects_a_trivial_pair(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.mining.sim_bridge = True
    cfg.mining.sim_bridge_low_percentile = 50
    corpus = _folder_without_a_repeated_entity()
    ids = [c.id for c in corpus.all_chunks()]
    dense = _dense_at_angles(ids, [0, 40, 88, 89])

    cfg.mining.sim_bridge_exclude_top_k = 3
    assert not mine_subgraphs(cfg, corpus, dense), "every partner is a top-3 neighbour here"
    cfg.mining.sim_bridge_exclude_top_k = 0
    assert mine_subgraphs(cfg, corpus, dense), "the rank ceiling is what rejected it"


def test_similarity_channel_needs_the_dense_index(tmp_path):
    """Enabled without an index the stage degrades to the entity channel rather
    than mining nothing."""
    cfg = _cfg(tmp_path)
    cfg.mining.sim_bridge = True
    corpus = SidCorpus.load(CORPUS)
    assert mine_subgraphs(cfg, corpus, None)


def test_entity_bridges_are_spent_before_similarity_ones(tmp_path):
    """A named bridge says what the chunks have in common; a similarity bridge
    only asserts that something does. The folder's budget goes to the former."""
    cfg = _cfg(tmp_path)
    cfg.mining.sim_bridge = True
    cfg.mining.sim_bridge_low_percentile = 50
    cfg.mining.sim_bridge_exclude_top_k = 0
    cfg.mining.max_subgraphs_per_path = 1
    chunks = [Chunk(file_name=f"d{i}.txt", index=0,
                    raw_text=_filler("Изделие «Тритон-9» прошло приёмку."
                                     if i in (2, 3) else f"Позиция «Объект-{i}»."),
                    title=f"База/Море/Отчёты/Документ {i}") for i in range(4)]
    corpus = SidCorpus(chunks)
    dense = _dense_at_angles([c.id for c in chunks], [0, 40, 88, 89])
    subgraphs = mine_subgraphs(cfg, corpus, dense)
    assert [s.bridge_kind for s in subgraphs] == ["entity"]


def test_shared_anchor_guard_holds_similarity_subgraphs_only():
    """S3, not S1, decides whether a doc2doc pair has anything to build a link
    on — the facts are extracted either way and they carry the paraphrase."""
    facts = {
        "a::0": [{"fact_id": "f1", "chunk_id": "a::0", "entities": ["Дефекту 16471"],
                  "discriminating_attributes": ["date:2019"]}],
        "b::0": [{"fact_id": "f2", "chunk_id": "b::0", "entities": ["дефект 16471"],
                  "discriminating_attributes": []}],
        "c::0": [{"fact_id": "f3", "chunk_id": "c::0", "entities": ["Эквайринг"],
                  "discriminating_attributes": []}],
    }
    assert has_shared_anchor({"chunks": ["a::0", "b::0"]}, facts), "inflection is not a difference"
    assert not has_shared_anchor({"chunks": ["a::0", "c::0"]}, facts)
    # an entity subgraph is never asked: its bridge IS the anchor
    cfg = SidConfig()
    sim = {"subgraph_id": "s1", "chunks": ["a::0", "c::0"], "bridge_kind": "similarity"}
    ent = {"subgraph_id": "s2", "chunks": ["a::0", "c::0"], "bridge_kind": "entity"}
    kept = {s["subgraph_id"] for _, members in plan_batches(cfg, [sim, ent], facts)
            for _, s in members}
    assert kept == {"s2"}


def test_per_file_quota_counts_subgraphs_not_member_chunks(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.mining.max_subgraphs_per_file = 2
    corpus = SidCorpus.load(CORPUS)
    counts = {}
    for s in mine_subgraphs(cfg, corpus):
        for doc in {c.split("::")[0] for c in s.chunks}:
            counts[doc] = counts.get(doc, 0) + 1
    assert counts, "the quota must not empty the pool"
    assert max(counts.values()) <= cfg.mining.max_subgraphs_per_file


# --------------------------------------------------------------------------- #
# S3 — facts
# --------------------------------------------------------------------------- #
def test_verbatim_span_check():
    chunk = "Компания  «Север»   была основана в 1992 году."
    assert span_is_verbatim("была основана в 1992 году", chunk)   # whitespace-normalised
    assert not span_is_verbatim("была основана в 1993 году", chunk)
    assert not span_is_verbatim("", chunk)


def test_section_is_prompt_context_but_never_a_fact_source():
    """The breadcrumb is given to the extractor so `fact_normalized` can be
    self-contained, and the verbatim check still runs against the chunk text
    alone — so a span lifted from the heading is dropped, not repaired."""
    section = "База/Дефекты/СберБизнес/Дефект DCBHCK6-16471"
    text = "Статус дефекта: открыт. Обходное решение отсутствует."
    prompt = facts_user("f.html::0", text, 4, section)
    assert "СберБизнес" in prompt and "Дефект DCBHCK6-16471" in prompt
    assert not span_is_verbatim("Дефект DCBHCK6-16471", text)
    assert span_is_verbatim("Обходное решение отсутствует", text)


def test_composer_is_shown_the_section_of_every_fact():
    facts = [{"fact_id": "f_1", "chunk_id": "a::0", "fact_normalized": "первый факт",
              "verbatim_span": "цитата один", "section": "База/Море/Отчёты/Док"},
             {"fact_id": "f_2", "chunk_id": "b::0", "fact_normalized": "второй факт",
              "verbatim_span": "цитата два"}]
    block = _facts_block(facts)
    assert "раздел: База > Море > Отчёты > Док" in block
    assert block.count("раздел:") == 1, "a fact without a section renders no line"


# --------------------------------------------------------------------------- #
# §4.3 — gap aggregation
# --------------------------------------------------------------------------- #
def test_gaps_take_min_inside_a_group_and_max_across_groups():
    probe = Probe(query="q")
    probe.fused_gap = {"easy": 0.1, "dup_of_easy": 0.9, "hard": 0.5}
    probe.lex_gap = dict(probe.fused_gap)
    probe.dense_gap = dict(probe.fused_gap)
    groups = [["easy", "dup_of_easy"], ["hard"]]
    out = aggregate_gaps_over_groups(probe, groups)
    # the redundant duplicate must not make an easy fact look hard
    assert out["fused_gap"] == 0.5
    assert gap_bin(out["fused_gap"], {"low": 0.33, "mid": 0.66}) == "mid"


# --------------------------------------------------------------------------- #
# S4/S5 — gates
# --------------------------------------------------------------------------- #
def _candidate(corpus: SidCorpus, chunk_ids, question="Тестовый вопрос?"):
    facts = [{"fact_id": f"f{i}", "chunk_id": c,
              "verbatim_span": corpus.text(c)[:80],
              "fact_normalized": corpus.text(c)[:80],
              "entities": [], "discriminating_attributes": []}
             for i, c in enumerate(chunk_ids)]
    return Candidate(candidate_id="c1", batch_id="b1", instantiation_rank=0,
                     subgraph_id="sg1", corpus="demo", language="ru",
                     question=question, answer="ответ", facts=facts,
                     mechanic="entity_chain", submechanic="x", has_negation=False,
                     hop_depth=len(facts))


def test_g_broad_rejects_a_question_that_already_returns_all_gold(tmp_path):
    cfg = _cfg(tmp_path)
    env = asyncio.run(build_env(cfg, version="v0"))
    corpus = env.corpus
    gold = ["org_00.txt::0", "org_00.txt::1"]
    # a question that is literally the two gold chunks retrieves both -> trivial
    trivial = corpus.text(gold[0]) + " " + corpus.text(gold[1])
    res = asyncio.run(cheap_gates(cfg, env, _candidate(corpus, gold, trivial)))
    assert res["broad_ok"] is False
    assert res["metrics"]["broad_query_hit_at_k"] == 1.0
    # a vague question does not, so G_BROAD lets it through
    res2 = asyncio.run(cheap_gates(cfg, env, _candidate(corpus, gold, "что известно?")))
    assert res2["broad_ok"] is True
    asyncio.run(env.aclose())


def test_g_reach_probes_each_gold_chunk(tmp_path):
    cfg = _cfg(tmp_path)
    env = asyncio.run(build_env(cfg, version="v0"))
    cand = _candidate(env.corpus, ["org_00.txt::0", "org_02.txt::2"], "вопрос?")
    res = asyncio.run(cheap_gates(cfg, env, cand))
    # probes come from the facts' own spans, so a real chunk is reachable
    assert res["reach_ok"] is True and res["unreachable"] == []
    asyncio.run(env.aclose())


def test_g_min_removes_a_redundant_fact_and_stops_at_the_floor():
    """Leave-one-fact-out with recheck: a critic that always says 'solvable'
    must still not strip the task below the fact floor."""
    cfg = SidConfig()
    cfg.gates.min_facts_after_min = 2
    facts = [{"fact_id": f"f{i}", "chunk_id": f"c{i}", "verbatim_span": "s",
              "fact_normalized": "n"} for i in range(4)]

    class AlwaysSolvable:
        async def complete_json(self, *_a, **_k):
            return {"solvable": True, "answer_correct": True}

    kept, removed = asyncio.run(g_min(cfg, AlwaysSolvable(), "q", "a", facts))
    assert len(kept) == 2 and removed == 2

    class NeverSolvable:
        async def complete_json(self, *_a, **_k):
            return {"solvable": False, "answer_correct": True}

    kept2, removed2 = asyncio.run(g_min(cfg, NeverSolvable(), "q", "a", facts))
    assert len(kept2) == 4 and removed2 == 0     # nothing is redundant


def test_repair_loop_rewrites_against_critic_feedback():
    from arqg.sid.gates import recompose
    cfg = SidConfig()
    cfg.llm.backend = "mock"
    facts = [{"fact_id": "f_0", "chunk_id": "a::0", "verbatim_span": "x",
              "fact_normalized": "Компания «Север» основана в 1992 году"},
             {"fact_id": "f_1", "chunk_id": "b::0", "verbatim_span": "y",
              "fact_normalized": "Буй «Тритон-3» вышел на рынок в 2001 году"}]
    cand = Candidate(candidate_id="c1", batch_id="b1", instantiation_rank=0,
                     subgraph_id="sg1", corpus="d", language="ru",
                     question="плохой вопрос?", answer="a", facts=facts,
                     mechanic="comparison", submechanic="s", has_negation=False,
                     hop_depth=2)
    fixed = asyncio.run(recompose(cfg, make_sid_client(cfg.llm), cand,
                                  "вопрос содержит ответ", 2))
    assert fixed is not None
    assert fixed.candidate_id == cand.candidate_id     # same candidate, retried
    assert fixed.question != cand.question
    assert fixed.compose_iters == 2


def test_remeasure_uses_the_final_gold_set(tmp_path):
    cfg = _cfg(tmp_path)
    env = asyncio.run(build_env(cfg, version="v0"))
    rec = {"question": "буй Тритон-3", "gold_chunk_ids": ["org_00.txt::1"],
           "fact_groups": [["org_00.txt::1"]]}
    m = asyncio.run(remeasure(cfg, env, rec))
    assert 0.0 <= m["fused_gap"] <= 1.0
    assert m["broad_query_hit_at_k"] == 1.0      # its own text retrieves it
    asyncio.run(env.aclose())


# --------------------------------------------------------------------------- #
# §7.1 — density
# --------------------------------------------------------------------------- #
def test_density_thresholds_are_percentiles_and_task_density_is_a_min(tmp_path):
    cfg = _cfg(tmp_path)
    env = asyncio.run(build_env(cfg, version="v0"))
    model = fit_density_model(cfg, env, None)
    assert model.tau_low < model.tau_sim <= 1.0
    assert model.similarity_shape["p95"] == pytest.approx(model.tau_sim, abs=1e-3)
    # the working norm stays provisional until the reach population is big enough
    assert model.reach_median_is_working is False
    assert model.median == model.density_median_all

    a, b = "org_00.txt::0", "park_00.txt::1"
    da = task_density(env, [a], model.tau_sim)
    db = task_density(env, [b], model.tau_sim)
    assert task_density(env, [a, b], model.tau_sim) == min(da, db)
    asyncio.run(env.aclose())


# --------------------------------------------------------------------------- #
# §7.5 — distractor verification
# --------------------------------------------------------------------------- #
def test_answer_leak_detection():
    assert _answer_leaks("четыре миллиарда рублей",
                         "Оборот составил четыре миллиарда рублей в 2021 году.")
    assert not _answer_leaks("четыре миллиарда рублей", "Компания открыла завод.")


def test_l2_attribute_must_be_observable_in_the_question():
    assert _attribute_in_question("date:2015 -> date:2018",
                                  "Что изменилось в 2015 году?")
    assert not _attribute_in_question("date:2015 -> date:2018",
                                      "Где находится подразделение?")
    # the attribute *type* must not satisfy the check on its own
    assert not _attribute_in_question("date:2015", "какая date указана?")


def test_verification_rejects_out_of_neighbourhood_candidates(tmp_path):
    cfg = _cfg(tmp_path)
    env = asyncio.run(build_env(cfg, version="v0"))
    model = fit_density_model(cfg, env, None)
    judge = make_sid_client(cfg.judge)
    rec = {"question": "вопрос про буй?", "answer": "ответ",
           "gold_chunk_ids": ["org_00.txt::1"], "facts": []}
    far = DistractorCandidate(text="Совершенно посторонний текст о погоде.",
                              level="L1_transplant", dtype="topical_lure",
                              source_chunk_id="park_00.txt::0")
    inj = TaskInjection(task_id="t1")
    kept = asyncio.run(verify_candidates(cfg, judge, env, rec, [far], model, inj))
    assert kept == [] and inj.rejected["neighborhood"] == 1
    asyncio.run(env.aclose())


def test_injected_chunk_is_embedded_the_way_the_index_holds_it(tmp_path):
    """A distractor is a chunk of this index like any other.

    With `embed_with_title` on, every v0 chunk is embedded as "title\ntext".
    Embedding a distractor as bare text puts it somewhere else in the space
    than where it will sit once v1 is rebuilt from disk — so §7.5's
    "does it land in the gold neighbourhood?" would be measured on a vector
    nothing ever retrieves against.
    """
    from arqg.sid.distractors import _passage, inject

    cfg = _cfg(tmp_path)
    assert cfg.embed.embed_with_title
    env = asyncio.run(build_env(cfg, version="v0"))
    donor = env.corpus.get("org_00.txt::1")
    assert donor.title, "the fixture must carry a breadcrumb title"

    cand = DistractorCandidate(text="Совсем другой текст про измерения.",
                               level="L2_perturbed", dtype="near_duplicate",
                               source_chunk_id=donor.id, title=donor.title)
    assert _passage(env, cand) == f"{donor.title}\n{cand.text}"

    vecs = asyncio.run(env.embedder.embed([_passage(env, cand)], kind="passage"))
    rec = {"task_id": "t1", "question": "вопрос?", "answer": "ответ",
           "gold_chunk_ids": ["org_00.txt::0"]}
    summary = inject(cfg, env, rec, TaskInjection(task_id="t1", accepted=[cand]), vecs)
    cid = summary["chunk_ids"][0]

    with_title = asyncio.run(
        env.embedder.embed([f"{donor.title}\n{cand.text}"], kind="passage"))[0]
    without = asyncio.run(env.embedder.embed([cand.text], kind="passage"))[0]
    assert np.allclose(env.dense.vec(cid), with_title)
    assert not np.allclose(env.dense.vec(cid), without)
    # and the title the chunk carries is the one that was embedded, so a
    # rebuild of v1 from corpus_injected.jsonl reproduces the same vector
    assert env.corpus.get(cid).title == donor.title
    asyncio.run(env.aclose())


def test_generated_distractor_inherits_its_donor_title(tmp_path):
    from arqg.sid.distractors import _gen_l2

    cfg = _cfg(tmp_path)
    env = asyncio.run(build_env(cfg, version="v0"))
    gen = make_sid_client(cfg.llm)
    rec = {"question": "вопрос?", "answer": "ответ",
           "facts": [{"discriminating_attributes": ["date:2015"]}]}
    cand = asyncio.run(_gen_l2(gen, env, rec, "org_00.txt::1"))
    assert cand is not None
    assert cand.title == env.corpus.get("org_00.txt::1").title
    asyncio.run(gen.aclose())
    asyncio.run(env.aclose())


def test_v1_index_is_saved_so_the_next_stage_does_not_re_embed(tmp_path):
    """The dense cache is keyed on the corpus checksum, which every injection
    changes. If S6 does not save the index it mutated, S7 — and any resumed
    `distract` — re-embeds the whole corpus for vectors the process was
    already holding."""
    import shutil

    from arqg.sid.distractors import save_index

    class CountingEmbedder(MockEmbedder):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.embedded = 0

        async def embed(self, texts, kind):
            if kind == "passage":
                self.embedded += len(texts)
            return await super().embed(texts, kind)

    cfg = _cfg(tmp_path)
    env = asyncio.run(build_env(cfg, version="v0"))
    chunk = env.corpus.inject(donor_file="org_00.txt", text="Текст-дистрактор для t1.",
                              task_id="t1", level="L2_perturbed",
                              dtype="near_duplicate", source_chunk_id="org_00.txt::1",
                              title=env.corpus.get("org_00.txt::1").title)
    vec = asyncio.run(env.embedder.embed([chunk.raw_text], kind="passage"))
    env.searcher.add_documents([chunk.id], [chunk.raw_text], vec)
    save_index(cfg, env)
    asyncio.run(env.aclose())

    counting = CountingEmbedder(cfg.embed)
    env2 = asyncio.run(build_env(cfg, version="v1", embedder=counting))
    assert counting.embedded == 0, "v1 must come from cache, not from the embedder"
    assert chunk.id in env2.dense.ids and len(env2.corpus) == len(env2.dense.ids)
    asyncio.run(env2.aclose())

    # without the saved index the same stage pays for the whole corpus again —
    # that is the regression this guards
    shutil.rmtree(cfg.paths.dense_dir("v1"))
    counting2 = CountingEmbedder(cfg.embed)
    env3 = asyncio.run(build_env(cfg, version="v1", embedder=counting2))
    assert counting2.embedded == len(env3.corpus)
    asyncio.run(env3.aclose())


# --------------------------------------------------------------------------- #
# §2.3 — versioning and marker containment
# --------------------------------------------------------------------------- #
def test_injection_is_additive_and_markers_stay_internal(tmp_path):
    corpus = SidCorpus.load(CORPUS)
    n_before = len(corpus)
    chunk = corpus.inject(donor_file="org_00.txt", text="Новый текст-дистрактор.",
                          task_id="t1", level="L2_perturbed", dtype="near_duplicate",
                          source_chunk_id="org_00.txt::1")
    assert chunk is not None
    assert len(corpus) == n_before + 1               # additive: nothing removed
    assert corpus.is_synthetic(chunk.id)

    public = str(tmp_path / "pub.jsonl")
    corpus.export_public(public)
    rows = list(read_jsonl(public))
    assert len(rows) == n_before + 1
    blob = json.dumps(rows, ensure_ascii=False)
    for marker in ("synthetic", "injected_for_task", "L2_perturbed", "near_duplicate"):
        assert marker not in blob, f"{marker} leaked into the agent-visible corpus"

    ledger = str(tmp_path / "ledger.jsonl")
    corpus.save_ledger(ledger)
    entries = list(read_jsonl(ledger))
    assert entries[0]["task_id"] == "t1" and entries[0]["level"] == "L2_perturbed"


def test_duplicate_text_is_rejected_by_content_hash():
    corpus = SidCorpus.load(CORPUS)
    existing = corpus.text("org_00.txt::0")
    assert corpus.has_duplicate_text(existing)
    assert corpus.inject(donor_file="org_00.txt", text=existing, task_id="t",
                         level="L1_transplant", dtype="topical_lure",
                         source_chunk_id="org_00.txt::0") is None
    assert content_hash("  A  b ") == content_hash("a b")


# --------------------------------------------------------------------------- #
# S8 — export
# --------------------------------------------------------------------------- #
def test_minhash_dedup_drops_near_duplicate_questions():
    rows = [{"question": "Какой буй выпускает компания Север и когда он вышел?"},
            {"question": "Какой буй выпускает компания Север и когда он вышел."},
            {"question": "Сколько туристов посещает национальный парк ежегодно?"}]
    kept, dropped = dedup(rows, 0.8)
    assert dropped == 1 and len(kept) == 2
    assert minhash("a b c d") == minhash("a b c d")


def test_stats_demand_group_granularity_when_groups_are_not_singletons():
    def task(groups):
        return {"task_id": "t", "coverage": {"A1_mechanic": "comparison",
                                             "A1_submechanic": "s", "has_negation": False},
                "complexity": {"hop_depth": 2, "fused_gap_bin": "mid"},
                "gold_chunk_ids": [c for g in groups for c in g],
                "fact_groups": groups, "distractors": {}}
    singleton = datamix_stats([task([["a"], ["b"]])])
    assert singleton["gold"]["share_singleton_groups"] == 1.0
    assert singleton["gold"]["ndcg_granularity"] == "chunk"
    grouped = datamix_stats([task([["a", "a2"], ["b"]])])
    assert grouped["gold"]["ndcg_granularity"] == "fact_group"


def test_split_is_train_holdout_with_no_gold_leak():
    cfg = SidConfig()
    cfg.export.holdout_size = 2
    tasks = [{"task_id": f"t{i}",
              "coverage": {"A1_mechanic": "comparison", "A1_submechanic": "s",
                           "has_negation": False},
              "complexity": {"hop_depth": 2, "fused_gap_bin": "mid"},
              "gold_chunk_ids": [f"c{i}", "shared" if i % 2 else f"x{i}"],
              "fact_groups": [[f"c{i}"]], "distractors": {}} for i in range(12)]
    splits = split_pool(cfg, tasks)
    assert set(splits) == {"train", "holdout"}        # no SFT/RL boundary here
    assert len(splits["holdout"]) > 0 and len(splits["train"]) > 0
    train_gold = {c for t in splits["train"] for c in t["gold_chunk_ids"]}
    holdout_gold = {c for t in splits["holdout"] for c in t["gold_chunk_ids"]}
    assert not (train_gold & holdout_gold)
    ids = {t["task_id"] for t in splits["train"]} | {t["task_id"] for t in splits["holdout"]}
    assert len(ids) == len(splits["train"]) + len(splits["holdout"])


# --------------------------------------------------------------------------- #
# Mock backend and end to end
# --------------------------------------------------------------------------- #
def test_mock_dispatches_on_the_prompt_tag():
    from arqg.sid.prompts import FACTS_SYS, SOLVE_SYS, facts_user
    out = sid_mock_handler(FACTS_SYS, facts_user("c::0",
                           "Компания «Север» основана в 1992 году в Архангельске. "
                           "Предприятие занималось ремонтом навигационного оборудования.", 3))
    assert out["facts"] and all("verbatim_span" in f for f in out["facts"])
    assert sid_mock_handler(SOLVE_SYS, "x")["solvable"] is True


def test_load_chunks_lifts_extra_record_fields_into_meta(tmp_path):
    """A corpus that inlines facets on the record itself (rather than keeping
    them in a separate sidecar) needs no extra step — anything beyond the five
    core fields lands on `Chunk.meta` for free."""
    from arqg.data import load_chunks

    path = tmp_path / "inline.jsonl"
    path.write_text(json.dumps({"file_name": "d.txt", "index": 0, "raw_text": "x",
                                "region": "77", "price_bucket": "small"},
                               ensure_ascii=False) + "\n", encoding="utf-8")
    chunks = load_chunks(str(path))
    assert len(chunks) == 1
    assert chunks[0].meta == {"region": "77", "price_bucket": "small"}


def test_corpus_load_merges_a_metadata_sidecar_by_chunk_id(tmp_path):
    """zakupki's `merge` keeps the corpus file to the five core fields and
    publishes everything else in a separate `*_meta.jsonl` keyed by
    `chunk_id` — `SidCorpus.load(..., meta_path=...)` folds it back on."""
    corpus_path = tmp_path / "corpus.jsonl"
    meta_path = tmp_path / "corpus_meta.jsonl"
    corpus_path.write_text(
        "\n".join(json.dumps({"file_name": "d.txt", "index": i, "raw_text": f"x{i}"},
                             ensure_ascii=False) for i in range(2)) + "\n",
        encoding="utf-8")
    meta_path.write_text(
        json.dumps({"chunk_id": "d.txt::0", "region": "77", "customer": "ООО Ромашка"},
                  ensure_ascii=False) + "\n",
        encoding="utf-8")

    corpus = SidCorpus.load(str(corpus_path), meta_path=str(meta_path))
    assert corpus.get("d.txt::0").meta == {"region": "77", "customer": "ООО Ромашка"}
    assert corpus.get("d.txt::1").meta == {}, "a chunk absent from the sidecar stays untouched"


def test_index_fields_reports_meta_facets_and_the_configured_scope(tmp_path):
    """S0's compat report is what a corpus operator reads to pick a scope: the
    facets an index actually carries, their coverage and cardinality, and
    which one `mining.scope_field`/`scope_strategy` currently points at."""
    from arqg.sid.compat import build_index_fields

    chunks = [Chunk(file_name=f"d{i}.txt", index=0, raw_text="x",
                    meta={"region": "77" if i < 2 else "78"}) for i in range(3)]
    corpus = SidCorpus(chunks)
    cfg = SidConfig()
    cfg.mining.scope_field = "region"
    cfg.mining.scope_strategy = "exact"

    fields = build_index_fields(corpus, cfg)
    assert fields["meta_fields"]["region"] == {"coverage": 1.0, "n_distinct": 2}
    assert fields["scope"] == {"field": "region", "strategy": "exact",
                               "source": "meta_fields"}


def test_compat_report_and_manifest(tmp_path):
    cfg = _cfg(tmp_path)
    report = run_compat(cfg)
    assert report["n_chunks"] == 54
    assert report["unit"]["units_aligned"] is True
    assert os.path.exists(cfg.paths.index_fields)
    manifest = json.load(open(cfg.paths.manifest, encoding="utf-8"))
    assert manifest["index_version"] == "v0" and manifest["n_injected"] == 0


def test_end_to_end_mock(tmp_path):
    from arqg.sid import pipeline
    cfg = _cfg(tmp_path)
    stats = asyncio.run(pipeline.run_all(cfg))

    tasks = list(read_jsonl(cfg.paths.tasks))
    assert tasks, "the pipeline must produce at least one task"
    assert stats["n_tasks"] == len(tasks)
    for t in tasks:
        assert t["question"] and t["answer"]
        assert len(t["gold_chunk_ids"]) >= 2
        # gold is exactly the union of the fact groups
        assert set(t["gold_chunk_ids"]) == {c for g in t["fact_groups"] for c in g}
        assert t["complexity"]["hop_depth"] >= 2
        assert t["complexity"]["fused_gap_bin"] in ("low", "mid", "high")
        assert t["coverage"]["A1_mechanic"]
        for gate in ("G_BROAD", "G_REACH", "G_SOLVE", "G_MIN", "G_REP"):
            assert gate in t["provenance"]["gates_passed"]
        assert t["index_version"] == "v1"

    # every artefact the plan names is on disk
    for path in (cfg.paths.compat_report, cfg.paths.index_fields, cfg.paths.manifest,
                 cfg.paths.subgraphs, cfg.paths.gate_stats, cfg.paths.density_stats,
                 cfg.paths.isolation_report, cfg.paths.stats):
        assert os.path.exists(path), path

    gate_stats = json.load(open(cfg.paths.gate_stats, encoding="utf-8"))
    assert gate_stats["funnel"]["G_BROAD"]["seen"] > 0
    assert gate_stats["funnel"]["G_MIN"]["seen"] > 0


def test_gate_winners_survive_judge_outage(tmp_path):
    """Cheap gates (G_BROAD/G_REACH) need the embedder; G_SOLVE needs the judge.
    If those two sit behind different network paths and the judge is briefly
    unreachable, a 1-of-N winner must be cached and retried later — not
    silently recorded as though the critic had ruled it unsolvable."""
    from arqg.sid import pipeline
    from arqg.sid.compose import candidates_from_dicts
    from arqg.sid.env import build_env
    from arqg.sid.gates import run_gates
    from arqg.sid.subgraphs import run_mining
    from arqg.llm import BaseLLM, LLMConnectionError

    cfg = _cfg(tmp_path)
    run_compat(cfg)
    asyncio.run(run_mining(cfg))
    asyncio.run(pipeline.stage_facts(cfg))
    asyncio.run(pipeline.stage_compose(cfg))
    candidates = candidates_from_dicts(list(read_jsonl(cfg.paths.candidates)))
    assert candidates, "need at least one composed candidate to gate"

    class FlakyJudge(BaseLLM):
        """Simulates a judge that is simply unreachable — not a judge that
        looked at the question and objected."""
        def __init__(self, cfg):
            super().__init__(cfg)
            self.calls = 0

        async def complete_json(self, system, user, **kw):
            self.calls += 1
            raise LLMConnectionError("simulated: judge network unreachable")

    async def _run(judge: BaseLLM) -> list[dict]:
        env = await build_env(cfg, version="v0")
        gen = make_sid_client(cfg.llm)
        try:
            return await run_gates(cfg, env, gen, judge, candidates)
        finally:
            await gen.aclose()
            await env.aclose()

    flaky = FlakyJudge(cfg.judge)
    records = asyncio.run(_run(flaky))
    assert records == [], "nothing can pass G_SOLVE while the judge is down"
    assert flaky.calls > 0, "at least one winner should have reached G_SOLVE"

    winners_cached = list(read_jsonl(cfg.paths.gate_winners))
    assert winners_cached, "1-of-N winners must be cached, not lost, on outage"
    decisions = list(read_jsonl(cfg.paths.gate_decisions))
    assert all(d["outcome"] != "rejected_G_SOLVE" for d in decisions), (
        "a judge outage must never be recorded as a genuine G_SOLVE rejection")

    # the judge recovers; a fresh run must finish the cached winners via
    # G_SOLVE without needing to repeat their (embedding-only) cheap gates
    records2 = asyncio.run(_run(make_sid_client(cfg.judge)))
    assert records2, "cached winners must pass G_SOLVE once the judge is back"
    assert any(r["candidate_id"] in {c.candidate_id for c in candidates}
               for r in records2)


def test_resume_is_idempotent(tmp_path):
    from arqg.sid import pipeline
    cfg = _cfg(tmp_path)
    asyncio.run(pipeline.run_all(cfg))
    n1 = len(list(read_jsonl(cfg.paths.tasks)))
    counts = {p: len(list(read_jsonl(p)))
              for p in (cfg.paths.candidates, cfg.paths.gated, cfg.paths.minimized)}
    asyncio.run(pipeline.run_all(cfg))
    assert len(list(read_jsonl(cfg.paths.tasks))) == n1
    for p, n in counts.items():
        assert len(list(read_jsonl(p))) == n, f"{p} duplicated on resume"
