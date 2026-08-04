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

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arqg.sid.compat import run_compat
from arqg.sid.config import SidConfig
from arqg.sid.corpus import SidCorpus, content_hash
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
from arqg.sid.retrieval import Probe, aggregate_gaps_over_groups, gap_bin
from arqg.sid.schema import Candidate
from arqg.sid.subgraphs import mine_subgraphs
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
# S3 — facts
# --------------------------------------------------------------------------- #
def test_verbatim_span_check():
    chunk = "Компания  «Север»   была основана в 1992 году."
    assert span_is_verbatim("была основана в 1992 году", chunk)   # whitespace-normalised
    assert not span_is_verbatim("была основана в 1993 году", chunk)
    assert not span_is_verbatim("", chunk)


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
