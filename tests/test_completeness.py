"""S3c — set-completeness: the checker's arithmetic, the hard rule on gold that
violates its own question, the repair loop's exits, the augmentation branch's
all-or-nothing conditions, and the funnel bookkeeping S3c's audit exposed.

All offline: the corpus is a zakupki-shaped miniature built in memory, and the
LLMs are scripted so every test asserts not only the outcome but also how many
model calls it was allowed to cost.
"""
import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arqg.llm import BaseLLM
from arqg.schema import Chunk
from arqg.sid.completeness import (CompletenessChecker, check_clean,
                                   completeness_active, doc_key_of,
                                   ensure_complete)
from arqg.sid.config import SidConfig
from arqg.sid.corpus import SidCorpus
from arqg.sid.schema import Candidate
from arqg.sid.taxonomy import Cell
from arqg.utils import read_jsonl


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _doc(n: str, *, region="Красноярский край", customer="КГБУЗ Больница №1",
         okpd2="80.10.12.000", law="44-ФЗ", year="2019", price="1500000.00",
         published="2019-06-11", text="") -> Chunk:
    body = text or (f"Извещение о закупке услуг охраны, реестровый номер {n}. "
                    f"Начальная цена контракта установлена документацией. "
                    f"Определение поставщика проводится электронным аукционом.")
    return Chunk(file_name=f"{n}.txt", index=0, raw_text=body,
                 document_id=n, title=f"Закупка № {n}",
                 meta={"purchase_number": n, "region": region,
                       "customer": customer, "okpd2_code": okpd2, "law": law,
                       "year": year, "price_start": price,
                       "published": published, "section": "Объект закупки"})


def _cfg(tmp_path) -> SidConfig:
    cfg = SidConfig()
    cfg.paths.out_dir = str(tmp_path)
    cfg.corpus_name = "zt"
    cc = cfg.completeness
    cc.doc_field = "purchase_number"
    cc.filter_fields = ["region", "customer", "okpd2_code", "law", "year",
                        "price_start", "published"]
    cfg.facets.fields = ["region", "customer", "okpd2_code", "law", "year"]
    cfg.facets.labels = {"region": "Регион", "customer": "Заказчик",
                         "okpd2_code": "ОКПД2", "law": "Закон", "year": "Год"}
    return cfg


def _cand(chunks: list[Chunk], filt: list[dict], *, mechanic="set_aggregation",
          answer_field="", question="В каких регионах проходили закупки охраны?",
          answer="Красноярский край") -> Candidate:
    facts = [{"fact_id": f"f{i}", "chunk_id": c.id,
              "verbatim_span": c.raw_text[:60],
              "fact_normalized": f"Закупка {c.meta['purchase_number']} "
                                 f"проводится в регионе {c.meta['region']}",
              "entities": [], "discriminating_attributes": [],
              "section": c.title, "facets": ""}
             for i, c in enumerate(chunks)]
    return Candidate(candidate_id="c1", batch_id="b1", instantiation_rank=0,
                     subgraph_id="sg1", corpus="zt", language="ru",
                     question=question, answer=answer, facts=facts,
                     mechanic=mechanic, submechanic="все объекты одного класса",
                     has_negation=False, hop_depth=len(facts),
                     filter=filt, answer_field=answer_field)


class ScriptedLLM(BaseLLM):
    """Returns the queued responses in order and counts every call — so a test
    can assert a decision cost zero (or exactly N) model calls."""

    def __init__(self, responses=()):
        from arqg.config import LLMConfig
        super().__init__(LLMConfig(backend="mock"))
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    async def complete_json(self, system, user, **kw):
        self.calls.append((system, user))
        assert self.responses, "scripted LLM ran out of responses"
        r = self.responses.pop(0)
        return r(system, user) if callable(r) else r


CELL = Cell("set_aggregation", "все объекты одного класса", False)


def _run(cfg, checker, gen, judge, cand, pool=None):
    return asyncio.run(ensure_complete(cfg, checker, gen, judge, cand, CELL,
                                       pool if pool is not None else cand.facts))


# --------------------------------------------------------------------------- #
# checker arithmetic
# --------------------------------------------------------------------------- #
def test_filter_ops_prefix_threshold_and_months(tmp_path):
    cfg = _cfg(tmp_path)
    corpus = SidCorpus([
        _doc("pn1", okpd2="80.10.12.000", price="1500000.00", published="2019-06-11"),
        _doc("pn2", okpd2="80.10.19.000", price="271206.00", published="2019-04-02"),
        _doc("pn3", okpd2="74.90.20.000", price="2000000.00", published="2019-11-20"),
    ])
    ch = CompletenessChecker(cfg, corpus)
    assert len(ch.docs) == 3

    filt, err = ch.validate([{"field": "okpd2_code", "op": "prefix", "value": "80.1"}])
    assert not err
    assert {d.key for d in ch.docs.values() if ch.matches_all(d, filt)} == {"pn1", "pn2"}

    filt, _ = ch.validate([{"field": "price_start", "op": "gte", "value": 1000000}])
    assert {d.key for d in ch.docs.values() if ch.matches_all(d, filt)} == {"pn1", "pn3"}

    filt, _ = ch.validate([{"field": "published", "op": "month_not_in",
                            "value": [3, 4, 5]}])
    assert {d.key for d in ch.docs.values() if ch.matches_all(d, filt)} == {"pn1", "pn3"}

    # unknown fields and wrong-op combinations are refused, not guessed at
    assert ch.validate([{"field": "colour", "op": "eq", "value": "x"}])[1]
    assert ch.validate([{"field": "region", "op": "gte", "value": 3}])[1]
    assert ch.validate([{"field": "price_start", "op": "gte", "value": "дорого"}])[1]
    assert ch.validate([])[1] == "empty filter"


def test_check_reports_excess_and_gold_violations(tmp_path):
    cfg = _cfg(tmp_path)
    gold = [_doc("pn1"), _doc("pn2", customer="МБОУ Школа №3")]
    cheap = _doc("pn3", price="271206.00")     # the «фон отрицания» document
    other = _doc("pn4", region="Брянская область")
    corpus = SidCorpus(gold + [cheap, other])
    ch = CompletenessChecker(cfg, corpus)

    filt = [{"field": "law", "op": "eq", "value": "44-ФЗ"},
            {"field": "price_start", "op": "gte", "value": 1000000}]
    res = ch.check(_cand(gold + [cheap], filt))
    # pn3 is gold but the question excludes everything under a million
    assert res.gold_violations == ["pn3"]
    # pn4 matches the filter but is not gold: the answer is not exhaustive
    assert [d.key for d in res.excess] == ["pn4"]
    assert res.truth_keys == ["pn1", "pn2", "pn4"]


def test_separable_requires_a_distinguishing_facet(tmp_path):
    cfg = _cfg(tmp_path)
    gold = [_doc("pn1"), _doc("pn2")]
    twin = _doc("pn3")                          # same facets on every field
    far = _doc("pn4", region="Брянская область")
    ch = CompletenessChecker(cfg, SidCorpus(gold + [twin, far]))
    gold_docs = [ch.docs["pn1"], ch.docs["pn2"]]
    assert ch.separable(ch.docs["pn4"], gold_docs) is True
    assert ch.separable(ch.docs["pn3"], gold_docs) is False


# --------------------------------------------------------------------------- #
# ensure_complete — accept / hard rule / repair / reject
# --------------------------------------------------------------------------- #
def test_exact_candidate_costs_zero_llm_calls(tmp_path):
    cfg = _cfg(tmp_path)
    gold = [_doc("pn1"), _doc("pn2", customer="МБОУ Школа №3")]
    corpus = SidCorpus(gold + [_doc("pn3", okpd2="74.90.20.000")])
    ch = CompletenessChecker(cfg, corpus)
    gen = ScriptedLLM()
    cand = _cand(gold, [{"field": "okpd2_code", "op": "prefix", "value": "80.1"}])
    out = _run(cfg, ch, gen, None, cand)
    assert out is not None
    assert out.completeness["status"] == "exact"
    assert out.completeness["iters"] == 0
    assert gen.calls == []
    rows = list(read_jsonl(cfg.paths.completeness_log))
    assert rows and rows[0]["outcome"] == "accepted"


def test_out_of_scope_mechanic_is_untouched(tmp_path):
    cfg = _cfg(tmp_path)
    gold = [_doc("pn1"), _doc("pn2")]
    ch = CompletenessChecker(cfg, SidCorpus(gold + [_doc("pn3")]))
    cand = _cand(gold, [], mechanic="comparison")
    out = _run(cfg, ch, ScriptedLLM(), None, cand)
    assert out is cand and out.completeness == {}


def test_gold_violating_own_constraint_is_dropped_from_gold(tmp_path):
    """The hard rule: a gold chunk the question itself excludes either leaves
    the gold set or fails the task — it never stays a retrieval target."""
    cfg = _cfg(tmp_path)
    gold = [_doc("pn1"), _doc("pn2", customer="МБОУ Школа №3")]
    cheap = _doc("pn3", price="271206.00")
    corpus = SidCorpus(gold + [cheap])
    ch = CompletenessChecker(cfg, corpus)
    filt = [{"field": "price_start", "op": "gte", "value": 1000000}]

    out = _run(cfg, ch, ScriptedLLM(), None, _cand(gold + [cheap], filt))
    assert out is not None
    assert out.completeness["dropped_gold_docs"] == ["pn3"]
    assert "pn3.txt::0" not in out.chunk_ids
    assert out.completeness["status"] == "repaired"

    # ... and when dropping it would leave a single chunk, the task fails
    out2 = _run(cfg, ch, ScriptedLLM(), None, _cand([gold[0], cheap], filt))
    assert out2 is None
    last = list(read_jsonl(cfg.paths.completeness_log))[-1]
    assert last["outcome"] == "rejected" and last["exit"] == "gold_violation"


def test_repair_loop_tightens_the_question(tmp_path):
    cfg = _cfg(tmp_path)
    gold = [_doc("pn1"), _doc("pn2")]           # same customer, same region
    stray = _doc("pn3", customer="МБОУ Школа №3")
    ch = CompletenessChecker(cfg, SidCorpus(gold + [stray]))
    base = [{"field": "region", "op": "eq", "value": "Красноярский край"}]

    def repaired(_system, user):
        # a cooperative composer: the feedback names the excess and carries the
        # current filter, so it adds the separating constraint
        assert "pn3" in user and "Текущий фильтр" in user
        return {"question": "В каких закупках охраны у больницы №1?",
                "answer": "ответ", "used_fact_ids": ["f0", "f1"],
                "filter": base + [{"field": "customer", "op": "eq",
                                   "value": "КГБУЗ Больница №1"}],
                "answer_field": "region"}

    gen = ScriptedLLM([repaired])
    out = _run(cfg, ch, gen, None, _cand(gold, base))
    assert out is not None
    assert out.completeness["status"] == "repaired"
    assert out.completeness["iters"] == 1
    assert out.completeness["added_constraints"] == 1
    assert len(gen.calls) == 1


def test_unrepairable_excess_rejects_before_any_llm_call(tmp_path):
    """An excess document that coincides with the gold on every facet cannot be
    excluded by any rephrasing — reject at zero cost, not at the limit."""
    cfg = _cfg(tmp_path)
    gold = [_doc("pn1"), _doc("pn2")]
    twin = _doc("pn3")
    ch = CompletenessChecker(cfg, SidCorpus(gold + [twin]))
    gen = ScriptedLLM()
    out = _run(cfg, ch, gen, None,
               _cand(gold, [{"field": "region", "op": "eq",
                             "value": "Красноярский край"}]))
    assert out is None
    assert gen.calls == []
    last = list(read_jsonl(cfg.paths.completeness_log))[-1]
    assert last["exit"] == "unrepairable" and last["iters"] == 0


def test_no_progress_exits_before_the_iteration_limit(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.completeness.max_repair_iters = 3
    gold = [_doc("pn1"), _doc("pn2")]
    stray = _doc("pn3", customer="МБОУ Школа №3")
    ch = CompletenessChecker(cfg, SidCorpus(gold + [stray]))
    base = [{"field": "region", "op": "eq", "value": "Красноярский край"}]
    same = {"question": "тот же вопрос?", "answer": "ответ",
            "used_fact_ids": ["f0", "f1"], "filter": base, "answer_field": ""}
    gen = ScriptedLLM([same, same, same])       # would be asked 3 times if allowed
    out = _run(cfg, ch, gen, None, _cand(gold, base))
    assert out is None
    assert len(gen.calls) == 1, "no shrink after one round must end the loop"
    last = list(read_jsonl(cfg.paths.completeness_log))[-1]
    assert last["exit"] == "no_progress" and last["iters"] == 1


def test_constraint_cap_bounds_the_result_not_the_iterations(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.completeness.max_added_constraints = 1
    gold = [_doc("pn1"), _doc("pn2")]
    s1 = _doc("pn3", customer="МБОУ Школа №3")
    s2 = _doc("pn4", customer="МБОУ Школа №3", year="2020", published="2020-02-11")
    ch = CompletenessChecker(cfg, SidCorpus(gold + [s1, s2]))
    base = [{"field": "region", "op": "eq", "value": "Красноярский край"}]
    overloaded = {"question": "перегруженный вопрос?", "answer": "ответ",
                  "used_fact_ids": ["f0", "f1"],
                  "filter": base + [{"field": "year", "op": "eq", "value": "2019"},
                                    {"field": "customer", "op": "eq",
                                     "value": "КГБУЗ Больница №1"}],
                  "answer_field": ""}
    gen = ScriptedLLM([overloaded])
    out = _run(cfg, ch, gen, None, _cand(gold, base))
    assert out is None
    last = list(read_jsonl(cfg.paths.completeness_log))[-1]
    assert last["exit"] == "constraint_cap"


def test_missing_filter_gets_one_chance_then_rejects(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.completeness.max_repair_iters = 1
    gold = [_doc("pn1"), _doc("pn2")]
    ch = CompletenessChecker(cfg, SidCorpus(gold))
    still_none = {"question": "вопрос?", "answer": "ответ",
                  "used_fact_ids": ["f0", "f1"]}
    gen = ScriptedLLM([still_none])
    out = _run(cfg, ch, gen, None, _cand(gold, []))
    assert out is None
    last = list(read_jsonl(cfg.paths.completeness_log))[-1]
    assert last["exit"] == "no_filter"
    # ... unless verification is explicitly waived
    cfg.completeness.require_filter = False
    out2 = _run(cfg, ch, ScriptedLLM(), None, _cand(gold, []))
    assert out2 is not None and out2.completeness["status"] == "unverified"


# --------------------------------------------------------------------------- #
# augmentation — the narrow safe branch
# --------------------------------------------------------------------------- #
def _augment_setup(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.completeness.max_repair_iters = 0       # force the bail path directly
    gold = [_doc("pn1"), _doc("pn2", region="Брянская область")]
    extra = _doc("pn3", region="Республика Хакасия")
    ch = CompletenessChecker(cfg, SidCorpus(gold + [extra]))
    filt = [{"field": "okpd2_code", "op": "prefix", "value": "80.1"}]
    cand = _cand(gold, filt, answer_field="region")
    return cfg, ch, gold, extra, cand


def test_augmentation_rebuilds_the_answer_mechanically(tmp_path):
    cfg, ch, gold, extra, cand = _augment_setup(tmp_path)
    judge = ScriptedLLM([{"matches": True, "reason": "охрана"}])
    span = extra.raw_text[:50]
    gen = ScriptedLLM([{"facts": [{"verbatim_span": span,
                                   "fact_normalized": "Закупка pn3 в Хакасии",
                                   "entities": [], "discriminating_attributes": []}]}])
    out = _run(cfg, ch, gen, judge, cand)
    assert out is not None
    assert out.completeness["status"] == "augmented"
    assert out.completeness["added_docs"] == ["pn3"]
    assert "pn3.txt::0" in out.chunk_ids
    # the answer is a projection of the metadata over the WHOLE truth set
    assert out.answer == "Брянская область, Красноярский край, Республика Хакасия"
    assert len(judge.calls) == 1
    # the extracted fact lands in the shared cache for later stages
    assert any(r.get("chunk_id") == "pn3.txt::0" for r in read_jsonl(cfg.paths.facts))


def test_augmentation_is_all_conditions_or_nothing(tmp_path):
    # subject check fails -> no augmentation, task rejected
    cfg, ch, gold, extra, cand = _augment_setup(tmp_path)
    judge = ScriptedLLM([{"matches": False, "reason": "не охрана"}])
    out = _run(cfg, ch, ScriptedLLM(), judge, cand)
    assert out is None
    assert list(read_jsonl(cfg.paths.completeness_log))[-1]["exit"] == "limit"

    # no mechanical answer field -> the branch is not even attempted
    cfg2, ch2, gold2, _, cand2 = _augment_setup(tmp_path / "b")
    cand2.answer_field = ""
    judge2 = ScriptedLLM()
    assert _run(cfg2, ch2, ScriptedLLM(), judge2, cand2) is None
    assert judge2.calls == []

    # truth set too large -> same
    cfg3, ch3, gold3, _, cand3 = _augment_setup(tmp_path / "c")
    cfg3.completeness.augment_max_docs = 2
    judge3 = ScriptedLLM()
    assert _run(cfg3, ch3, ScriptedLLM(), judge3, cand3) is None
    assert judge3.calls == []


# --------------------------------------------------------------------------- #
# hooks: S4 re-check and S5 doc keys
# --------------------------------------------------------------------------- #
def test_check_clean_flags_a_loosened_repair(tmp_path):
    cfg = _cfg(tmp_path)
    gold = [_doc("pn1"), _doc("pn2")]
    stray = _doc("pn3", customer="МБОУ Школа №3")
    ch = CompletenessChecker(cfg, SidCorpus(gold + [stray]))
    tight = [{"field": "region", "op": "eq", "value": "Красноярский край"},
             {"field": "customer", "op": "eq", "value": "КГБУЗ Больница №1"}]
    ok, _ = check_clean(cfg, ch, _cand(gold, tight))
    assert ok is True
    loose = tight[:1]
    ok2, why = check_clean(cfg, ch, _cand(gold, loose))
    assert ok2 is False and "beyond gold" in why
    # mechanics outside the scope are never blocked
    ok3, _ = check_clean(cfg, ch, _cand(gold, loose, mechanic="comparison"))
    assert ok3 is True


def test_doc_key_falls_back_to_document_identity(tmp_path):
    cfg = _cfg(tmp_path)
    plain = Chunk(file_name="x.txt", index=0, raw_text="текст", document_id="dX")
    assert doc_key_of(cfg, plain) == "dX"
    assert completeness_active(cfg) is True
    cfg.completeness.filter_fields = []
    assert completeness_active(cfg) is False


# --------------------------------------------------------------------------- #
# funnel bookkeeping (the 258 -> 196 and 17-vs-15 audit findings)
# --------------------------------------------------------------------------- #
def test_funnel_accounts_for_selection_and_counts_solve_once(tmp_path):
    """G_REACH.passed - SELECT_1_OF_N non-winners == G_SOLVE.seen, and a repair
    attempt must not inflate G_SOLVE.seen past what selection passed."""
    from arqg.sid import pipeline
    from arqg.sid.compat import run_compat
    from arqg.sid.compose import candidates_from_dicts
    from arqg.sid.env import build_env
    from arqg.sid.gates import run_gates
    from arqg.sid.mockllm import make_sid_client
    from arqg.sid.subgraphs import run_mining

    cfg = SidConfig()
    cfg.paths.corpus = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "sample_corpus_sid.jsonl")
    cfg.paths.out_dir = str(tmp_path)
    cfg.llm.backend = cfg.judge.backend = "mock"
    cfg.embed.backend = "mock"
    cfg.corpus_name = "demo"

    run_compat(cfg)
    asyncio.run(run_mining(cfg))
    asyncio.run(pipeline.stage_facts(cfg))
    asyncio.run(pipeline.stage_compose(cfg))
    candidates = candidates_from_dicts(list(read_jsonl(cfg.paths.candidates)))
    assert candidates

    class RejectingJudge(BaseLLM):
        """Objects to every question — so every candidate runs the full repair
        loop, which under per-attempt accounting used to double-count."""
        def __init__(self, cfg):
            super().__init__(cfg)

        async def complete_json(self, system, user, **kw):
            return {"solvable": False, "answer_correct": False,
                    "leaks_answer": False, "leaks_intermediate": False,
                    "standalone": True, "needs_multiple_chunks": True,
                    "reason": "scripted rejection"}

    async def _run():
        env = await build_env(cfg, version="v0")
        gen = make_sid_client(cfg.llm)
        try:
            return await run_gates(cfg, env, gen, RejectingJudge(cfg.judge),
                                   candidates)
        finally:
            await gen.aclose()
            await env.aclose()

    records = asyncio.run(_run())
    assert records == []
    funnel = json.load(open(cfg.paths.gate_stats, encoding="utf-8"))["funnel"]
    assert funnel["SELECT_1_OF_N"]["seen"] == funnel["G_REACH"]["passed"], \
        "every candidate that clears G_REACH must be visible to selection"
    assert funnel["G_SOLVE"]["seen"] == funnel["SELECT_1_OF_N"]["passed"], \
        "a repair attempt is not a second candidate"
    assert funnel["G_SOLVE"]["passed"] == 0
