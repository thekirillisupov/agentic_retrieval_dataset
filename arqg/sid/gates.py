"""S4/S5 — the gates (plan §6 and §7.0).

    G_SOLVE  solvable from the full gold set, and neither the answer nor the
             intermediate entities are given away by the question
    G_BROAD  the whole question as ONE query must NOT return the whole gold set
    G_REACH  every gold chunk must be reachable by at least one probe
    G_MIN    leave-one-**fact**-out minimisation (not leave-one-chunk-out)
    G_REP    every chunk stating a surviving fact joins that fact's group

`G_BROAD` and `G_REACH` are the two ends of one interval: not trivial, not above
the environment's ceiling. Both are retrieval-only — `G_REACH` probes with the
fact's verbatim span and its normalised paraphrase, which S3 already produced —
so the cheap gates cost no LLM calls at all and run before the 1-of-N selection.

Why minimisation is per fact: redundancy lives *between chunks of one fact*, not
between facts. Leave-one-chunk-out would drop a whole redundant group one member
at a time; leave-one-fact-out cannot, because facts are atomic by construction.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..llm import BaseLLM
from ..utils import append_jsonl, load_done_keys, log, read_jsonl
from .config import SidConfig
from .env import Env
from .prompts import (ENTAIL_SYS, SOLVE_DEVIL_SYS, SOLVE_SYS, entail_user,
                      solve_user)
from .retrieval import aggregate_gaps, aggregate_gaps_over_groups
from .schema import Candidate, sid_hash

GATE_ORDER = ["G_BROAD", "G_REACH", "G_SOLVE", "G_MIN", "G_REP"]


@dataclass
class GateStats:
    seen: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    passed: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_mechanic: dict[str, dict[str, list[int]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    dual_critic_calls: int = 0
    dual_critic_disagreements: int = 0

    def record(self, gate: str, mechanic: str, ok: bool) -> None:
        self.seen[gate] += 1
        self.passed[gate] += int(ok)
        cell = self.by_mechanic[mechanic][gate]
        cell[0] += 1
        cell[1] += int(ok)

    def to_dict(self, corpus: str) -> dict[str, Any]:
        rate = {g: round(self.passed[g] / self.seen[g], 4)
                for g in GATE_ORDER if self.seen.get(g)}
        per_mech = {
            m: {g: {"seen": v[0], "passed": v[1],
                    "pass_rate": round(v[1] / v[0], 4) if v[0] else None}
                for g, v in gates.items()}
            for m, gates in self.by_mechanic.items()}
        funnel = {g: {"seen": self.seen.get(g, 0), "passed": self.passed.get(g, 0)}
                  for g in GATE_ORDER}
        return {
            "corpus": corpus,
            "funnel": funnel,
            "pass_rate": rate,
            "by_mechanic": per_mech,
            "dual_critic": {
                "calls": self.dual_critic_calls,
                "disagreements": self.dual_critic_disagreements,
                "disagreement_rate": round(
                    self.dual_critic_disagreements / self.dual_critic_calls, 4)
                if self.dual_critic_calls else None,
            },
        }


# --------------------------------------------------------------------------- #
# Cheap gates: G_BROAD + G_REACH (retrieval only)
# --------------------------------------------------------------------------- #
async def cheap_gates(cfg: SidConfig, env: Env, cand: Candidate) -> dict[str, Any]:
    k = cfg.gates.top_k
    gold = cand.chunk_ids

    broad = await env.searcher.probe(cand.question, k, gold)
    hits = set(broad.hit_ids)
    gaps = aggregate_gaps(broad, gold)
    broad_hit = len([g for g in gold if g in hits]) / max(1, len(gold))
    broad_ok = not all(g in hits for g in gold)

    # G_REACH: per gold chunk, one verbatim-oriented and one paraphrased probe.
    reach_queries: list[str] = []
    owner: list[str] = []
    for f in cand.facts:
        for q in (f.get("verbatim_span", ""), f.get("fact_normalized", "")):
            if q:
                reach_queries.append(q)
                owner.append(f["chunk_id"])
    probes = await env.searcher.probe_many(reach_queries, k) if reach_queries else []
    reachable: dict[str, bool] = {g: False for g in gold}
    for cid, p in zip(owner, probes):
        if cid in reachable and cid in set(p.hit_ids):
            reachable[cid] = True
    reach_ok = all(reachable.values()) if reachable else False

    return {
        "broad_ok": broad_ok,
        "reach_ok": reach_ok,
        "unreachable": [c for c, ok in reachable.items() if not ok],
        "metrics": {
            "broad_query_hit_at_k": round(broad_hit, 4),
            "top_k": k,
            **{key: round(v, 4) for key, v in gaps.items()},
            "per_chunk_gaps": {
                g: {"lex_gap": round(broad.lex_gap.get(g, 1.0), 4),
                    "dense_gap": round(broad.dense_gap.get(g, 1.0), 4),
                    "fused_gap": round(broad.fused_gap.get(g, 1.0), 4)}
                for g in gold},
        },
    }


async def remeasure(cfg: SidConfig, env: Env, rec: dict) -> dict[str, Any]:
    """Re-probe a task whose gold set changed under G_MIN / G_REP.

    Gaps are aggregated over *fact groups*, not raw chunks: retrieving any
    member satisfies the fact, so a redundant duplicate that ranks poorly must
    not make an easy task look hard."""
    gold = rec["gold_chunk_ids"]
    groups = rec.get("fact_groups") or [[c] for c in gold]
    probe = await env.searcher.probe(rec["question"], cfg.gates.top_k, gold)
    hits = set(probe.hit_ids)
    gaps = aggregate_gaps_over_groups(probe, groups)
    covered = sum(1 for g in groups if any(c in hits for c in g))
    return {
        "broad_query_hit_at_k": round(covered / max(1, len(groups)), 4),
        "top_k": cfg.gates.top_k,
        **{k: round(v, 4) for k, v in gaps.items()},
        "per_chunk_gaps": {
            g: {"lex_gap": round(probe.lex_gap.get(g, 1.0), 4),
                "dense_gap": round(probe.dense_gap.get(g, 1.0), 4),
                "fused_gap": round(probe.fused_gap.get(g, 1.0), 4)}
            for g in gold},
    }


# --------------------------------------------------------------------------- #
# G_SOLVE
# --------------------------------------------------------------------------- #
async def g_solve(cfg: SidConfig, judge: BaseLLM, question: str, answer: str,
                  facts: list[dict], stats: GateStats | None = None) -> tuple[bool, str, bool]:
    """Returns (passed, reason, dual_critic_agreement)."""
    try:
        v = await judge.complete_json(SOLVE_SYS, solve_user(question, answer, facts))
    except Exception as e:                                  # noqa: BLE001
        return False, f"critic error: {e}", True

    problems = []
    if not v.get("solvable"):
        problems.append("ответ не выводится однозначно из gold-фактов")
    if not v.get("answer_correct", True):
        problems.append("предложенный ответ не следует из фактов")
    if v.get("leaks_answer"):
        problems.append("вопрос содержит ответ")
    if v.get("leaks_intermediate"):
        problems.append("вопрос называет промежуточные сущности")
    if not v.get("standalone", True):
        problems.append("вопрос не самодостаточен")
    if not v.get("needs_multiple_chunks", True):
        problems.append("хватает одного фрагмента")
    reason = "; ".join(problems) or str(v.get("reason", ""))[:200]
    ok = not problems
    agreement = True

    if cfg.gates.dual_critic:
        try:
            d = await judge.complete_json(SOLVE_DEVIL_SYS,
                                          solve_user(question, answer, facts))
            agreement = bool(d.get("solvable")) == bool(v.get("solvable"))
        except Exception:                                   # noqa: BLE001
            agreement = True
        if stats is not None:
            stats.dual_critic_calls += 1
            stats.dual_critic_disagreements += int(not agreement)
        if not agreement and cfg.gates.drop_on_disagreement:
            ok, reason = False, "критики разошлись в оценке решаемости"
    return ok, reason, agreement


async def _solvable_without(judge: BaseLLM, question: str, answer: str,
                            facts: list[dict]) -> bool:
    """G_MIN's probe: the same standard of unambiguity as G_SOLVE."""
    if len(facts) < 1:
        return False
    try:
        v = await judge.complete_json(SOLVE_SYS, solve_user(question, answer, facts))
    except Exception:                                       # noqa: BLE001
        return False
    return bool(v.get("solvable")) and bool(v.get("answer_correct", True))


# --------------------------------------------------------------------------- #
# G_MIN — leave-one-fact-out
# --------------------------------------------------------------------------- #
async def g_min(cfg: SidConfig, judge: BaseLLM, question: str, answer: str,
                facts: list[dict]) -> tuple[list[dict], int]:
    remaining = list(facts)
    removed = 0
    floor = max(1, cfg.gates.min_facts_after_min)
    changed = True
    while changed and len(remaining) > floor:
        changed = False
        for f in list(remaining):
            subset = [x for x in remaining if x["fact_id"] != f["fact_id"]]
            if len(subset) < floor:
                continue
            if await _solvable_without(judge, question, answer, subset):
                remaining = subset
                removed += 1
                changed = True
                break          # restart: re-check every survivor after a removal
    return remaining, removed


# --------------------------------------------------------------------------- #
# G_REP — fact groups
# --------------------------------------------------------------------------- #
async def g_rep(cfg: SidConfig, judge: BaseLLM, env: Env,
                facts: list[dict]) -> list[list[str]]:
    """Every chunk that states a surviving fact joins its group. Collapsing a
    group to one representative would penalise a rollout that returned an
    equally valid member (plan §7.0)."""
    groups: list[list[str]] = []
    if not cfg.gates.run_rep:
        return [[f["chunk_id"]] for f in facts]

    queries = [f["fact_normalized"] for f in facts]
    probes = await env.searcher.probe_many(queries, cfg.gates.rep_top_k)
    for f, probe in zip(facts, probes):
        group = [f["chunk_id"]]
        candidates = [cid for cid, _ in probe.hits
                      if cid != f["chunk_id"] and not env.corpus.is_synthetic(cid)]
        # prefilter by similarity to the source chunk before spending a judge call
        src_vec = env.dense.vec(f["chunk_id"])
        sims = env.dense.scores_for(src_vec, candidates) if src_vec is not None else {}
        ranked = [c for c in sorted(candidates, key=lambda c: -sims.get(c, 0.0))
                  if sims.get(c, 0.0) >= cfg.gates.rep_min_score]
        ranked = ranked[: cfg.gates.rep_max_judges_per_fact]

        async def check(cid: str) -> str | None:
            try:
                v = await judge.complete_json(
                    ENTAIL_SYS, entail_user(f["fact_normalized"], cid, env.corpus.text(cid)))
            except Exception:                               # noqa: BLE001
                return None
            return cid if v.get("states_fact") else None

        for cid in await asyncio.gather(*(check(c) for c in ranked)):
            if cid:
                group.append(cid)
        groups.append(sorted(set(group)))
    return groups


# --------------------------------------------------------------------------- #
# Stage driver
# --------------------------------------------------------------------------- #
def _pick_batch_winners(cfg: SidConfig, scored: list[tuple[Candidate, dict]]
                        ) -> list[tuple[Candidate, dict]]:
    """1-of-N: keep the hardest survivors of each batch. `fused_gap` is the axis
    that actually predicts retrieval difficulty (§4.3), so it ranks."""
    by_batch: dict[str, list[tuple[Candidate, dict]]] = defaultdict(list)
    for cand, res in scored:
        by_batch[cand.batch_id].append((cand, res))
    winners: list[tuple[Candidate, dict]] = []
    for members in by_batch.values():
        members.sort(key=lambda cr: (-cr[1]["metrics"]["fused_gap"], -len(cr[0].facts)))
        winners.extend(members[: max(1, cfg.taxonomy.keep_per_batch)])
    return winners


def _decide(path: str, key: str, ident: str, outcome: str, **extra: Any) -> None:
    append_jsonl(path, {key: ident, "outcome": outcome, **extra})


async def recompose(cfg: SidConfig, gen: BaseLLM, cand: Candidate,
                    feedback: str, iters: int) -> Candidate | None:
    """Rewrite a question against the critic's objection, keeping its cell and
    its facts. This is the v1 stand-in for the plan's incremental composition
    (§5.2): the same corrective signal at a fraction of the calls."""
    from .compose import compose_one
    from .taxonomy import Cell
    cell = Cell(cand.mechanic, cand.submechanic, cand.has_negation)
    stub = {"subgraph_id": cand.subgraph_id, "chunks": cand.chunk_ids}
    fixed = await compose_one(cfg, gen, cand.batch_id, cand.instantiation_rank,
                              cell, stub, cand.facts, feedback=feedback, iters=iters)
    if fixed is not None:
        # a repair is the same candidate, so the decision log and the gated
        # record stay keyed on one id across attempts
        fixed.candidate_id = cand.candidate_id
    return fixed


async def run_gates(cfg: SidConfig, env: Env, gen: BaseLLM, judge: BaseLLM,
                    candidates: list[Candidate]) -> list[dict]:
    stats = GateStats()
    seen = load_done_keys(cfg.paths.gate_decisions, "candidate_id")
    todo = [c for c in candidates if c.candidate_id not in seen]
    log.info("S4: %d candidates to gate (%d already decided)", len(todo), len(seen))
    decisions = cfg.paths.gate_decisions

    # ---- cheap gates on everything, then 1-of-N selection ----------------- #
    cheap = await asyncio.gather(*(cheap_gates(cfg, env, c) for c in todo))
    survivors: list[tuple[Candidate, dict]] = []
    for cand, res in zip(todo, cheap):
        if cfg.gates.run_broad:
            stats.record("G_BROAD", cand.mechanic, res["broad_ok"])
            if not res["broad_ok"]:
                _decide(decisions, "candidate_id", cand.candidate_id, "rejected_G_BROAD")
                continue
        if cfg.gates.run_reach:
            stats.record("G_REACH", cand.mechanic, res["reach_ok"])
            if not res["reach_ok"]:
                _decide(decisions, "candidate_id", cand.candidate_id, "rejected_G_REACH")
                append_jsonl(cfg.paths.ceiling_pool, {
                    "candidate_id": cand.candidate_id, "question": cand.question,
                    "mechanic": cand.mechanic, "unreachable": res["unreachable"],
                    "index_version": env.corpus.version})
                continue
        survivors.append((cand, res))

    winners = _pick_batch_winners(cfg, survivors)
    won = {c.candidate_id for c, _ in winners}
    for cand, _ in survivors:
        if cand.candidate_id not in won:
            # its batch already produced a winner; resurrecting it later would
            # defeat the point of 1-of-N selection
            _decide(decisions, "candidate_id", cand.candidate_id, "not_selected_1_of_N")
    log.info("S4: cheap gates kept %d/%d; 1-of-N selection kept %d",
             len(survivors), len(todo), len(winners))

    # ---- expensive gate: G_SOLVE with a bounded repair loop --------------- #
    async def solve_stage(cand: Candidate, res: dict) -> dict | None:
        for attempt in range(1, max(1, cfg.compose.max_compose_iters) + 1):
            ok, reason, agreement = await g_solve(cfg, judge, cand.question,
                                                  cand.answer, cand.facts, stats)
            stats.record("G_SOLVE", cand.mechanic, ok)
            if ok:
                return _to_record(cfg, cand, res, agreement)
            if attempt >= cfg.compose.max_compose_iters:
                break
            # feed the critic's objection back to the composer instead of
            # discarding the subgraph: the facts are fine, the phrasing is not
            repaired = await recompose(cfg, gen, cand, reason, attempt + 1)
            if repaired is None:
                break
            cand = repaired
            res = await cheap_gates(cfg, env, cand)      # the question changed
            if not (res["broad_ok"] and res["reach_ok"]):
                _decide(decisions, "candidate_id", cand.candidate_id,
                        "rejected_after_repair",
                        reason="broad" if not res["broad_ok"] else "reach")
                return None
        log.debug("S4: G_SOLVE rejected %s — %s", cand.candidate_id, reason)
        _decide(decisions, "candidate_id", cand.candidate_id,
                "rejected_G_SOLVE", reason=reason[:200])
        return None

    records = [r for r in await asyncio.gather(*(solve_stage(c, r) for c, r in winners))
               if r is not None]
    for rec in records:
        append_jsonl(cfg.paths.gated, rec)
        _decide(decisions, "candidate_id", rec["candidate_id"], "passed",
                task_id=rec["task_id"])

    _dump_stats(cfg, stats)
    all_records = list(read_jsonl(cfg.paths.gated))
    log.info("S4: %d candidates passed G_BROAD/G_REACH/G_SOLVE", len(all_records))
    return all_records


def _to_record(cfg: SidConfig, cand: Candidate, res: dict, agreement: bool) -> dict:
    return {
        "task_id": f"syn_{cfg.language}_{cfg.corpus_name}_"
                   f"{sid_hash(cand.candidate_id, n=8)}",
        "candidate_id": cand.candidate_id,
        "batch_id": cand.batch_id,
        "corpus": cand.corpus,
        "language": cand.language,
        "index_version": "v0",
        "question": cand.question,
        "answer": cand.answer,
        "facts": cand.facts,
        "gold_chunk_ids": cand.chunk_ids,
        "coverage": {
            "A1_mechanic": cand.mechanic,
            "A1_submechanic": cand.submechanic,
            "has_negation": cand.has_negation,
        },
        "complexity": {
            "hop_depth": cand.hop_depth,
            "hop_depth_recomputed_after_g_min": False,
            **{k: v for k, v in res["metrics"].items()
               if k in ("lex_gap", "dense_gap", "fused_gap")},
        },
        "metrics": res["metrics"],
        "provenance": {
            "subgraph_id": cand.subgraph_id,
            "instantiation_rank": cand.instantiation_rank,
            "compose_iters": cand.compose_iters,
            "gates_passed": ["G_BROAD", "G_REACH", "G_SOLVE"],
            "dual_critic_agreement": agreement,
            "generator_model": cand.generator_model,
            "taxonomy_version": cfg.taxonomy_version,
        },
    }


def _dump_stats(cfg: SidConfig, stats: GateStats) -> None:
    """gate_stats is cumulative from day one (plan §6): a resumed run only gates
    what is new, so the funnel is merged with what was already recorded."""
    import json
    import os
    from ..utils import ensure_parent
    payload = stats.to_dict(cfg.corpus_name)
    if os.path.exists(cfg.paths.gate_stats):
        with open(cfg.paths.gate_stats, "r", encoding="utf-8") as f:
            prev = json.load(f)
        for gate, cur in payload["funnel"].items():
            old = prev.get("funnel", {}).get(gate, {})
            cur["seen"] += old.get("seen", 0)
            cur["passed"] += old.get("passed", 0)
        payload["pass_rate"] = {g: round(v["passed"] / v["seen"], 4)
                                for g, v in payload["funnel"].items() if v["seen"]}
        merged = prev.get("by_mechanic", {})
        for mech, gates in payload["by_mechanic"].items():
            slot = merged.setdefault(mech, {})
            for gate, cur in gates.items():
                old = slot.get(gate, {})
                seen = cur["seen"] + old.get("seen", 0)
                passed = cur["passed"] + old.get("passed", 0)
                slot[gate] = {"seen": seen, "passed": passed,
                              "pass_rate": round(passed / seen, 4) if seen else None}
        payload["by_mechanic"] = merged
        for key in ("calls", "disagreements"):
            payload["dual_critic"][key] += prev.get("dual_critic", {}).get(key, 0)
        calls = payload["dual_critic"]["calls"]
        payload["dual_critic"]["disagreement_rate"] = (
            round(payload["dual_critic"]["disagreements"] / calls, 4) if calls else None)
    ensure_parent(cfg.paths.gate_stats)
    with open(cfg.paths.gate_stats, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# S5 driver — G_MIN + G_REP
# --------------------------------------------------------------------------- #
async def run_minimize(cfg: SidConfig, env: Env, judge: BaseLLM,
                       records: list[dict]) -> list[dict]:
    seen = load_done_keys(cfg.paths.minimize_decisions, "task_id")
    todo = [r for r in records if r["task_id"] not in seen]
    stats = GateStats()
    log.info("S5: minimising %d tasks (%d already decided)", len(todo), len(seen))

    async def one(rec: dict) -> dict | None:
        mechanic = rec["coverage"]["A1_mechanic"]
        facts = rec["facts"]
        removed = 0
        if cfg.gates.run_min:
            facts, removed = await g_min(cfg, judge, rec["question"], rec["answer"], facts)
        chunk_ids = list(dict.fromkeys(f["chunk_id"] for f in facts))
        ok = len(facts) >= cfg.gates.min_facts_after_min and len(chunk_ids) >= 2
        stats.record("G_MIN", mechanic, ok)
        if not ok:
            log.debug("S5: %s collapsed below the fact floor", rec["task_id"])
            _decide(cfg.paths.minimize_decisions, "task_id", rec["task_id"],
                    "rejected_G_MIN", facts_left=len(facts))
            return None

        groups = await g_rep(cfg, judge, env, facts)
        stats.record("G_REP", mechanic, True)   # G_REP builds labels, never rejects
        gold = sorted({cid for g in groups for cid in g})
        if max((len(g) for g in groups), default=0) > 1 + cfg.gates.rep_max_judges_per_fact // 2:
            log.warning("S5: %s has a fact group of %d chunks — check the "
                        "entailment judge before trusting NDCG on this pool",
                        rec["task_id"], max(len(g) for g in groups))

        rec = dict(rec)
        rec["facts"] = facts
        rec["fact_groups"] = groups
        rec["gold_chunk_ids"] = gold
        # hop_depth was "known by construction"; a removed fact means the hop it
        # carried was not load-bearing, so the axis has to be recomputed (§7.0)
        rec["complexity"] = {**rec["complexity"],
                             "hop_depth": len(facts),
                             "hop_depth_recomputed_after_g_min": removed > 0}
        # gold changed → the gaps measured on the old gold set no longer describe
        # this task; one extra probe is cheaper than a wrong difficulty axis
        metrics = await remeasure(cfg, env, rec)
        rec["metrics"] = {**rec["metrics"], **metrics}
        rec["complexity"].update({k: metrics[k]
                                  for k in ("lex_gap", "dense_gap", "fused_gap")})
        rec["provenance"] = {**rec["provenance"],
                             "gates_passed": rec["provenance"]["gates_passed"] + ["G_MIN", "G_REP"],
                             "facts_removed_by_g_min": removed,
                             "fact_group_sizes": [len(g) for g in groups]}
        return rec

    results = [r for r in await asyncio.gather(*(one(r) for r in todo)) if r]
    for rec in results:
        append_jsonl(cfg.paths.minimized, rec)
        _decide(cfg.paths.minimize_decisions, "task_id", rec["task_id"], "passed")
    _dump_stats(cfg, stats)
    out = list(read_jsonl(cfg.paths.minimized))
    log.info("S5: %d tasks minimised -> %s", len(out), cfg.paths.minimized)
    return out
