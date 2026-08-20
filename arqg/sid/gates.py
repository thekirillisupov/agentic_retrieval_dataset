"""S4/S5 — the gates (plan §6 and §7.0).

    G_SOLVE  solvable from the full gold set, and neither the answer nor the
             intermediate entities are given away by the question
    G_BROAD  the whole question as ONE query must NOT return the whole gold set
    G_REACH  every gold chunk must be reachable by at least one probe
    G_MIN    leave-one-**fact**-out minimisation (not leave-one-chunk-out)
    G_REP    every chunk stating a surviving fact joins that fact's group

`G_BROAD` and `G_REACH` are the two ends of one interval: not trivial, not above
the environment's ceiling. Both are retrieval-only — `G_REACH` probes with the
normalised paraphrase S3 already produced (see `reach_probe_fields`) — so the
cheap gates cost no LLM calls at all and run before the 1-of-N selection.

Why minimisation is per fact: redundancy lives *between chunks of one fact*, not
between facts. Leave-one-chunk-out would drop a whole redundant group one member
at a time; leave-one-fact-out cannot, because facts are atomic by construction.
"""
from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..llm import BaseLLM, LLMConnectionError
from ..utils import append_jsonl, load_done_keys, log, read_jsonl
from .config import SidConfig
from .env import Env, chunk_passage_text
from .prompts import (ENTAIL_SYS, SOLVE_DEVIL_SYS, SOLVE_SYS, entail_user,
                      solve_user)
from .retrieval import aggregate_gaps, aggregate_gaps_over_groups, gap_bin
from .scoping import facet_header
from .schema import Candidate, sid_hash

# SELECT_1_OF_N is not a gate but it *is* a stage of the funnel: candidates that
# clear G_REACH and then lose their 1-of-N batch used to vanish from the stats
# (G_REACH.passed > G_SOLVE.seen with nothing in between to explain the gap),
# which made the funnel unreadable exactly when someone audited it.
# G_AMBIG / G_VERBATIM are mechanic-scoped cheap gates: only disambiguation_first
# candidates are seen by the former, only verbatim_lookup by the latter.
GATE_ORDER = ["G_BROAD", "G_REACH", "G_AMBIG", "G_VERBATIM", "SELECT_1_OF_N",
              "G_SOLVE", "G_MIN", "G_REP"]


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
#: `gates.reach_probe` -> the fact fields G_REACH probes with
_REACH_FIELDS = {
    "paraphrase": ("fact_normalized",),
    "verbatim": ("verbatim_span",),
    "both": ("verbatim_span", "fact_normalized"),
}


def reach_probe_fields(cfg: SidConfig) -> tuple[str, ...]:
    """Which fact field(s) G_REACH queries with.

    The default is the paraphrase alone. `verbatim_span` is an exact substring
    of the very chunk it has to retrieve, so probing with it asks whether BM25
    can find a document from its own text — nearly always yes, whatever the
    environment's real ceiling is. With the old `any`-over-both rule the
    paraphrase could therefore never change an outcome, which quietly turned
    G_REACH into a no-op: `environment_ceiling_pool` stayed empty, the
    "low G_REACH pass-rate ⇒ the environment's ceiling" reading of
    `gates.funnel` could not fire, and §7.1's reach-conditioned density median
    collapsed onto the unconditioned one it is supposed to be compared against.
    """
    try:
        return _REACH_FIELDS[cfg.gates.reach_probe]
    except KeyError:
        raise SystemExit(
            f"unknown gates.reach_probe: {cfg.gates.reach_probe!r} "
            f"(expected one of {', '.join(_REACH_FIELDS)})") from None


async def cheap_gates(cfg: SidConfig, env: Env, cand: Candidate) -> dict[str, Any]:
    k = cfg.gates.top_k
    gold = cand.chunk_ids

    broad = await env.searcher.probe(cand.question, k, gold)
    hits = set(broad.hit_ids)
    gaps = aggregate_gaps(broad, gold)
    broad_hit = len([g for g in gold if g in hits]) / max(1, len(gold))
    broad_ok = not all(g in hits for g in gold)

    # G_REACH: per gold chunk, one probe per fact (see `reach_probe_fields`).
    reach_queries: list[str] = []
    owner: list[str] = []
    for f in cand.facts:
        for field in reach_probe_fields(cfg):
            q = f.get(field, "")
            if q:
                reach_queries.append(q)
                owner.append(f["chunk_id"])
    probes = await env.searcher.probe_many(reach_queries, k) if reach_queries else []
    reachable: dict[str, bool] = {g: False for g in gold}
    for cid, p in zip(owner, probes):
        if cid in reachable and cid in set(p.hit_ids):
            reachable[cid] = True
    reach_ok = all(reachable.values()) if reachable else False

    res: dict[str, Any] = {
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
    # Mechanic-scoped cheap gates. Computed here rather than in the stage
    # driver so a repaired question (solve_stage re-runs cheap_gates) is
    # re-checked automatically — the descriptor / identifier may have changed
    # with the phrasing.
    if cand.mechanic == "disambiguation_first" and cfg.gates.run_ambiguity:
        amb = await ambiguity_check(cfg, env, cand)
        res["ambig_ok"] = amb.pop("ok")
        res["metrics"]["ambiguity"] = amb
    if cand.mechanic == "verbatim_lookup" and cfg.gates.run_verbatim:
        verb = verbatim_check(cfg, env, cand, res["metrics"]["per_chunk_gaps"])
        res["verbatim_ok"] = verb.pop("ok")
        res["metrics"]["verbatim"] = verb
    return res


def _doc_of(env: Env, chunk_id: str) -> str:
    c = env.corpus.get(chunk_id)
    return (c.document_id or c.file_name) if c is not None else chunk_id


async def ambiguity_check(cfg: SidConfig, env: Env, cand: Candidate) -> dict[str, Any]:
    """G_AMBIG — the descriptor must denote >= 2 plausible referents.

    ``disambiguation_first`` claims its starting entity is ambiguous, but the
    composer's claim is a prompt instruction, not a property of the corpus: a
    "descriptor" matching exactly one document is a paraphrase (or a leaked
    identifier — «закупка с кодом pn_lot_…»), and every later gate happily
    passes it. The check is retrieval-only. The *referent set* is the documents
    scoring within ``ambiguity_sim_ratio`` of the descriptor probe's best hit;
    the gate demands the gold's document inside it (the descriptor does point
    at the referent) and at least ``ambiguity_min_referents − 1`` competing
    documents alongside — an ambiguity the agent actually has to resolve.
    Counting top-k hits without the score band would be vacuous (any query
    returns k hits), and a band anchored on the gold rather than the best hit
    would let a descriptor that describes the referent *badly* pass on the
    crowd of equally-bad matches. A hit nearly identical to a gold chunk
    (``ambiguity_dup_ceiling``) is a restatement, not a competitor.
    """
    g = cfg.gates
    desc = (cand.descriptor or "").strip()
    if not desc:
        return {"ok": False, "reason": "no_descriptor", "n_referents": 0}
    gold = cand.chunk_ids
    gold_docs = {_doc_of(env, c) for c in gold}
    probe = await env.searcher.probe(desc, max(g.ambiguity_top_k, g.top_k), gold)
    hit_ids = probe.hit_ids
    qvec = (await env.embedder.embed([desc], kind="query"))[0]
    sims = env.dense.scores_for(qvec, list(dict.fromkeys(hit_ids + list(gold))))
    best = max(sims.values(), default=0.0)
    best_gold = max((sims.get(c, 0.0) for c in gold), default=0.0)
    threshold = best * g.ambiguity_sim_ratio
    if best <= 0.0 or best_gold < threshold:
        return {"ok": False, "reason": "descriptor_misses_referent",
                "n_referents": 0, "best_gold_sim": round(best_gold, 4),
                "best_hit_sim": round(best, 4)}
    # near-duplicates of the gold (сводные обзоры, mirrored notices) restate
    # the referent instead of competing with it
    dup: dict[str, float] = {h: 0.0 for h in hit_ids}
    for c in gold:
        gvec = env.dense.vec(c)
        if gvec is None:
            continue
        for h, s in env.dense.scores_for(gvec, hit_ids).items():
            dup[h] = max(dup[h], s)
    competitors = {_doc_of(env, h) for h in hit_ids
                   if _doc_of(env, h) not in gold_docs
                   and sims.get(h, 0.0) >= threshold
                   and dup[h] < g.ambiguity_dup_ceiling}
    n_referents = 1 + len(competitors)
    ok = n_referents >= max(2, g.ambiguity_min_referents)
    return {"ok": ok,
            "reason": "" if ok else "descriptor_is_a_paraphrase",
            "n_referents": n_referents,
            "best_gold_sim": round(best_gold, 4),
            "best_hit_sim": round(best, 4),
            "competitor_docs": sorted(competitors)[:5]}


def verbatim_check(cfg: SidConfig, env: Env, cand: Candidate,
                   per_chunk_gaps: dict[str, dict[str, float]]) -> dict[str, Any]:
    """G_VERBATIM — the identifier is real, and the branch asymmetry holds.

    The mechanic exists to teach tool choice, so its acceptance condition is
    the *inverse* of what every other mechanic wants from the entry chunk:
    trivial for the lexical branch (``lex_gap`` at or below the low bin —
    grep/must_contain solves it in one shot) while the dense branch alone
    misses it (``dense_gap`` high — the лексический промах эмбеддера the
    never-retrieved bucket is made of). Both gaps were already measured by the
    G_BROAD probe, so the gate adds no retrieval call; what it adds is the
    harness check that the identifier occurs verbatim in the question AND in a
    gold chunk's passage *as the index holds it* (title and facets included —
    zakupki packs the purchase number into the title).
    """
    g = cfg.gates
    ident = (cand.identifier or "").strip()
    if not ident:
        return {"ok": False, "reason": "no_identifier"}
    if ident not in cand.question:
        return {"ok": False, "reason": "identifier_not_in_question"}
    entry = []
    for cid in cand.chunk_ids:
        c = env.corpus.get(cid)
        if c is not None and ident in chunk_passage_text(c, cfg):
            entry.append(cid)
    if not entry:
        return {"ok": False, "reason": "identifier_not_in_gold_passage"}
    cid = min(entry, key=lambda c: per_chunk_gaps.get(c, {}).get("lex_gap", 1.0))
    gaps = per_chunk_gaps.get(cid, {})
    lex = gaps.get("lex_gap", 1.0)
    dense = gaps.get("dense_gap", 0.0)
    detail = {"entry_chunk": cid, "entry_lex_gap": round(lex, 4),
              "entry_dense_gap": round(dense, 4)}
    if lex > g.verbatim_lex_gap_max:
        return {"ok": False, "reason": "lexical_misses_entry", **detail}
    if dense < g.verbatim_dense_gap_min:
        return {"ok": False, "reason": "dense_already_finds_entry", **detail}
    return {"ok": True, "reason": "", **detail}


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
                  facts: list[dict], stats: GateStats | None = None,
                  declared_filter: str = "") -> tuple[bool, str, bool]:
    """Returns (passed, reason, dual_critic_agreement).

    Raises `LLMConnectionError` as-is rather than turning it into a verdict:
    the critic being unreachable is not the same claim as the critic having
    looked at the question and found it unsolvable, and the caller needs to
    tell them apart to avoid permanently rejecting a candidate over a network
    blip."""
    try:
        v = await judge.complete_json(
            SOLVE_SYS, solve_user(question, answer, facts, declared_filter))
    except LLMConnectionError:
        raise
    except Exception as e:                                  # noqa: BLE001
        return False, f"critic error: {e}", True

    problems = []
    if declared_filter and not v.get("filter_matches_question", True):
        problems.append("заявленный фильтр не совпадает с ограничениями вопроса")
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
def _facets_of(cfg: SidConfig, env: Env, chunk_id: str) -> str:
    """The candidate chunk's facet header, as the prompts elsewhere show it."""
    f = cfg.facets
    c = env.corpus.get(chunk_id)
    if not (f.fields and f.in_prompts) or c is None:
        return ""
    return facet_header(c, f.fields, f.labels, f.max_value_chars)


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
                    ENTAIL_SYS, entail_user(f["fact_normalized"], cid,
                                            env.corpus.text(cid),
                                            _facets_of(cfg, env, cid)))
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
def _bin_of(cfg: SidConfig, fused_gap: float) -> str:
    return gap_bin(float(fused_gap), cfg.export.fused_gap_bins)


def prior_bins(cfg: SidConfig) -> Counter:
    """The bins this selector has already committed to.

    Seeded from `gate_winners.jsonl`, which records every selection ever made,
    so a resumed run keeps balancing where it left off instead of restarting
    the tally and re-skewing the pool.
    """
    bins: Counter = Counter()
    for row in read_jsonl(cfg.paths.gate_winners):
        gap = row.get("res", {}).get("metrics", {}).get("fused_gap")
        if gap is not None:
            bins[_bin_of(cfg, gap)] += 1
    return bins


def _pick_batch_winners(cfg: SidConfig, scored: list[tuple[Candidate, dict]],
                        prior: Counter | None = None
                        ) -> list[tuple[Candidate, dict]]:
    """1-of-N selection (plan §4.5).

    `fused_gap` is the axis that predicts retrieval difficulty (§4.3), and
    ranking a batch by it keeps the hardest candidate. But the members of a
    batch come from *different* subgraphs with *different* submechanics — they
    are N different tasks, not N phrasings of one — so "keep the hardest" is a
    difficulty filter applied to the whole pool, and it pulls directly against
    the datamix the export grades the pool by (`target_fused_gap_share`, 30/40/30
    by default): the `low` bin can only ever be filled by a batch whose every
    member is easy.

    So the default picks, from each batch, the member whose bin is furthest
    below its target share, breaking ties by difficulty. Batches are walked in
    `batch_id` order and the running tally carries across runs, so the result
    does not depend on how the pool was chunked into runs.
    """
    by_batch: dict[str, list[tuple[Candidate, dict]]] = defaultdict(list)
    for cand, res in scored:
        by_batch[cand.batch_id].append((cand, res))
    keep = max(1, cfg.taxonomy.keep_per_batch)
    mode = cfg.taxonomy.batch_selection

    if mode == "hardest":
        winners: list[tuple[Candidate, dict]] = []
        for members in by_batch.values():
            members.sort(key=lambda cr: (-cr[1]["metrics"]["fused_gap"], -len(cr[0].facts)))
            winners.extend(members[:keep])
        return winners
    if mode != "datamix":
        raise SystemExit(f"unknown taxonomy.batch_selection: {mode!r} "
                         f"(expected 'datamix' or 'hardest')")

    counts: Counter = Counter(prior or {})
    target = cfg.export.target_fused_gap_share
    winners = []
    for batch_id in sorted(by_batch):
        members = by_batch[batch_id]
        left = list(range(len(members)))
        for _ in range(min(keep, len(members))):
            total = sum(counts.values())

            def rank(i: int) -> tuple:
                cand, res = members[i]
                gap = res["metrics"]["fused_gap"]
                b = _bin_of(cfg, gap)
                share = counts[b] / total if total else 0.0
                # rounded so a deficit tie falls through to difficulty rather
                # than to float noise
                return (round(target.get(b, 0.0) - share, 6), gap,
                        len(cand.facts), cand.candidate_id)

            pick = max(left, key=rank)
            left.remove(pick)
            counts[_bin_of(cfg, members[pick][1]["metrics"]["fused_gap"])] += 1
            winners.append(members[pick])
    return winners


def _decide(path: str, key: str, ident: str, outcome: str, **extra: Any) -> None:
    append_jsonl(path, {key: ident, "outcome": outcome, **extra})


async def recompose(cfg: SidConfig, gen: BaseLLM, cand: Candidate,
                    feedback: str, iters: int,
                    filter_spec: str = "") -> Candidate | None:
    """Rewrite a question against the critic's objection, keeping its cell and
    its facts. This is the v1 stand-in for the plan's incremental composition
    (§5.2): the same corrective signal at a fraction of the calls."""
    from .compose import compose_one
    from .taxonomy import Cell
    cell = Cell(cand.mechanic, cand.submechanic, cand.has_negation)
    stub = {"subgraph_id": cand.subgraph_id, "chunks": cand.chunk_ids}
    fixed = await compose_one(cfg, gen, cand.batch_id, cand.instantiation_rank,
                              cell, stub, cand.facts, feedback=feedback,
                              iters=iters, filter_spec=filter_spec)
    if fixed is not None:
        # a repair is the same candidate, so the decision log and the gated
        # record stay keyed on one id across attempts
        fixed.candidate_id = cand.candidate_id
        fixed.completeness = cand.completeness
    return fixed


async def run_gates(cfg: SidConfig, env: Env, gen: BaseLLM, judge: BaseLLM,
                    candidates: list[Candidate]) -> list[dict]:
    stats = GateStats()
    seen = load_done_keys(cfg.paths.gate_decisions, "candidate_id")
    # Winners of a previous run that never made it past G_SOLVE (e.g. the
    # embedding backend and the LLM gateway are reachable over two different
    # network paths and only one was up) are cached by candidate_id, so this
    # run does not have to repeat their (embedding-only) cheap gates.
    id_to_cand = {c.candidate_id: c for c in candidates}
    cached_winners: list[tuple[Candidate, dict]] = [
        (id_to_cand[r["candidate_id"]], r["res"])
        for r in read_jsonl(cfg.paths.gate_winners)
        if r["candidate_id"] not in seen and r["candidate_id"] in id_to_cand]
    cached_ids = {c.candidate_id for c, _ in cached_winners}
    todo = [c for c in candidates
            if c.candidate_id not in seen and c.candidate_id not in cached_ids]
    log.info("S4: %d candidates to cheap-gate (%d already decided, %d cached "
             "winners awaiting G_SOLVE)", len(todo), len(seen), len(cached_ids))
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
        # mechanic-scoped cheap gates: the keys exist only for the mechanic
        # they police, so other candidates are neither seen nor counted
        if "ambig_ok" in res:
            stats.record("G_AMBIG", cand.mechanic, res["ambig_ok"])
            if not res["ambig_ok"]:
                _decide(decisions, "candidate_id", cand.candidate_id,
                        "rejected_G_AMBIG",
                        reason=res["metrics"].get("ambiguity", {}).get("reason", ""))
                continue
        if "verbatim_ok" in res:
            stats.record("G_VERBATIM", cand.mechanic, res["verbatim_ok"])
            if not res["verbatim_ok"]:
                _decide(decisions, "candidate_id", cand.candidate_id,
                        "rejected_G_VERBATIM",
                        reason=res["metrics"].get("verbatim", {}).get("reason", ""))
                continue
        survivors.append((cand, res))

    new_winners = _pick_batch_winners(cfg, survivors, prior_bins(cfg))
    won = {c.candidate_id for c, _ in new_winners}
    for cand, _ in survivors:
        # selection is a funnel stage like any other: without this row the
        # candidates a batch discards simply vanish between G_REACH and G_SOLVE
        stats.record("SELECT_1_OF_N", cand.mechanic, cand.candidate_id in won)
        if cand.candidate_id not in won:
            # its batch already produced a winner; resurrecting it later would
            # defeat the point of 1-of-N selection
            _decide(decisions, "candidate_id", cand.candidate_id, "not_selected_1_of_N")
    for cand, res in new_winners:
        append_jsonl(cfg.paths.gate_winners, {"candidate_id": cand.candidate_id, "res": res})
    winners = cached_winners + new_winners
    log.info("S4: cheap gates kept %d/%d; 1-of-N selection kept %d new (+%d cached) "
             "= %d awaiting G_SOLVE", len(survivors), len(todo), len(new_winners),
             len(cached_winners), len(winners))

    # ---- expensive gate: G_SOLVE with a bounded repair loop --------------- #
    deferred = 0
    # S3c re-check material: a repaired question is a *new* question, and for
    # aggregation-type mechanics it must still denote exactly its gold set.
    from .completeness import (CompletenessChecker, check_clean,
                               completeness_active, in_scope)
    checker = (CompletenessChecker(cfg, env.corpus)
               if completeness_active(cfg) else None)

    def _declared_filter(cand: Candidate) -> str:
        if checker is None or not in_scope(cfg, cand.mechanic) or not cand.filter:
            return ""
        import json as _json
        return _json.dumps(cand.filter, ensure_ascii=False)

    async def solve_stage(cand: Candidate, res: dict) -> dict | None:
        nonlocal deferred
        # G_SOLVE is recorded once per candidate with its final outcome: a
        # repair attempt is not a second candidate, and counting each attempt
        # made `seen` exceed what SELECT_1_OF_N passed (the 17-vs-15 kind of
        # bookkeeping hole an audit rightly flags).
        reason = ""
        try:
            for attempt in range(1, max(1, cfg.compose.max_compose_iters) + 1):
                ok, reason, agreement = await g_solve(
                    cfg, judge, cand.question, cand.answer, cand.facts, stats,
                    declared_filter=_declared_filter(cand))
                if ok:
                    stats.record("G_SOLVE", cand.mechanic, True)
                    return _to_record(cfg, cand, res, agreement)
                if attempt >= cfg.compose.max_compose_iters:
                    break
                # feed the critic's objection back to the composer instead of
                # discarding the subgraph: the facts are fine, the phrasing is not
                spec = (checker.filter_spec()
                        if checker is not None and in_scope(cfg, cand.mechanic)
                        else "")
                repaired = await recompose(cfg, gen, cand, reason, attempt + 1,
                                           filter_spec=spec)
                if repaired is None:
                    break
                cand = repaired
                res = await cheap_gates(cfg, env, cand)      # the question changed
                failed = next((name for name, ok in (
                    ("broad", res["broad_ok"]), ("reach", res["reach_ok"]),
                    ("ambiguity", res.get("ambig_ok", True)),
                    ("verbatim", res.get("verbatim_ok", True))) if not ok), "")
                if failed:
                    stats.record("G_SOLVE", cand.mechanic, False)
                    _decide(decisions, "candidate_id", cand.candidate_id,
                            "rejected_after_repair", reason=failed)
                    return None
                clean, why = check_clean(cfg, checker, cand)
                if not clean:
                    stats.record("G_SOLVE", cand.mechanic, False)
                    _decide(decisions, "candidate_id", cand.candidate_id,
                            "rejected_after_repair",
                            reason=f"completeness: {why}"[:200])
                    return None
                if cand.completeness:
                    cand.completeness = {**cand.completeness,
                                         "filter": cand.filter,
                                         "answer_field": cand.answer_field}
        except LLMConnectionError as e:
            # the judge/generator was unreachable, not "the critic said no" —
            # leave this candidate undecided (it stays cached in gate_winners,
            # not gate_decisions) so a later run retries G_SOLVE for it instead
            # of losing it permanently to what may be a one-sided network path.
            # Nothing is recorded in the funnel either: an undecided candidate
            # is seen again — and counted — on the run that decides it.
            deferred += 1
            log.warning("S4: G_SOLVE deferred for %s — judge unreachable: %r",
                        cand.candidate_id, e)
            return None
        stats.record("G_SOLVE", cand.mechanic, False)
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
    log.info("S4: %d candidates passed G_BROAD/G_REACH/G_SOLVE (%d deferred, "
             "judge unreachable — rerun to retry them)", len(all_records), deferred)
    return all_records


def _to_record(cfg: SidConfig, cand: Candidate, res: dict, agreement: bool) -> dict:
    passed = ["G_BROAD", "G_REACH"]
    if res.get("ambig_ok"):
        passed.append("G_AMBIG")
    if res.get("verbatim_ok"):
        passed.append("G_VERBATIM")
    passed.append("G_SOLVE")
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
            "bridge_kind": cand.bridge_kind,
            "instantiation_rank": cand.instantiation_rank,
            "compose_iters": cand.compose_iters,
            "gates_passed": passed,
            "dual_critic_agreement": agreement,
            "generator_model": cand.generator_model,
            "taxonomy_version": cfg.taxonomy_version,
            # S3c verdict (status, exit reason, iterations, declared filter);
            # {} for mechanics outside completeness.mechanics or when off
            "completeness": cand.completeness,
            # G_AMBIG / G_VERBATIM material, for eval tooling and audits
            **({"descriptor": cand.descriptor} if cand.descriptor else {}),
            **({"identifier": cand.identifier} if cand.identifier else {}),
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

    from .completeness import completeness_active, doc_keys_of

    async def one(rec: dict) -> dict | None:
        mechanic = rec["coverage"]["A1_mechanic"]
        facts = rec["facts"]
        removed = 0
        if cfg.gates.run_min:
            minimized, removed = await g_min(cfg, judge, rec["question"],
                                             rec["answer"], facts)
            # For a completeness-verified task the gold set is defined by the
            # question's own filter (gold docs == truth docs), not by what the
            # solvability judge happens to find redundant: a removal that drops
            # a whole document would un-answer the enumeration, so it is vetoed.
            filt = (rec.get("provenance", {}).get("completeness") or {}).get("filter")
            if removed and filt and completeness_active(cfg):
                before = doc_keys_of(cfg, env.corpus, {f["chunk_id"] for f in facts})
                after = doc_keys_of(cfg, env.corpus, {f["chunk_id"] for f in minimized})
                if after != before:
                    log.info("S5: %s — G_MIN removal would drop gold document(s) "
                             "%s from a completeness-verified task; vetoed",
                             rec["task_id"], sorted(before - after))
                    minimized, removed = facts, 0
            # For verbatim_lookup the identifier's carrier is load-bearing by
            # definition — the question addresses the entry document by its
            # code, and G_VERBATIM verified the code sits in a gold passage.
            # A removal that leaves no gold chunk carrying it would break the
            # property the task exists to teach, whatever the judge thinks.
            ident = str(rec.get("provenance", {}).get("identifier") or "")
            if removed and ident:
                def _carries(fs: list[dict]) -> bool:
                    for f in fs:
                        c = env.corpus.get(f["chunk_id"])
                        if c is not None and ident in chunk_passage_text(c, cfg):
                            return True
                    return False
                if _carries(facts) and not _carries(minimized):
                    log.info("S5: %s — G_MIN removal would drop the identifier's "
                             "carrier chunk; vetoed", rec["task_id"])
                    minimized, removed = facts, 0
            facts = minimized
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
