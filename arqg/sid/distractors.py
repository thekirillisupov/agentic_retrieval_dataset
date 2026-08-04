"""S6 — conditional distractor generation, verification and injection
(plan §7.2–§7.6).

Volume is **normalisation, not amplification**:

    n_distractors = clip(density_median − density_current, 0, n_max)

The point is to pull sparse outliers up to *this corpus's* median, not to build
an artificial difficulty race — otherwise distractors quietly become a
difficulty knob and leave-one-corpus-out transfer degrades.

The cascade differs by *how much real text survives* and *how precisely we
control the discriminating feature*, not by "modified / unmodified". A real
chunk taken unchanged cannot be a distractor at all: if it already satisfies
`sim > τ_sim` it is **already counted** in `density_v0`, so a copy is either a
no-op or a content-hash duplicate.

    L1 transplant  real chunk from outside the neighbourhood, surface entities
                   swapped for the gold neighbourhood's → maximal real text,
                   weak control over the discriminating axis
    L2 perturb     real chunk from the [τ_low, τ_sim) band with one
                   discriminating attribute perturbed → near-miss, precise
                   control; the main source of near_duplicate / partial_answer
    L3 generate    full generation from a corpus template — fallback only,
                   because synthetic surface statistics invite the shortcut
                   "looks generated → ignore"
"""
from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..llm import BaseLLM
from ..utils import append_jsonl, load_done_keys, log, read_jsonl
from .config import SidConfig
from .density import DensityModel
from .env import Env
from .prompts import (DISTRACTOR_CHECK_SYS, GENERATE_SYS, PERTURB_SYS,
                      TRANSPLANT_SYS, distractor_check_user,
                      generate_distractor_user, perturb_user, transplant_user)

_WS = re.compile(r"\s+")


def _norm(t: str) -> str:
    return _WS.sub(" ", (t or "")).strip().lower()


@dataclass
class DistractorCandidate:
    text: str
    level: str
    dtype: str
    source_chunk_id: str
    perturbed_attribute: str = ""
    sim_to_gold: float = 0.0


@dataclass
class TaskInjection:
    task_id: str
    accepted: list[DistractorCandidate] = field(default_factory=list)
    rejected: dict[str, int] = field(default_factory=lambda: {
        "content_hash": 0, "answer_in_text": 0, "attribute_not_in_question": 0,
        "neighborhood": 0, "alternative_path": 0, "generation_failed": 0})
    l2_band_size: int = 0


# --------------------------------------------------------------------------- #
# Candidate pools
# --------------------------------------------------------------------------- #
def _pools(env: Env, gold: list[str], model: DensityModel,
           rng: random.Random, cfg: SidConfig) -> tuple[list[str], list[str]]:
    """(L2 band pool, L1 far pool) relative to the gold set."""
    gold_vecs = env.dense.vecs(gold)
    if gold_vecs.size == 0:
        return [], []
    sims = env.dense.matrix @ gold_vecs.T          # (N, |gold|)
    best = sims.max(axis=1)
    gold_set = set(gold)

    band, far = [], []
    for i, cid in enumerate(env.dense.ids):
        if cid in gold_set or env.corpus.is_synthetic(cid):
            continue
        s = float(best[i])
        if model.tau_low <= s < model.tau_sim:
            band.append((cid, s))
        elif s < model.tau_low:
            far.append(cid)
    # top of the band first: those need the least push to cross τ_sim
    band.sort(key=lambda cs: -cs[1])
    band_ids = [c for c, _ in band][: cfg.distractors.l2_pool]
    rng.shuffle(far)
    return band_ids, far[: cfg.distractors.l1_pool]


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
async def _gen_l2(llm: BaseLLM, env: Env, rec: dict, source_id: str) -> DistractorCandidate | None:
    attrs: list[str] = []
    for f in rec["facts"]:
        attrs.extend(f.get("discriminating_attributes", []))
    try:
        obj = await llm.complete_json(
            PERTURB_SYS, perturb_user(rec["question"], rec["answer"],
                                      env.corpus.text(source_id), attrs[:6]))
    except Exception as e:                                   # noqa: BLE001
        log.debug("L2 generation failed: %s", e)
        return None
    text = str(obj.get("text", "")).strip()
    if not text:
        return None
    return DistractorCandidate(
        text=text, level="L2_perturbed",
        dtype=str(obj.get("distractor_type", "near_duplicate")),
        source_chunk_id=source_id,
        perturbed_attribute=str(obj.get("perturbed_attribute", "")))


async def _gen_l1(llm: BaseLLM, env: Env, rec: dict, source_id: str,
                  entities: list[str]) -> DistractorCandidate | None:
    try:
        obj = await llm.complete_json(
            TRANSPLANT_SYS, transplant_user(rec["question"], rec["answer"],
                                            env.corpus.text(source_id), entities))
    except Exception as e:                                   # noqa: BLE001
        log.debug("L1 generation failed: %s", e)
        return None
    text = str(obj.get("text", "")).strip()
    if not text:
        return None
    return DistractorCandidate(
        text=text, level="L1_transplant",
        dtype=str(obj.get("distractor_type", "topical_lure")),
        source_chunk_id=source_id)


async def _gen_l3(llm: BaseLLM, env: Env, rec: dict, template_id: str) -> DistractorCandidate | None:
    try:
        obj = await llm.complete_json(
            GENERATE_SYS, generate_distractor_user(rec["question"], rec["answer"],
                                                   env.corpus.text(template_id)))
    except Exception as e:                                   # noqa: BLE001
        log.debug("L3 generation failed: %s", e)
        return None
    text = str(obj.get("text", "")).strip()
    if not text:
        return None
    return DistractorCandidate(
        text=text, level="L3_generated",
        dtype=str(obj.get("distractor_type", "topical_lure")),
        source_chunk_id=template_id)


# --------------------------------------------------------------------------- #
# Verification (§7.5) — cheap checks first, LLM last
# --------------------------------------------------------------------------- #
def _answer_leaks(answer: str, text: str) -> bool:
    a, t = _norm(answer), _norm(text)
    if not a:
        return False
    if a in t:
        return True
    words = [w for w in re.findall(r"[^\W_]{4,}", a)]
    if not words:
        return False
    hit = sum(1 for w in words if w in t)
    return hit / len(words) >= 0.9


def _attribute_in_question(attr: str, question: str) -> bool:
    """§7.5 p.4 — a near-miss only teaches discrimination if the feature that
    separates it from gold is observable to the agent through the question.
    If the question never mentions the date, and the distractor differs from
    gold only by date, no query can tell them apart and NDCG punishes what the
    observation does not determine.

    Only the *values* count. Attributes arrive as "type:value" ("date:2019 ->
    date:2021"), and matching the type name would let "date" in the prompt
    vocabulary pass a check about the year.
    """
    q = _norm(question)
    values: list[str] = []
    for part in re.split(r"->|,|;", _norm(attr)):
        part = part.strip()
        values.append(part.split(":", 1)[1] if ":" in part else part)
    tokens = [t for v in values for t in re.findall(r"[^\W_]{2,}", v)]
    return any(t in q for t in tokens if not t.isalpha() or len(t) > 3)


async def verify_candidates(cfg: SidConfig, judge: BaseLLM, env: Env, rec: dict,
                            cands: list[DistractorCandidate],
                            model: DensityModel, inj: TaskInjection
                            ) -> list[DistractorCandidate]:
    d = cfg.distractors
    stage1: list[DistractorCandidate] = []
    for c in cands:
        if env.corpus.has_duplicate_text(c.text):                 # p.3
            inj.rejected["content_hash"] += 1
            continue
        if _answer_leaks(rec["answer"], c.text):                  # p.1 (cheap pass)
            inj.rejected["answer_in_text"] += 1
            continue
        if (d.require_l2_attribute_in_question and c.level == "L2_perturbed"
                and c.perturbed_attribute
                and not _attribute_in_question(c.perturbed_attribute, rec["question"])):
            inj.rejected["attribute_not_in_question"] += 1
            continue
        stage1.append(c)
    if not stage1:
        return []

    # p.5 — must actually land in the gold neighbourhood, or the whole injection
    # budget is spent without moving density. One score against embeddings we
    # already have; must run *before* anything is written to the index.
    vecs = await env.embedder.embed([c.text for c in stage1], kind="passage")
    gold_vecs = env.dense.vecs(rec["gold_chunk_ids"])
    stage2: list[DistractorCandidate] = []
    for c, v in zip(stage1, vecs):
        sim = float((gold_vecs @ v).max()) if gold_vecs.size else 0.0
        c.sim_to_gold = sim
        if d.require_neighborhood_hit and sim <= model.tau_sim:
            inj.rejected["neighborhood"] += 1
            continue
        stage2.append(c)
    if not stage2:
        return []

    async def check(c: DistractorCandidate) -> DistractorCandidate | None:
        try:
            v = await judge.complete_json(
                DISTRACTOR_CHECK_SYS,
                distractor_check_user(rec["question"], rec["answer"], c.text))
        except Exception:                                     # noqa: BLE001
            return None
        if v.get("contains_answer") or v.get("valid_alternative_path"):
            return None
        return c

    checked = await asyncio.gather(*(check(c) for c in stage2))
    out = [c for c in checked if c is not None]
    inj.rejected["alternative_path"] += len(stage2) - len(out)
    return out


# --------------------------------------------------------------------------- #
# Per-task driver
# --------------------------------------------------------------------------- #
async def build_for_task(cfg: SidConfig, gen: BaseLLM, judge: BaseLLM, env: Env,
                         rec: dict, model: DensityModel,
                         rng: random.Random) -> TaskInjection:
    d = cfg.distractors
    inj = TaskInjection(task_id=rec["task_id"])
    target = int(rec.get("_n_distractors_target", 0))
    if target <= 0:
        return inj

    band, far = _pools(env, rec["gold_chunk_ids"], model, rng, cfg)
    inj.l2_band_size = len(band)
    entities: list[str] = []
    for f in rec["facts"]:
        entities.extend(f.get("entities", []))
    entities = list(dict.fromkeys(entities))[:5]

    generated = 0
    bi = fi = 0
    # a level that keeps producing candidates none of which survive verification
    # is retired, so its failures cannot eat the whole generation budget and
    # starve the next level in the cascade
    tried: dict[str, int] = {"L2_perturbed": 0, "L1_transplant": 0}
    retired: set[str] = set()
    max_barren_waves = 2

    while len(inj.accepted) < target and generated < d.max_candidates_per_task:
        need = target - len(inj.accepted)
        jobs, wave_levels = [], []

        # quota, not strict priority: >= min_l2 from L2 when the band allows it
        n_l2_accepted = sum(1 for c in inj.accepted if c.level == "L2_perturbed")
        want_l2 = 0 if "L2_perturbed" in retired else \
            max(0, min(d.min_l2 - n_l2_accepted, need))
        while want_l2 > 0 and bi < len(band):
            jobs.append(_gen_l2(gen, env, rec, band[bi]))
            wave_levels.append("L2_perturbed")
            bi += 1
            want_l2 -= 1
        while len(jobs) < need and fi < len(far) and "L1_transplant" not in retired:
            jobs.append(_gen_l1(gen, env, rec, far[fi], entities))
            wave_levels.append("L1_transplant")
            fi += 1
        while len(jobs) < need and bi < len(band) and "L2_perturbed" not in retired:
            jobs.append(_gen_l2(gen, env, rec, band[bi]))
            wave_levels.append("L2_perturbed")
            bi += 1
        if not jobs and d.allow_l3:
            template = rec["gold_chunk_ids"][0]
            jobs = [_gen_l3(gen, env, rec, template) for _ in range(need)]
            wave_levels = ["L3_generated"] * need
        if not jobs:
            break

        generated += len(jobs)
        cands = [c for c in await asyncio.gather(*jobs) if c is not None]
        inj.rejected["generation_failed"] += len(jobs) - len(cands)
        accepted = await verify_candidates(cfg, judge, env, rec, cands, model, inj)
        inj.accepted.extend(accepted)

        won = {c.level for c in accepted}
        for level in set(wave_levels):
            if level in ("L1_transplant", "L2_perturbed"):
                tried[level] = 0 if level in won else tried[level] + 1
                if tried[level] >= max_barren_waves:
                    retired.add(level)
        if "L3_generated" in wave_levels and not accepted:
            break                       # the fallback failed too; give up on this task

    inj.accepted = inj.accepted[:target]
    return inj


def inject(cfg: SidConfig, env: Env, rec: dict, inj: TaskInjection,
           vecs: np.ndarray, version: str = "v1") -> dict:
    """Write accepted distractors into the corpus and both index branches."""
    added_ids, added_texts, added_vecs = [], [], []
    levels = {"L1_transplant": 0, "L2_perturbed": 0, "L3_generated": 0}
    types: dict[str, int] = {}
    for c, v in zip(inj.accepted, vecs):
        donor = env.corpus.get(c.source_chunk_id)
        chunk = env.corpus.inject(
            donor_file=donor.file_name if donor else rec["gold_chunk_ids"][0].split("::")[0],
            text=c.text, task_id=rec["task_id"], level=c.level, dtype=c.dtype,
            source_chunk_id=c.source_chunk_id,
            document_id=donor.document_id if donor else "",
            title=donor.title if donor else "",
            perturbed_attribute=c.perturbed_attribute,
            sim_to_gold=round(c.sim_to_gold, 4), version=version)
        if chunk is None:
            continue
        added_ids.append(chunk.id)
        added_texts.append(chunk.raw_text)
        added_vecs.append(v)
        levels[c.level] = levels.get(c.level, 0) + 1
        types[c.dtype] = types.get(c.dtype, 0) + 1

    if added_ids:
        env.searcher.add_documents(added_ids, added_texts, np.asarray(added_vecs))

    return {
        "injected": bool(added_ids),
        "n_injected": len(added_ids),
        "chunk_ids": added_ids,
        "levels": levels,
        "types": types,
        "all_in_neighborhood": all(c.sim_to_gold > 0 for c in inj.accepted),
        "rejected": inj.rejected,
        "l2_band_size": inj.l2_band_size,
        "l2_attributes_in_question": all(
            c.level != "L2_perturbed" or c.perturbed_attribute == "" or
            _attribute_in_question(c.perturbed_attribute, rec["question"])
            for c in inj.accepted),
    }


async def run_distractors(cfg: SidConfig, env: Env, gen: BaseLLM, judge: BaseLLM,
                          records: list[dict], model: DensityModel) -> list[dict]:
    if not cfg.distractors.enabled:
        log.info("S6: distractor injection disabled — tasks stay on v0")
        return [{**r, "distractors": {"injected": False, "n_injected": 0}} for r in records]

    done = load_done_keys(cfg.paths.injected_tasks, "task_id")
    rng = random.Random(cfg.density.seed + 7)
    out: list[dict] = []
    todo = [r for r in records if r["task_id"] not in done]
    log.info("S6: %d tasks to densify (%d done)", len(todo), len(done))

    # sequential per task: each injection changes the index the next task sees
    for rec in todo:
        target = int(rec.get("_n_distractors_target", 0))
        inj = await build_for_task(cfg, gen, judge, env, rec, model, rng)
        vecs = (await env.embedder.embed([c.text for c in inj.accepted], kind="passage")
                if inj.accepted else np.zeros((0, 1), dtype="float32"))
        summary = inject(cfg, env, rec, inj, vecs)
        env.corpus.version = "v1"

        rec = {k: v for k, v in rec.items() if k != "_n_distractors_target"}
        rec["distractors"] = summary
        rec["index_version"] = "v1"
        rec["metrics"] = {
            **rec["metrics"],
            "n_distractors_target": target,
            "neighborhood_density_v1": _density_after(env, rec, model),
        }
        append_jsonl(cfg.paths.injected_tasks, rec)
        out.append(rec)

    all_out = list(read_jsonl(cfg.paths.injected_tasks))
    env.corpus.export_public(cfg.paths.injected_corpus, only_injected=True)
    env.corpus.save_ledger(cfg.paths.injection_ledger)
    env.corpus.write_manifest(cfg.paths.manifest, cfg.corpus_name,
                              extra={"taxonomy_version": cfg.taxonomy_version})
    n_inj = sum(r["distractors"]["n_injected"] for r in all_out)
    log.info("S6: injected %d distractor chunks over %d tasks -> %s",
             n_inj, len(all_out), cfg.paths.injected_corpus)
    return all_out


def _density_after(env: Env, rec: dict, model: DensityModel) -> int:
    from .density import task_density
    return task_density(env, rec["gold_chunk_ids"], model.tau_sim)


# --------------------------------------------------------------------------- #
# §7.6 — sampled reachability re-measurement (a metric, not a gate)
# --------------------------------------------------------------------------- #
async def reach_recheck(cfg: SidConfig, env: Env, records: list[dict]) -> dict[str, Any]:
    """A gold chunk that became unreachable *for every rollout* is common-mode:
    it scales every NDCG in the group by the same constant and cancels under
    group-normalised advantage. So this is an indicator of injection volume, not
    a gate over the pool."""
    n = cfg.distractors.reach_recheck_sample
    if not n:
        return {}
    rng = random.Random(cfg.density.seed + 11)
    injected = [r for r in records if r.get("distractors", {}).get("injected")]
    plain = [r for r in records if not r.get("distractors", {}).get("injected")]
    sample = (rng.sample(injected, min(len(injected), n // 2 or len(injected)))
              + rng.sample(plain, min(len(plain), n // 2 or len(plain))))
    if not sample:
        return {}

    lost = 0
    for rec in sample:
        queries, owners = [], []
        for f in rec["facts"]:
            for q in (f.get("verbatim_span", ""), f.get("fact_normalized", "")):
                if q:
                    queries.append(q)
                    owners.append(f["chunk_id"])
        probes = await env.searcher.probe_many(queries, cfg.gates.top_k)
        reachable = {f["chunk_id"]: False for f in rec["facts"]}
        for cid, p in zip(owners, probes):
            if cid in reachable and cid in set(p.hit_ids):
                reachable[cid] = True
        if not all(reachable.values()):
            lost += 1

    rate = lost / len(sample)
    log.info("§7.6: g3_loss_rate_after_injection = %.3f over %d sampled tasks "
             "(target <= 0.05)", rate, len(sample))
    return {"sampled": len(sample), "lost": lost,
            "g3_loss_rate_after_injection": round(rate, 4)}
