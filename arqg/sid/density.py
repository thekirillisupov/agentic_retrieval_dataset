"""§7.1 — neighbourhood density and the corpus norm it is compared against.

    neighborhood_density(c)    = |{c' : sim(c, c') > τ_sim, c' ∉ gold}|
    neighborhood_density(task) = min over c in gold_set

`min`, because the narrow chunk sets the task's difficulty — symmetric to the
`max` rule for gaps (§4.3).

τ_sim is an **intra-corpus percentile**, never an absolute number: embedding
geometry is corpus-specific, and a threshold calibrated on one corpus gives a
systematically biased density on another.

The norm has to be computed on the population it is applied to:

* ``density_median_all``   — provisional, bootstrapped from pseudo-gold sets of
  2–5 random chunks (same *statistic* as the task-level min, so it is comparable
  even before tasks exist);
* ``density_median_reach`` — the working norm: median task-level density over
  tasks that passed ``G_REACH`` on v0. Reachability and density are negatively
  related, so the reach-passing population sits at lower density than the corpus
  at large; normalising to the corpus-wide median would push tasks into a
  neighbourhood denser than typical for their own population.

The gap between the two numbers is itself a diagnostic: it measures how hard
``G_REACH`` is biasing the pool.
"""
from __future__ import annotations

import json
import random
import statistics as st
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..utils import ensure_parent, log
from .config import SidConfig
from .dense import pairwise_sample
from .env import Env


@dataclass
class DensityModel:
    tau_sim: float
    tau_low: float
    density_median_all: float
    density_median_reach: float | None = None
    n_reach_tasks: int = 0
    # the reach-conditioned median is only the working norm once its population
    # is big enough to estimate; below that it is recorded as a diagnostic only
    reach_median_is_working: bool = False
    similarity_shape: dict[str, float] | None = None

    @property
    def median(self) -> float:
        """The working norm (§7.3 only holds for the reach-conditioned one)."""
        if self.reach_median_is_working and self.density_median_reach is not None:
            return self.density_median_reach
        return self.density_median_all

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau_sim": round(self.tau_sim, 6),
            "tau_low": round(self.tau_low, 6),
            "density_median_all": self.density_median_all,
            "density_median_reach": self.density_median_reach,
            "reach_median_is_working": self.reach_median_is_working,
            "density_median_used": self.median,
            "n_reach_tasks": self.n_reach_tasks,
            "similarity_shape": self.similarity_shape,
        }


def density_of(env: Env, chunk_id: str, tau: float, exclude: set[str]) -> int:
    vec = env.dense.vec(chunk_id)
    if vec is None:
        return 0
    sims = env.dense.sims_to(vec)
    mask = sims > tau
    n = int(mask.sum())
    for cid in exclude | {chunk_id}:                       # self + gold don't count
        i = env.dense._pos.get(cid)
        if i is not None and sims[i] > tau:
            n -= 1
    return max(0, n)


def task_density(env: Env, gold: list[str], tau: float) -> int:
    if not gold:
        return 0
    exclude = set(gold)
    return min(density_of(env, g, tau, exclude) for g in gold)


def fit_density_model(cfg: SidConfig, env: Env,
                      reach_task_golds: list[list[str]] | None = None) -> DensityModel:
    d = cfg.density
    sims = pairwise_sample(env.dense, d.sample_chunks, d.seed)
    if sims.size == 0:
        return DensityModel(tau_sim=1.0, tau_low=1.0, density_median_all=0.0)

    tau_sim = float(np.percentile(sims, d.tau_sim_percentile))
    tau_low = float(np.percentile(sims, d.tau_low_percentile))
    shape = {
        "mean": round(float(sims.mean()), 4),
        "std": round(float(sims.std()), 4),
        "p50": round(float(np.percentile(sims, 50)), 4),
        "p80": round(tau_low, 4),
        "p95": round(tau_sim, 4),
        "p99": round(float(np.percentile(sims, 99)), 4),
        "n_pairs": int(sims.size),
    }

    # provisional norm: min-density over pseudo-gold sets of 2–5 random chunks
    rng = random.Random(d.seed + 1)
    ids = env.dense.ids
    vals: list[int] = []
    n_sets = min(d.pseudo_gold_sets, max(50, len(ids) * 4))
    for _ in range(n_sets):
        k = rng.randint(2, min(5, len(ids)))
        pseudo = rng.sample(ids, k)
        vals.append(task_density(env, pseudo, tau_sim))
    median_all = float(st.median(vals)) if vals else 0.0

    model = DensityModel(tau_sim=tau_sim, tau_low=tau_low,
                         density_median_all=median_all, similarity_shape=shape)

    if reach_task_golds:
        reach_vals = [task_density(env, g, tau_sim) for g in reach_task_golds]
        model.n_reach_tasks = len(reach_vals)
        if reach_vals:
            # always recorded: the distance to density_median_all measures how
            # far G_REACH shifts the population, a ceiling diagnostic in itself
            model.density_median_reach = float(st.median(reach_vals))
        model.reach_median_is_working = len(reach_vals) >= d.min_reach_tasks_for_median
        if not model.reach_median_is_working:
            log.info("§7.1: only %d G_REACH-passing tasks (< %d) — using the "
                     "provisional median; recompute once the pool grows",
                     len(reach_vals), d.min_reach_tasks_for_median)
    log.info("§7.1: τ_sim=%.4f (p%.0f) τ_low=%.4f | median_all=%.1f median_reach=%s",
             tau_sim, d.tau_sim_percentile, tau_low, median_all,
             model.density_median_reach)
    return model


def annotate_density(cfg: SidConfig, env: Env, records: list[dict],
                     model: DensityModel) -> list[dict]:
    low_thr = model.median * cfg.distractors.low_threshold_ratio
    out = []
    for rec in records:
        gold = rec["gold_chunk_ids"]
        dens = task_density(env, gold, model.tau_sim)
        rec = dict(rec)
        rec["metrics"] = {
            **rec.get("metrics", {}),
            "neighborhood_density_v0": dens,
            "density_median_reach": model.density_median_reach,
            "density_median_all": model.density_median_all,
            "tau_sim_percentile": cfg.density.tau_sim_percentile,
            "tau_low_percentile": cfg.density.tau_low_percentile,
        }
        rec["complexity"] = {**rec.get("complexity", {}),
                             "sparse_origin": dens < low_thr}
        # §7.3 — normalise to the median, never amplify past it
        need = int(max(0, round(model.median - dens)))
        rec["_n_distractors_target"] = min(need, cfg.distractors.n_max)
        out.append(rec)
    return out


def save_density(cfg: SidConfig, model: DensityModel, extra: dict | None = None) -> None:
    ensure_parent(cfg.paths.density_stats)
    with open(cfg.paths.density_stats, "w", encoding="utf-8") as f:
        json.dump({**model.to_dict(), **(extra or {})}, f, ensure_ascii=False, indent=2)
