"""Stage orchestration.

    S0 compat → S1 mine → S3 facts → S3 compose → S4 gates → S5 minimize
       → S6 density → S6 distract (v0 → v1) → S7 isolate → S8 export

Every stage reads and writes JSONL and resumes by skipping keys already present
in its output, so a crashed run costs only the stage it died in.
"""
from __future__ import annotations

import os
from typing import Any

from ..utils import log, read_jsonl
from .compat import run_compat
from .compose import candidates_from_dicts, compose_candidates
from .config import SidConfig
from .density import (DensityModel, annotate_density, fit_density_model,
                      save_density)
from .distractors import reach_recheck, run_distractors
from .env import build_env
from .export import run_export
from .facts import extract_facts, load_facts
from .gates import run_gates, run_minimize
from .isolation import run_isolation
from .mockllm import make_sid_client
from .subgraphs import run_mining


def _subgraphs(cfg: SidConfig) -> list[dict]:
    rows = list(read_jsonl(cfg.paths.subgraphs))
    if not rows:
        raise SystemExit(f"no subgraphs at {cfg.paths.subgraphs} — run the `mine` stage")
    return rows


async def stage_facts(cfg: SidConfig) -> dict[str, list[dict]]:
    from .corpus import SidCorpus
    corpus = SidCorpus.load(cfg.paths.corpus)
    chunk_ids = [c for s in _subgraphs(cfg) for c in s["chunks"]]
    llm = make_sid_client(cfg.llm)
    try:
        return await extract_facts(cfg, llm, corpus, chunk_ids)
    finally:
        await llm.aclose()


async def stage_compose(cfg: SidConfig) -> list[dict]:
    facts = {cid: rows for cid, rows in load_facts(cfg.paths.facts).items()}
    facts = {cid: [f for f in rows if f.get("fact_id")] for cid, rows in facts.items()}
    llm = make_sid_client(cfg.llm)
    try:
        cands = await compose_candidates(cfg, llm, _subgraphs(cfg), facts)
    finally:
        await llm.aclose()
    return cands


async def stage_gates(cfg: SidConfig) -> list[dict]:
    rows = list(read_jsonl(cfg.paths.candidates))
    if not rows:
        raise SystemExit("no candidates — run the `compose` stage")
    gen, judge = make_sid_client(cfg.llm), make_sid_client(cfg.judge)
    env = await build_env(cfg, version="v0")
    try:
        return await run_gates(cfg, env, gen, judge, candidates_from_dicts(rows))
    finally:
        await gen.aclose()
        await judge.aclose()
        await env.aclose()


async def stage_minimize(cfg: SidConfig) -> list[dict]:
    rows = list(read_jsonl(cfg.paths.gated))
    if not rows:
        raise SystemExit("nothing passed the gates — run the `gates` stage")
    judge = make_sid_client(cfg.judge)
    env = await build_env(cfg, version="v0")
    try:
        return await run_minimize(cfg, env, judge, rows)
    finally:
        await judge.aclose()
        await env.aclose()


async def stage_density(cfg: SidConfig) -> tuple[DensityModel, list[dict]]:
    """§7.1 — fit τ_sim and the corpus norm on the reach-passing population,
    then annotate every task with its density and its injection budget."""
    from ..utils import write_jsonl
    rows = list(read_jsonl(cfg.paths.minimized))
    if not rows:
        raise SystemExit("no minimised tasks — run the `minimize` stage")
    env = await build_env(cfg, version="v0")
    try:
        model = fit_density_model(cfg, env, [r["gold_chunk_ids"] for r in rows])
        annotated = annotate_density(cfg, env, rows, model)
        save_density(cfg, model, extra={"n_tasks": len(annotated)})
        write_jsonl(cfg.paths.densified, annotated)
    finally:
        await env.aclose()
    return model, annotated


async def stage_distract(cfg: SidConfig) -> list[dict]:
    rows = list(read_jsonl(cfg.paths.densified))
    if not rows:
        raise SystemExit("no densified tasks — run the `density` stage")
    model = _load_density(cfg)
    version = "v1" if os.path.exists(cfg.paths.injected_corpus) else "v0"
    env = await build_env(cfg, version=version)
    gen, judge = make_sid_client(cfg.llm), make_sid_client(cfg.judge)
    try:
        out = await run_distractors(cfg, env, gen, judge, rows, model)
        recheck = await reach_recheck(cfg, env, out)
        if recheck:
            save_density(cfg, model, extra={"reach_recheck": recheck})
    finally:
        await gen.aclose()
        await judge.aclose()
        await env.aclose()
    return out


async def stage_isolate(cfg: SidConfig) -> list[dict]:
    rows = list(read_jsonl(cfg.paths.injected_tasks)) or list(read_jsonl(cfg.paths.densified))
    if not rows:
        raise SystemExit("no tasks to isolate — run the `distract` stage")
    version = "v1" if os.path.exists(cfg.paths.injected_corpus) else "v0"
    env = await build_env(cfg, version=version)
    judge = make_sid_client(cfg.judge)
    try:
        return await run_isolation(cfg, env, judge, rows)
    finally:
        await judge.aclose()
        await env.aclose()


def stage_export(cfg: SidConfig) -> dict[str, Any]:
    # Take the most downstream file that has actually run. Falling through a
    # stage that ran and rejected everything would silently export unisolated
    # tasks, so an existing-but-empty isolation output is honoured, not skipped.
    for path in (cfg.paths.isolated, cfg.paths.injected_tasks, cfg.paths.densified):
        if os.path.exists(path):
            rows = list(read_jsonl(path))
            if not rows:
                raise SystemExit(
                    f"{path} exists but is empty — the stage that writes it "
                    f"rejected every task; fix that before exporting")
            break
    else:
        raise SystemExit("nothing to export — run the pipeline first")
    extra: dict[str, Any] = {}
    if os.path.exists(cfg.paths.density_stats):
        import json
        with open(cfg.paths.density_stats, "r", encoding="utf-8") as f:
            extra["density"] = json.load(f)
    if os.path.exists(cfg.paths.gate_stats):
        import json
        with open(cfg.paths.gate_stats, "r", encoding="utf-8") as f:
            extra["gates"] = json.load(f)
    return run_export(cfg, rows, extra)


def _load_density(cfg: SidConfig) -> DensityModel:
    import json
    with open(cfg.paths.density_stats, "r", encoding="utf-8") as f:
        d = json.load(f)
    return DensityModel(
        tau_sim=d["tau_sim"], tau_low=d["tau_low"],
        density_median_all=d["density_median_all"],
        density_median_reach=d.get("density_median_reach"),
        n_reach_tasks=d.get("n_reach_tasks", 0),
        reach_median_is_working=d.get("reach_median_is_working", False),
        similarity_shape=d.get("similarity_shape"))


async def run_all(cfg: SidConfig) -> dict[str, Any]:
    run_compat(cfg)
    run_mining(cfg)
    await stage_facts(cfg)
    await stage_compose(cfg)
    await stage_gates(cfg)
    await stage_minimize(cfg)
    await stage_density(cfg)
    await stage_distract(cfg)
    await stage_isolate(cfg)
    stats = stage_export(cfg)
    log.info("done: %d tasks", stats.get("n_tasks", 0))
    return stats
