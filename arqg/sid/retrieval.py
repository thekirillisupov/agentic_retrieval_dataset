"""Hybrid retrieval (BM25 + dense, fused with RRF) and the query↔chunk gaps.

This is the environment the gates measure against, so it must be the *same*
retriever the agent will use at rollout time. One probe yields the ranking and
all three gaps of §4.3 — ``lex_gap`` / ``dense_gap`` / ``fused_gap`` — because
they only need the per-branch scores that the fusion already computed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..embeddings import BaseEmbedder
from .dense import DenseIndex
from .lexical import BM25Index


def _gap(target: float, best: float) -> float:
    """1 − score/best, clamped to [0, 1]. 1.0 = target scores nothing."""
    if best <= 0:
        return 1.0
    return float(min(1.0, max(0.0, 1.0 - max(0.0, target) / best)))


@dataclass
class Probe:
    query: str
    hits: list[tuple[str, float]] = field(default_factory=list)   # fused, descending
    lex_gap: dict[str, float] = field(default_factory=dict)
    dense_gap: dict[str, float] = field(default_factory=dict)
    fused_gap: dict[str, float] = field(default_factory=dict)

    @property
    def hit_ids(self) -> list[str]:
        return [c for c, _ in self.hits]

    def hit_rank(self, cid: str) -> int:
        for i, (c, _) in enumerate(self.hits):
            if c == cid:
                return i + 1
        return 0


class HybridSearcher:
    def __init__(self, bm25: BM25Index, dense: DenseIndex, embedder: BaseEmbedder,
                 rrf_k: int = 60, candidates: int = 100):
        self.bm25 = bm25
        self.dense = dense
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.candidates = candidates

    # ---- mutation (injection) ------------------------------------------- #
    def add_documents(self, ids: list[str], texts: list[str], vecs: np.ndarray) -> None:
        self.bm25.add_many(list(zip(ids, texts)))
        self.dense.add(ids, vecs)

    # ---- probing --------------------------------------------------------- #
    async def probe(self, query: str, top_k: int,
                    targets: Sequence[str] = ()) -> Probe:
        return (await self.probe_many([query], top_k, [list(targets)]))[0]

    async def probe_many(self, queries: list[str], top_k: int,
                         targets: list[list[str]] | None = None) -> list[Probe]:
        if not queries:
            return []
        targets = targets or [[] for _ in queries]
        qvecs = await self.embedder.embed(queries, kind="query")
        out: list[Probe] = []
        for q, qvec, tgt in zip(queries, qvecs, targets):
            out.append(self._fuse(q, qvec, tgt, top_k))
        return out

    def _fuse(self, query: str, qvec: np.ndarray, targets: list[str], top_k: int) -> Probe:
        lex_hits, lex_target = self.bm25.search_with_targets(
            query, self.candidates, targets)
        dense_hits = self.dense.search(qvec, self.candidates)
        dense_target = self.dense.scores_for(qvec, targets)

        rrf: dict[str, float] = {}
        for rank, (cid, _) in enumerate(lex_hits, start=1):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, (cid, _) in enumerate(dense_hits, start=1):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        fused = sorted(rrf.items(), key=lambda kv: -kv[1])

        best_lex = lex_hits[0][1] if lex_hits else 0.0
        best_dense = dense_hits[0][1] if dense_hits else 0.0
        best_fused = fused[0][1] if fused else 0.0

        p = Probe(query=query, hits=fused[:top_k])
        for t in targets:
            p.lex_gap[t] = _gap(lex_target.get(t, 0.0), best_lex)
            p.dense_gap[t] = _gap(dense_target.get(t, 0.0), best_dense)
            p.fused_gap[t] = _gap(rrf.get(t, 0.0), best_fused)
        return p


def aggregate_gaps(probe: Probe, gold: list[str]) -> dict[str, float]:
    """Task-level gaps: ``max`` over the gold set — the narrow chunk sets the
    difficulty, mirroring the ``min`` rule for density (plan §4.3)."""
    return aggregate_gaps_over_groups(probe, [[g] for g in gold])


def aggregate_gaps_over_groups(probe: Probe, groups: list[list[str]]) -> dict[str, float]:
    """Same rule, one level up, once ``G_REP`` has built fact groups.

    A fact is satisfied by *any* member of its group, so the group's gap is the
    ``min`` over its members — the easiest way in. The task's gap is then the
    ``max`` over groups: the hardest fact still sets the difficulty. Taking
    ``max`` over raw chunks instead would report a task as hard because some
    redundant duplicate of an easy fact ranks poorly, which is not a fact about
    the task at all.
    """
    groups = [g for g in groups if g]
    if not groups:
        return {"lex_gap": 1.0, "dense_gap": 1.0, "fused_gap": 1.0}
    out = {}
    for name, table in (("lex_gap", probe.lex_gap), ("dense_gap", probe.dense_gap),
                        ("fused_gap", probe.fused_gap)):
        out[name] = max(min(table.get(c, 1.0) for c in group) for group in groups)
    return out


def gap_bin(value: float, bins: dict[str, float]) -> str:
    low = bins.get("low", 0.33)
    mid = bins.get("mid", 0.66)
    if value <= low:
        return "low"
    if value <= mid:
        return "mid"
    return "high"
