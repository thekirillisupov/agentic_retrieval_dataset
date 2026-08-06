"""S1 second channel — bridging two chunks by doc2doc similarity.

The entity channel asks "which rare surface form do these two chunks share?".
That question has a hard ceiling, and on `ckr` it is reached early: of 800
section scopes only 283 contain an entity repeated in two of their chunks, and
lifting τ_df to infinity buys 352. The other 517 folders (5 431 chunks) are not
filtered out by a threshold — **no surface form repeats in them at all**. Two
chunks about one subject worded differently are invisible to an exact-match
bridge, and that is most of a knowledge base written by different people.

So a folder can also be bridged by the embedder: pairs the retriever itself
considers related, which is the relation the task will actually be measured in.

**The band is bounded above by a rank, not by a cosine.** Raw similarity does
not separate good tasks from bad ones — mined subgraphs that reached a task
averaged 0.674 against 0.682 for the ones that died in the gates — but it
predicts *triviality* sharply, because two chunks similar enough are returned by
one query and G_BROAD rejects the task by definition:

    partner in the other's top-3 neighbours   G_BROAD passed 0.22
    partner beyond neighbour 50               G_BROAD passed 0.55

An absolute cosine ceiling cannot express that here: the corpus distribution is
compressed exactly where the ceiling belongs (p96 = 0.54, p98 = 0.80), so a
threshold moved by one percentile moves by 0.18 of cosine, and a folder of
near-identical defect cards sits above any ceiling calibrated corpus-wide.
Neighbour rank is scale-free and local, which is what the failure mode is.

The lower edge stays a corpus percentile of the *same* pairwise sample §7.1 fits
τ_sim on: below it a pair is no more related than two random chunks, and being
in one folder is not enough to make a question out of it.

What this channel cannot supply is an *anchor*. An entity bridge names the thing
the two chunks have in common; a similarity bridge only asserts that something
is. Composing on that invites the invented link that section scoping exists to
remove — so the anchor is required later, from the facts, in `compose.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils import log
from .dense import DenseIndex, pairwise_sample


@dataclass
class SimBand:
    """The similarity window a bridge pair has to land in."""
    low: float
    exclude_top_k: int
    percentile: float

    def to_dict(self) -> dict[str, float]:
        return {"sim_bridge_low": round(self.low, 6),
                "sim_bridge_low_percentile": self.percentile,
                "sim_bridge_exclude_top_k": self.exclude_top_k}


def fit_band(dense: DenseIndex, percentile: float, exclude_top_k: int,
             sample_chunks: int, seed: int) -> SimBand:
    sims = pairwise_sample(dense, sample_chunks, seed)
    low = float(np.percentile(sims, percentile)) if sims.size else 1.0
    log.info("S1/sim: band low=%.4f (p%.0f of %d sampled pairs), "
             "excluding partners inside the top-%d neighbours",
             low, percentile, sims.size, exclude_top_k)
    return SimBand(low=low, exclude_top_k=exclude_top_k, percentile=percentile)


class CoRetrievability:
    """Is one chunk already inside the other's top-k nearest neighbours?

    The k-th neighbour's similarity is cached per chunk rather than the whole
    neighbour list: the test is ``sim(a, b) >= kth(a)``, one full row of the
    corpus per chunk asked about. Rows are computed on demand, because only the
    pairs a scope actually tries to emit need one — computing them for every
    candidate pair up front is a corpus-sized matrix multiplication spent mostly
    on pairs the per-folder budget will never reach.
    """

    def __init__(self, dense: DenseIndex, top_k: int):
        self.dense = dense
        self.top_k = top_k
        self._kth: dict[str, float] = {}

    def _kth_sim(self, cid: str) -> float:
        if cid in self._kth:
            return self._kth[cid]
        vec = self.dense.vec(cid)
        if vec is None or self.top_k <= 0:
            val = 1.1                              # nothing can reach it
        else:
            sims = self.dense.sims_to(vec)
            k = min(self.top_k, len(sims) - 1)     # rank 0 is the chunk itself
            val = float(np.partition(sims, -(k + 1))[-(k + 1)]) if k >= 0 else 1.1
        self._kth[cid] = val
        return val

    def co_retrievable(self, a: str, b: str, sim: float) -> bool:
        return sim >= self._kth_sim(a) or sim >= self._kth_sim(b)


def scope_pairs(dense: DenseIndex, cids: list[str], band: SimBand,
                limit: int) -> list[tuple[tuple[str, str], float]]:
    """Candidate similarity pairs of one scope, least co-retrievable first.

    Ordered by ascending similarity and de-overlapped the way the entity channel
    orders its own pairs: inside a folder topicality is already settled by the
    scope, so the axis left to spend the budget on is how far apart the two
    chunks are for the retriever.
    """
    usable = [c for c in cids if dense.has(c)]
    if len(usable) < 2 or limit <= 0:
        return []
    mat = dense.vecs(usable)
    sims = mat @ mat.T
    iu = np.triu_indices(len(usable), k=1)
    vals = sims[iu]
    keep = np.where(vals >= band.low)[0]
    if keep.size == 0:
        return []
    keep = keep[np.argsort(vals[keep])]
    rows, cols = iu[0][keep], iu[1][keep]

    out: list[tuple[tuple[str, str], float]] = []
    used: set[str] = set()
    overflow: list[tuple[tuple[str, str], float]] = []
    for i, j, s in zip(rows, cols, vals[keep]):
        pair = (usable[int(i)], usable[int(j)])
        if used & set(pair):
            overflow.append((pair, float(s)))
            continue
        out.append((pair, float(s)))
        used |= set(pair)
        if len(out) >= limit:
            return out
    for item in overflow:                          # overlap rather than starve
        if len(out) >= limit:
            break
        out.append(item)
    return out
