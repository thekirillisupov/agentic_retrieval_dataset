"""Dense side of the index: embedding matrix per index version, with caching.

The plan makes the index mutable (v0 → vN, §2.3), so unlike ``arqg.index`` this
one is append-friendly and caches per version.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..embeddings import BaseEmbedder
from ..utils import ensure_parent, log


class DenseIndex:
    def __init__(self, ids: list[str], matrix: np.ndarray):
        self.ids = ids
        self.matrix = matrix if matrix.size else np.zeros((0, 1), dtype="float32")
        self._pos = {cid: i for i, cid in enumerate(ids)}

    # ---- persistence ----------------------------------------------------- #
    def save(self, index_dir: str, signature: dict) -> None:
        ensure_parent(os.path.join(index_dir, "x"))
        np.save(os.path.join(index_dir, "embeddings.npy"), self.matrix)
        with open(os.path.join(index_dir, "ids.json"), "w", encoding="utf-8") as f:
            json.dump(self.ids, f)
        with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"signature": signature, "dim": int(self.matrix.shape[1])}, f)

    @classmethod
    def load(cls, index_dir: str, signature: dict) -> "DenseIndex | None":
        meta_p = os.path.join(index_dir, "meta.json")
        emb_p = os.path.join(index_dir, "embeddings.npy")
        ids_p = os.path.join(index_dir, "ids.json")
        if not all(os.path.exists(p) for p in (meta_p, emb_p, ids_p)):
            return None
        with open(meta_p, "r", encoding="utf-8") as f:
            if json.load(f).get("signature") != signature:
                return None
        with open(ids_p, "r", encoding="utf-8") as f:
            ids = json.load(f)
        return cls(ids, np.load(emb_p))

    # ---- mutation -------------------------------------------------------- #
    def add(self, ids: list[str], vecs: np.ndarray) -> None:
        if not ids:
            return
        keep = [i for i, cid in enumerate(ids) if cid not in self._pos]
        if not keep:
            return
        new_ids = [ids[i] for i in keep]
        new_vecs = vecs[keep]
        base = len(self.ids)
        self.ids.extend(new_ids)
        for j, cid in enumerate(new_ids):
            self._pos[cid] = base + j
        self.matrix = (new_vecs.astype("float32") if self.matrix.size == 0
                       else np.vstack([self.matrix, new_vecs.astype("float32")]))

    # ---- lookup ---------------------------------------------------------- #
    def has(self, cid: str) -> bool:
        return cid in self._pos

    def vec(self, cid: str) -> np.ndarray | None:
        i = self._pos.get(cid)
        return None if i is None else self.matrix[i]

    def vecs(self, cids: list[str]) -> np.ndarray:
        rows = [self._pos[c] for c in cids if c in self._pos]
        if not rows:
            return np.zeros((0, self.matrix.shape[1]), dtype="float32")
        return self.matrix[rows]

    def search(self, qvec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self.matrix.size == 0:
            return []
        sims = self.matrix @ np.asarray(qvec, dtype="float32")
        k = min(top_k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
        idx = idx[np.argsort(-sims[idx])]
        return [(self.ids[int(i)], float(sims[int(i)])) for i in idx]

    def sims_to(self, qvec: np.ndarray) -> np.ndarray:
        if self.matrix.size == 0:
            return np.zeros(0, dtype="float32")
        return self.matrix @ np.asarray(qvec, dtype="float32")

    def scores_for(self, qvec: np.ndarray, cids: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for c in cids:
            v = self.vec(c)
            if v is not None:
                out[c] = float(np.dot(v, qvec))
        return out


async def build_dense(embedder: BaseEmbedder, ids: list[str], texts: list[str],
                      index_dir: str, signature: dict,
                      rebuild: bool = False) -> DenseIndex:
    if not rebuild:
        cached = DenseIndex.load(index_dir, signature)
        if cached is not None:
            log.info("dense: loaded %d cached embeddings from %s", len(cached.ids), index_dir)
            return cached
    log.info("dense: embedding %d chunks ...", len(ids))
    matrix = await embedder.embed(texts, kind="passage")
    idx = DenseIndex(list(ids), matrix)
    idx.save(index_dir, signature)
    log.info("dense: cached %s -> %s", matrix.shape, index_dir)
    return idx
