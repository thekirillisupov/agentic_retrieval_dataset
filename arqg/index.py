"""Dense embedding index over the corpus, with on-disk caching.

Embedding the whole corpus is the expensive part, so the matrix is cached under
``index_dir`` and reused unless the corpus signature (path/size/mtime + count) or
the embedding model changes.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .config import RetrieveConfig
from .data import ChunkStore
from .embeddings import BaseEmbedder
from .schema import Chunk
from .utils import ensure_parent, log


def _passage_text(c: Chunk, with_title: bool) -> str:
    if with_title and c.title:
        return f"{c.title}\n{c.raw_text}"
    return c.raw_text


def _signature(chunks_path: str, model: str, n: int) -> dict:
    sig = {"model": model, "n": n, "path": os.path.abspath(chunks_path)}
    try:
        st = os.stat(chunks_path)
        sig["size"] = st.st_size
        sig["mtime"] = int(st.st_mtime)
    except OSError:
        pass
    return sig


class EmbeddingIndex:
    def __init__(self, chunk_ids: list[str], matrix: np.ndarray, meta: list[dict]):
        self.chunk_ids = chunk_ids
        self.matrix = matrix              # (N, D) normalised float32
        self.meta = meta                  # per-chunk {document_id, title, file_name, index}
        self._pos = {cid: i for i, cid in enumerate(chunk_ids)}

    # ---- persistence ---------------------------------------------------- #
    def save(self, index_dir: str, signature: dict) -> None:
        ensure_parent(os.path.join(index_dir, "x"))
        np.save(os.path.join(index_dir, "embeddings.npy"), self.matrix)
        with open(os.path.join(index_dir, "ids.jsonl"), "w", encoding="utf-8") as f:
            for cid, m in zip(self.chunk_ids, self.meta):
                f.write(json.dumps({"chunk_id": cid, **m}, ensure_ascii=False) + "\n")
        with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"signature": signature, "dim": int(self.matrix.shape[1])}, f)

    @classmethod
    def load(cls, index_dir: str) -> "EmbeddingIndex | None":
        emb_p = os.path.join(index_dir, "embeddings.npy")
        ids_p = os.path.join(index_dir, "ids.jsonl")
        if not (os.path.exists(emb_p) and os.path.exists(ids_p)):
            return None
        matrix = np.load(emb_p)
        chunk_ids, meta = [], []
        with open(ids_p, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                chunk_ids.append(rec.pop("chunk_id"))
                meta.append(rec)
        return cls(chunk_ids, matrix, meta)

    @staticmethod
    def cached_signature(index_dir: str) -> dict | None:
        p = os.path.join(index_dir, "meta.json")
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f).get("signature")

    # ---- search --------------------------------------------------------- #
    def search(self, query_vecs: np.ndarray, top_k: int) -> list[list[tuple[str, float]]]:
        """For each query row return up to top_k (chunk_id, score), descending."""
        if self.matrix.size == 0 or query_vecs.size == 0:
            return [[] for _ in range(len(query_vecs))]
        sims = query_vecs @ self.matrix.T              # (Q, N)
        k = min(top_k, self.matrix.shape[0])
        out: list[list[tuple[str, float]]] = []
        for row in sims:
            idx = np.argpartition(-row, k - 1)[:k] if k < len(row) else np.arange(len(row))
            idx = idx[np.argsort(-row[idx])]
            out.append([(self.chunk_ids[int(i)], float(row[int(i)])) for i in idx])
        return out

    def meta_for(self, chunk_id: str) -> dict:
        i = self._pos.get(chunk_id)
        return self.meta[i] if i is not None else {}


async def build_or_load_index(cfg: RetrieveConfig, embedder: BaseEmbedder,
                              store: ChunkStore, chunks_path: str,
                              index_dir: str) -> EmbeddingIndex:
    chunks = store.all_chunks()
    signature = _signature(chunks_path, cfg.model, len(chunks))

    if not cfg.rebuild_index and EmbeddingIndex.cached_signature(index_dir) == signature:
        idx = EmbeddingIndex.load(index_dir)
        if idx is not None:
            log.info("index: loaded cached embeddings (%d chunks) from %s",
                     len(idx.chunk_ids), index_dir)
            return idx

    log.info("index: embedding %d corpus chunks with %s ...", len(chunks), cfg.model)
    texts = [_passage_text(c, cfg.embed_with_title) for c in chunks]
    matrix = await embedder.embed(texts, kind="passage")
    chunk_ids = [c.id for c in chunks]
    meta = [{"document_id": c.document_id, "title": c.title,
             "file_name": c.file_name, "index": c.index} for c in chunks]
    idx = EmbeddingIndex(chunk_ids, matrix, meta)
    idx.save(index_dir, signature)
    log.info("index: built and cached %d x %d embeddings -> %s",
             matrix.shape[0], matrix.shape[1], index_dir)
    return idx
