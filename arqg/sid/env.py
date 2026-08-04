"""Assembling the retrieval environment for a given index version.

Every stage that measures anything (gates, density, distractors, isolation)
needs the same triple: corpus + BM25 + dense index, wired into one hybrid
searcher. Building it in one place keeps train/measure consistency, which the
plan makes a blocking requirement (§9.1).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from ..embeddings import BaseEmbedder, make_embedder
from ..utils import log
from .config import SidConfig
from .corpus import SidCorpus, load_corpus
from .dense import DenseIndex, build_dense
from .lexical import BM25Index
from .retrieval import HybridSearcher


def _passage_text(c, with_title: bool) -> str:
    return f"{c.title}\n{c.raw_text}" if (with_title and c.title) else c.raw_text


def build_bm25(corpus: SidCorpus) -> BM25Index:
    idx = BM25Index()
    idx.add_many([(c.id, c.raw_text) for c in corpus.all_chunks()])
    log.info("bm25: %d docs, avgdl=%.1f", idx.n_docs, idx.avgdl)
    return idx


@dataclass
class Env:
    cfg: SidConfig
    corpus: SidCorpus
    bm25: BM25Index
    dense: DenseIndex
    embedder: BaseEmbedder
    searcher: HybridSearcher

    async def aclose(self) -> None:
        await self.embedder.aclose()


async def build_env(cfg: SidConfig, *, version: str = "v0",
                    embedder: BaseEmbedder | None = None) -> Env:
    corpus = load_corpus(cfg, with_injections=(version != "v0"))
    corpus.version = version
    emb = embedder or make_embedder(cfg.embed)
    chunks = corpus.all_chunks()
    signature = {
        "model": cfg.embed.model,
        "backend": cfg.embed.backend,
        "n": len(chunks),
        "version": version,
        "corpus": os.path.abspath(cfg.paths.corpus),
        "checksum": corpus.checksum(),
    }
    dense = await build_dense(
        emb, [c.id for c in chunks],
        [_passage_text(c, cfg.embed.embed_with_title) for c in chunks],
        cfg.paths.dense_dir(version), signature, rebuild=cfg.embed.rebuild_index)
    bm25 = build_bm25(corpus)
    searcher = HybridSearcher(bm25, dense, emb, rrf_k=cfg.rrf_k,
                              candidates=cfg.fusion_candidates)
    return Env(cfg=cfg, corpus=corpus, bm25=bm25, dense=dense,
               embedder=emb, searcher=searcher)
