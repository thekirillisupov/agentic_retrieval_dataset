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


def passage_text(title: str, raw_text: str, with_title: bool) -> str:
    """The exact string a passage is embedded as.

    Public because injection has to reproduce it. A distractor is a chunk of
    this index like any other: embedding it as bare text while every v0 chunk
    carries its title puts it in a different place of the space than the one
    it will occupy once the v1 index is rebuilt from disk — so the
    neighbourhood check of §7.5 would be measured on a vector nobody ever
    retrieves against.
    """
    return f"{title}\n{raw_text}" if (with_title and title) else raw_text


def chunk_passage_text(c, with_title: bool) -> str:
    return passage_text(c.title, c.raw_text, with_title)


def dense_signature(cfg: SidConfig, corpus: SidCorpus, version: str) -> dict:
    """What the dense cache is keyed on. Any injection changes the checksum, so
    a version whose index is not saved after injection can never be resumed
    into — it re-embeds the whole corpus instead (see `distractors.save_index`).
    """
    return {
        "model": cfg.embed.model,
        "backend": cfg.embed.backend,
        "n": len(corpus),
        "version": version,
        "corpus": os.path.abspath(cfg.paths.corpus),
        "checksum": corpus.checksum(),
    }


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


async def build_dense_for(cfg: SidConfig, corpus: SidCorpus, version: str,
                          embedder: BaseEmbedder) -> DenseIndex:
    """The dense index of one corpus version, cached on disk.

    Split out of ``build_env`` because S1 wants the embeddings without the rest
    of the environment, and it has to hit the *same* cache: the signature is
    what makes the mining stage and the gates share one embedding bill.
    """
    chunks = corpus.all_chunks()
    return await build_dense(
        embedder, [c.id for c in chunks],
        [chunk_passage_text(c, cfg.embed.embed_with_title) for c in chunks],
        cfg.paths.dense_dir(version), dense_signature(cfg, corpus, version),
        rebuild=cfg.embed.rebuild_index)


async def build_env(cfg: SidConfig, *, version: str = "v0",
                    embedder: BaseEmbedder | None = None) -> Env:
    corpus = load_corpus(cfg, with_injections=(version != "v0"))
    corpus.version = version
    emb = embedder or make_embedder(cfg.embed)
    dense = await build_dense_for(cfg, corpus, version, emb)
    bm25 = build_bm25(corpus)
    searcher = HybridSearcher(bm25, dense, emb, rrf_k=cfg.rrf_k,
                              candidates=cfg.fusion_candidates)
    return Env(cfg=cfg, corpus=corpus, bm25=bm25, dense=dense,
               embedder=emb, searcher=searcher)
