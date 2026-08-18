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
from .scoping import facet_header


def passage_text(title: str, raw_text: str, with_title: bool,
                 facets: str = "") -> str:
    """The exact string a passage is *held as* — by both index branches.

    Public because injection has to reproduce it. A distractor is a chunk of
    this index like any other: embedding it as bare text while every v0 chunk
    carries its title puts it in a different place of the space than the one
    it will occupy once the v1 index is rebuilt from disk — so the
    neighbourhood check of §7.5 would be measured on a vector nobody ever
    retrieves against.

    `facets` is the header built by `scoping.facet_header` from the fields
    `facets.fields` names. It sits between the title and the body because it
    labels the whole passage, the way the breadcrumb does.
    """
    head = [p for p in ((title if with_title else ""), facets) if p]
    return "\n".join(head + [raw_text]) if head else raw_text


def passage_facets(c, cfg: SidConfig) -> str:
    """The facet header as the INDEX holds it, or ``""`` when `in_passage` is
    off. The prompt side asks for the same header separately (see facts.py):
    the two switches are independent, so neither may read the other's flag."""
    f = cfg.facets
    if not (f.fields and f.in_passage):
        return ""
    return facet_header(c, f.fields, f.labels, f.max_value_chars)


def chunk_passage_text(c, cfg: SidConfig) -> str:
    return passage_text(c.title, c.raw_text, cfg.embed.embed_with_title,
                        passage_facets(c, cfg))


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
        # ... and how a chunk is *rendered* before it is embedded. The checksum
        # covers the corpus text, not the string built from it: without these,
        # turning facets (or the title) on reuses vectors built without them,
        # and every later measurement is taken against an index that does not
        # match the one the config describes.
        "with_title": cfg.embed.embed_with_title,
        "facets": cfg.facets.signature(),
    }


def build_bm25(corpus: SidCorpus, cfg: SidConfig) -> BM25Index:
    """The lexical branch over the SAME string the dense branch embeds.

    It used to index `raw_text` while the dense side embedded `title + text`,
    which is not a choice between two indexing policies — it is one policy
    applied to half the retriever. On a corpus whose title carries the facets
    (zakupki packs the purchase number, customer, region and year into it) the
    effect is pointed: a fact paraphrased with the customer's name is probed
    against a lexical index in which that name does not occur, so BM25 spends
    the query on documents that merely mention it in prose and ranks the gold
    below them. Both branches now see one passage — `passage_text`.
    """
    idx = BM25Index()
    idx.add_many([(c.id, chunk_passage_text(c, cfg)) for c in corpus.all_chunks()])
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
        [chunk_passage_text(c, cfg) for c in chunks],
        cfg.paths.dense_dir(version), dense_signature(cfg, corpus, version),
        rebuild=cfg.embed.rebuild_index)


async def build_env(cfg: SidConfig, *, version: str = "v0",
                    embedder: BaseEmbedder | None = None) -> Env:
    corpus = load_corpus(cfg, with_injections=(version != "v0"))
    corpus.version = version
    emb = embedder or make_embedder(cfg.embed)
    dense = await build_dense_for(cfg, corpus, version, emb)
    bm25 = build_bm25(corpus, cfg)
    searcher = HybridSearcher(bm25, dense, emb, rrf_k=cfg.rrf_k,
                              candidates=cfg.fusion_candidates)
    return Env(cfg=cfg, corpus=corpus, bm25=bm25, dense=dense,
               embedder=emb, searcher=searcher)
