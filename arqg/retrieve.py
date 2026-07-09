"""Stage 4b — dense retrieval over the corpus to fill the collect round.

Reads ``retrieval_requests.jsonl`` (one query per clue), runs each query against
the embedding index, and writes ``retrieval_results.jsonl`` in exactly the format
``collect-positives`` consumes. This automates what was previously a manual step.
"""
from __future__ import annotations

import os

from .config import Config
from .data import ChunkStore
from .embeddings import make_embedder
from .index import build_or_load_index
from .utils import ensure_parent, load_done_keys, log, read_jsonl, write_jsonl


def _index_dir(cfg: Config) -> str:
    return cfg.retrieve.index_dir or os.path.join(cfg.paths.out_dir, "index")


async def retrieve(cfg: Config, store: ChunkStore) -> None:
    requests = list(read_jsonl(cfg.paths.retrieval_requests))
    if not requests:
        log.warning("retrieve: no requests at %s (run the clues stage first)",
                    cfg.paths.retrieval_requests)
        return

    done = load_done_keys(cfg.paths.retrieval_results, "clue_id")
    todo = [r for r in requests if r.get("clue_id") not in done]
    log.info("retrieve: %d requests, %d done, %d to do", len(requests), len(done), len(todo))
    if not todo:
        return

    embedder = make_embedder(cfg.retrieve)
    try:
        idx = await build_or_load_index(
            cfg.retrieve, embedder, store, cfg.paths.corpus, _index_dir(cfg))

        queries = [r.get("query", "") for r in todo]
        q_vecs = await embedder.embed(queries, kind="query")
        default_k = cfg.retrieve.top_k or cfg.collect.top_k
        # group by requested top_k so we fetch enough for each row
        max_k = max((int(r.get("top_k") or default_k) for r in todo), default=default_k)
        hits = idx.search(q_vecs, max_k)
    finally:
        await embedder.aclose()

    ensure_parent(cfg.paths.retrieval_results)
    written = 0
    for r, row in zip(todo, hits):
        k = int(r.get("top_k") or default_k)
        passages = []
        for cid, score in row[:k]:
            m = idx.meta_for(cid)
            passages.append({
                "chunk_id": cid,
                "document_id": m.get("document_id", ""),
                "title": m.get("title", ""),
                "file_name": m.get("file_name", ""),
                "index": m.get("index"),
                "score": round(score, 6),
            })
        write_jsonl(cfg.paths.retrieval_results,
                    [{"clue_id": r["clue_id"], "passages": passages}], mode="a")
        written += 1
    log.info("retrieve: wrote results for %d clues -> %s", written, cfg.paths.retrieval_results)
