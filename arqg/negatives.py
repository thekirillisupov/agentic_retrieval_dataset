"""Stage 4 (optional) — mine hard negatives with a multilingual embedder.

Hard negatives are chunks that are semantically close to the question but are
NOT gold. They make the dataset far more useful for *training/evaluating*
retrievers. This stage is optional and only runs if `negatives.enabled`.

Requires: sentence-transformers (+ torch). A Russian-capable model such as
``intfloat/multilingual-e5-large`` is recommended.
"""
from __future__ import annotations

from .config import Config
from .data import ChunkStore
from .schema import DatasetItem
from .utils import log, read_jsonl, write_jsonl


def mine_negatives(cfg: Config, store: ChunkStore) -> None:
    if not cfg.negatives.enabled:
        log.info("negatives: disabled, skipping")
        return
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pip install sentence-transformers numpy — required for negative mining"
        ) from e

    src = cfg.paths.items_source()   # collected (with positives) if present, else verified
    items = [DatasetItem(**d) for d in read_jsonl(src)]
    if not items:
        log.warning("negatives: no items found in %s", src)
        return

    ncfg = cfg.negatives
    model = SentenceTransformer(ncfg.embedding_model, device=ncfg.device)

    chunks = store.all_chunks()
    chunk_ids = [c.id for c in chunks]
    id_to_pos = {cid: i for i, cid in enumerate(chunk_ids)}

    # e5-family models expect "passage:" / "query:" prefixes.
    is_e5 = "e5" in ncfg.embedding_model.lower()
    passages = [(f"passage: {c.raw_text}" if is_e5 else c.raw_text) for c in chunks]
    log.info("negatives: embedding %d chunks", len(passages))
    chunk_emb = model.encode(passages, batch_size=ncfg.batch_size,
                             normalize_embeddings=True, show_progress_bar=True)
    chunk_emb = np.asarray(chunk_emb, dtype="float32")

    queries = [(f"query: {it.question}" if is_e5 else it.question) for it in items]
    q_emb = model.encode(queries, batch_size=ncfg.batch_size,
                         normalize_embeddings=True, show_progress_bar=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    sims = q_emb @ chunk_emb.T  # cosine (normalized)
    out = []
    over_fetch = ncfg.top_k + 64
    for i, it in enumerate(items):
        # never mine a validated positive (near-duplicate source) as a negative
        excluded = (set(it.gold_chunk_ids) | set(it.window_chunk_ids)
                    | set(it.positive_chunk_ids))
        row = sims[i]
        order = np.argsort(-row)[:over_fetch]
        negs: list[str] = []
        for pos in order:
            cid = chunk_ids[int(pos)]
            if cid in excluded:
                continue
            if ncfg.exclude_same_file and store.get_by_id(cid).file_name == it.file_name:
                continue
            negs.append(cid)
            if len(negs) >= ncfg.top_k:
                break
        it.hard_negative_ids = negs
        out.append(it.to_dict())

    write_jsonl(cfg.paths.dataset, out)
    log.info("negatives: wrote %d items with hard negatives -> %s", len(out), cfg.paths.dataset)
