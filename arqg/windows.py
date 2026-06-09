"""Stage 1 — build contiguous neighbour windows from the chunk store.

A window is a run of consecutive chunks within a single document. It is the
context handed to the generator; the *required* subset becomes gold after
verification. We bias toward small windows (2-3) because tight neighbours give
the cleanest multi-hop bridges, and we cap windows per file so large documents
don't dominate the dataset.
"""
from __future__ import annotations

import random

from .config import WindowConfig, FilterConfig
from .data import ChunkStore, is_eligible_seed
from .schema import Chunk, Window
from .utils import log


def build_windows(store: ChunkStore, wcfg: WindowConfig, fcfg: FilterConfig) -> list[Window]:
    rng = random.Random(wcfg.seed)
    files = store.files
    rng.shuffle(files)

    windows: list[Window] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()

    for file_name in files:
        indices = store.file_indices(file_name)
        if len(indices) < 2:
            continue  # need at least two neighbours for a multi-chunk question
        per_file = 0
        # iterate seeds by position so windows are contiguous in document order
        for pos in range(0, len(indices), wcfg.stride):
            if per_file >= wcfg.max_windows_per_file:
                break
            start_index = indices[pos]
            seed = store.get(file_name, start_index)
            if seed is None or not is_eligible_seed(seed, fcfg):
                continue
            size = _sample_size(rng, wcfg)
            chunks = store.contiguous_window(file_name, start_index, size)
            if chunks is None:
                # shrink to the largest contiguous run we can still form (>=2)
                chunks = _largest_contiguous(store, file_name, start_index, size)
                if chunks is None:
                    continue
            total = sum(c.n_chars for c in chunks)
            if total > wcfg.max_window_chars and len(chunks) > 2:
                chunks = chunks[:2]
                total = sum(c.n_chars for c in chunks)
            if len(chunks) < 2 or total > wcfg.max_window_chars:
                continue
            key = (file_name, tuple(c.index for c in chunks))
            if key in seen:
                continue
            seen.add(key)
            windows.append(_to_window(file_name, chunks, total))
            per_file += 1

    rng.shuffle(windows)
    if wcfg.target_windows and len(windows) > wcfg.target_windows:
        windows = windows[: wcfg.target_windows]
    log.info("built %d windows from %d files", len(windows), len(files))
    return windows


def _sample_size(rng: random.Random, wcfg: WindowConfig) -> int:
    return rng.choices(wcfg.window_sizes, weights=wcfg.window_size_weights, k=1)[0]


def _largest_contiguous(store: ChunkStore, file_name: str, start_index: int, size: int):
    for s in range(size - 1, 1, -1):
        chunks = store.contiguous_window(file_name, start_index, s)
        if chunks is not None:
            return chunks
    return None


def _to_window(file_name: str, chunks: list[Chunk], total: int) -> Window:
    indices = [c.index for c in chunks]
    return Window(
        window_id=Window.make_id(file_name, indices),
        file_name=file_name,
        indices=indices,
        chunk_ids=[c.id for c in chunks],
        texts=[c.raw_text for c in chunks],
        n_chars=total,
    )
