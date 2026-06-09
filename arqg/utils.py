"""Small IO / checkpointing helpers shared across stages."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Iterable, Iterator

log = logging.getLogger("arqg")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_parent(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)


def read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, records: Iterable[dict[str, Any]], mode: str = "w") -> int:
    ensure_parent(path)
    n = 0
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str, record: dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_done_keys(path: str, key: str) -> set[str]:
    """Read an output JSONL and return the set of already-processed keys.

    Enables idempotent resume: re-running a stage skips items already written.
    """
    done: set[str] = set()
    for rec in read_jsonl(path):
        if key in rec:
            done.add(rec[key])
    return done
