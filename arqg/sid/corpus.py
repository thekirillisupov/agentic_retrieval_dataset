"""Versioned corpus (plan §2.3).

``v0`` is the source corpus; ``vN`` is v0 plus N waves of *additive* distractor
injection. Injection provenance (``synthetic``, ``injected_for_task``, cascade
level, distractor type) lives in a **side ledger**, never on the chunk record —
if the marker reached the agent through search/read/grep the policy would learn
"synthetic → ignore", which is a shortcut instead of a skill.

``export_public()`` writes exactly what an agent may see; ``ledger`` is ours.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

from ..data import load_chunks
from ..schema import Chunk
from ..utils import ensure_parent, log, read_jsonl, write_jsonl

PUBLIC_FIELDS = ("file_name", "index", "raw_text", "document_id", "title")


def content_hash(text: str) -> str:
    norm = " ".join((text or "").lower().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def load_meta_sidecar(path: str) -> dict[str, dict[str, Any]]:
    """``chunk_id -> facets`` from a metadata sidecar.

    Some corpora keep their facets on the chunk record itself (``load_chunks``
    already lifts anything beyond the five core fields onto ``Chunk.meta``);
    others — zakupki's ``merge`` output, for instance — keep the corpus file
    to exactly those five fields and publish everything else in a *separate*
    ``*_meta.jsonl``, one record per chunk, addressed by ``chunk_id``. This
    reads that file so `SidCorpus.load` can fold it back onto the chunks.
    """
    out: dict[str, dict[str, Any]] = {}
    for rec in read_jsonl(path):
        cid = rec.get("chunk_id")
        if not cid:
            continue
        out[cid] = {k: v for k, v in rec.items()
                    if k not in ("chunk_id", "file_name", "index")}
    return out


@dataclass
class InjectionRecord:
    chunk_id: str
    task_id: str
    level: str            # L1_transplant | L2_perturbed | L3_generated
    dtype: str            # near_duplicate | homonym | topical_lure | partial_answer
    source_chunk_id: str
    index_version: str
    perturbed_attribute: str = ""
    sim_to_gold: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class SidCorpus:
    """Chunks addressed by ``{file_name}::{index}`` across index versions."""

    def __init__(self, chunks: list[Chunk], version: str = "v0"):
        self.version = version
        self._by_id: dict[str, Chunk] = {}
        self._by_file: dict[str, list[int]] = {}
        self._hashes: set[str] = set()
        self.ledger: dict[str, InjectionRecord] = {}
        for c in chunks:
            self._register(c)

    # ---- construction ---------------------------------------------------- #
    @classmethod
    def load(cls, path: str, version: str = "v0", meta_path: str = "") -> "SidCorpus":
        chunks = load_chunks(path)
        if meta_path:
            sidecar = load_meta_sidecar(meta_path)
            n = 0
            for c in chunks:
                extra = sidecar.get(c.id)
                if extra:
                    c.meta.update(extra)
                    n += 1
            log.info("corpus: merged metadata for %d/%d chunks from %s",
                     n, len(chunks), meta_path)
        return cls(chunks, version=version)

    def _register(self, c: Chunk) -> bool:
        if c.id in self._by_id:
            return False
        self._by_id[c.id] = c
        self._by_file.setdefault(c.file_name, []).append(c.index)
        self._hashes.add(content_hash(c.raw_text))
        return True

    # ---- access ---------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, cid: str) -> bool:
        return cid in self._by_id

    def get(self, cid: str) -> Chunk | None:
        return self._by_id.get(cid)

    def text(self, cid: str) -> str:
        c = self._by_id.get(cid)
        return c.raw_text if c else ""

    def ids(self) -> list[str]:
        return list(self._by_id)

    def all_chunks(self) -> list[Chunk]:
        return list(self._by_id.values())

    @property
    def files(self) -> list[str]:
        return list(self._by_file)

    def is_synthetic(self, cid: str) -> bool:
        return cid in self.ledger

    def v0_ids(self) -> list[str]:
        """Subgraphs may only be built on v0 chunks (plan §3.2)."""
        return [cid for cid in self._by_id if cid not in self.ledger]

    def has_duplicate_text(self, text: str) -> bool:
        return content_hash(text) in self._hashes

    # ---- mutation -------------------------------------------------------- #
    def next_index(self, file_name: str) -> int:
        idx = self._by_file.get(file_name)
        return (max(idx) + 1) if idx else 0

    def inject(self, *, donor_file: str, text: str, task_id: str, level: str,
               dtype: str, source_chunk_id: str, document_id: str = "",
               title: str = "", perturbed_attribute: str = "",
               sim_to_gold: float = 0.0, version: str = "v1") -> Chunk | None:
        """Append a distractor to the corpus. Returns None on a content-hash
        duplicate (§7.5 p.3)."""
        if self.has_duplicate_text(text):
            return None
        c = Chunk(file_name=donor_file, index=self.next_index(donor_file),
                  raw_text=text, document_id=document_id, title=title)
        if not self._register(c):
            return None
        self.ledger[c.id] = InjectionRecord(
            chunk_id=c.id, task_id=task_id, level=level, dtype=dtype,
            source_chunk_id=source_chunk_id, index_version=version,
            perturbed_attribute=perturbed_attribute, sim_to_gold=sim_to_gold)
        return c

    # ---- persistence ----------------------------------------------------- #
    def export_public(self, path: str, only_injected: bool = False) -> int:
        """Agent-visible view: no synthetic markers anywhere."""
        def rows() -> Iterable[dict[str, Any]]:
            for cid, c in self._by_id.items():
                if only_injected and cid not in self.ledger:
                    continue
                yield {k: getattr(c, k) for k in PUBLIC_FIELDS}
        return write_jsonl(path, rows())

    def save_ledger(self, path: str) -> int:
        return write_jsonl(path, (r.to_dict() for r in self.ledger.values()))

    def load_ledger(self, path: str) -> int:
        n = 0
        for rec in read_jsonl(path):
            self.ledger[rec["chunk_id"]] = InjectionRecord(**rec)
            n += 1
        return n

    def write_manifest(self, path: str, corpus_name: str, extra: dict[str, Any] | None = None) -> None:
        payload = {
            "corpus": corpus_name,
            "index_version": self.version,
            "n_chunks": len(self._by_id),
            "n_injected": len(self.ledger),
            "checksum": self.checksum(),
            **(extra or {}),
        }
        ensure_parent(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        log.info("manifest: %s (%s, %d chunks, %d injected)",
                 path, self.version, len(self._by_id), len(self.ledger))

    def checksum(self) -> str:
        h = hashlib.sha1()
        for cid in sorted(self._by_id):
            h.update(cid.encode("utf-8"))
            h.update(content_hash(self._by_id[cid].raw_text).encode("utf-8"))
        return h.hexdigest()[:16]


def load_corpus(cfg, *, with_injections: bool = False) -> SidCorpus:
    """Load v0 and, when asked, replay the injected delta on top of it."""
    corpus = SidCorpus.load(cfg.paths.corpus, version="v0", meta_path=cfg.paths.meta)
    if not with_injections:
        return corpus
    inj_path = cfg.paths.injected_corpus
    if not os.path.exists(inj_path):
        log.warning("no injected corpus at %s — staying on v0", inj_path)
        return corpus
    added = 0
    for rec in read_jsonl(inj_path):
        c = Chunk(file_name=rec["file_name"], index=int(rec["index"]),
                  raw_text=rec.get("raw_text", ""),
                  document_id=rec.get("document_id", ""), title=rec.get("title", ""))
        if corpus._register(c):
            added += 1
    corpus.load_ledger(cfg.paths.injection_ledger)
    corpus.version = "v1"
    log.info("corpus: v1 = %d chunks (+%d injected)", len(corpus), added)
    return corpus
