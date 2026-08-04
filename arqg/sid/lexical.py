"""In-memory BM25 over the corpus.

Needed for three things the plan leans on and the existing pipeline does not
provide: the lexical branch of the hybrid retriever, ``lex_gap`` (§4.3), and the
entity IDF used to decide whether an entity is niche enough to bridge a subgraph
(§3.2). Small enough to stay dependency-free; a 60k-chunk corpus indexes in a
couple of seconds and scores a query in milliseconds.

Documents can be *appended* after construction, because the index is mutated by
distractor injection (v0 → vN).
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

# Light Russian suffix stripping. Not a real stemmer — just enough that
# "компания"/"компании"/"компанию" collide, which is what BM25 needs on
# inflected text. Longest suffix wins, so the list is length-sorted: matching
# "ю" before "ию" would leave "компани" next to "компан".
# Inflectional endings only. Derivational ones ("-ание", "-ость", "-ация")
# are deliberately absent: stripping them merges genuinely different lemmas.
_SUFFIXES = tuple(sorted({
    "иями", "ями", "ами", "ому", "ему", "ого", "его", "ыми", "ими",
    "ах", "ях", "ам", "ям", "ов", "ев", "ий", "ый", "ой", "ая", "яя",
    "ые", "ие", "ем", "ём", "ом", "ую", "юю", "ей", "ии", "ию", "ия",
    "а", "я", "о", "е", "у", "ю", "ы", "и", "й", "ь",
}, key=len, reverse=True))


def stem(token: str) -> str:
    if len(token) <= 4 or token.isdigit():
        return token
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 4:
            return token[: -len(suf)]
    return token


def tokenize(text: str, do_stem: bool = True) -> list[str]:
    toks = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    if not do_stem:
        return toks
    return [stem(t) for t in toks]


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_ids: list[str] = []
        self._pos: dict[str, int] = {}
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._len: list[int] = []
        self._total_len = 0

    # ---- build ----------------------------------------------------------- #
    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._pos:
            return
        i = len(self.doc_ids)
        self.doc_ids.append(doc_id)
        self._pos[doc_id] = i
        tf: dict[str, int] = defaultdict(int)
        for tok in tokenize(text):
            tf[tok] += 1
        for tok, n in tf.items():
            self._postings[tok].append((i, n))
        length = sum(tf.values())
        self._len.append(length)
        self._total_len += length

    def add_many(self, docs: list[tuple[str, str]]) -> None:
        for did, text in docs:
            self.add(did, text)

    # ---- stats ----------------------------------------------------------- #
    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    @property
    def avgdl(self) -> float:
        return (self._total_len / self.n_docs) if self.n_docs else 0.0

    def df(self, token: str) -> int:
        return len(self._postings.get(token, ()))

    def idf(self, token: str) -> float:
        """Robertson/Sparck-Jones idf, floored at 0 for ubiquitous terms."""
        n, df = self.n_docs, self.df(token)
        if n == 0:
            return 0.0
        return max(0.0, math.log((n - df + 0.5) / (df + 0.5) + 1.0))

    # ---- scoring --------------------------------------------------------- #
    def scores(self, query: str) -> dict[int, float]:
        """Sparse BM25 scores keyed by internal doc position."""
        out: dict[int, float] = defaultdict(float)
        avgdl = self.avgdl or 1.0
        qtf: dict[str, int] = defaultdict(int)
        for tok in tokenize(query):
            qtf[tok] += 1
        for tok in qtf:
            postings = self._postings.get(tok)
            if not postings:
                continue
            idf = self.idf(tok)
            if idf <= 0:
                continue
            for i, tf in postings:
                dl = self._len[i] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
                out[i] += idf * (tf * (self.k1 + 1)) / denom
        return out

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        scored = self.scores(query)
        ranked = sorted(scored.items(), key=lambda kv: -kv[1])[:top_k]
        return [(self.doc_ids[i], s) for i, s in ranked]

    def search_with_targets(self, query: str, top_k: int,
                            targets: list[str]) -> tuple[list[tuple[str, float]], dict[str, float]]:
        """One pass returning the ranking *and* each target's raw score, so
        ``lex_gap`` costs nothing extra beyond the probe itself (plan §4.3)."""
        scored = self.scores(query)
        ranked = sorted(scored.items(), key=lambda kv: -kv[1])[:top_k]
        hits = [(self.doc_ids[i], s) for i, s in ranked]
        tscores = {t: scored.get(self._pos[t], 0.0) for t in targets if t in self._pos}
        return hits, tscores

    def contains(self, doc_id: str) -> bool:
        return doc_id in self._pos
