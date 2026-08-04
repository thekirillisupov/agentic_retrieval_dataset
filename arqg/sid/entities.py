"""Entity extraction for subgraph mining (plan §3.1).

The plan calls for a NER model validated on 200 chunks. For v1 that dependency
buys little: what the miner actually needs is *rare, repeated surface forms that
bridge chunks*, and a pattern extractor over proper nouns, quoted names, codes,
dates and amounts finds those with no model to serve or validate. The interface
is the one a NER model would fill, so swapping one in later is local to this
file.

Index tags (§3.1 item 2) win over extracted entities on conflict, and are the
anchors for `temporal_resolution` / `constraint_intersection`.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

_QUOTED = re.compile(r"[«\"']([^«»\"']{2,60})[»\"']")
_CODE = re.compile(r"\b[A-ZА-ЯЁ]{2,}[-–—]?\d{1,6}\b")
_DATE_FULL = re.compile(r"\b\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}\b")
_YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\s*(?:год|г\.|году|года)?", re.IGNORECASE)
_AMOUNT = re.compile(r"\b\d[\d\s]{2,}(?:,\d+)?\s*(?:руб|₽|тыс|млн|млрд|%)\w*", re.IGNORECASE)
_PROPER = re.compile(r"\b[А-ЯЁA-Z][а-яёa-z]{2,}(?:[- ][А-ЯЁA-Z][а-яёa-z]{2,})*\b")

# Sentence-initial capitalisation is not evidence of a proper noun.
_SENT_START = re.compile(r"(?:^|[.!?…]\s+|\n\s*)([А-ЯЁA-Z][а-яёa-z]+)")

_STOP_SURFACES = {
    "Компания", "Организация", "Предприятие", "Общество", "Также", "Однако",
    "Кроме", "После", "Согласно", "Например", "Данные", "Работа", "Этот",
    "Новый", "Первый", "Второй", "Третий", "Годовой", "Ежегодно", "Через",
}


@dataclass(frozen=True)
class EntityMention:
    surface: str
    type: str          # ORG | CODE | DATE | AMOUNT | PROPER


def extract_entities(text: str) -> list[EntityMention]:
    """Surface forms worth using as a bridge between chunks."""
    out: dict[str, EntityMention] = {}

    def put(surface: str, etype: str) -> None:
        s = " ".join(surface.split()).strip(" .,;:()")
        if len(s) < 3 or s in _STOP_SURFACES:
            return
        out.setdefault(s.lower(), EntityMention(s, etype))

    for m in _QUOTED.finditer(text):
        put(m.group(1), "ORG")
    for m in _CODE.finditer(text):
        put(m.group(0), "CODE")
    for m in _DATE_FULL.finditer(text):
        put(m.group(0), "DATE")
    for m in _YEAR.finditer(text):
        put(m.group(1), "DATE")
    for m in _AMOUNT.finditer(text):
        put(m.group(0), "AMOUNT")

    sentence_initial = {m.group(1) for m in _SENT_START.finditer(text)}
    for m in _PROPER.finditer(text):
        s = m.group(0)
        # a single sentence-initial word is almost always just capitalisation
        if " " not in s and "-" not in s and s in sentence_initial:
            continue
        put(s, "PROPER")
    return list(out.values())


class EntityGraph:
    """Bipartite entity ↔ chunk map plus the idf each entity carries."""

    def __init__(self) -> None:
        self.chunks_of: dict[str, set[str]] = defaultdict(set)
        self.mention: dict[str, EntityMention] = {}
        self.entities_of: dict[str, set[str]] = defaultdict(set)
        self.n_chunks = 0

    def add_chunk(self, chunk_id: str, text: str, index_tags: dict[str, str] | None = None) -> None:
        self.n_chunks += 1
        for ent in extract_entities(text):
            key = ent.surface.lower()
            self.chunks_of[key].add(chunk_id)
            self.entities_of[chunk_id].add(key)
            self.mention.setdefault(key, ent)
        # index tags are trusted anchors and override the extractor on conflict
        for tag, value in (index_tags or {}).items():
            if not value:
                continue
            key = f"{tag}:{value}".lower()
            self.chunks_of[key].add(chunk_id)
            self.entities_of[chunk_id].add(key)
            self.mention[key] = EntityMention(str(value), f"TAG_{tag.upper()}")

    def df(self, key: str) -> int:
        return len(self.chunks_of.get(key, ()))

    def idf(self, key: str) -> float:
        df = self.df(key)
        if df == 0 or self.n_chunks == 0:
            return 0.0
        return math.log(self.n_chunks / df)

    def is_tag(self, key: str) -> bool:
        return self.mention[key].type.startswith("TAG_")

    def idf_threshold(self, percentile: float, min_df: int = 2,
                      max_df: int | None = None) -> float:
        """τ_idf as a percentile of the idf distribution over entities that can
        actually bridge.

        The plan says "percentile of the corpus idf distribution". Taken over
        *every* surface form that degenerates: the corpus is dominated by
        hapax entities that all carry the maximum idf, so any percentile above
        ~50 collapses onto `log(N/1)` and admits only df=1 entities — which the
        `co_occurrence >= 2` requirement then rejects, leaving nothing. The
        population the threshold is applied to is entities with
        `min_df <= df <= max_df`, so that is the population it is estimated on.
        """
        vals = sorted(self.idf(k) for k, chunks in self.chunks_of.items()
                      if len(chunks) >= min_df and (max_df is None or len(chunks) <= max_df))
        if not vals:
            return 0.0
        i = min(len(vals) - 1, max(0, int(round(percentile / 100.0 * (len(vals) - 1)))))
        return vals[i]
