"""Entity extraction for subgraph mining (plan §3.1).

The plan calls for a NER model validated on 200 chunks. For v1 that dependency
buys little: what the miner actually needs is *rare, repeated surface forms that
bridge chunks*, and a pattern extractor over proper nouns, quoted names, codes,
dates and amounts finds those with no model to serve or validate. The interface
is the one a NER model would fill, so swapping one in later is local to this
file.

Index tags (§3.1 item 2) win over extracted entities on conflict, and are the
anchors for `temporal_resolution` / `constraint_intersection`.

A bare capitalised word is weak evidence on its own: this corpus is heavy on
markdown tables, bullet lists and link text, all of which capitalise the first
word of a cell/item/label for reasons that have nothing to do with being a
name (`| Статус | ... |`, `- Проверь клиента`, `[Постановлением ...]`). Rather
than enumerate every such layout as a regex special case — which only ever
covers the layouts seen so far — `common_lc` gives extraction a corpus-wide
signal: a token that is ever spelled with a lowercase first letter somewhere
in the same corpus is, by Russian orthography, not a proper noun, regardless
of why *this* occurrence happened to be capitalised. `_SENT_START` stays as a
cheap first pass (it also has to work standalone, one chunk at a time, with no
corpus context); `common_lc` is the one that actually holds against layout it
has never seen.

The lowercase attestation has to survive inflection: "Аудиту" (dative) will
never itself appear lowercase if the word only ever shows up lowercase as
"аудит" or "аудита" elsewhere. Comparing exact surface forms would miss that,
so both the vocabulary and the candidate are compared after the same light
suffix-stripping `lexical.stem` already uses for BM25 — one normaliser, not a
second one invented here.

Neither of the above catches a word that is *always* capitalised in this
corpus and never declines into anything attested lowercase — mostly
imperative verbs used as UI action labels ("Скачать", "Настроить"). No amount
of case-based heuristics settles "is this a verb", because that is not what
case encodes. `pymorphy3` is a dictionary + FST morphological analyser (not a
model that needs serving, training or GPU — a bundled OpenCorpora dictionary,
same class of tool as the stemmer above, just POS-aware) and it answers that
question directly, so a bare single word whose most likely analysis is a
verb, adverb, conjunction, preposition, particle, pronoun or numeral is
rejected regardless of what the corpus attests elsewhere. It is optional: if
it is not installed, extraction falls back to the case-based filters only.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from .lexical import stem

try:
    import pymorphy3
    _MORPH: "pymorphy3.MorphAnalyzer | None" = pymorphy3.MorphAnalyzer()
except Exception:                                            # noqa: BLE001
    _MORPH = None

# Parts of speech no proper noun is ever tagged as on its own. Adjectives
# (ADJF/ADJS) are excluded too: a bare adjective is not a useful bridging
# entity by itself, and multi-word names ("Северный поток") never reach this
# check — it only applies to single bare words.
_NON_ENTITY_POS = {
    "VERB", "INFN", "GRND", "PRTF", "PRTS", "ADVB",
    "CONJ", "PREP", "PRCL", "INTJ", "NPRO", "NUMR", "ADJF", "ADJS",
}


@lru_cache(maxsize=200_000)
def _is_non_entity_pos(word_lower: str) -> bool:
    if _MORPH is None:
        return False
    return _MORPH.parse(word_lower)[0].tag.POS in _NON_ENTITY_POS

_QUOTED = re.compile(r"[«\"']([^«»\"']{2,60})[»\"']")
_CODE = re.compile(r"\b[A-ZА-ЯЁ]{2,}[-–—]?\d{1,6}\b")
_DATE_FULL = re.compile(r"\b\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}\b")
_YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\s*(?:год|г\.|году|года)?", re.IGNORECASE)
_AMOUNT = re.compile(r"\b\d[\d\s]{2,}(?:,\d+)?\s*(?:руб|₽|тыс|млн|млрд|%)\w*", re.IGNORECASE)
_PROPER = re.compile(r"\b[А-ЯЁA-Z][а-яёa-z]{2,}(?:[- ][А-ЯЁA-Z][а-яёa-z]{2,})*\b")

# Sentence-initial capitalisation is not evidence of a proper noun — including
# the markdown layouts (`|`, `[`, list bullets) that also reset it without a
# "real" sentence boundary.
_SENT_START = re.compile(
    r"(?:^|[.!?…]\s+|\n\s*|\n[ \t]*[-•*][ \t]+|[|\[]\s*)([А-ЯЁA-Z][а-яёa-z]+)")

_WORD = re.compile(r"[А-ЯЁа-яёA-Za-z]+")

_STOP_SURFACES = {
    "Компания", "Организация", "Предприятие", "Общество", "Также", "Однако",
    "Кроме", "После", "Согласно", "Например", "Данные", "Работа", "Этот",
    "Новый", "Первый", "Второй", "Третий", "Годовой", "Ежегодно", "Через",
}


def corpus_lowercase_vocab(texts: Iterable[str]) -> frozenset[str]:
    """Stems attested with a lowercase first letter anywhere in the corpus.

    A single pass over raw text, independent of the (capitalised) entity
    extraction below — this is what lets `extract_entities` reject an
    ordinary word that only *happens* to sit capitalised in a table cell,
    list item or link, without having to special-case every layout that
    produces that effect. Stemmed so "аудит"/"аудита" seen lowercase also
    disqualifies "Аудиту" seen capitalised — same word, different case
    ending, and inflection is not evidence of a different word.
    """
    out: set[str] = set()
    for text in texts:
        for m in _WORD.finditer(text or ""):
            w = m.group(0)
            if w[0].islower():
                out.add(stem(w.lower()))
    return frozenset(out)


@dataclass(frozen=True)
class EntityMention:
    surface: str
    type: str          # ORG | CODE | DATE | AMOUNT | PROPER


def extract_entities(text: str, common_lc: frozenset[str] | None = None) -> list[EntityMention]:
    """Surface forms worth using as a bridge between chunks.

    ``common_lc``, if given, is the corpus-wide set of stems attested with a
    lowercase first letter (see `corpus_lowercase_vocab`): a bare single-word
    candidate whose stem is also spelled lowercase elsewhere in the corpus is
    an ordinary word wearing incidental capitalisation, not a name.
    """
    out: dict[str, EntityMention] = {}

    def normalize(surface: str) -> str:
        return " ".join(surface.split()).strip(" .,;:()")

    def is_ordinary_bare_word(s: str) -> bool:
        """A single word (quoted or not) that is either attested lowercase
        elsewhere in the corpus, or whose own morphology says it is not a
        noun — a name does not stop being an ordinary word just because
        someone put it in quotes (button labels are quoted too). ``s`` must
        already be whitespace-normalised — `« Выбрать »` keeps its padding
        spaces straight out of `_QUOTED`, which would otherwise look like a
        multi-word phrase and skip this check entirely."""
        if " " in s or "-" in s:
            return False
        if common_lc is not None and stem(s.lower()) in common_lc:
            return True
        return _is_non_entity_pos(s.lower())

    def put(surface: str, etype: str) -> None:
        s = normalize(surface)
        if len(s) < 3 or s in _STOP_SURFACES:
            return
        out.setdefault(s.lower(), EntityMention(s, etype))

    for m in _QUOTED.finditer(text):
        s = normalize(m.group(1))
        if is_ordinary_bare_word(s):
            continue
        put(s, "ORG")
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
        # a single sentence(-or-cell-or-item)-initial word is almost always
        # just capitalisation — a position-based signal, checked separately
        # from is_ordinary_bare_word's corpus/morphology-based ones
        if " " not in s and "-" not in s and s in sentence_initial:
            continue
        if is_ordinary_bare_word(s):
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

    def add_chunk(self, chunk_id: str, text: str, index_tags: dict[str, str] | None = None,
                  common_lc: frozenset[str] | None = None) -> None:
        self.n_chunks += 1
        for ent in extract_entities(text, common_lc=common_lc):
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
