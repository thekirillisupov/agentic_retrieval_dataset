"""Offline mock backend for the SID prompts.

Lets the whole pipeline — mining, composition, all five gates, distractor
cascade, injection, isolation, export — run end to end with no network, no GPU
and no API key. That is what makes the first version verifiable: `--backend mock`
exercises every stage's plumbing, and swapping in a real model changes only the
config.

Dispatch is on the ``[[SID:<tag>]]`` marker each system prompt carries.
"""
from __future__ import annotations

import re
from typing import Any

from ..config import LLMConfig
from ..llm import BaseLLM, MockLLM, make_client

_TAG_RE = re.compile(r"\[\[SID:([a-z_]+)\]\]")
_FACT_LINE = re.compile(r"^- \[(\S+) @ ([^\]]+)\]\s*(.+)$", re.MULTILINE)
_CHUNK_BLOCK = re.compile(r"\[CHUNK ([^\]]+)\]:?\n(.*?)(?=\n\n\[CHUNK |\n\nИзвлеки |\Z)",
                          re.DOTALL)
_YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_CAPS = re.compile(r"[«\"]([^»\"]{2,40})[»\"]|\b([А-ЯЁ][а-яё]{3,})\b")


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[^\W_]{4,}", text.lower())}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 30]


_TERM_STOP = {"Согласно", "Компания", "Через", "Ежегодно", "Устройство", "Изделие",
              "Основными", "Первоначально", "Национальный", "Главным", "Общий",
              "Совместный", "Отчёт", "Финансирование", "Самый", "Испытания"}


def _key_terms(text: str, n: int = 2) -> list[str]:
    """Quoted names first — they are the corpus's real handles; bare capitalised
    words are a weak fallback and never a sentence-opening stopword."""
    quoted, bare = [], []
    for m in _CAPS.finditer(text):
        if m.group(1):
            quoted.append(m.group(1))
        elif m.group(2) and m.group(2) not in _TERM_STOP:
            bare.append(m.group(2))
    out: list[str] = []
    for term in quoted + bare:
        if term not in out:
            out.append(term)
        if len(out) >= n:
            break
    return out


def sid_mock_handler(system: str, user: str) -> dict[str, Any]:
    m = _TAG_RE.search(system)
    tag = m.group(1) if m else ""
    return _HANDLERS.get(tag, _fallback)(user)


# --------------------------------------------------------------------------- #
def _facts(user: str) -> dict[str, Any]:
    blocks = _CHUNK_BLOCK.findall(user)
    text = blocks[0][1] if blocks else user
    facts = []
    for sent in _sentences(text)[:3]:
        facts.append({
            "verbatim_span": sent,                      # exact substring by construction
            "fact_normalized": f"Согласно источнику, {sent[0].lower()}{sent[1:]}",
            "entities": _key_terms(sent, 3),
            "discriminating_attributes": [f"date:{y}" for y in _YEAR.findall(sent)[:1]],
        })
    return {"facts": facts}


def _compose(user: str) -> dict[str, Any]:
    rows = _FACT_LINE.findall(user)
    picked: list[tuple[str, str, str]] = []
    seen_chunks: set[str] = set()
    for fid, cid, text in rows:                          # prefer distinct chunks
        if cid not in seen_chunks:
            picked.append((fid, cid, text))
            seen_chunks.add(cid)
    for fid, cid, text in rows:
        if len(picked) >= 2:
            break
        if fid not in {p[0] for p in picked}:
            picked.append((fid, cid, text))
    picked = picked[:3]
    terms = []
    for _, _, text in picked:
        terms.extend(_key_terms(text, 1))
    subject = terms[0] if terms else "объект"
    other = terms[1] if len(terms) > 1 else "смежный объект"
    m = re.search(r"^Механика поиска: (\w+)", user, re.MULTILINE)
    mechanic = m.group(1) if m else "entity_chain"
    # vary the phrasing by mechanic, otherwise every mock task is one template
    # and MinHash de-duplication (correctly) collapses the whole pool to one
    templates = {
        "entity_chain": "Через какой объект связаны {a} и {b}?",
        "constraint_intersection": "Что относится одновременно к {a} и к {b}?",
        "set_aggregation": "Перечислите объекты, относящиеся к {a} наряду с {b}.",
        "comparison": "Чем {a} отличается от {b} по основному показателю?",
        "temporal_resolution": "Какое значение для {a} актуально позже, чем у {b}?",
        "disambiguation_first": "О каком объекте идёт речь, если он связан с {a}, но не с {b}?",
    }
    q = templates.get(mechanic, templates["entity_chain"]).format(a=subject, b=other)
    out = {
        "question": q,
        "answer": f"{subject}, связанный с {other}.",
        "used_fact_ids": [p[0] for p in picked],
        "reasoning": f"mock composition ({mechanic}) over facts from distinct chunks",
    }
    out.update(_mock_filter(user))
    return out


def _mock_filter(user: str) -> dict[str, Any]:
    """When the prompt carries the S3c filter spec, declare a filter the way a
    cooperative model would: constrain the first available facet, and add one
    more field per repair round (the feedback carries the current filter)."""
    if "СТРУКТУРИРОВАННЫЙ ФИЛЬТР" not in user:
        return {}
    import json as _json
    fields = re.findall(r"^- (\w+) \(([^)]*)\)$", user, re.MULTILINE)
    attrs: dict[str, str] = {}
    m = re.search(r"атрибуты: (.+)$", user, re.MULTILINE)
    if m:
        for part in m.group(1).split(" | "):
            label, _, value = part.partition(": ")
            if value:
                attrs[label.strip()] = value.strip()
    current: list[dict] = []
    m = re.search(r"Текущий фильтр: (\[.*\])", user)
    if m:
        try:
            current = [c for c in _json.loads(m.group(1)) if isinstance(c, dict)]
        except Exception:                                   # noqa: BLE001
            current = []
    used = {c.get("field") for c in current}
    filt = list(current)
    for fld, label in fields:
        if fld in used:
            continue
        value = attrs.get(label.strip()) or attrs.get(fld)
        if value:
            filt.append({"field": fld, "op": "eq", "value": value})
            break
    return {"filter": filt, "answer_field": fields[0][0] if fields else ""}


def _solve(_user: str) -> dict[str, Any]:
    return {"solvable": True, "answer_correct": True, "leaks_answer": False,
            "leaks_intermediate": False, "standalone": True,
            "needs_multiple_chunks": True, "reason": "mock"}


def _solve_devil(_user: str) -> dict[str, Any]:
    return {"solvable": True, "reason": "mock: no honest objection"}


def _entail(user: str) -> dict[str, Any]:
    """A real judge checks entailment; the mock approximates it with lexical
    overlap *plus* agreement on numbers. Without the numeric condition a
    templated corpus makes every sibling document 'state' every fact, and the
    fact groups explode."""
    fact = user.split("\n", 1)[0]
    blocks = _CHUNK_BLOCK.findall(user)
    chunk = blocks[0][1] if blocks else ""
    fw, cw = _words(fact), _words(chunk)
    overlap = len(fw & cw) / max(1, len(fw))
    f_nums = set(re.findall(r"\d+", fact))
    c_nums = set(re.findall(r"\d+", chunk))
    numbers_ok = not f_nums or bool(f_nums & c_nums)
    return {"states_fact": overlap >= 0.75 and numbers_ok,
            "reason": f"mock overlap={overlap:.2f} numbers_ok={numbers_ok}"}


def _perturb(user: str) -> dict[str, Any]:
    src = user.split("ИСХОДНЫЙ ФРАГМЕНТ:\n", 1)[-1]
    years = _YEAR.findall(src)
    if years:
        old = years[0]
        new = str(int(old) + 3)
        return {"text": src.replace(old, new, 1),
                "perturbed_attribute": f"date:{old} -> date:{new}",
                "distractor_type": "near_duplicate"}
    return {"text": re.sub(r"\b\d+\b", lambda mm: str(int(mm.group(0)) + 7), src, count=1),
            "perturbed_attribute": "amount:changed",
            "distractor_type": "near_duplicate"}


def _transplant(user: str) -> dict[str, Any]:
    src = user.split("ИСХОДНЫЙ ФРАГМЕНТ:\n", 1)[-1]
    ents_line = ""
    for line in user.splitlines():
        if line.startswith("СУЩНОСТИ ДЛЯ ПОДСТАНОВКИ:"):
            ents_line = line.split(":", 1)[1]
    ents = [e.strip() for e in ents_line.split(",") if e.strip() and e.strip() != "(нет)"]
    text = src
    for i, term in enumerate(_key_terms(src, len(ents) or 1)):
        if i < len(ents):
            text = text.replace(term, ents[i])
    return {"text": text, "distractor_type": "topical_lure"}


def _generate(user: str) -> dict[str, Any]:
    tpl = user.split("ОБРАЗЕЦ СТИЛЯ:\n", 1)[-1]
    return {"text": "В отраслевом обзоре отмечается следующее. " + tpl,
            "distractor_type": "topical_lure"}


def _distractor_check(user: str) -> dict[str, Any]:
    answer = ""
    for line in user.splitlines():
        if line.startswith("ОТВЕТ:"):
            answer = line.split(":", 1)[1].strip()
    body = user.split("КАНДИДАТ:\n", 1)[-1]
    contains = bool(answer) and answer.lower()[:40] in body.lower()
    return {"contains_answer": contains, "valid_alternative_path": False,
            "reason": "mock"}


def _isolation(_user: str) -> dict[str, Any]:
    return {"alternative_path_chunk_ids": [], "reason": "mock"}


def _subject_match(_user: str) -> dict[str, Any]:
    return {"matches": True, "reason": "mock"}


def _fallback(_user: str) -> dict[str, Any]:
    return {"ok": True}


_HANDLERS = {
    "facts": _facts,
    "compose": _compose,
    "solve": _solve,
    "solve_devil": _solve_devil,
    "entail": _entail,
    "perturb": _perturb,
    "transplant": _transplant,
    "generate_distractor": _generate,
    "distractor_check": _distractor_check,
    "isolation": _isolation,
    "subject_match": _subject_match,
}


def make_sid_client(cfg: LLMConfig) -> BaseLLM:
    """Like ``arqg.llm.make_client`` but with the SID-aware mock."""
    if cfg.backend.lower() == "mock":
        return MockLLM(cfg, handler=sid_mock_handler)
    return make_client(cfg)
