"""S3c — set-completeness for aggregation-type questions.

The defect this stage closes: a ``set_aggregation`` («перечислите все…»,
«в каких регионах…») or ``constraint_intersection`` question is composed from a
2-chunk subgraph, and the composer describes *its subgraph* as if it were the
whole corpus. Dozens of other documents satisfy the same constraints, so the
"gold" answer is not exhaustive — and no existing gate can notice, because
G_MIN checks minimality and G_REACH reachability; nothing checks that the gold
set equals the set the question actually denotes.

The fix is arithmetic, not judgement. For the mechanics in
``completeness.mechanics`` the composer must also emit a **structured filter**
— the question's constraints over the corpus's facet metadata — and this module
evaluates that filter over every document of the corpus:

* ``truth == gold``           → the task is exact; accept.
* a gold document *violates* the filter (the «фон отрицания» case: a chunk the
  question explicitly excludes was left in the gold set) → it is dropped from
  the gold when enough facts remain, otherwise the task is rejected. A gold
  chunk that breaks its own question's constraint never ships.
* ``truth ⊃ gold``            → the candidate enters a bounded **repair loop**:
  the composer is shown the excess documents with their facets and asked to add
  a natural distinguishing constraint (month, region, price threshold,
  customer…). Before the first LLM call a pure-metadata *repairability check*
  runs: an excess document that matches the gold on every available facet can
  never be excluded by rephrasing, so the task is rejected at zero cost.
  Between iterations the excess must strictly shrink (``no_progress`` exit),
  the iteration count is capped, and so is the number of constraints added on
  top of the original filter — a question with four bolted-on filters reads as
  a structured query, not as a user.
* the narrow safe case — small truth set, an answer that is mechanically a
  projection of one facet field, and a judge confirming each excess document's
  subject matches the question — goes to the **augmentation branch** instead:
  the excess documents join the gold (facts extracted, answer rebuilt from
  metadata), and the augmented candidate then passes through G_REACH / G_SOLVE
  / G_MIN / G_REP like any other, so nothing enters the pool unverified.

Every decision (repaired / augmented / unrepairable / no_progress / limit /
constraint_cap / no_filter / gold_violation) is logged with the iterations
spent, so one run shows whether the limits are set right.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..llm import BaseLLM
from ..schema import Chunk
from ..utils import append_jsonl, log
from .config import SidConfig
from .corpus import SidCorpus
from .prompts import FACTS_SYS, SUBJECT_MATCH_SYS, facts_user, subject_match_user
from .schema import Candidate, Fact, sid_hash
from .scoping import facet_header, field_value

CAT_OPS = ("eq", "neq", "in", "not_in", "prefix")
NUM_OPS = ("eq", "neq", "lt", "lte", "gt", "gte", "between", "in", "not_in")
DATE_OPS = ("eq", "neq", "lt", "lte", "gt", "gte", "between", "prefix",
            "month_in", "month_not_in")

_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def completeness_active(cfg: SidConfig) -> bool:
    cc = cfg.completeness
    return bool(cc.enabled and cc.filter_fields and cc.mechanics)


def in_scope(cfg: SidConfig, mechanic: str) -> bool:
    return completeness_active(cfg) and mechanic in cfg.completeness.mechanics


def _num(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    m = _NUM_RE.search(str(raw).replace(" ", "").replace(" ", ""))
    return float(m.group(0).replace(",", ".")) if m else None


def _cat(raw: Any) -> str:
    s = " ".join(str(raw or "").split()).lower().strip()
    return s.rstrip("…").strip("«»\"'")


def _cat_eq(a: str, b: str) -> bool:
    """Facet headers truncate long values («…»), so equality on long strings
    tolerates one side being a prefix of the other."""
    if a == b:
        return True
    return min(len(a), len(b)) >= 16 and (a.startswith(b) or b.startswith(a))


def _month(raw: Any) -> int | None:
    s = str(raw or "")
    if len(s) >= 7 and s[4] == "-" and s[5:7].isdigit():
        return int(s[5:7])
    return None


@dataclass
class DocView:
    """One corpus document as the checker sees it: its key, its chunks, and
    the union of its chunks' facet metadata."""
    key: str
    chunk_ids: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckResult:
    error: str = ""                       # non-empty = the filter is unusable
    constraints: list[dict] = field(default_factory=list)
    gold_docs: list[DocView] = field(default_factory=list)
    gold_violations: list[str] = field(default_factory=list)   # doc keys
    truth_keys: list[str] = field(default_factory=list)
    excess: list[DocView] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.error and not self.gold_violations and not self.excess


def doc_key_of(cfg: SidConfig, chunk: Chunk) -> str:
    return (field_value(chunk, cfg.completeness.doc_field).strip()
            or chunk.document_id or chunk.file_name)


class CompletenessChecker:
    """Evaluates declared filters against the whole v0 corpus, per document."""

    def __init__(self, cfg: SidConfig, corpus: SidCorpus):
        self.cfg = cfg
        self.cc = cfg.completeness
        self.corpus = corpus
        self.docs: dict[str, DocView] = {}
        self._doc_of_chunk: dict[str, str] = {}
        for cid in corpus.v0_ids():
            c = corpus.get(cid)
            if c is None:
                continue
            key = doc_key_of(cfg, c)
            if not key:
                continue
            doc = self.docs.setdefault(key, DocView(key=key))
            doc.chunk_ids.append(cid)
            for k, v in (c.meta or {}).items():
                if v not in (None, "") and not doc.meta.get(k):
                    doc.meta[k] = v
            self._doc_of_chunk[cid] = key

    # ---- value access ----------------------------------------------------- #
    def raw(self, doc: DocView, fld: str) -> Any:
        v = doc.meta.get(fld)
        if v not in (None, ""):
            return v
        c = self.corpus.get(doc.chunk_ids[0]) if doc.chunk_ids else None
        return field_value(c, fld) if c is not None else ""

    def _kind(self, fld: str) -> str:
        if fld in self.cc.numeric_fields:
            return "num"
        if fld in self.cc.date_fields:
            return "date"
        return "cat"

    # ---- filter validation ------------------------------------------------ #
    def validate(self, filt: list[dict] | None) -> tuple[list[dict], str]:
        if not filt:
            return [], "empty filter"
        allowed = set(self.cc.filter_fields)
        out: list[dict] = []
        for c in filt:
            if not isinstance(c, dict):
                return [], "constraint is not an object"
            fld = str(c.get("field", "")).strip()
            op = str(c.get("op", "")).strip()
            value = c.get("value")
            if fld not in allowed:
                return [], f"unknown field {fld!r}"
            kind = self._kind(fld)
            ops = {"num": NUM_OPS, "date": DATE_OPS, "cat": CAT_OPS}[kind]
            if op not in ops:
                return [], f"op {op!r} not valid for {fld!r}"
            if op in ("in", "not_in", "month_in", "month_not_in", "between"):
                if not isinstance(value, (list, tuple)) or not value:
                    return [], f"op {op!r} needs a non-empty list value"
                if op == "between" and len(value) != 2:
                    return [], "between needs [lo, hi]"
            elif value in (None, ""):
                return [], f"constraint on {fld!r} has no value"
            if kind == "num" and op not in ("in", "not_in"):
                vals = value if isinstance(value, (list, tuple)) else [value]
                if any(_num(v) is None for v in vals):
                    return [], f"non-numeric value for {fld!r}"
            out.append({"field": fld, "op": op, "value": value})
        return out, ""

    # ---- matching --------------------------------------------------------- #
    def match(self, doc: DocView, c: dict) -> bool:
        fld, op, value = c["field"], c["op"], c["value"]
        raw = self.raw(doc, fld)
        kind = self._kind(fld)
        if kind == "num":
            v = _num(raw)
            if op in ("in", "not_in"):
                hits = any(v is not None and _num(x) == v for x in value)
                return hits if op == "in" else (v is not None and not hits)
            if v is None:
                return False
            if op == "between":
                lo, hi = _num(value[0]), _num(value[1])
                return lo is not None and hi is not None and lo <= v <= hi
            w = _num(value)
            return {"eq": v == w, "neq": v != w, "lt": v < w, "lte": v <= w,
                    "gt": v > w, "gte": v >= w}[op]
        if kind == "date":
            s = str(raw or "")
            if op in ("month_in", "month_not_in"):
                m = _month(s)
                if m is None:
                    return False
                months = {int(_num(x)) for x in value if _num(x) is not None}
                return (m in months) if op == "month_in" else (m not in months)
            if not s:
                return False
            if op == "prefix":
                return s.startswith(str(value))
            if op == "between":
                return str(value[0]) <= s[:10] <= str(value[1])
            w = str(value)
            return {"eq": s[:10] == w, "neq": s[:10] != w, "lt": s[:10] < w,
                    "lte": s[:10] <= w, "gt": s[:10] > w, "gte": s[:10] >= w}[op]
        # categorical
        v = _cat(raw)
        if op == "in":
            return any(v and _cat_eq(v, _cat(x)) for x in value)
        if op == "not_in":
            return bool(v) and not any(_cat_eq(v, _cat(x)) for x in value)
        w = _cat(value)
        if op == "eq":
            return bool(v) and _cat_eq(v, w)
        if op == "neq":
            return bool(v) and not _cat_eq(v, w)
        if op == "prefix":
            return bool(v) and v.startswith(w)
        return False

    def matches_all(self, doc: DocView, constraints: list[dict]) -> bool:
        return all(self.match(doc, c) for c in constraints)

    # ---- the check -------------------------------------------------------- #
    def doc_of(self, chunk_id: str) -> str:
        key = self._doc_of_chunk.get(chunk_id)
        if key:
            return key
        c = self.corpus.get(chunk_id)
        return doc_key_of(self.cfg, c) if c is not None else chunk_id

    def check(self, cand: Candidate) -> CheckResult:
        constraints, err = self.validate(cand.filter)
        if err:
            return CheckResult(error=err)
        gold_keys = list(dict.fromkeys(self.doc_of(c) for c in cand.chunk_ids))
        gold_docs = [self.docs.get(k, DocView(key=k)) for k in gold_keys]
        violations = [d.key for d in gold_docs if not self.matches_all(d, constraints)]
        truth = [d for d in self.docs.values() if self.matches_all(d, constraints)]
        gold_set = set(gold_keys)
        excess = sorted((d for d in truth if d.key not in gold_set),
                        key=lambda d: d.key)
        return CheckResult(constraints=constraints, gold_docs=gold_docs,
                           gold_violations=violations,
                           truth_keys=sorted(d.key for d in truth), excess=excess)

    # ---- repairability ---------------------------------------------------- #
    def separable(self, doc: DocView, gold_docs: list[DocView]) -> bool:
        """Can *some* facet constraint exclude ``doc`` while keeping every gold
        document? Pure metadata arithmetic — no LLM call. A document that
        coincides with the gold on every available facet is unrepairable by
        rephrasing, whatever the composer tries."""
        for fld in self.cc.filter_fields:
            gvals = [self.raw(g, fld) for g in gold_docs]
            if any(v in (None, "") for v in gvals):
                continue          # a constraint here would break the gold too
            dval = self.raw(doc, fld)
            kind = self._kind(fld)
            if kind == "num":
                gn = [_num(v) for v in gvals]
                if any(v is None for v in gn):
                    continue
                dn = _num(dval)
                if dn is None or dn < min(gn) or dn > max(gn):
                    return True
            elif kind == "date":
                gs = [str(v)[:10] for v in gvals]
                ds = str(dval or "")[:10]
                if not ds or ds < min(gs) or ds > max(gs):
                    return True
                gm = {_month(v) for v in gvals}
                if None not in gm and _month(dval) not in gm:
                    return True
            else:
                gset = {_cat(v) for v in gvals}
                dc = _cat(dval)
                if not dc or not any(_cat_eq(dc, g) for g in gset):
                    return True
        return False

    # ---- prompt material -------------------------------------------------- #
    def filter_spec(self) -> str:
        labels = self.cfg.facets.labels
        lines = "\n".join(f"- {f} ({labels.get(f, f)})"
                          for f in self.cc.filter_fields)
        num = ", ".join(f for f in self.cc.filter_fields
                        if f in self.cc.numeric_fields) or "—"
        dates = ", ".join(f for f in self.cc.filter_fields
                          if f in self.cc.date_fields) or "—"
        return f"""СТРУКТУРИРОВАННЫЙ ФИЛЬТР (обязателен для этой механики):
дополнительно верни в JSON поле "filter" — список условий по карточным полям
документов, ТОЧНО повторяющий ограничения из текста вопроса (не больше и не
меньше: каждое условие фильтра названо в вопросе, каждое ограничение вопроса
есть в фильтре). Формат условия: {{"field": "...", "op": "...", "value": ...}}.
Доступные поля:
{lines}
Операторы для текстовых полей: eq, neq, in, not_in, prefix (prefix — для кодов,
например ОКПД2 "80.1"). Для числовых полей ({num}): eq, lt, lte, gt, gte,
between [lo, hi]. Для дат ({dates}): те же сравнения по ISO-дате, prefix
("2019"), month_in / month_not_in (список номеров месяцев).
Если ответ на вопрос — перечисление значений одного поля (например, регионов
или номеров закупок), верни также "answer_field": "<имя поля>"."""

    def describe_doc(self, doc: DocView, max_fields: int = 8) -> str:
        labels = self.cfg.facets.labels
        parts = []
        for fld in self.cc.filter_fields[:max_fields]:
            v = self.raw(doc, fld)
            if v not in (None, ""):
                parts.append(f"{labels.get(fld, fld)}: {v}")
        return f"№ {doc.key}: " + "; ".join(parts)

    def repair_feedback(self, res: CheckResult, max_docs: int = 8) -> str:
        listed = "\n".join(f"- {self.describe_doc(d)}"
                           for d in res.excess[:max_docs])
        more = (f"\n… и ещё {len(res.excess) - max_docs} документов."
                if len(res.excess) > max_docs else "")
        return (f"вопрос покрывает больше документов корпуса, чем размечено: под "
                f"его ограничения подпадают {len(res.truth_keys)} документов, "
                f"лишние ({len(res.excess)}):\n{listed}{more}\n"
                f"Текущий фильтр: {json.dumps(res.constraints, ensure_ascii=False)}\n"
                f"Добавь в ТЕКСТ вопроса одно естественное различающее ограничение "
                f"(порог цены, месяц, регион, заказчик и т.п.), которое исключает "
                f"лишние документы, но сохраняет целевые, и обнови \"filter\" "
                f"соответственно. Целевые документы и ответ не меняй.")


# --------------------------------------------------------------------------- #
# the repair / augment driver
# --------------------------------------------------------------------------- #
def _finish(cfg: SidConfig, cand: Candidate | None, base: Candidate, *,
            status: str, exit_reason: str, iters: int, res: CheckResult | None,
            dropped_gold: list[str], added_docs: list[str],
            base_len: int | None) -> Candidate | None:
    final = cand if cand is not None else base
    n_constraints = len(res.constraints) if res is not None else len(final.filter or [])
    prov = {
        "status": status,
        "exit": exit_reason,
        "iters": iters,
        "n_truth": len(res.truth_keys) if res is not None else None,
        "n_excess": len(res.excess) if res is not None else None,
        "added_constraints": max(0, n_constraints - base_len) if base_len is not None else 0,
        "dropped_gold_docs": dropped_gold,
        "added_docs": added_docs,
        "answer_rebuilt": bool(added_docs),
        "filter": final.filter,
        "answer_field": final.answer_field,
    }
    final.completeness = prov
    append_jsonl(cfg.paths.completeness_log, {
        "candidate_id": base.candidate_id, "mechanic": base.mechanic,
        "outcome": "accepted" if cand is not None else "rejected", **prov})
    if cand is None:
        log.info("S3c: %s rejected (%s) after %d repair iteration(s)",
                 base.candidate_id, exit_reason, iters)
    return cand


def _drop_violating_gold(cfg: SidConfig, checker: CompletenessChecker,
                         cand: Candidate, violations: list[str]) -> Candidate | None:
    """The hard rule: a gold chunk that breaks its own question's constraint is
    dropped from the gold; if too little remains, the task fails."""
    bad = set(violations)
    kept = [f for f in cand.facts if checker.doc_of(f["chunk_id"]) not in bad]
    chunks = {f["chunk_id"] for f in kept}
    if len(kept) < cfg.compose.min_facts_per_question or len(chunks) < 2:
        return None
    cand.facts = kept
    cand.hop_depth = len(kept)
    return cand


async def _recompose(cfg: SidConfig, gen: BaseLLM, cand: Candidate, cell,
                     facts_pool: list[dict], feedback: str, spec: str,
                     iters: int) -> Candidate | None:
    from .compose import compose_one
    stub = {"subgraph_id": cand.subgraph_id, "chunks": cand.chunk_ids}
    fixed = await compose_one(cfg, gen, cand.batch_id, cand.instantiation_rank,
                              cell, stub, facts_pool, feedback=feedback,
                              iters=iters, filter_spec=spec)
    if fixed is not None:
        fixed.candidate_id = cand.candidate_id
    return fixed


async def ensure_complete(cfg: SidConfig, checker: CompletenessChecker,
                          gen: BaseLLM, judge: BaseLLM | None, cand: Candidate,
                          cell, facts_pool: list[dict],
                          facts_by_chunk: dict[str, list[dict]] | None = None
                          ) -> Candidate | None:
    """Drive one candidate to a state where its gold set IS the truth set of
    its own declared filter — by repair, by augmentation, or not at all."""
    if not in_scope(cfg, cand.mechanic):
        return cand
    cc = cfg.completeness
    spec = checker.filter_spec()
    iters = 0
    dropped_gold: list[str] = []
    base_len: int | None = None
    cur = cand

    while True:
        res = checker.check(cur)
        if res.error:
            if not cc.require_filter:
                return _finish(cfg, cur, cand, status="unverified",
                               exit_reason="no_filter", iters=iters, res=None,
                               dropped_gold=dropped_gold, added_docs=[],
                               base_len=base_len)
            if iters >= cc.max_repair_iters:
                return _finish(cfg, None, cand, status="rejected",
                               exit_reason="no_filter", iters=iters, res=None,
                               dropped_gold=dropped_gold, added_docs=[],
                               base_len=base_len)
            fb = (f"структурированный фильтр отсутствует или невалиден "
                  f"({res.error}). Верни поле \"filter\" по инструкции.")
            nxt = await _recompose(cfg, gen, cur, cell, facts_pool, fb, spec,
                                   cur.compose_iters + 1)
            iters += 1
            if nxt is None:
                return _finish(cfg, None, cand, status="rejected",
                               exit_reason="no_filter", iters=iters, res=None,
                               dropped_gold=dropped_gold, added_docs=[],
                               base_len=base_len)
            cur = nxt
            continue

        if base_len is None:
            base_len = len(res.constraints)

        if res.gold_violations:
            trimmed = _drop_violating_gold(cfg, checker, cur, res.gold_violations)
            if trimmed is None:
                return _finish(cfg, None, cand, status="rejected",
                               exit_reason="gold_violation", iters=iters,
                               res=res, dropped_gold=dropped_gold,
                               added_docs=[], base_len=base_len)
            dropped_gold += res.gold_violations
            cur = trimmed
            continue

        if not res.excess:
            status = "repaired" if iters or dropped_gold else "exact"
            return _finish(cfg, cur, cand, status=status, exit_reason="repaired",
                           iters=iters, res=res, dropped_gold=dropped_gold,
                           added_docs=[], base_len=base_len)

        # ---- excess exists: decide between repair, augment and reject ----- #
        async def bail(reason: str) -> Candidate | None:
            aug = await _try_augment(cfg, checker, gen, judge, cur, res,
                                     facts_by_chunk or {})
            if aug is not None:
                return _finish(cfg, aug, cand, status="augmented",
                               exit_reason=reason, iters=iters, res=res,
                               dropped_gold=dropped_gold,
                               added_docs=[d.key for d in res.excess],
                               base_len=base_len)
            return _finish(cfg, None, cand, status="rejected",
                           exit_reason=reason, iters=iters, res=res,
                           dropped_gold=dropped_gold, added_docs=[],
                           base_len=base_len)

        if any(not checker.separable(d, res.gold_docs) for d in res.excess):
            return await bail("unrepairable")
        if iters >= cc.max_repair_iters:
            return await bail("limit")
        if len(res.constraints) - base_len >= cc.max_added_constraints:
            return await bail("constraint_cap")

        fb = checker.repair_feedback(res)
        nxt = await _recompose(cfg, gen, cur, cell, facts_pool, fb, spec,
                               cur.compose_iters + 1)
        iters += 1
        if nxt is None:
            return await bail("no_progress")
        nres = checker.check(nxt)
        if nres.error or len(nres.excess) >= len(res.excess):
            return await bail("no_progress")
        if len(nres.constraints) > base_len + cc.max_added_constraints:
            return await bail("constraint_cap")
        cur = nxt


# --------------------------------------------------------------------------- #
# augmentation — the narrow safe branch
# --------------------------------------------------------------------------- #
def _pick_chunk(cfg: SidConfig, checker: CompletenessChecker,
                doc: DocView) -> Chunk | None:
    hint = cfg.completeness.augment_section_hint.lower()
    chunks = [checker.corpus.get(c) for c in doc.chunk_ids]
    chunks = [c for c in chunks if c is not None and c.raw_text]
    if not chunks:
        return None
    if hint:
        for c in chunks:
            if hint in field_value(c, "section").lower():
                return c
    return chunks[0]


def _facets_of(cfg: SidConfig, chunk: Chunk) -> str:
    f = cfg.facets
    if not (f.fields and f.in_prompts):
        return ""
    return facet_header(chunk, f.fields, f.labels, f.max_value_chars)


async def _fact_for_chunk(cfg: SidConfig, gen: BaseLLM, chunk: Chunk,
                          facts_by_chunk: dict[str, list[dict]]) -> dict | None:
    """One fact grounding the added document — from the cache when the chunk
    was already mined, extracted otherwise (and appended to the cache)."""
    from .facts import span_is_verbatim
    cached = facts_by_chunk.get(chunk.id) or []
    if cached:
        return dict(cached[0])
    try:
        obj = await gen.complete_json(
            FACTS_SYS, facts_user(chunk.id, chunk.raw_text,
                                  cfg.compose.max_facts_per_chunk,
                                  chunk.title, _facets_of(cfg, chunk)))
    except Exception as e:                                   # noqa: BLE001
        log.warning("S3c: fact extraction failed for %s: %s", chunk.id, e)
        return None
    first: dict | None = None
    for i, raw in enumerate(obj.get("facts", [])[: cfg.compose.max_facts_per_chunk]):
        span = str(raw.get("verbatim_span", ""))
        if cfg.compose.verbatim_match and not span_is_verbatim(span, chunk.raw_text):
            continue
        fact = Fact(
            fact_id=f"f_{sid_hash(chunk.id, i, span)}",
            chunk_id=chunk.id,
            verbatim_span=span,
            fact_normalized=str(raw.get("fact_normalized", "")).strip(),
            entities=[str(e) for e in raw.get("entities", [])][:8],
            discriminating_attributes=[
                str(a) for a in raw.get("discriminating_attributes", [])][:8],
            section=chunk.title,
            facets=_facets_of(cfg, chunk),
        ).to_dict()
        if not fact["fact_normalized"]:
            continue
        append_jsonl(cfg.paths.facts, fact)      # cache for later stages/runs
        if first is None:
            first = fact
    return first


def _rebuild_answer(checker: CompletenessChecker, truth_keys: list[str],
                    answer_field: str) -> str:
    seen: dict[str, str] = {}
    for key in truth_keys:
        doc = checker.docs.get(key)
        if doc is None:
            continue
        raw = key if answer_field == checker.cc.doc_field \
            else str(checker.raw(doc, answer_field) or "")
        if raw and _cat(raw) not in seen:
            seen[_cat(raw)] = raw
    return ", ".join(sorted(seen.values()))


async def _try_augment(cfg: SidConfig, checker: CompletenessChecker,
                       gen: BaseLLM, judge: BaseLLM | None, cand: Candidate,
                       res: CheckResult,
                       facts_by_chunk: dict[str, list[dict]]) -> Candidate | None:
    """All conditions at once, or no augmentation at all (plan: the excess is
    small, the answer is a mechanical projection of metadata, every added
    document passes a subject check, and the result re-enters the gates)."""
    cc = cfg.completeness
    allowed_answer = set(cc.filter_fields) | {cc.doc_field}
    if not (cc.augment and judge is not None and res.excess
            and len(res.truth_keys) <= cc.augment_max_docs
            and cand.answer_field in allowed_answer):
        return None

    new_facts: list[dict] = []
    for doc in res.excess:
        chunk = _pick_chunk(cfg, checker, doc)
        if chunk is None:
            return None
        try:
            v = await judge.complete_json(
                SUBJECT_MATCH_SYS,
                subject_match_user(cand.question, chunk.id, chunk.raw_text,
                                   _facets_of(cfg, chunk)))
        except Exception as e:                               # noqa: BLE001
            log.warning("S3c: subject check failed for %s: %s", chunk.id, e)
            return None
        if not v.get("matches"):
            log.info("S3c: %s — excess doc %s failed the subject check, "
                     "augmentation aborted", cand.candidate_id, doc.key)
            return None
        fact = await _fact_for_chunk(cfg, gen, chunk, facts_by_chunk)
        if fact is None:
            return None
        new_facts.append(fact)

    answer = _rebuild_answer(checker, res.truth_keys, cand.answer_field)
    if not answer:
        return None
    cand.facts = cand.facts + new_facts
    cand.answer = answer
    cand.hop_depth = len(cand.facts)
    return cand


# --------------------------------------------------------------------------- #
# hooks for the later stages
# --------------------------------------------------------------------------- #
def check_clean(cfg: SidConfig, checker: CompletenessChecker | None,
                cand: Candidate) -> tuple[bool, str]:
    """Post-repair (S4) re-check: a rewritten question must still denote
    exactly its gold set. No repair loop here — the S4 loop owns iteration."""
    if checker is None or not in_scope(cfg, cand.mechanic):
        return True, ""
    res = checker.check(cand)
    if res.error:
        return (not cfg.completeness.require_filter), f"filter: {res.error}"
    if res.gold_violations:
        return False, f"gold violates own constraints: {res.gold_violations}"
    if res.excess:
        return False, f"{len(res.excess)} documents beyond gold match the filter"
    return True, ""


def doc_keys_of(cfg: SidConfig, corpus: SidCorpus, chunk_ids) -> set[str]:
    out = set()
    for cid in chunk_ids:
        c = corpus.get(cid)
        if c is not None:
            out.add(doc_key_of(cfg, c))
    return out
