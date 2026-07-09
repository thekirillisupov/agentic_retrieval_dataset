"""Build a dialogue retrieval training set from MuSiQue multi-hop decompositions.

MuSiQue gives, for every multi-hop question, a ``question_decomposition``: an
ordered list of single-hop sub-questions where a later hop references an earlier
hop's answer with a ``#k`` placeholder (``k`` = 1-based hop position). Every hop
also carries ``paragraph_support_idx`` — the index of the single paragraph that
answers it. That is exactly the raw material for a conversational, anaphora-heavy
retrieval set:

    hop 1: client asks q1                       -> bot answers a1
    hop 2: client asks q2 where "#1" is replaced by a pronoun / definite
           description referring to a1 (which the bot already said)  -> a2
    ...

An N-hop question yields N turns, and we emit one training item per turn ``t``:

* ``question``  — the transcript so far in the ``<client>/<bot>`` format,
                  ending with the client's latest message (turn ``t``).
* ``answer``    — hop ``t``'s answer.
* ``gold_chunk_ids`` — **only** hop ``t``'s supporting paragraph.

The gold rule is a deliberate, frozen decision: the bot has already uttered the
earlier answers in the transcript, so hop ``t``'s supporting passage is the only
thing "relevant to the latest message" for a downstream retriever — the earlier
hops' documents are redundant to it.

Each MuSiQue example ships its own ~20 paragraphs (supporting + distractors);
their union is written out as the retrieval corpus, so ``gold_chunk_ids`` point
into a real pool that contains hard distractors for free.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Iterator

from .config import Config
from .generate import _gather
from .llm import BaseLLM, LLMError
from .prompts import ANAPHORA_SYSTEM, anaphora_user
from .schema import DatasetItem, chunk_id
from .utils import append_jsonl, ensure_parent, load_done_keys, log, read_jsonl, write_jsonl

# ``#3`` style back-reference to the answer of an earlier hop.
_REF_RE = re.compile(r"#(\d+)")

PROFILE = "musique_dialogue"


# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #
class MusiqueOptions:
    """Runtime knobs for the builder (populated from CLI args)."""

    def __init__(
        self,
        hf_name: str = "bdsaglam/musique",
        config_name: str = "answerable",
        split: str = "train",
        input_path: str = "",
        limit: int = 0,
        min_turn: int = 1,
        only_anaphora_turns: bool = False,
        anaphora: str = "llm",          # "llm" | "heuristic"
        corpus_path: str = "",
        dataset_path: str = "",
    ):
        self.hf_name = hf_name
        self.config_name = config_name
        self.split = split
        self.input_path = input_path
        self.limit = limit
        self.min_turn = max(1, min_turn)
        self.only_anaphora_turns = only_anaphora_turns
        self.anaphora = anaphora
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path


# --------------------------------------------------------------------------- #
# Loading MuSiQue
# --------------------------------------------------------------------------- #
def load_musique(opts: MusiqueOptions) -> list[dict[str, Any]]:
    """Load raw MuSiQue examples from a local file or the HF hub."""
    if opts.input_path:
        raw = list(_iter_local(opts.input_path))
        log.info("musique: loaded %d examples from %s", len(raw), opts.input_path)
    else:
        raw = _load_from_hub(opts)
    examples = [ex for ex in raw if _is_usable(ex)]
    log.info("musique: %d/%d examples usable (answerable + full support)",
             len(examples), len(raw))
    if opts.limit and opts.limit > 0:
        examples = examples[: opts.limit]
        log.info("musique: capped to %d examples (--limit)", len(examples))
    return examples


def _iter_local(path: str) -> Iterator[dict[str, Any]]:
    """Read examples from a .jsonl / .json file or a directory of them."""
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.endswith((".json", ".jsonl")):
                yield from _iter_local(os.path.join(path, name))
        return
    if path.endswith(".jsonl"):
        for rec in read_jsonl(path):
            yield rec
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        yield from (data if isinstance(data, list) else [data])


def _load_from_hub(opts: MusiqueOptions) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - exercised only without the lib
        raise ImportError(
            "pip install datasets — required to pull MuSiQue from the HF hub, "
            "or pass --input <local musique.jsonl>"
        ) from e
    log.info("musique: loading %s config=%s split=%s from the HF hub",
             opts.hf_name, opts.config_name, opts.split)
    ds = load_dataset(opts.hf_name, opts.config_name, split=opts.split)
    return [dict(ex) for ex in ds]


def _is_usable(ex: dict[str, Any]) -> bool:
    """Keep only answerable multi-hop examples whose every hop has an answer and
    a supporting paragraph — those are the ones that map cleanly to turns."""
    if not ex.get("answerable", True):
        return False
    hops = ex.get("question_decomposition") or []
    if len(hops) < 2:
        return False
    para_idxs = {p["idx"] for p in ex.get("paragraphs", []) if "idx" in p}
    for h in hops:
        if not (h.get("answer") or "").strip():
            return False
        sup = h.get("paragraph_support_idx")
        if sup is None or sup not in para_idxs:
            return False
    return True


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #
def write_corpus(examples: list[dict[str, Any]], path: str) -> int:
    """Write every example's paragraphs as chunk records for retrieval.

    ``chunk_id == "{example_id}::{paragraph_idx}"`` so a hop's
    ``paragraph_support_idx`` maps straight onto a gold chunk id.
    """
    def rows() -> Iterator[dict[str, Any]]:
        for ex in examples:
            ex_id = str(ex["id"])
            for p in ex.get("paragraphs", []):
                if "idx" not in p:
                    continue
                yield {
                    "file_name": ex_id,
                    "index": int(p["idx"]),
                    "raw_text": p.get("paragraph_text", "") or "",
                    "title": p.get("title", "") or "",
                    "document_id": ex_id,
                    "is_supporting": bool(p.get("is_supporting", False)),
                }

    n = write_jsonl(path, rows())
    log.info("musique: wrote %d corpus chunks -> %s", n, path)
    return n


# --------------------------------------------------------------------------- #
# Transcript rendering
# --------------------------------------------------------------------------- #
def render_transcript(turns: list[tuple[str, str]], upto: int) -> str:
    """Render the ``<client>/<bot>`` transcript up to (and including) the client
    message of turn ``upto`` (1-based), with no trailing bot reply.

    ``turns`` is a list of ``(client_message, bot_answer)`` for turns 1..upto.
    """
    lines: list[str] = []
    for i in range(upto):
        client, bot = turns[i]
        lines.append(f"<client>{client}</client>")
        if i < upto - 1:                      # every turn but the latest gets a reply
            lines.append(f"<bot>{bot}</bot>")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Anaphora: replace #k with a pronoun / definite description
# --------------------------------------------------------------------------- #
def _refs(question: str) -> list[int]:
    """1-based hop positions referenced by ``#k`` tokens, in first-seen order."""
    seen: list[int] = []
    for m in _REF_RE.findall(question):
        k = int(m)
        if k not in seen:
            seen.append(k)
    return seen


def _describe(hop_question: str) -> str:
    """A generic definite description for the entity a hop asks about, used by the
    heuristic rewriter and as a safety net for leftover placeholders."""
    q = hop_question.strip().lower()
    if q.startswith("who"):
        return "that person"
    if q.startswith("when"):
        return "that date"
    if q.startswith("where"):
        return "that place"
    if q.startswith(("what", "which")):
        toks = hop_question.split()
        stop = {"is", "was", "are", "were", "did", "does", "do",
                "the", "a", "an", "of", "in", "'s"}
        if len(toks) >= 2 and toks[1].lower().strip("?,.") not in stop:
            return "that " + toks[1].lower().strip("?,.")
        return "that"
    return "it"


def heuristic_rewrite(raw_question: str, decomposition: list[dict[str, Any]]) -> str:
    """Deterministic, model-free anaphora: swap each ``#k`` for a definite
    description derived from hop ``k``'s question."""
    def sub(m: re.Match) -> str:
        k = int(m.group(1))
        if 1 <= k <= len(decomposition):
            return _describe(decomposition[k - 1].get("question", ""))
        return "it"
    return _REF_RE.sub(sub, raw_question).strip()


def strip_placeholders(text: str, decomposition: list[dict[str, Any]]) -> str:
    """Safety net: guarantee no raw ``#k`` survives into the final message even if
    the LLM left one behind."""
    if "#" not in text:
        return text
    return heuristic_rewrite(text, decomposition)


async def rewrite_turn(
    llm: BaseLLM | None,
    opts: MusiqueOptions,
    transcript: str,
    raw_question: str,
    decomposition: list[dict[str, Any]],
) -> str:
    """Produce the client's next message for a hop that references earlier answers.

    Turn 1 (and any hop without a ``#k``) is emitted verbatim by the caller; this
    is only invoked when there is anaphora to resolve.
    """
    refs = _refs(raw_question)
    if opts.anaphora == "heuristic" or llm is None:
        return heuristic_rewrite(raw_question, decomposition)

    ref_lines = []
    for k in refs:
        if 1 <= k <= len(decomposition):
            ref_lines.append(f'    #{k} = "{decomposition[k - 1].get("answer", "")}"')
    ref_map = "\n".join(ref_lines)
    try:
        obj = await llm.complete_json(
            ANAPHORA_SYSTEM, anaphora_user(transcript, raw_question, ref_map))
        msg = (obj.get("message") or "").strip()
    except LLMError as e:
        log.warning("anaphora rewrite failed (%s); using heuristic fallback", e)
        msg = ""
    if not msg:
        msg = heuristic_rewrite(raw_question, decomposition)
    return strip_placeholders(msg, decomposition)


# --------------------------------------------------------------------------- #
# Turn / item construction
# --------------------------------------------------------------------------- #
async def build_turns(
    llm: BaseLLM | None, opts: MusiqueOptions, ex: dict[str, Any]
) -> list[tuple[str, str]]:
    """Build the ordered ``(client_message, bot_answer)`` list for one example.

    Turns are built front-to-back because a later turn's anaphora rewrite is
    conditioned on the transcript produced by the earlier turns.
    """
    hops = ex["question_decomposition"]
    turns: list[tuple[str, str]] = []
    for i, hop in enumerate(hops):
        raw_q = (hop.get("question") or "").strip()
        answer = (hop.get("answer") or "").strip()
        if _REF_RE.search(raw_q):
            transcript = render_transcript(turns + [(raw_q, answer)], i + 1)
            client_msg = await rewrite_turn(llm, opts, transcript, raw_q, hops)
        else:
            client_msg = raw_q
        turns.append((client_msg, answer))
    return turns


def items_for_example(
    ex: dict[str, Any], turns: list[tuple[str, str]], opts: MusiqueOptions
) -> list[dict[str, Any]]:
    """Emit one dataset record per eligible turn of a single example."""
    ex_id = str(ex["id"])
    hops = ex["question_decomposition"]
    n_hops = len(hops)
    aliases = ex.get("answer_aliases") or []
    out: list[dict[str, Any]] = []

    for t in range(1, n_hops + 1):
        hop = hops[t - 1]
        raw_q = hop.get("question") or ""
        has_anaphora = bool(_REF_RE.search(raw_q))
        if t < opts.min_turn:
            continue
        if opts.only_anaphora_turns and not has_anaphora:
            continue

        transcript = render_transcript(turns, t)
        gold_id = chunk_id(ex_id, int(hop["paragraph_support_idx"]))
        # window = the supporting passages of the whole reasoning chain up to t
        chain_ids = [chunk_id(ex_id, int(hops[j]["paragraph_support_idx"]))
                     for j in range(t)]

        item = DatasetItem(
            id=f"{ex_id}__t{t}",
            question=transcript,
            answer=turns[t - 1][1],
            gold_chunk_ids=[gold_id],
            file_name=ex_id,
            question_type="multi_hop",
            question_style="dialogue",
            num_gold=1,
            window_chunk_ids=chain_ids,
            profile=PROFILE,
            difficulty="hard",
            positive_chunk_ids=[gold_id],
            num_positives=1,
            # provenance lives in the free-form dict so records stay valid
            # DatasetItem dicts (usable by stats / negatives / finalize).
            verification={
                "musique": {
                    "orig_id": ex_id,
                    "orig_question": ex.get("question", ""),
                    "orig_answer": ex.get("answer", ""),
                    "num_hops": n_hops,
                    "turn": t,
                    "has_anaphora": has_anaphora,
                    "hop_question_raw": raw_q,
                    "hop_answer_aliases": aliases if t == n_hops else [],
                }
            },
        )
        out.append(item.to_dict())
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
async def build(cfg: Config, llm: BaseLLM | None, opts: MusiqueOptions) -> None:
    corpus_path = opts.corpus_path or os.path.join(cfg.paths.out_dir, "musique_corpus.jsonl")
    dataset_path = opts.dataset_path or os.path.join(cfg.paths.out_dir, "musique_dialogues.jsonl")

    examples = load_musique(opts)
    if not examples:
        log.warning("musique: no usable examples; nothing to do")
        return

    # Corpus is deterministic and cheap — (re)build it over ALL examples so it is
    # always complete, independent of the resumable item pass below.
    write_corpus(examples, corpus_path)

    done = load_done_keys(dataset_path, "file_name")
    todo = [ex for ex in examples if str(ex["id"]) not in done]
    log.info("musique: %d examples, %d already built, %d to build",
             len(examples), len(examples) - len(todo), len(todo))
    ensure_parent(dataset_path)

    lock = asyncio.Lock()
    written = 0

    async def worker(ex: dict[str, Any]) -> None:
        nonlocal written
        try:
            turns = await build_turns(llm, opts, ex)
        except LLMError as e:
            log.error("musique: turn build failed for %s: %s", ex.get("id"), e)
            return
        records = items_for_example(ex, turns, opts)
        if not records:
            return
        async with lock:
            for rec in records:
                append_jsonl(dataset_path, rec)
                written += 1

    concurrency = cfg.llm.max_concurrency if opts.anaphora == "llm" else 32
    await _gather(todo, worker, concurrency)
    log.info("musique: wrote %d dialogue items -> %s", written, dataset_path)
    _log_stats(dataset_path)


def _log_stats(path: str) -> None:
    items = list(read_jsonl(path))
    if not items:
        return
    turns: dict[int, int] = {}
    hops: dict[int, int] = {}
    anaphora = 0
    for it in items:
        meta = it.get("verification", {}).get("musique", {})
        turns[meta.get("turn", 0)] = turns.get(meta.get("turn", 0), 0) + 1
        hops[meta.get("num_hops", 0)] = hops.get(meta.get("num_hops", 0), 0) + 1
        anaphora += bool(meta.get("has_anaphora"))
    log.info("musique: %d items; %d with anaphora (%.0f%%)",
             len(items), anaphora, 100 * anaphora / len(items))
    log.info("musique: turns  " + ", ".join(f"t{k}={turns[k]}" for k in sorted(turns)))
    log.info("musique: n_hops " + ", ".join(f"{k}hop={hops[k]}" for k in sorted(hops)))
