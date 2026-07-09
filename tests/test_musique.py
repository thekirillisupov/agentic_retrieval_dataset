"""Offline tests for the MuSiQue -> dialogue builder (mock LLM, no network)."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arqg.config import Config
from arqg.llm import make_client
from arqg.musique import (
    MusiqueOptions,
    build,
    heuristic_rewrite,
    items_for_example,
    load_musique,
    render_transcript,
    _refs,
)
from arqg.utils import read_jsonl

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "sample_musique.jsonl")


def _cfg(tmp_path) -> Config:
    cfg = Config()
    cfg.paths.out_dir = str(tmp_path)
    cfg.llm.backend = "mock"
    return cfg


def _opts(tmp_path, **kw) -> MusiqueOptions:
    base = dict(
        input_path=SAMPLE,
        corpus_path=str(tmp_path / "corpus.jsonl"),
        dataset_path=str(tmp_path / "dialogues.jsonl"),
    )
    base.update(kw)
    return MusiqueOptions(**base)


def test_load_filters_unanswerable():
    opts = MusiqueOptions(input_path=SAMPLE)
    exs = load_musique(opts)
    ids = {e["id"] for e in exs}
    assert ids == {"2hop__strat", "3hop__river"}  # unanswerable one dropped


def test_refs_and_heuristic_rewrite():
    assert _refs("How long is #2 near #1 and #2?") == [2, 1]
    decomp = [
        {"question": "What is the capital of Egypt?", "answer": "Cairo"},
        {"question": "What river does #1 sit on?", "answer": "the Nile"},
    ]
    # #1 -> "that" derived from a "What <noun>" question, no placeholder remains
    out = heuristic_rewrite("What river does #1 sit on?", decomp)
    assert "#" not in out and "sit on" in out


def test_render_transcript_shape():
    turns = [("q1", "a1"), ("q2", "a2"), ("q3", "a3")]
    t = render_transcript(turns, 2)
    assert t == "<client>q1</client>\n<bot>a1</bot>\n<client>q2</client>"
    # the latest client message never gets a trailing bot reply
    assert not t.rstrip().endswith("</bot>")


def test_corpus_and_items_end_to_end(tmp_path):
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    asyncio.run(build(cfg, make_client(cfg.llm), opts))

    corpus = list(read_jsonl(opts.corpus_path))
    # 4 paragraphs for each of the 2 usable examples
    assert len(corpus) == 8
    ids = {f"{c['file_name']}::{c['index']}" for c in corpus}
    assert "2hop__strat::2" in ids and "3hop__river::0" in ids

    items = list(read_jsonl(opts.dataset_path))
    # 2 turns + 3 turns
    assert len(items) == 5
    by_id = {it["id"]: it for it in items}

    # gold for a turn is exactly that hop's supporting paragraph
    t2 = by_id["2hop__strat__t2"]
    assert t2["gold_chunk_ids"] == ["2hop__strat::2"]
    assert t2["num_gold"] == 1 and t2["answer"] == "Leo Fender"
    # transcript carries the earlier bot answer and ends on the client's message
    assert "<bot>Fender</bot>" in t2["question"]
    assert t2["question"].rstrip().endswith("</client>")
    # the anaphora rewrite removed the raw #1 back-reference
    assert "#1" not in t2["question"]

    # window is the reasoning chain up to the turn; profile/style are set
    assert t2["window_chunk_ids"] == ["2hop__strat::1", "2hop__strat::2"]
    assert t2["profile"] == "musique_dialogue" and t2["question_style"] == "dialogue"

    # 3-hop, final turn: gold is only the third hop's passage
    t3 = by_id["3hop__river__t3"]
    assert t3["gold_chunk_ids"] == ["3hop__river::2"]
    assert t3["verification"]["musique"]["num_hops"] == 3
    assert t3["verification"]["musique"]["hop_answer_aliases"] == ["6650 km"]
    # the last turn's transcript has two prior bot replies
    assert t3["question"].count("<bot>") == 2


def test_items_are_valid_dataset_items(tmp_path):
    """Records must round-trip through DatasetItem so stats/negatives/finalize
    can consume them (extra provenance lives in the free-form dict)."""
    from arqg.schema import DatasetItem
    exs = load_musique(_opts(tmp_path))
    ex = next(e for e in exs if e["id"] == "2hop__strat")
    turns = asyncio.run(build_turns_heuristic(ex))
    recs = items_for_example(ex, turns, _opts(tmp_path, min_turn=1))
    for r in recs:
        DatasetItem(**r)  # must not raise on unexpected keys


async def build_turns_heuristic(ex):
    from arqg.musique import build_turns
    return await build_turns(None, MusiqueOptions(anaphora="heuristic"), ex)


def test_min_turn_and_only_anaphora(tmp_path):
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path, min_turn=2, only_anaphora_turns=True, anaphora="heuristic")
    asyncio.run(build(cfg, None, opts))
    items = list(read_jsonl(opts.dataset_path))
    # first (standalone) hops are dropped; only referencing turns remain
    turns = {it["id"] for it in items}
    assert turns == {"2hop__strat__t2", "3hop__river__t2", "3hop__river__t3"}
    for it in items:
        assert it["verification"]["musique"]["has_anaphora"]


def test_resume_is_idempotent(tmp_path):
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path, anaphora="heuristic")
    asyncio.run(build(cfg, None, opts))
    n1 = len(list(read_jsonl(opts.dataset_path)))
    asyncio.run(build(cfg, None, opts))  # second run skips done examples
    n2 = len(list(read_jsonl(opts.dataset_path)))
    assert n1 == n2 == 5
