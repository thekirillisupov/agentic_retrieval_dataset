"""End-to-end pipeline test on the offline mock backend (no network/GPU).

Verifies: chunk loading, filtering, window construction, JSON extraction, the
async generate/verify stages, and that final items satisfy the multi-chunk
(>= 2 gold) invariant.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arqg.config import Config
from arqg.data import ChunkStore, load_chunks, is_eligible_seed, cyrillic_ratio
from arqg.generate import generate as run_generate
from arqg.llm import make_client, extract_json
from arqg.verify import verify as run_verify
from arqg.windows import build_windows
from arqg.utils import read_jsonl, write_jsonl

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "sample_chunks.jsonl")


def _cfg(tmpdir) -> Config:
    cfg = Config()
    cfg.paths.chunks = SAMPLE
    cfg.paths.out_dir = str(tmpdir)
    cfg.llm.backend = "mock"
    cfg.verify.judge.backend = "mock"
    cfg.windows.target_windows = 0
    return cfg


def test_load_and_filter():
    chunks = load_chunks(SAMPLE)
    assert len(chunks) == 8
    store = ChunkStore(chunks)
    assert len(store) == 8
    assert store.file_indices("doc_a.txt") == [0, 1, 2, 3]
    win = store.contiguous_window("doc_a.txt", 1, 3)
    assert [c.index for c in win] == [1, 2, 3]
    # all sample chunks are eligible Russian text
    fcfg = Config().filters
    assert all(is_eligible_seed(c, fcfg) for c in chunks)
    assert cyrillic_ratio("Привет мир") > 0.5


def test_filter_rejects_junk():
    from arqg.schema import Chunk
    fcfg = Config().filters
    assert not is_eligible_seed(Chunk("f", 0, "12345 67890 ..."), fcfg)   # too short / non-alpha
    assert not is_eligible_seed(Chunk("f", 0, "hello world " * 50), fcfg)  # not Cyrillic


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('бла бла {"a": {"b": 2}} конец') == {"a": {"b": 2}}
    assert extract_json("not json at all") is None


def test_windows_are_contiguous_multichunk():
    cfg = _cfg("/tmp/arqg_win")
    store = ChunkStore(load_chunks(cfg.paths.chunks))
    windows = build_windows(store, cfg.windows, cfg.filters)
    assert windows
    for w in windows:
        assert len(w.indices) >= 2
        assert w.indices == list(range(w.indices[0], w.indices[0] + len(w.indices)))
        assert len(set(c.split("::")[0] for c in w.chunk_ids)) == 1  # single document


def test_end_to_end_mock(tmp_path):
    cfg = _cfg(tmp_path)
    store = ChunkStore(load_chunks(cfg.paths.chunks))

    windows = build_windows(store, cfg.windows, cfg.filters)
    write_jsonl(cfg.paths.windows, (w.to_dict() for w in windows))

    gen = make_client(cfg.llm)
    asyncio.run(run_generate(cfg, gen))
    candidates = list(read_jsonl(cfg.paths.candidates))
    assert candidates, "expected generated candidates"
    for c in candidates:
        assert len(c["required_chunk_ids"]) >= 2

    judge = make_client(cfg.verify.judge)
    asyncio.run(run_verify(cfg, judge, store))
    items = list(read_jsonl(cfg.paths.verified))
    assert items, "expected verified dataset items"
    for it in items:
        assert it["num_gold"] >= 2
        assert len(it["gold_chunk_ids"]) >= 2
        # gold chunks must be a subset of the window context
        assert set(it["gold_chunk_ids"]).issubset(set(it["window_chunk_ids"]))


def test_style_sampling_deterministic_and_injected():
    from arqg.generate import _sample_style
    from arqg.prompts import STYLES, gen_user
    from arqg.schema import Chunk

    cfg = Config()
    s1 = _sample_style(cfg, "w_abc", 0)
    s2 = _sample_style(cfg, "w_abc", 0)
    assert s1 == s2 and s1 in STYLES          # deterministic, valid style
    # weights restricted to one style force that style
    cfg.generate.styles = {"search_query": 1.0}
    assert _sample_style(cfg, "w_any", 0) == "search_query"
    # unknown styles in config are ignored, falling back gracefully
    cfg.generate.styles = {"bogus": 1.0}
    assert _sample_style(cfg, "w_any", 0) == "simple_user"
    # the style instruction text is actually injected into the prompt
    chunk = Chunk("f", 0, "текст")
    assert STYLES["expert"][:30] in gen_user([chunk], "expert")


def test_dataset_items_carry_style(tmp_path):
    cfg = _cfg(tmp_path)
    store = ChunkStore(load_chunks(cfg.paths.chunks))
    windows = build_windows(store, cfg.windows, cfg.filters)
    write_jsonl(cfg.paths.windows, (w.to_dict() for w in windows))
    asyncio.run(run_generate(cfg, make_client(cfg.llm)))
    asyncio.run(run_verify(cfg, make_client(cfg.verify.judge), store))
    from arqg.prompts import STYLES
    items = list(read_jsonl(cfg.paths.verified))
    assert items and all(i["question_style"] in STYLES for i in items)


def test_doc_units_span_whole_documents():
    from arqg.docunits import build_doc_units
    cfg = Config()
    cfg.paths.chunks = SAMPLE
    store = ChunkStore(load_chunks(SAMPLE))
    units = build_doc_units(store, cfg.docgen, cfg.filters)
    # small sample docs (4 chunks each) fit the caps -> one unit per document
    assert len(units) == 2
    for u in units:
        assert u.window_id.startswith("d_")           # distinct id namespace
        assert u.indices == [0, 1, 2, 3]              # whole document
        assert len(set(c.split("::")[0] for c in u.chunk_ids)) == 1


def test_doc_units_split_oversized_documents():
    from arqg.docunits import build_doc_units
    from arqg.schema import Chunk
    para = "Это предложение содержит несколько обычных русских слов для теста. " * 18
    chunks = [Chunk("big.txt", i, para) for i in range(10)]   # ~1200 chars each
    store = ChunkStore(chunks)
    cfg = Config()
    cfg.docgen.units.max_doc_chars = 2500   # ~2 chunks per span
    cfg.docgen.units.max_units_per_file = 10
    units = build_doc_units(store, cfg.docgen, cfg.filters)
    assert len(units) >= 4                  # 10 chunks split into several spans
    for u in units:
        assert u.n_chars <= 2500 or len(u.indices) == 1


def test_doc_end_to_end_simple_and_hard(tmp_path):
    from arqg.docunits import build_doc_units
    from arqg.generate_docs import generate_docs as run_gen_docs
    cfg = _cfg(tmp_path)
    cfg.generate.enabled = False
    cfg.docgen.enabled = True
    # force a 50/50 deterministic mix by running enough units (sample is 2 docs)
    cfg.docgen.questions_per_unit = 1
    store = ChunkStore(load_chunks(cfg.paths.chunks))

    units = build_doc_units(store, cfg.docgen, cfg.filters)
    write_jsonl(cfg.paths.docunits, (u.to_dict() for u in units))
    asyncio.run(run_gen_docs(cfg, make_client(cfg.llm)))

    cands = list(read_jsonl(cfg.paths.candidates))
    assert cands
    for c in cands:
        assert c["profile"] == "doc_simple_hard"
        assert c["difficulty"] in ("simple", "hard")
        if c["difficulty"] == "simple":
            assert len(c["required_chunk_ids"]) == 1
            assert c["min_gold"] == 1 and c["run_minimality"] is False
        else:
            assert len(c["required_chunk_ids"]) >= 2

    asyncio.run(run_verify(cfg, make_client(cfg.verify.judge), store))
    items = list(read_jsonl(cfg.paths.verified))
    assert items
    for it in items:
        # gold is always a subset of the unit and respects the difficulty floor
        assert set(it["gold_chunk_ids"]).issubset(set(it["window_chunk_ids"]))
        if it["difficulty"] == "simple":
            assert it["num_gold"] == 1
        else:
            assert it["num_gold"] >= 2


def test_verify_keeps_single_gold_when_policy_allows(tmp_path):
    """A simple (gold=1) candidate must survive verify, unlike the default
    multi-hop policy which would drop single-chunk items."""
    from arqg.schema import Candidate
    cfg = _cfg(tmp_path)
    store = ChunkStore(load_chunks(cfg.paths.chunks))
    simple = Candidate(
        candidate_id="x__c0", window_id="d_x", file_name="doc_a.txt",
        window_chunk_ids=["doc_a.txt::0", "doc_a.txt::1"],
        question="вопрос?", answer="ответ", required_chunk_ids=["doc_a.txt::0"],
        question_type="factoid", profile="doc_simple_hard", difficulty="simple",
        min_gold=1, enforce_multi_chunk=False, run_minimality=False,
    )
    write_jsonl(cfg.paths.candidates, [simple.to_dict()])
    asyncio.run(run_verify(cfg, make_client(cfg.verify.judge), store))
    items = list(read_jsonl(cfg.paths.verified))
    assert len(items) == 1 and items[0]["num_gold"] == 1


def _seed_verified(cfg, store):
    """Run the neighbour pipeline up to verified.jsonl for collect tests."""
    windows = build_windows(store, cfg.windows, cfg.filters)
    write_jsonl(cfg.paths.windows, (w.to_dict() for w in windows))
    asyncio.run(run_generate(cfg, make_client(cfg.llm)))
    asyncio.run(run_verify(cfg, make_client(cfg.verify.judge), store))


def test_clues_and_retrieval_requests(tmp_path):
    from arqg.clues import make_clues
    cfg = _cfg(tmp_path)
    store = ChunkStore(load_chunks(cfg.paths.chunks))
    _seed_verified(cfg, store)

    asyncio.run(make_clues(cfg, make_client(cfg.llm), store))
    clues = list(read_jsonl(cfg.paths.clues))
    reqs = list(read_jsonl(cfg.paths.retrieval_requests))
    assert clues and len(reqs) == len(clues)
    # the request format I hand the user
    r = reqs[0]
    assert set(r) == {"clue_id", "item_id", "query", "top_k"}
    assert r["top_k"] == cfg.collect.top_k
    # each clue attributes to gold chunks of its item
    items = {i["id"]: i for i in read_jsonl(cfg.paths.verified)}
    for c in clues:
        assert set(c["source_gold_ids"]).issubset(set(items[c["item_id"]]["gold_chunk_ids"]))


def test_collect_positives_adds_near_duplicates(tmp_path):
    from arqg.clues import make_clues
    from arqg.collect import collect_positives
    # corpus = sample + a near-duplicate of doc_a under a different file name
    rows = list(read_jsonl(SAMPLE))
    dup = [{**r, "file_name": "doc_a_DUP.txt"} for r in rows if r["file_name"] == "doc_a.txt"]
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl(str(corpus), rows + dup)

    cfg = _cfg(tmp_path)
    cfg.paths.chunks = str(corpus)
    cfg.collect.enabled = True
    cfg.verify.judge.backend = cfg.collect.judge.backend = "mock"
    store = ChunkStore(load_chunks(cfg.paths.chunks))
    _seed_verified(cfg, store)
    asyncio.run(make_clues(cfg, make_client(cfg.llm), store))

    # simulate the user's retrieval: for each clue return the dup of its gold source
    results = []
    for c in read_jsonl(cfg.paths.clues):
        passages = []
        for gid in c["source_gold_ids"]:
            fn, idx = gid.split("::")
            if fn == "doc_a.txt":
                passages.append({"chunk_id": f"doc_a_DUP.txt::{idx}", "score": 0.9})
        results.append({"clue_id": c["clue_id"], "passages": passages})
    write_jsonl(cfg.paths.retrieval_results, results)

    asyncio.run(collect_positives(cfg, make_client(cfg.collect.judge), store))
    items = list(read_jsonl(cfg.paths.collected))
    assert items
    expanded = 0
    for it in items:
        # gold is preserved and is a subset of positives
        assert set(it["gold_chunk_ids"]).issubset(set(it["positive_chunk_ids"]))
        assert it["num_positives"] == len(it["positive_chunk_ids"])
        if any("doc_a.txt" in g for g in it["gold_chunk_ids"]):
            assert any(p.startswith("doc_a_DUP.txt::") for p in it["positive_chunk_ids"])
            expanded += 1
    assert expanded > 0   # at least one item gained a near-duplicate positive


def test_index_ranks_exact_match_first(tmp_path):
    from arqg.embeddings import make_embedder
    from arqg.index import build_or_load_index
    cfg = _cfg(tmp_path)
    cfg.retrieve.backend = "mock"
    store = ChunkStore(load_chunks(cfg.paths.chunks))
    embedder = make_embedder(cfg.retrieve)
    idx = asyncio.run(build_or_load_index(
        cfg.retrieve, embedder, store, cfg.paths.chunks, str(tmp_path / "index")))
    # a query identical to a passage must retrieve that passage at rank 1
    target = store.get("doc_b.txt", 2)
    q = asyncio.run(embedder.embed([target.raw_text], kind="query"))
    hits = idx.search(q, top_k=3)[0]
    assert hits[0][0] == target.id
    # the cached index loads back without re-embedding
    idx2 = asyncio.run(build_or_load_index(
        cfg.retrieve, embedder, store, cfg.paths.chunks, str(tmp_path / "index")))
    assert idx2.chunk_ids == idx.chunk_ids


def test_loader_reads_document_id_and_title(tmp_path):
    from arqg.schema import Chunk
    p = tmp_path / "c.jsonl"
    write_jsonl(str(p), [{"file_name": "f.txt", "index": 0, "raw_text": "текст",
                          "document_id": "D1", "title": "Заголовок"}])
    c = load_chunks(str(p))[0]
    assert c.document_id == "D1" and c.title == "Заголовок" and c.id == "f.txt::0"


def test_retrieve_fills_results_format(tmp_path):
    from arqg.retrieve import retrieve as run_retrieve
    cfg = _cfg(tmp_path)
    cfg.retrieve.backend = "mock"
    cfg.collect.top_k = 3
    store = ChunkStore(load_chunks(cfg.paths.chunks))
    # a request whose query is a real passage text -> that passage should appear
    target = store.get("doc_a.txt", 1)
    write_jsonl(cfg.paths.retrieval_requests,
                [{"clue_id": "q__k0", "item_id": "q", "query": target.raw_text, "top_k": 3}])
    asyncio.run(run_retrieve(cfg, store))
    results = list(read_jsonl(cfg.paths.retrieval_results))
    assert len(results) == 1
    passages = results[0]["passages"]
    assert 1 <= len(passages) <= 3
    assert set(passages[0]) >= {"chunk_id", "document_id", "title", "file_name", "index", "score"}
    assert any(p["chunk_id"] == target.id for p in passages)


def test_full_auto_collect_with_retrieval(tmp_path):
    """End-to-end: clues -> retrieve (mock index) -> collect-positives."""
    from arqg.clues import make_clues
    from arqg.retrieve import retrieve as run_retrieve
    from arqg.collect import collect_positives
    cfg = _cfg(tmp_path)
    cfg.collect.enabled = True
    cfg.retrieve.backend = "mock"
    cfg.collect.judge.backend = "mock"
    store = ChunkStore(load_chunks(cfg.paths.chunks))
    _seed_verified(cfg, store)
    asyncio.run(make_clues(cfg, make_client(cfg.llm), store))
    asyncio.run(run_retrieve(cfg, store))
    assert list(read_jsonl(cfg.paths.retrieval_results))   # retrieval produced results
    asyncio.run(collect_positives(cfg, make_client(cfg.collect.judge), store))
    items = list(read_jsonl(cfg.paths.collected))
    assert items
    for it in items:
        assert set(it["gold_chunk_ids"]).issubset(set(it["positive_chunk_ids"]))


def test_resume_is_idempotent(tmp_path):
    cfg = _cfg(tmp_path)
    store = ChunkStore(load_chunks(cfg.paths.chunks))
    windows = build_windows(store, cfg.windows, cfg.filters)
    write_jsonl(cfg.paths.windows, (w.to_dict() for w in windows))

    gen = make_client(cfg.llm)
    asyncio.run(run_generate(cfg, gen))
    n1 = len(list(read_jsonl(cfg.paths.candidates)))
    # second run should skip everything (already done) and not duplicate
    asyncio.run(run_generate(cfg, gen))
    n2 = len(list(read_jsonl(cfg.paths.candidates)))
    assert n1 == n2
