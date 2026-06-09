# Agentic Retrieval — Synthetic Dataset Pipeline

Generate a high-quality synthetic dataset for **agentic / multi-hop retrieval** from a
chunked, document-based knowledge base (Russian). Each produced item is:

```json
{"question": "...", "answer": "...", "gold_chunk_ids": ["file::3", "file::4"], ...}
```

The defining property: **every question requires combining information from several
*neighbouring* chunks**, and the gold set is **verified to be minimal and strictly
necessary** — so a retriever genuinely has to find *all* the gold chunks, not just one.

---

## Why this design (quality first)

A naïve "ask an LLM to write a question about this chunk" approach produces noisy data:
questions answerable from a single chunk, ungrounded answers, gold sets full of
unnecessary chunks, and questions that leak answers or reference "the text". This
pipeline attacks each failure mode explicitly:

| Risk | Mitigation |
|---|---|
| Single-chunk questions | Generator is forced to use ≥2 neighbours; a **minimality judge** rejects anything where one chunk alone suffices |
| Bloated gold sets | The minimality judge **shrinks gold to the strictly-necessary subset** before it's written |
| Hallucinated answers | **Groundedness judge** checks the answer is fully supported by the gold chunks |
| "Согласно тексту…" / context-dependent | Generator prompt forbids it; judge enforces a **standalone** check |
| Trivia answerable without retrieval | Judge flags **answerable-from-world-knowledge** and drops it |
| Junk chunks (headers, page numbers, min=1 char) | Seed eligibility filter (length / word / Cyrillic ratios); junk stays usable only as *neighbour context* |
| Giant un-split chunks (max≈39k) / huge docs dominating | `max_chars` seed cap, `max_window_chars`, `max_windows_per_file` |

The generator and judge can be **different models** (a stronger judge reduces self-bias).

---

## Pipeline stages

```
chunks.jsonl
   │  windows    contiguous neighbour windows within a single document
   ▼
windows.jsonl
   │  generate   LLM writes a multi-chunk question + claimed required chunks
   ▼
candidates.jsonl
   │  verify     judge#1 minimality (→ minimal gold) + judge#2 groundedness
   ▼
verified.jsonl
   │  negatives  (optional) embed all chunks, attach hard negatives per question
   ▼
dataset.jsonl   ← final
```

Every stage reads/writes JSONL and is **independently resumable**: re-running a stage
skips items already in its output file (crash-safe, append-only).

---

## Install

```bash
pip install -r requirements.txt
# optional backends:
pip install anthropic                 # native Anthropic
pip install sentence-transformers     # hard-negative mining
```

## Input format

One JSON object per line (`.jsonl`), a `.json` array, or a directory of either:

```json
{"file_name": "doc_a.txt", "index": 0, "raw_text": "…"}
```

`index` is the chunk's position in its document; neighbours are `index ± 1`.

---

## Choosing the LLM: API vs vLLM (same code path)

Generation and verification talk to an **OpenAI-compatible** endpoint, so the *only*
difference between "I already have a model via API" and "I'll run vLLM separately" is
the `base_url`/`model` in your config — no code changes.

### (A) You already have a model via API
```yaml
llm:
  backend: openai
  base_url: https://api.your-provider.com/v1   # OpenAI, DeepSeek, Together, OpenRouter, Mistral, a gateway…
  model: <provider-model-id>
  api_key_env: OPENAI_API_KEY
```
```bash
export OPENAI_API_KEY=sk-...
```
(For native Anthropic instead, set `backend: anthropic`, `api_key_env: ANTHROPIC_API_KEY`.)

### (B) Run vLLM separately
```bash
pip install vllm
MODEL=Qwen/Qwen2.5-72B-Instruct TP=2 bash scripts/serve_vllm.sh   # exposes http://localhost:8000/v1
```
```yaml
llm:
  backend: openai
  base_url: http://localhost:8000/v1
  model: Qwen/Qwen2.5-72B-Instruct
  api_key_env: OPENAI_API_KEY          # any value; vLLM ignores it
```

**Model recommendation (Russian):** for both generation and judging, prefer a strong
multilingual instruct model — e.g. `Qwen/Qwen2.5-72B-Instruct` (or 32B for smaller
GPUs), `Llama-3.1-70B-Instruct`, or a top hosted API model. Quality of the *judge*
matters most for dataset cleanliness; use the strongest model you can there.

---

## Run

```bash
cp config.example.yaml config.yaml      # edit paths + llm
python run_pipeline.py all --config config.yaml

# or stage by stage (each resumable):
python run_pipeline.py windows  --config config.yaml
python run_pipeline.py generate --config config.yaml
python run_pipeline.py verify   --config config.yaml
python run_pipeline.py negatives --config config.yaml   # optional
python run_pipeline.py stats    --config config.yaml
```

**Dry run with no model/GPU** (uses the offline `mock` backend):
```bash
python run_pipeline.py all --config config.example.yaml --backend mock \
  --chunks tests/sample_chunks.jsonl --out-dir out_demo
```

---

## Output schema (`out/dataset.jsonl`)

```json
{
  "id": "w_7c8dd3bee49d__c0",
  "question": "…",
  "answer": "…",
  "gold_chunk_ids": ["doc_b.txt::1", "doc_b.txt::2"],
  "file_name": "doc_b.txt",
  "question_type": "multi_hop | aggregation | comparison | condition",
  "num_gold": 2,
  "window_chunk_ids": ["doc_b.txt::1", "doc_b.txt::2", "doc_b.txt::3"],
  "hard_negative_ids": ["other::5", "…"],
  "verification": { "minimality": {...}, "groundedness": {...} },
  "generation_model": "…",
  "judge_model": "…"
}
```

`gold_chunk_ids` is the **minimal verified** set. `window_chunk_ids` is the full context
the question was generated from (a superset of gold). `chunk_id == "{file_name}::{index}"`.

---

## Scaling to your corpus (60,855 chunks / 8,257 docs)

- `windows.target_windows` caps how many windows you build (default 5,000). Increase for
  a larger dataset; set `0` to use all eligible seeds.
- `max_windows_per_file` keeps long documents from dominating.
- Throughput is governed by `llm.max_concurrency` (in-flight requests) — raise it for a
  fast API or a well-provisioned vLLM server.
- Cost ≈ `target_windows × (1 generation + ~2 judge calls)`. Disable a judge in
  `verify:` to trade quality for cost, but the minimality judge is what guarantees the
  multi-hop property — keep it on.

## Tests

```bash
python -m pytest tests/ -q
```
Covers loading, filtering, window contiguity, JSON extraction, and a full mock-backed
generate→verify run asserting the ≥2-gold invariant.
