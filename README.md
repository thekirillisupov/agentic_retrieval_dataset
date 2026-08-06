# Agentic Retrieval — Synthetic Dataset Pipeline

Generate a high-quality synthetic dataset for **agentic / multi-hop retrieval** from a
chunked, document-based knowledge base (Russian). Each produced item is:

```json
{"question": "...", "answer": "...", "gold_chunk_ids": ["file::3", "file::4"], ...}
```

The defining property: **every question requires combining information from several
*neighbouring* chunks**, and the gold set is **verified to be minimal and strictly
necessary** — so a retriever genuinely has to find *all* the gold chunks, not just one.

> **Looking for the SID task factory?** The pipeline built from
> `synthetic_data_plan_sid1_v0.8` — entity subgraph mining, an A1 coverage
> taxonomy, five gates, measured retrieval gaps, a *mutable* index with
> conditional distractor injection and post-injection isolation — lives beside
> this one and is documented in **[README_SID.md](README_SID.md)**
> (`run_sid.py`, `config_sid.yaml`, `arqg/sid/`). The two share the LLM,
> embedding and IO layers but are otherwise independent.

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
| Exam-style phrasing unlike real users | **Style sampling**: each question is generated in one of several realistic user styles (simple user, novice, expert, search query), with few-shot good/bad examples in the prompt |
| Question copies rare phrases verbatim (retrieval becomes trivial lexical match) | Prompt enforces paraphrasing in the asker's own words |
| Junk chunks (headers, page numbers, min=1 char) | Seed eligibility filter (length / word / Cyrillic ratios); junk stays usable only as *neighbour context* |
| Giant un-split chunks (max≈39k) / huge docs dominating | `max_chars` seed cap, `max_window_chars`, `max_windows_per_file` |

The generator and judge can be **different models** (a stronger judge reduces self-bias).

---

## Two generation processes (split by config)

There are two independently-switchable generators. Both emit the **same record
format** (`question` + `gold_chunk_ids`) and share one `verify`/`finalize`/`negatives`
tail, so you can run either or both and they merge into one dataset.

| Generator | Config block | Unit of context | Gold (anchor) chunks |
|---|---|---|---|
| **#1 neighbour multi-hop** | `generate` | tight run of 2–4 *adjacent* chunks | always ≥ 2, minimised |
| **#2 document simple/hard** | `docgen` | a *whole document* (split into spans if huge) | **no limit** — `simple` = 1 passage, `hard` = many across the doc |

Toggle with `generate.enabled` / `docgen.enabled`. Each dataset item records its
`profile` (`neighbor_multihop` / `doc_simple_hard`) and `difficulty` (`simple`/`hard`).

## Pipeline stages

```
chunks.jsonl
   ├─ generator #1 (generate.enabled) ────────────────────────────┐
   │     windows    contiguous neighbour windows (2–4 chunks)      │
   │       ▼                                                       │
   │     windows.jsonl                                             │
   │       ▼ generate    multi-chunk question (gold ≥ 2)           │
   │                                                               ▼
   ├─ generator #2 (docgen.enabled) ──────────────────────►  candidates.jsonl
   │     docunits   whole-document units (spans for huge docs)     ▲
   │       ▼                                                       │
   │     docunits.jsonl                                            │
   │       ▼ gen-docs    simple (1 passage) OR hard (many) ────────┘
   ▼
candidates.jsonl
   │  verify     per-item policy: judge minimality (→ minimal gold) + groundedness
   ▼              (simple items keep gold=1; hard/neighbour items are minimised to ≥2)
verified.jsonl
   │  clues             decompose each question into atomic facts -> retrieval_requests.jsonl
   │  retrieve          embed + index the corpus, top-k passages per clue -> retrieval_results.jsonl
   │  collect-positives  entailment-judge each passage; keep those that state the fact
   ▼
collected.jsonl   (gold preserved + positive_chunk_ids / positive_groups added)
   │  negatives  (optional) embed all chunks, attach hard negatives per question
   ▼              (validated positives are excluded from negatives)
dataset.jsonl   ← final
```

Every stage reads/writes JSONL and is **independently resumable**: re-running a stage
skips items already in its output file (crash-safe, append-only).

### Collect-all-positives (final validation)

The same fact often lives in several near-duplicate documents in a corpus. If only
the one chunk we generated from is marked gold, a retriever that finds an equally
valid duplicate gets unfairly penalised. This step fixes the labels:

1. `clues` — each verified question is decomposed into **atomic clues** (self-contained
   facts, one per hop), written to `retrieval_requests.jsonl` (one query per clue).
2. `retrieve` — the corpus is embedded once (GigaEmbeddings), indexed (cached on disk),
   and the top-k passages per clue are written to `retrieval_results.jsonl`.
3. `collect-positives` — an entailment judge checks each retrieved passage against the
   clue's fact; every passage that truly states it (plus the original gold) becomes a
   positive. Output keeps `gold_chunk_ids` and adds `positive_chunk_ids` (the full
   relevant set) and `positive_groups` (per-clue alternates — score a multi-hop hit as
   "≥1 chunk from each group").

With `collect.enabled` + `retrieve.enabled` (default), `all` runs steps 1–3 automatically:
```bash
export GIGACHAT_TOKEN=...            # or GIGACHAT_AUTH_KEY for OAuth (see below)
python run_pipeline.py all --config config.yaml
```

**Corpus schema for retrieval** — records may carry the extra fields; `chunk_id` is
still `"{file_name}::{index}"`, while `document_id`/`title` are kept as metadata and the
title is (optionally) prepended before embedding:
```json
{"raw_text": "…", "document_id": "D123", "title": "…", "file_name": "doc_a.txt", "index": 3}
```

**GigaChat embeddings** (`retrieve.backend: gigachat`, model `GigaEmbeddings-3B-2025-09`):
provide auth via **either** a static token (`GIGACHAT_TOKEN`) **or** OAuth client
credentials (`GIGACHAT_AUTH_KEY` = base64 `client_id:secret`; the token is fetched from
`oauth_url` and auto-refreshed). GigaChat's TLS uses the Russian Trusted CA — set
`verify_ssl: false` or point `ca_bundle` at the CA `.pem`. Queries are sent with the
GigaEmbeddings instruction prefix, passages without it. The embedding matrix is cached
under `index_dir` (default `<out_dir>/index`) and reused until the corpus or model changes.

**Prefer to retrieve yourself?** Set `retrieve.enabled: false`; `all` then stops after
`clues`, and you fill `retrieval_results.jsonl` by hand:
```json
{"clue_id": "w_ab12__c0__k0", "passages": [{"chunk_id": "doc_c.txt::9", "score": 0.81}]}
```
Then run `collect-positives` and `finalize`. `chunk_id` (`"{file_name}::{index}"`) is all
that's needed. Any `clue_id` with no entry keeps its original gold as the only positive.

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

## Corpus source: ЕИС / zakupki.gov.ru (44-ФЗ)

Procurement documentation is the densest **near-duplicate** material available in
Russian. Every notification is generated from the same regulated template, so
thousands of documents differ only in the customer, the object of purchase, the
deadlines and the amounts — exactly the distractor structure SID otherwise has to
synthesise (`L1_transplant` / `L2_perturbed`). A retriever that survives this
corpus is not matching on surface form.

`scripts/build_zakupki_corpus.py` has two halves that are deliberately separate,
because the machine with ЕИС access usually is not the machine that builds
datasets:

| | needs | does |
|---|---|---|
| `fetch` | token **and** a Russian IP | calls the data services, saves raw archives + a resume manifest |
| `build` / `stats` | nothing | archives → chunks JSONL + a near-duplicate report |

### Access (2025+ — every older guide is wrong here)

* **The FTP dump is gone.** `ftp://fz223free@ftp.zakupki.gov.ru`, which every
  Habr/vc.ru recipe uses, was shut down on **2025-01-01**. Those recipes do not
  work, at all.
* Everything now goes through the data services at
  `https://int44.zakupki.gov.ru/eis-integration/services/`, and every call needs
  credentials:
  * **физлицо** — a token from <https://zakupki.gov.ru/pmd/auth/welcome> (login
    via Госуслуги → «Регистрация нового потребителя машиночитаемых данных» →
    «Физическое лицо»), sent in the `individualPerson_token` SOAP header.
    Endpoint `getDocsIP`. This is the path implemented here.
  * **юрлицо** — requests signed with a qualified ЭЦП against `getDocsLE2`.
    The ГОСТ-signing infrastructure is *not* implemented; only the token flow is.
* The token is needed **again** on the archive download, as an HTTP header — the
  `archiveUrl` you get back is not public.
* **Geo-block.** zakupki.gov.ru resets TLS from non-Russian address space, so
  `fetch` must run from a Russian IP whatever your credentials say. `build` never
  touches the network.
### No token? Third-party dumps (`from-table`)

A ЕИС token needs a Госуслуги-verified account and does not arrive the same day,
so there is a second path that works immediately. Several extracts of
zakupki.gov.ru are published elsewhere and still download without credentials —
these three were checked by hand and have a profile in `arqg/zakupki/tabular.py`:

| profile | source | size | licence | content |
|---|---|---|---|---|
| `kaggle_biggest` | `kaggle.com/datasets/dadalyndell/russian-biggest-government-procurement-contracts` | 0.7 MB | **ODC PDDL** (public domain) | 4.5k tenders over 500 mln ₽, 2019-2023, 84 regions, winner + final price |
| `zakupkihack` | `kaggle.com/datasets/mrmorj/zakupkihack-recsys` | 153 MB | **not declared** — check before redistributing | ~0.5M+ lots, 2019-2020, 44/223-ФЗ, ОКПД2 + item descriptions; procurement numbers anonymised |
| `hf_medicines` | `huggingface.co/datasets/zavzyatiy/medicines_from_zakupki_gov_ru` | 63 MB | **Apache-2.0** | drug procurement only, 2015-2025, НМЦК + contract price |

Kaggle serves these over its public API without a key:

```bash
curl -L -o dump.zip \
  https://www.kaggle.com/api/v1/datasets/download/dadalyndell/russian-biggest-government-procurement-contracts
unzip dump.zip

python scripts/build_zakupki_corpus.py from-table \
    --input tender_data.csv --out-dir data/zakupki
```

The profile is detected from the header; anything else goes through
`--profile generic` with explicit `--column canonical=source` pairs. Unmapped
columns are ignored rather than guessed at — a mis-mapped ИНН would quietly
poison entity mining downstream. Reading `.xlsx` needs `openpyxl`.

**What you give up.** These are *record cards, not documents*: a row carries the
notification's key fields, not its text, so it renders as three to five short
sections where a real XML document gives a dozen. Chunks are ~180-270 characters
against ~730 for the XML path, and there is no обеспечение/требования/сроки
prose. The near-duplicate property survives intact — better than intact, in fact,
because the cards are pure template:

| corpus | docs | chunks | avg chunk | exact dup | structural dup | Jaccard |
|---|---|---|---|---|---|---|
| `kaggle_biggest` | 4 519 | 22 595 | 204 | 39.1 % | 82.4 % | 0.89 |
| `zakupkihack` (200k rows) | 200 000 | 612 246 | 268 | 15.1 % | 71.0 % | 0.94 |
| `hf_medicines` (30k rows) | 30 000 | 149 987 | 177 | 30.0 % | 89.3 % | 0.95 |

So: use a dump to develop and calibrate the pipeline today, and re-run the XML
path once a token arrives if you need the full document text. `from-table`
defaults to `--merge-below 0` because a card merged into one chunk can never form
a neighbour window; the command warns if any document ends up with a single
chunk.

Dead ends, so you don't retry them: the FTP dump (closed 2025-01-01),
`opendata.gov.ru` and `data.gov.ru` (unreachable), `api.clearspending.ru` (bare
nginx — the Госзатраты API is gone). Scraping the portal's server-rendered HTML
still works for anyone with a Russian IP, but it is not implemented here.

### Run

```bash
# 0. keep the live schema next to the data: element order is validated
#    positionally and the schemas are revised a few times a year
python scripts/build_zakupki_corpus.py xsd --out data/zakupki/getDocsIP.xsd

# 1. a month of electronic-auction notifications for two regions
#    (needs the token; see "No token?" above for the dump path)
export EIS_TOKEN=5d035886-...
python scripts/build_zakupki_corpus.py fetch \
    --regions 72,77 --doc-types epNotificationEF2020,epNotificationEOK2020 \
    --date-from 2025-06-01 --date-to 2025-06-30 --raw-dir data/zakupki/raw

# 2. build the corpus — offline, no token, resumable input
python scripts/build_zakupki_corpus.py build \
    --raw-dir data/zakupki/raw --out-dir data/zakupki

# 3. feed SID
python run_sid.py all --config config_sid.yaml \
    --corpus data/zakupki/zakupki_index.jsonl
```

`fetch` walks the region × document-type × day grid (the service's own
granularity is one day), paces itself against the per-consumer quota, skips
archives already in the manifest, and keeps going when a single cell fails.

### What `build` produces

XML → ordered, Russian-labelled sections → chunks in the pipeline's input format.
The parser is **schema-tolerant on purpose**: 44-ФЗ schemas are versioned
(`…EF2020`, `…EF2023`) and revised often, so instead of a per-type field map it
walks any document generically, translating tags through a dictionary and
de-camel-casing the ones it does not know. A schema revision costs you an uglier
label, not a dropped field. Signature and certificate blobs are discarded;
repeated elements (lots, positions, requirements) stay numbered so two positions
of the same notification remain distinguishable.

```
data/zakupki/zakupki_index.jsonl      chunks: file_name / index / raw_text / document_id / title
data/zakupki/zakupki_documents.jsonl  document-level dump (sections, metadata)
data/zakupki/zakupki_report.json      near-duplicate report
```

### The near-duplicate report

Two notions, and the difference between them is the point:

* **exact** — identical after whitespace/case normalisation. Verbatim boilerplate
  (единые требования к участникам, условия обеспечения). Genuinely
  indistinguishable; no retriever can be blamed for confusing them.
* **structural** — identical once every digit is masked. Same template,
  *different* customer, deadline and price. Telling those apart is the skill the
  dataset trains, so `structural_duplicates.share_of_chunks` is the number to
  look at when deciding whether a slice of ЕИС is worth generating over.

```
— выгрузка ЕИС —
документов:            60
чанков:                312
точные дубликаты:      3 групп, 39.1% чанков
шаблонные дубликаты:   40 групп, 84.6% чанков, средний Jaccard 0.66
```

---

## Third generator: MuSiQue → multi-hop dialogue

`scripts/build_musique_dialogues.py` builds a **conversational, anaphora-heavy**
retrieval set from [MuSiQue](https://huggingface.co/datasets/bdsaglam/musique)
instead of a raw corpus. MuSiQue ships, for every multi-hop question, a
`question_decomposition` — an ordered list of single-hop sub-questions where a
later hop references an earlier hop's answer with a `#k` token (`k` = 1-based hop
position), and every hop carries `paragraph_support_idx` (the one paragraph that
answers it). That is exactly the material for a dialogue with anaphora:

```
turn 1: <client> asks q1                         <bot> answers a1
turn 2: <client> asks q2, where "#1" is rewritten as a pronoun / definite
        description referring to a1 (the bot already said it)   <bot> a2
turn 3: ...
```

matching the app's dialogue format (`<client>…</client><bot>…</bot>…` ending on
the client's latest message). A 3- or 4-hop question yields 3–4 turns.

**One item per turn.** For turn `t` we emit a record whose `question` is the
transcript so far (ending on the client's latest message), `answer` is hop `t`'s
answer, and:

> **Frozen gold rule:** `gold_chunk_ids` for turn `t` is **only hop `t`'s
> supporting paragraph.** The bot has already uttered the earlier answers in the
> transcript, so the earlier hops' documents are redundant to a downstream
> consumer — hop `t`'s passage is the only thing "relevant to the latest
> message". Gold comes straight from `paragraph_support_idx`, no LLM verify step.

Each example's ~20 paragraphs (supporting **and** distractors) are written out as
the retrieval corpus (`chunk_id = "{example_id}::{paragraph_idx}"`), so the gold
ids point into a real pool with hard distractors for free.

The `#k` → anaphora rewrite reuses the same LLM client as the rest of the
pipeline (configure it via `--config`); `--anaphora heuristic` gives a
deterministic, model-free rewrite, and `--backend mock` runs fully offline.

```bash
# LLM-rewritten anaphora, pulling MuSiQue (answerable subset) from the HF hub
pip install datasets
export OPENAI_API_KEY=sk-...
python scripts/build_musique_dialogues.py --config config.yaml --out-dir out_mq

# offline dry run (no network / GPU) on a local MuSiQue dump
python scripts/build_musique_dialogues.py --backend mock \
    --input data/musique_ans_v1.0_train.jsonl --limit 50 --out-dir out_demo
```

Output (`out_mq/`): `musique_corpus.jsonl` (retrieval pool) and
`musique_dialogues.jsonl` (the dialogue items, **same schema as `dataset.jsonl`**
below, with per-turn provenance under `verification.musique`). Useful flags:
`--min-turn 2` (skip the standalone first hop), `--only-anaphora-turns` (keep only
turns that actually reference an earlier answer), `--config-name default` (include
unanswerable-derived examples; the builder still keeps only fully-supported ones).
The output is a valid `DatasetItem` file, so you can point the optional
`negatives` stage at it (`paths.chunks: out_mq/musique_corpus.jsonl`,
`paths.index`/item source = `musique_dialogues.jsonl`) to mine hard negatives, or
run `stats` on it.

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
python run_pipeline.py windows  --config config.yaml   # generator #1
python run_pipeline.py generate --config config.yaml
python run_pipeline.py docunits --config config.yaml   # generator #2
python run_pipeline.py gen-docs --config config.yaml
python run_pipeline.py verify   --config config.yaml
python run_pipeline.py negatives --config config.yaml  # optional
python run_pipeline.py stats    --config config.yaml
```

`all` runs whichever generators are enabled in config. To produce **only** the
simple/hard document dataset, set `generate.enabled: false` and `docgen.enabled: true`
(or just run the four `docunits → gen-docs → verify → finalize` stages).

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
  "question_type": "factoid | multi_hop | aggregation | comparison | condition",
  "question_style": "simple_user | novice | expert | search_query",
  "profile": "neighbor_multihop | doc_simple_hard",
  "difficulty": "simple | hard",
  "num_gold": 2,
  "window_chunk_ids": ["doc_b.txt::1", "doc_b.txt::2", "doc_b.txt::3"],
  "positive_chunk_ids": ["doc_b.txt::1", "doc_b.txt::2", "dup.txt::4"],
  "positive_groups": [{"clue": "…", "chunk_ids": ["doc_b.txt::1", "dup.txt::4"]}],
  "num_positives": 3,
  "hard_negative_ids": ["other::5", "…"],
  "verification": { "minimality": {...}, "groundedness": {...} },
  "generation_model": "…",
  "judge_model": "…"
}
```

`gold_chunk_ids` is the **minimal verified** set. `positive_chunk_ids` (present after
collect-all-positives) is the full set of acceptable relevant chunks incl. near-duplicate
sources — score retrieval against this. `window_chunk_ids` is the full context the
question was generated from (a superset of gold). `chunk_id == "{file_name}::{index}"`.

---

## Question styles (matching real user queries)

Real users don't ask exam questions. Each generated question is written in a sampled
style (recorded as `question_style`, weights configurable in `generate.styles`):

| Style | Sounds like | Default weight |
|---|---|---|
| `simple_user` | short colloquial question, support-chat / search-bar tone | 0.45 |
| `novice` | no domain terminology — describes things in everyday words | 0.20 |
| `expert` | precise domain phrasing | 0.20 |
| `search_query` | 3–8 word search-bar query, may lack a verb or "?" | 0.15 |

Sampling is deterministic per window (`style_seed`), so resumed runs keep the same mix.
The groundedness judge is explicitly told that colloquial/short phrasing is *not* a
defect, so simple questions aren't filtered out for style. Tune the weights to match
your production query logs; set a weight to `0` to disable a style.

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

`tests/test_zakupki.py` covers the ЕИС export offline: envelope tag order, SOAP
fault handling, nested-zip walking, XML → sections, chunk contiguity and the
duplicate report. The live call is not tested — zakupki.gov.ru is unreachable
from outside Russia, so that path has to be exercised by hand from a Russian IP.
