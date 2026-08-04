# SID — synthetic task factory for agentic retrieval

First working version of the pipeline described in
`synthetic_data_plan_sid1_v0.8`. It produces, from a chunked corpus:

```
(question, versioned index, minimal-sufficient gold set,
 coverage + complexity labels, optionally injected distractors)
```

It is a **separate pipeline** from the one in `README.md`. That one generates
questions over neighbouring chunks and verifies them; this one builds a task
*environment*: a mutable index, a measured retrieval difficulty per task, and a
gold set that is minimal by construction rather than by silver labelling.

```bash
# offline dry run — no model, no GPU, no API key
python scripts/make_sid_demo_corpus.py tests/sample_corpus_sid.jsonl
python run_sid.py all --config config_sid.yaml --backend mock \
    --corpus tests/sample_corpus_sid.jsonl --out-dir out_sid_demo

# real run
cp config_sid.yaml config_mine.yaml     # point paths.corpus at your chunks
export OPENAI_API_KEY=... GIGACHAT_AUTH_KEY=...
python run_sid.py all --config config_mine.yaml
```

---

## Pipeline

```
S0 compat    index/unit compatibility, available fields, v0 manifest
   ↓
S1 mine      entity ↔ chunk subgraphs (rare bridging entities, v0 only)
   ↓
S3 facts     atomic facts with verbatim spans, cached per chunk
   compose   1-of-N questions per coverage cell, from different subgraphs
   ↓
S4 gates     G_BROAD + G_REACH (retrieval-only) → 1-of-N selection → G_SOLVE
   ↓
S5 minimize  G_MIN (leave-one-fact-out) → G_REP (fact groups)
   ↓
§7.1 density τ_sim / τ_low percentiles, corpus norm, per-task injection budget
   ↓
S6 distract  transplant → perturb → generate, verified, injected: v0 → v1
   ↓
S7 isolate   cross-task isolation ON THE POST-INJECTION INDEX
   ↓
S8 export    final pool, train/holdout split, datamix stats
```

Every stage reads and writes JSONL and resumes by skipping what it already
decided, so a crashed run costs only the stage it died in. Stages can be run
individually: `python run_sid.py gates --config …`.

### Why the order is what it is

- **Cheap gates before expensive ones.** `G_BROAD` and `G_REACH` need no LLM
  call at all — `G_REACH` probes with the fact's own verbatim span and the
  normalised paraphrase that S3 already produced. So they run on the full
  1-of-N batch, and only the surviving winner pays for a critic.
- **Minimisation before injection.** Minimality is a property of the labels;
  injection does not change it, so `G_MIN`/`G_REP` never re-run.
- **Isolation after injection, never before.** A distractor built for task A
  lives in the shared index and is a candidate in the results for B. If it
  happens to be a valid alternative path for B, that is exactly the labelling
  hole isolation exists to close, and a check on v0 cannot see it.

---

## What the gates enforce

| Gate | Check | Rejects |
|---|---|---|
| `G_BROAD` | the whole question as ONE query must not return the whole gold set | trivial tasks |
| `G_REACH` | every gold chunk reachable by at least one probe | tasks above the environment's ceiling → `environment_ceiling_pool.jsonl` |
| `G_SOLVE` | answer uniquely derivable from the gold facts; question leaks neither the answer nor the intermediate entities | unsolvable, distorted, self-answering |
| `G_MIN` | leave-one-**fact**-out, greedy, re-checking every survivor after each removal | bloated gold sets |
| `G_REP` | every chunk that states a surviving fact joins that fact's group | nothing — it builds labels |

`G_BROAD` and `G_REACH` are the two ends of one interval: not trivial, not
unreachable.

**Minimisation is per fact, not per chunk.** Redundancy lives between the
chunks of one fact, not between facts. If a fact is covered by `{c₁, c₂, c₃}`,
leave-one-chunk-out finds every single one individually unnecessary and strips
the whole group. Facts are atomic by construction, so leave-one-fact-out cannot
do that. When a fact is removed, `hop_depth` is recomputed — the hop it carried
was not load-bearing, and that axis is what difficulty binning reads.

**Fact groups are never collapsed to one representative.** The other members
stay in the index and will be retrieved; a rollout that returned an equally
valid member would score zero for a correctly found fact.

---

## Reading the output

`out_sid/tasks.jsonl` — one record per task, matching plan §10. Splits are in
`split_train.jsonl` / `split_holdout.jsonl`: holdout is stratified over
mechanic × difficulty with the hard tail up-weighted, and no train task shares a
gold chunk with a holdout task.

There is **no SFT/RL split**. That boundary is drawn by what each half will be
used for, and nothing in this pipeline collects trajectories yet — so it would
be drawn on no evidence, and drawing it early costs tasks, since enforcing
disjoint gold sets across it discards whichever side loses the overlap. Split
the pool when there is a trainer to split it for.

`out_sid/stats.json` carries the numbers that decide things:

| Field | What it decides |
|---|---|
| `gold.share_singleton_groups` | **< 0.95 ⇒ NDCG must be computed over fact groups**, with `B_i` counted in groups. At chunk granularity the metric penalises correct behaviour. The field `gold.ndcg_granularity` states the verdict. |
| `complexity.fused_gap_share` | the datamix balancing axis; target is 30/40/30 low/mid/high |
| `complexity.lexicon_arm_size` | tasks with `lex_gap` high **and** `fused_gap` high — the only population where a `lexicon` field in `<state>` can show an effect. If it is empty, that hypothesis is untestable on this corpus, and `dense_gap` says why. |
| `distractors.share_L3` | > 0.10 means full generation is being used as a crutch |
| `distractors.share_tasks_with_empty_L2_band` | > 0.25 means the corpus is too sparse for this class of task — revisit subgraph selection rather than flooding the index with synthetic text |
| `gates.funnel` / `gates.by_mechanic` | low `G_REACH` pass-rate with the rest normal ⇒ the environment's ceiling; `G_REACH` fine but `G_SOLVE`/`G_BROAD` sagging ⇒ the generator. Also doubles as the cell × corpus feasibility matrix. |
| `density.density_median_all` vs `density_median_reach` | how far `G_REACH` biases the pool |

Expect a low yield. The plan budgets 30–40% through the funnel; generate with a
×3 margin.

---

## Distractors

Volume is normalisation, not amplification:

```
n_distractors = clip(density_median − density_current, 0, n_max)
```

The point is to pull sparse outliers up to *this corpus's* median, not to build
an artificial difficulty race — otherwise distractors quietly become a
difficulty knob and leave-one-corpus-out transfer degrades.

Three modes, by measured neighbourhood density: at or above the median, nothing
is injected; below it, top up to the median; far below, top up and mark the task
`sparse_origin: true` so the ablation can check whether the policy behaves
differently there.

The cascade differs by how much *real* text survives, not by "modified vs not":

| Level | Source | Modification |
|---|---|---|
| L1 transplant | real chunk from **outside** the gold neighbourhood | surface entities swapped for the neighbourhood's |
| L2 perturb | real chunk from the `[τ_low, τ_sim)` band | one discriminating attribute perturbed → near-miss |
| L3 generate | corpus template | full generation — fallback only |

A real chunk taken *unchanged* cannot be a distractor: if it already satisfies
`sim > τ_sim` it is already counted in `density_v0`, so a copy is either a no-op
or a content-hash duplicate.

Every candidate must clear five checks, cheap first: content hash, answer not
present, (L2 only) the perturbed attribute appears in the question text, the
candidate actually lands in the gold neighbourhood, and finally a critic
confirming it is not a valid alternative path. The fourth check matters more
than it looks — a near-miss only teaches discrimination if the feature
separating it from gold is observable through the question; otherwise no query
can tell them apart and NDCG punishes what the observation does not determine.

**Injection markers never reach the agent.** Injected chunks get a natural
`{file_name}::{index}` inside a real document; `synthetic`, `injected_for_task`,
cascade level and distractor type live in `injection_ledger.jsonl`, which is
ours. `corpus_injected.jsonl` is the agent-visible delta and carries none of it.
A leaked marker teaches "synthetic → ignore", which is a shortcut, not a skill.

---

## Where this departs from the plan, and why

| Plan | v1 | Reason |
|---|---|---|
| NER model, validated on 200 chunks | pattern extractor (proper nouns, quoted names, codes, dates, amounts) behind the same interface | the miner needs *rare repeated surface forms that bridge chunks*; a model to serve and validate buys little here. Swapping one in is local to `entities.py`. |
| τ_idf = 75th percentile of the corpus idf distribution | same percentile, taken over **bridge-eligible** entities (`df` between `min_co_occurrence` and `τ_df`) | over every surface form the corpus is dominated by hapax entities that all carry the maximum idf, so the percentile collapses onto `log(N/1)` and admits only `df = 1` entities — which `co_occurrence ≥ 2` then rejects, leaving nothing |
| incremental composition, base → +hop → +constraint, `max_compose_iters = 4` | one-shot composition + repair against the critic's objection, `max_compose_iters = 2` | same corrective signal, a fraction of the calls |
| `N = 6` candidates per cell | `N = 3` (configurable) | cost; the mechanism is what matters at v1 |
| LLM proposes submechanics per corpus | fixed grounded list per mechanic | they are local diversity, not cells; a corpus that cannot support a cell fails its gates and shows up in `gate_stats` |
| dual-critic on the pilot | implemented, off by default (`gates.dual_critic`) | it is a one-off purchase of information, per the plan itself |
| teacher trajectories, SFT filtering by teacher recall | **not implemented** | needs the RL harness — four tools, the `<state>` format, per-episode doc_id remapping. `export.py` produces exactly the pool those would be collected on. |
| SFT / RL pool split (§9.4) | **not implemented** — train/holdout only | the split is defined by downstream use; with no trajectories it would be arbitrary, and its disjoint-gold requirement silently discards tasks |
| `share_singleton_groups` decides NDCG granularity | measured and reported with the verdict; the metric itself lives in the trainer | out of this repo's scope |

`fused_gap` bins, the `lexicon` arm rule (`lex_gap` high **and** `fused_gap`
high), the `min`-over-gold density rule, the `max`-over-groups gap rule, the
reach-conditioned density median, and the §7.6 sampled reachability
re-measurement are all implemented as specified.

---

## Cost

Per candidate that reaches the expensive stages: 1 composition + 1 critic
(+1 if the repair fires) + up to `rep_max_judges_per_fact` entailment calls per
surviving fact + 1 distractor-check per candidate distractor + 1 isolation call.
Fact extraction is per *chunk* and cached, so a chunk shared by several
subgraphs is paid for once. The cheap gates and all three gap measurements cost
zero LLM calls.

Throughput is governed by `llm.max_concurrency` and `embed.max_concurrency`.

## Tests

```bash
python -m pytest tests/test_sid.py -q
```

25 tests, all offline on the mock backend: the BM25 branch, entity mining, the
gates (including that `G_BROAD` really rejects a question that retrieves its own
gold, and that `G_MIN` stops at the fact floor), gap aggregation, density
percentiles, distractor verification, marker containment, MinHash de-duplication,
a full end-to-end run and resume idempotency.
