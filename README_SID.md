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
S1 mine      entity ↔ chunk subgraphs + doc2doc pairs, WITHIN a section scope
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

**What isolation drops, and what it merely measures.** Two tasks with the *same*
gold set are one task labelled twice: one survives, chosen by `task_id` so the
result does not depend on the order the pool arrives in. Everything else in a
result list that is not this task's gold — another task's gold included — is a
*candidate* shortcut and goes to the judge; only a verdict that it actually
yields this answer drops the task. Rejecting on the mere presence of another
task's gold is a different test: it fires on co-retrieval, which is the normal
state of affairs once S1 mines within a folder — the better the scoping, the
more neighbouring tasks retrieve each other — and it fires symmetrically, so
both members of a pair die instead of one. It also could never see the case it
was named for, since a gold chunk shared with another task is *this* task's gold
too and was filtered out before the check ran. `isolation_report.json` now
carries `gold_overlap`, so how much the pool shares gold stays measured.

---

## Where S1 looks: section scopes

In these corpora `title` is not a headline, it is the document's path in the
knowledge base:

```
Общее пространство ЦКР (УСО, ТП, УУП)/Дефекты/Дефекты СберБизнес/<документ>
```

The last segment names the document; the prefix is the folder it was filed in.
That prefix is the only *editorial* topical grouping the index carries, and S1
confines its search to it. Run over the whole corpus instead, "rare entity
shared by two chunks" is satisfied by coincidence far more often than by subject
matter — on `ckr`, **51% of globally-mined subgraphs shared nothing but the
corpus root**, held together by словоформы like «Перестал» or «Сторону». Scoped,
that is 3%, and the bridges become «национальное бюро кредитных историй»,
«объединенное кредитное бюро», actual defect codes and dates.

The scope is `title` minus its leaf minus `path_scope_gap` further levels — `0`
is siblings only, `1` also admits cousins. A chunk whose title is too shallow to
name a folder falls back to the unscoped global search, so a corpus with flat
titles behaves exactly as it did before scoping existed.

**Confining the search forces τ_idf to be reinterpreted.** Global rarity was
only ever a proxy for "this entity discriminates", and inside a folder it is the
wrong proxy: an entity rare enough to clear τ_idf (df ≈ 2) lands twice in the
*same* folder only by coincidence, which collapsed a 1585-scope corpus onto 122
folders. Discrimination inside a folder is local — an entity in two of a
folder's twelve chunks separates them whatever its global df, and one in *most*
of the folder's chunks is the folder's subject rather than a bridge («Эквайринг»
inside the эквайринг folder). So `scope_df_ratio` is the upper bound, τ_df stays
as the ubiquity ceiling, and global idf is demoted to the ordering key.

`index` earns a job too. `require_cross_document` asked the wrong question: this
corpus has documents of 900+ chunks, where positions 40 and 300 are different
sections that need a second query, while positions 2 and 5 of a six-chunk page
are one read apart. So a same-document pair needs **both** a document worth
navigating (`same_doc_min_chunks`) and a real gap (`min_index_gap`).

Two quotas keep a folder from dominating: `max_subgraphs_per_path`, and
`max_bridge_type_share` — a *share* of the folder's budget rather than an
absolute count, because the failure modes are not symmetric. A folder of defect
cards whose only bridges are dates yields one question six times, but a folder
whose twenty bridges are twenty different people yields twenty different
questions; an absolute cap punishes both, and 221 of 419 productive scopes on
`ckr` carry bridges of exactly one type.

The scope is deliberately **not** registered as an index tag on the entity graph.
As a tag it would bypass τ_idf through `is_tag`, inflate the df of every folder
member and let the extension step pull in an arbitrary sibling — the same
failure that already keeps `document_id` out. A folder is *where to look*, not
*what ties the chunks together*.

`stats.json` reports `sections.share_root_only`, so the pathology this removes
stays measured rather than assumed.

### Scoping by any metadata facet, not only the title breadcrumb

`title` is `ckr`'s *only* editorial grouping, but it need not be the only
grouping S1 knows how to use. `mining.scope_field` names the field a chunk is
grouped by — a builtin chunk attribute (`title`, `document_id`, `file_name`)
or a key in `Chunk.meta` — and `mining.scope_strategy` says how a value
becomes a scope key (see `arqg/sid/scoping.py`):

| `scope_strategy` | how a scope key is built | fits |
|---|---|---|
| `"path"` (default) | `/`-separated breadcrumb → its folder, `path_scope_gap` deep (see `sections.py`) | a hierarchical field like `ckr`'s `title` |
| `"exact"` | the field's value, verbatim | a flat categorical facet: region, customer, ОКПД2 code, law, year, price bucket, … |

`Chunk.meta` is populated one of two ways: `load_chunks` lifts any field
beyond the pipeline's five core ones straight off the corpus record, or — for
a corpus that keeps its facets in a separate file, like zakupki's
`*_meta.jsonl` — point `paths.meta` (or `run_sid.py --meta`) at it and
`SidCorpus.load` merges it in by `chunk_id`. Either way, `python run_sid.py
compat` reports every facet it found under `index_fields.yaml`'s
`meta_fields`, with coverage and cardinality, so picking a scope is reading a
report rather than guessing:

```bash
python scripts/build_zakupki_corpus.py merge --input data/zakupki/dumps/*.csv:auto \
    --name zakupki_all --out-dir data/zakupki
python run_sid.py compat --config config_sid.yaml \
    --corpus data/zakupki/zakupki_all.jsonl --meta data/zakupki/zakupki_all_meta.jsonl \
    --out-dir out_sid_zakupki
# inspect out_sid_zakupki/index_fields.yaml -> meta_fields, then set in the config:
#   mining.scope_field: region        # or customer / okpd2_code / law / year / price_bucket
#   mining.scope_strategy: exact
python run_sid.py all --config config_sid.yaml \
    --corpus data/zakupki/zakupki_all.jsonl --meta data/zakupki/zakupki_all_meta.jsonl \
    --out-dir out_sid_zakupki
```

An existing config that never set these two keys mines exactly as before —
the defaults are `scope_field: title`, `scope_strategy: path`.

### Where the entity bridge runs out: the doc2doc channel

Scoping fixes *where* to look; it does not help with folders where the entity
miner finds nothing to look *at*. On `ckr` only 283 of 800 scopes contain a rare
surface form repeated in two of their chunks, and the reason is not a threshold
being strict — of the 517 that yield nothing, **451 contain no repeated surface
form at all**, 45 no extracted entity at all, and only 12 are blocked by τ_df or
the folder-subject ratio. Lifting τ_df to infinity buys 352 of 800. Two chunks
about one subject written by two people share no surface form, and that is most
of a knowledge base.

So `mining.sim_bridge` adds a second channel that bridges a folder by doc2doc
similarity — the relation the retriever itself will be measured in. It reaches
598 of the 800 scopes, and the mined pool goes from ~1100 subgraphs to ~2900.

**Its upper bound is a rank, not a cosine.** Similarity does not tell good tasks
from bad: subgraphs that reached a task averaged 0.674 against 0.682 for those
that died in the gates. What it predicts is *triviality*, because two chunks
close enough are returned by one query and that is exactly what `G_BROAD`
rejects — with the partner inside the top-3 neighbours `G_BROAD` passed 0.22 of
the time and 0.13 of candidates became tasks, against 0.55 / 0.30 beyond
neighbour 50. An absolute cosine cannot carry that here: the distribution is
compressed precisely where the ceiling belongs (p96 = 0.54, p98 = 0.80), so one
percentile of movement is 0.18 of cosine, and a folder of near-identical defect
cards sits above any corpus-wide ceiling. Rank is scale-free and local, which is
what the failure mode is. The lower edge stays a corpus percentile of the same
pairwise sample §7.1 fits τ_sim on. Pairs are tried lowest-similarity first, so
the emitted pool sits just above the lower edge (median 0.520) and the rank
ceiling only catches the residue.

What this channel cannot supply is an **anchor**. An entity bridge names what
two chunks have in common; a similarity bridge only asserts that something does,
and a composer handed two fragments with nothing it can point at will invent the
link — the failure section scoping exists to remove. The check is therefore
deferred to S3 (`compose.require_shared_anchor_for_sim`): a similarity subgraph
survives only if the facts of two different chunks name the same thing, stemmed
so «дефекту» and «дефект» count as one. The facts are extracted either way, they
are what the composer will actually see, and they carry the paraphrase the raw
text does not — so the guard costs no call that was not already being made.

Entity bridges are spent before similarity ones inside each folder, and the
channel is scope-local: corpus-wide, doc2doc would reintroduce the coincidental
pairing scoping was added to remove, with no folder to bound it. `stats.json`
reports `bridge_kind` over the surviving tasks, so the share of the pool the new
channel actually carries through the funnel stays measured too.

**The breadcrumb also goes into the prompts.** These corpora are largely
markdown tables whose subject lives in the title, not in the cells — a chunk
reading `| Номер дефекта | DCBHCK6-16471 |` gives a fact extractor no way to
know which product the defect belongs to, and the composer then invents the link
between two such fragments. S3 passes the path as context to both. It is context
only: `verbatim_span` is still checked against the chunk text alone, so a fact
lifted from the heading is dropped rather than repaired. Since `embed_with_title`
is on, the retriever already sees the path, so a question that references a
section stays reachable for `G_REACH`.

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
| `bridge_kind` | which S1 channel the surviving tasks came from. Against the channel's share of the *mined* pool (logged by S1) it says whether the folders only doc2doc can reach produce tasks or merely candidates |
| `sections.share_root_only` | gold sets whose fragments share nothing but the corpus root — bridged by coincidence, so the composer had to invent the link. Globally-mined `ckr` sat at 0.51; anything non-trivial means section scoping is not reaching the pool. |

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

**An injected chunk is embedded exactly like a v0 one.** With `embed_with_title`
on, every corpus chunk enters the dense index as `title\ntext`; a distractor
inherits its donor's title and is embedded the same way, both for the §7.5
neighbourhood check and for the vector written to the index. Embedded as bare
text it would sit somewhere other than where a rebuild of v1 from
`corpus_injected.jsonl` puts it — so "does it land in the gold neighbourhood?"
would be answered about a vector nothing ever retrieves against.

S6 also **saves the index it mutated** under `index/v1/`, together with the
injected delta, every `50` tasks and at the end. The dense cache is keyed on the
corpus checksum, which every injection changes: without that save, S7 — and any
resumed `distract` — re-embeds the entire corpus for vectors the process is
already holding, which on a 60k-chunk corpus behind a rate-limited embedder is
hours of API calls per resume.

---

## Where this departs from the plan, and why

| Plan | v1 | Reason |
|---|---|---|
| NER model, validated on 200 chunks | pattern extractor (proper nouns, quoted names, codes, dates, amounts) behind the same interface | the miner needs *rare repeated surface forms that bridge chunks*; a model to serve and validate buys little here. Swapping one in is local to `entities.py`. |
| τ_idf = 75th percentile of the corpus idf distribution | same percentile, taken over **bridge-eligible** entities (`df` between `min_co_occurrence` and `τ_df`) | over every surface form the corpus is dominated by hapax entities that all carry the maximum idf, so the percentile collapses onto `log(N/1)` and admits only `df = 1` entities — which `co_occurrence ≥ 2` then rejects, leaving nothing |
| mine bridges over the whole corpus | mine **within a `title` section scope**; τ_idf inside a scope replaced by scope-local `scope_df_ratio` with τ_df as the ceiling | 51% of globally-mined `ckr` subgraphs shared only the corpus root and were bridged by словоформы. Keeping global τ_idf *and* scoping collapses the pool onto 122 of 1585 folders — see "Where S1 looks" |
| a shared entity is the only thing that can bridge two chunks | a second **doc2doc** channel, bounded above by neighbour rank, with the anchor required from the facts at S3 | an entity bridge reaches 283 of 800 folders, and 451 of the rest contain no repeated surface form at all rather than one a threshold rejected — see "Where the entity bridge runs out" |
| `require_cross_document` as the "not reading on" test | `same_doc_min_chunks` **and** `min_index_gap` | the question is whether reading the document hands the agent both chunks, which depends on the document: 900-chunk pages have genuine sections, six-chunk pages do not |
| chunk text is the only prompt input | the section breadcrumb is passed to S3 as context, never as a fact source | a bare markdown table row does not say which product it belongs to, and the composer fills that gap by inventing. `verbatim_span` is still matched against the chunk alone. |
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

43 tests, all offline on the mock backend: the BM25 branch, entity mining,
section scoping (that no subgraph is held together by the corpus root alone, that
a folder's own subject is refused as a bridge, that a short document cannot
bridge to itself, and that flat titles still mine), the doc2doc channel (that it
bridges a folder no entity can, that the rank ceiling and not the cosine is what
rejects a trivial pair, that entity bridges are spent first, and that a
similarity subgraph without a shared anchor in its facts never reaches the
composer), the gates (including that
`G_BROAD` really rejects a question that retrieves its own gold, and that `G_MIN`
stops at the fact floor), gap aggregation, density percentiles, distractor
verification, marker containment, MinHash de-duplication, a full end-to-end run
and resume idempotency.
