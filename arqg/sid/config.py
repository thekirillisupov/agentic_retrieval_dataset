"""Configuration for the SID pipeline.

Reuses the existing ``arqg`` LLM / embeddings / filter configs so a single YAML
can drive both pipelines, and adds the SID-specific stage knobs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any

from ..config import FilterConfig, LLMConfig, RetrieveConfig, _merge_dataclass


@dataclass
class FacetConfig:
    """Which per-chunk facets are *surfaced*, and where (see scoping.py).

    `Chunk.meta` used to reach exactly two places: the scope key S1 groups on
    and S0's inventory report. Everything else in the pipeline — both branches
    of the retriever and every prompt — was blind to it, which is a problem the
    moment a corpus's identity lives in its facets rather than in its prose. A
    zakupki notice is distinguished from ten thousand near-identical ones by
    its region, customer and ОКПД2 code; if the index cannot see them, a fact
    paraphrased with them cannot retrieve its own chunk (G_REACH), and if the
    composer cannot see them it writes questions the environment cannot answer.

    The two switches are deliberately separate but meant to move together.
    `in_prompts` without `in_passage` is the harmful combination: the LLM
    starts phrasing facts and questions with facet vocabulary that exists in no
    indexed text, which makes G_REACH strictly worse and lets G_BROAD pass for
    the wrong reason — the query misses the gold because part of it addresses
    nothing at all. `run_sid.py` warns when a config asks for that.

    Empty `fields` = the whole feature is off and every stage behaves exactly
    as it did before, which is what corpora whose facets are already in the
    title or the text (ckr) want.
    """
    #: Facet keys to surface, in the order they are rendered. A builtin chunk
    #: attribute (title/document_id/file_name) is accepted too, though the
    #: title has its own switch in `embed.embed_with_title`. See S0's
    #: `index_fields.yaml` -> `meta_fields` for what a corpus actually carries.
    fields: list[str] = field(default_factory=list)
    #: Render them into the passage BOTH index branches hold (dense + BM25).
    in_passage: bool = True
    #: Show them to the LLM: fact extraction, composition, the G_SOLVE critic
    #: and the G_REP entailment judge all see the same header.
    in_prompts: bool = True
    #: Human labels for the header, `{field: label}`; the key itself is used
    #: when absent (`region: Регион`, `okpd2_code: ОКПД2`).
    labels: dict[str, str] = field(default_factory=dict)
    #: Long free-text facets (a customer's full legal name) are truncated, so
    #: one verbose field cannot outweigh the passage it is supposed to label.
    max_value_chars: int = 120

    def signature(self) -> dict[str, Any]:
        """What changes the rendered passage — folded into the dense cache key
        so editing this block re-embeds instead of silently reusing vectors
        built from a different string."""
        return {"fields": list(self.fields), "in_passage": self.in_passage,
                "labels": dict(self.labels), "max_value_chars": self.max_value_chars}


@dataclass
class MiningConfig:
    """S1 — entity ↔ chunk bipartite mining (plan §3)."""
    # entity is "niche enough": idf above this percentile of the corpus idf
    # distribution (plan §3.2 starts at the 75th percentile)
    idf_percentile: float = 75.0
    max_document_frequency: int = 40      # τ_df — entity must not be ubiquitous
    min_co_occurrence: int = 2            # entity must bridge >= 2 chunks
    min_chunks: int = 2                   # subgraph size bounds
    max_chunks: int = 4
    max_subgraphs_per_entity: int = 2
    max_subgraphs_per_file: int = 6       # keep large documents from dominating
    target_subgraphs: int = 2000          # 0 = unlimited
    cross_document_bonus: bool = True     # prefer subgraphs spanning >1 document
    # strict version: drop single-document subgraphs entirely. A question whose
    # gold sits in one document can often be answered by reading on rather than
    # issuing a second query, which is not the behaviour we are training.
    require_cross_document: bool = False
    seed: int = 11

    # ---- scope selection (see scoping.py) ---------------------------------- #
    # Mine within a *scope* — a group of chunks sharing one field's value —
    # instead of over the whole corpus. `scope_field` names the field: a
    # builtin chunk attribute (`title`, `document_id`, `file_name`) or a key in
    # `Chunk.meta` that the corpus's metadata sidecar carries (see S0's
    # `index_fields.yaml` -> `meta_fields` for what is actually available).
    # `scope_strategy` says how a value becomes a scope key: `"path"` reads it
    # as a `/`-separated breadcrumb and groups by folder (the original
    # title-scoping behaviour, and the default — an existing config keeps
    # mining exactly as before); `"exact"` groups chunks sharing the value
    # verbatim, the right shape for a flat categorical facet (zakupki's
    # `region`, `customer`, `okpd2_code`, `law`, `year`, ...) that is not a
    # path at all.
    scope_field: str = "title"
    scope_strategy: str = "path"          # "path" | "exact"
    # `"path"`-only: `None` disables scoping and restores the plain global
    # search; chunks whose value is too shallow to scope fall back to it
    # either way. 0 = immediate parent folder (siblings), 1 = one level up
    # (cousins). Ignored by `"exact"`, which has no notion of depth.
    path_scope_gap: int | None = 1
    min_scope_depth: int = 2              # "path" only: a 1–2 segment scope is a whole domain
    min_scope_chunks: int = 2
    # An entity in more than this fraction of the scope's chunks is the folder's
    # *subject*, not a bridge between two of its documents ("Эквайринг" inside
    # the эквайринг folder). This is what replaces global rarity as the
    # discrimination test once the search is confined to a folder.
    scope_df_ratio: float = 0.5
    # Global τ_idf collapses under scoping: a globally rare entity (df <= 2)
    # lands twice in the same folder only by coincidence, which concentrates the
    # whole pool in a handful of folders. Inside a scope the floor is therefore
    # a *separate*, much lower percentile; τ_df still applies, and the global
    # idf still orders candidates rarest-first.
    scope_idf_percentile: float = 0.0
    max_subgraphs_per_path: int = 8
    # ... and not all of them on the same kind of bridge. Whole folders of
    # defect cards share nothing but dates, and six date bridges from one folder
    # are one question six times. A *share* of the folder's budget rather than
    # an absolute count, because the two failure modes are not symmetric: an
    # absolute cap of 2 also punishes a folder whose twenty bridges are twenty
    # different people, which is twenty different questions. Type monoculture
    # only implies question monoculture where the type fixes the question shape.
    max_bridge_type_share: float = 0.5
    # Two chunks of ONE document are a real second query only if reading that
    # document would not hand the agent both for free. That takes two
    # conditions, not one: the document must be long enough to be worth
    # navigating at all, and the gap between the chunks wide enough. Positions
    # 2 and 5 of a six-chunk page are one read apart; positions 40 and 300 of a
    # 900-chunk page are different sections.
    same_doc_min_chunks: int = 20
    min_index_gap: int = 8

    # ---- doc2doc bridging (see simbridge.py) ------------------------------ #
    # A second bridge channel for the folders an entity bridge cannot reach.
    # Off by default: it needs the dense index, which the entity channel does
    # not, so a corpus that mines fine without an embedder keeps doing so.
    sim_bridge: bool = False
    # "related above the corpus background" — a percentile of the same pairwise
    # similarity sample §7.1 fits τ_sim on, so the two agree by construction.
    sim_bridge_low_percentile: float = 95.0
    # The upper bound is a *rank*, not a cosine. On `ckr` the cosine
    # distribution is compressed exactly where the ceiling has to sit (p96 =
    # 0.54, p98 = 0.80), so an absolute threshold there is unstable, while the
    # rank is scale-free and is what actually predicted the outcome: with the
    # partner inside the top-3 neighbours G_BROAD passed 0.22 of the time
    # against 0.55 beyond rank 50, because one query already returns both.
    sim_bridge_exclude_top_k: int = 3
    # Similarity pairs are tried low-first: within a folder topicality is
    # already guaranteed by the scope, so the remaining axis is co-retrievability
    # and less of it is better.
    sim_bridge_max_pairs_per_scope: int = 8


@dataclass
class TaxonomyConfig:
    """S2 — coverage cells (plan §4)."""
    # A1 mechanics to generate; empty = all six
    mechanics: list[str] = field(default_factory=list)
    negation_rate: float = 0.20           # plan §4.1 wants >= 15%
    # 1-of-N local diversity (plan §4.5). N candidates from *different*
    # subgraphs with *different* submechanics; the best `keep_per_batch` go on.
    candidates_per_cell: int = 3
    keep_per_batch: int = 1
    # How the batch's survivor is chosen. `"hardest"` takes the largest
    # `fused_gap`, which is a difficulty filter rather than a diversity one and
    # pulls against `export.target_fused_gap_share`: the "low" bin is starved by
    # construction, and the datamix then reports a skew the selector caused.
    # `"datamix"` picks the survivor whose bin is furthest below its target
    # share, so the pool the export has to balance is already balanced.
    batch_selection: str = "datamix"      # datamix | hardest
    seed: int = 17


@dataclass
class ComposeConfig:
    """S3 — facts and composition (plan §5)."""
    max_facts_per_chunk: int = 4
    min_facts_per_question: int = 2
    max_facts_per_question: int = 5
    max_compose_iters: int = 2            # plan says 4; 2 is enough for v1 cost
    verbatim_match: bool = True           # harness check: span occurs in chunk
    # A similarity-bridged subgraph carries no discrete anchor of its own, so
    # the composer has nothing to build the link *on* and will invent one. The
    # check is deferred to here rather than paid for at S1: the facts say what
    # the chunks actually have in common, and they are extracted either way.
    require_shared_anchor_for_sim: bool = True


@dataclass
class GatesConfig:
    """S4/S5 — gates (plan §6, §7.0)."""
    top_k: int = 10                       # top-k used by every gate probe
    # G_BROAD: reject when the whole question as one query already returns the
    # entire gold set (trivial task).
    run_broad: bool = True
    # G_REACH: every gold chunk must be retrievable by at least one probe.
    run_reach: bool = True
    # What that probe is made of. The fact's `verbatim_span` is by construction
    # an exact substring of the chunk it has to retrieve, so BM25 returns it
    # almost always and the gate answers a question nobody asked: an agent
    # never holds the gold's own wording. `"paraphrase"` probes with
    # `fact_normalized` — the closest thing to a query the agent could form —
    # which is what makes G_REACH a ceiling measurement rather than a
    # self-retrieval check, and halves its embedding bill. `"verbatim"` and
    # `"both"` (either probe suffices) keep the old behaviour available.
    reach_probe: str = "paraphrase"       # paraphrase | verbatim | both
    # G_SOLVE dual-critic — pilot only (plan §6). Off by default: one critic.
    dual_critic: bool = False
    drop_on_disagreement: bool = True
    # G_MIN: leave-one-fact-out minimisation.
    run_min: bool = True
    min_facts_after_min: int = 2          # drop tasks that collapse to one fact
    # G_REP: expand each surviving fact to every chunk that states it.
    run_rep: bool = True
    rep_top_k: int = 8                    # candidates retrieved per fact
    rep_max_judges_per_fact: int = 5      # cap entailment calls
    rep_min_score: float = 0.55           # dense prefilter before judging


@dataclass
class DensityConfig:
    """§7.1 — neighbourhood density and its corpus norm."""
    tau_sim_percentile: float = 95.0
    tau_low_percentile: float = 80.0
    sample_chunks: int = 600              # chunks sampled for the pairwise τ estimate
    pseudo_gold_sets: int = 2000          # bootstrap sets for density_median_all
    min_reach_tasks_for_median: int = 300  # below this, use the provisional median
    seed: int = 29


@dataclass
class DistractorConfig:
    """§7.2–7.5 — conditional injection."""
    enabled: bool = True
    n_max: int = 15                       # cap per task
    low_threshold_ratio: float = 0.34     # density < ratio*median => sparse_origin
    min_l2: int = 2                       # plan: >= 2 L2 distractors per task
    l1_pool: int = 40                     # candidate donor chunks sampled per task
    l2_pool: int = 20
    allow_l3: bool = True
    max_candidates_per_task: int = 24     # generation budget before giving up
    require_l2_attribute_in_question: bool = True   # §7.5 p.4
    require_neighborhood_hit: bool = True           # §7.5 p.5
    reach_recheck_sample: int = 500       # §7.6 sampled G_REACH re-run (0 = off)


@dataclass
class IsolationConfig:
    """S7 — cross-task isolation on the post-injection index (plan §8)."""
    enabled: bool = True
    top_k: int = 10
    judge_top_n: int = 4                  # non-gold hits handed to the judge
    judge_alternative_paths: bool = True  # set False for a cheap id-only check


@dataclass
class ExportConfig:
    """S8-lite — final pool, train/holdout split, datamix stats (plan §9.3, §9.5).

    No SFT/RL split: that boundary is drawn by what each half is used for, and
    nothing here collects trajectories yet.
    """
    holdout_size: int = 300
    dedup_threshold: float = 0.8          # MinHash Jaccard over question shingles
    fused_gap_bins: dict[str, float] = field(default_factory=lambda: {
        # upper bound of each bin; "high" takes the rest
        "low": 0.33,
        "mid": 0.66,
    })
    # The datamix the pool is aimed at (plan §9.3). Read both by the S8 report
    # and by S4's 1-of-N selection, so the target the stats grade against is the
    # one the selection was steering towards.
    target_fused_gap_share: dict[str, float] = field(default_factory=lambda: {
        "low": 0.30, "mid": 0.40, "high": 0.30,
    })
    seed: int = 31


@dataclass
class SidPaths:
    corpus: str = "data/chunks.jsonl"     # v0 corpus
    # Optional per-chunk metadata sidecar, one record per chunk keyed by
    # `chunk_id`, merged into `Chunk.meta` at load time (see corpus.py). Empty
    # = no sidecar; a corpus that inlines extra fields on the record itself
    # needs none — `load_chunks` already keeps those on `.meta`. zakupki's
    # `build_zakupki_corpus.py merge` writes exactly this shape.
    meta: str = ""
    out_dir: str = "out_sid"

    def _p(self, name: str) -> str:
        return os.path.join(self.out_dir, name)

    # S0
    @property
    def compat_report(self) -> str: return self._p("index_compat_report.json")
    @property
    def index_fields(self) -> str: return self._p("index_fields.yaml")
    @property
    def manifest(self) -> str: return self._p("index_manifest.json")
    # S1
    @property
    def subgraphs(self) -> str: return self._p("subgraphs.jsonl")
    # S3
    @property
    def facts(self) -> str: return self._p("facts.jsonl")
    @property
    def candidates(self) -> str: return self._p("candidates.jsonl")
    # S4/S5
    # Decision logs record EVERY candidate a stage looked at, pass or fail.
    # Resuming off the survivor file alone would re-process rejects — and for a
    # 1-of-N batch that means a runner-up gets promoted on the second run,
    # silently inflating the pool.
    @property
    def gate_decisions(self) -> str: return self._p("gate_decisions.jsonl")
    # Candidates that already cleared G_BROAD/G_REACH (embedding-only) and won
    # their 1-of-N batch, cached so a re-run that only has the LLM reachable
    # (e.g. embeddings need one network path, the LLM gateway a different one)
    # never has to repeat the embedding calls to get back to G_SOLVE.
    @property
    def gate_winners(self) -> str: return self._p("gate_winners.jsonl")
    @property
    def minimize_decisions(self) -> str: return self._p("minimize_decisions.jsonl")
    @property
    def isolation_decisions(self) -> str: return self._p("isolation_decisions.jsonl")
    @property
    def gated(self) -> str: return self._p("gated.jsonl")
    @property
    def gate_stats(self) -> str: return self._p("gate_stats.json")
    @property
    def ceiling_pool(self) -> str: return self._p("environment_ceiling_pool.jsonl")
    @property
    def minimized(self) -> str: return self._p("minimized.jsonl")
    # S6
    @property
    def density_stats(self) -> str: return self._p("density.json")
    @property
    def densified(self) -> str: return self._p("densified.jsonl")
    @property
    def injection_ledger(self) -> str: return self._p("injection_ledger.jsonl")
    @property
    def injected_corpus(self) -> str: return self._p("corpus_injected.jsonl")
    @property
    def injected_tasks(self) -> str: return self._p("injected.jsonl")
    # S7
    @property
    def isolation_report(self) -> str: return self._p("isolation_report.json")
    @property
    def isolated(self) -> str: return self._p("isolated.jsonl")
    # S8
    @property
    def tasks(self) -> str: return self._p("tasks.jsonl")
    @property
    def stats(self) -> str: return self._p("stats.json")

    def split(self, name: str) -> str:
        return self._p(f"split_{name}.jsonl")

    def dense_dir(self, version: str) -> str:
        return os.path.join(self.out_dir, "index", version)


@dataclass
class SidConfig:
    corpus_name: str = "corpus"
    language: str = "ru"
    taxonomy_version: str = "v0.8"

    llm: LLMConfig = field(default_factory=LLMConfig)
    judge: LLMConfig = field(default_factory=lambda: LLMConfig(temperature=0.0))
    embed: RetrieveConfig = field(default_factory=RetrieveConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    facets: FacetConfig = field(default_factory=FacetConfig)

    mining: MiningConfig = field(default_factory=MiningConfig)
    taxonomy: TaxonomyConfig = field(default_factory=TaxonomyConfig)
    compose: ComposeConfig = field(default_factory=ComposeConfig)
    gates: GatesConfig = field(default_factory=GatesConfig)
    density: DensityConfig = field(default_factory=DensityConfig)
    distractors: DistractorConfig = field(default_factory=DistractorConfig)
    isolation: IsolationConfig = field(default_factory=IsolationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    paths: SidPaths = field(default_factory=SidPaths)

    rrf_k: int = 60           # RRF constant for the hybrid fusion
    fusion_candidates: int = 100   # per-branch depth before fusion
    log_level: str = "INFO"

    # ---- loading --------------------------------------------------------- #
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SidConfig":
        cfg = cls()
        # a top-level `sid:` block is accepted so one YAML can hold both pipelines
        d = dict(d or {})
        if "sid" in d and isinstance(d["sid"], dict):
            merged = {k: v for k, v in d.items() if k != "sid"}
            merged.update(d["sid"])
            d = merged
        for f in fields(cls):
            if f.name not in d or d[f.name] is None:
                continue
            val, cur = d[f.name], getattr(cfg, f.name)
            if hasattr(cur, "__dataclass_fields__") and isinstance(val, dict):
                setattr(cfg, f.name, _merge_dataclass(cur, val))
            else:
                setattr(cfg, f.name, val)
        cfg._resolve_secrets()
        cfg._warn_on_facet_asymmetry()
        return cfg

    @classmethod
    def load(cls, path: str | None) -> "SidConfig":
        data: dict[str, Any] = {}
        if path:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def _warn_on_facet_asymmetry(self) -> None:
        """Facets in the prompts but not in the index is the one combination
        that makes things worse rather than differently good.

        The LLM starts phrasing facts and questions with facet vocabulary; the
        retriever holds passages in which that vocabulary does not occur. So
        G_REACH falls (the paraphrase probes tokens no document has) *and*
        G_BROAD rises (the question misses the gold for the same reason and
        looks non-trivial), which moves both ends of the interval §6 measures
        and makes the funnel unreadable. Not fatal, so not an error — a
        deliberate ablation is a legitimate reason to run it.
        """
        from ..utils import log
        f = self.facets
        if f.fields and f.in_prompts and not f.in_passage:
            log.warning(
                "facets: %d field(s) go to the prompts but not into the index "
                "(facets.in_passage=false). The composer will name attributes "
                "no passage carries — expect G_REACH to fall and G_BROAD to "
                "pass for the wrong reason.", len(f.fields))

    def _resolve_secrets(self) -> None:
        for llm in (self.llm, self.judge):
            if not llm.api_key:
                llm.api_key = os.environ.get(llm.api_key_env, "") or llm.default_dummy_key
            env_url = os.environ.get("ARQG_BASE_URL")
            if env_url:
                llm.base_url = env_url
