"""Configuration model. Loaded from YAML with environment-variable overrides
for secrets (API keys) and endpoints so the same config works against either a
hosted API or a locally-served vLLM instance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class LLMConfig:
    # backend: "openai" (OpenAI-compatible incl. vLLM), "gateway" (mTLS/custom HTTP),
    #          "anthropic", or "mock"
    backend: str = "openai"
    model: str = "Qwen/Qwen2.5-72B-Instruct"
    base_url: str = "http://localhost:8000/v1"   # vLLM default; gateway: full POST URL
    api_key_env: str = "OPENAI_API_KEY"           # name of env var holding the key
    api_key: str = ""                             # filled from env at load time
    temperature: float = 0.4
    max_tokens: int = 1536
    request_timeout: float = 120.0
    max_retries: int = 5
    max_concurrency: int = 16                     # in-flight requests
    # vLLM served models accept arbitrary keys; "EMPTY" is the conventional placeholder
    default_dummy_key: str = "EMPTY"
    # gateway backend: client-cert auth + custom JSON body (see config.example.yaml)
    cert_file: str = ""
    key_file: str = ""
    verify_ssl: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterConfig:
    """Seed-chunk eligibility. Junk chunks are kept in the store (so they remain
    valid neighbours for context) but are never used as window seeds/gold."""
    min_chars: int = 200
    max_chars: int = 4000          # avoid giant un-split blobs as seeds
    min_words: int = 25
    min_alpha_ratio: float = 0.5   # fraction of letters among non-space chars
    min_cyrillic_ratio: float = 0.3


@dataclass
class WindowConfig:
    window_sizes: list[int] = field(default_factory=lambda: [2, 3, 4])
    window_size_weights: list[float] = field(default_factory=lambda: [0.5, 0.35, 0.15])
    max_window_chars: int = 9000
    max_windows_per_file: int = 6  # diversity: cap dominance of huge documents
    target_windows: int = 5000     # total windows to build (0 = all eligible)
    stride: int = 1                # step between consecutive window seeds in a file
    seed: int = 13


@dataclass
class GenerateConfig:
    enabled: bool = True           # neighbour-window multi-hop generator
    questions_per_window: int = 1
    require_multi_chunk: bool = True
    min_gold_chunks: int = 2
    # Question style mix: style name -> sampling weight. Styles mimic real user
    # phrasing (see arqg/prompts.py STYLES). One style is sampled per question,
    # deterministically per window, and recorded in the dataset.
    styles: dict[str, float] = field(default_factory=lambda: {
        "simple_user": 0.45,
        "novice": 0.20,
        "expert": 0.20,
        "search_query": 0.15,
    })
    style_seed: int = 17


@dataclass
class DocUnitConfig:
    """How to turn a whole document into generation unit(s)."""
    max_doc_chars: int = 30000     # cap a unit's size (huge docs are split into spans)
    max_doc_chunks: int = 40       # cap a unit's chunk count
    min_unit_chunks: int = 1       # keep units with at least this many chunks
    max_units_per_file: int = 4    # how many spans to keep from a very large document
    target_units: int = 2000       # total units to build (0 = all)
    seed: int = 23


@dataclass
class DocGenConfig:
    """SECOND, separate generation process: simple/hard questions over a whole
    document with NO limit on the number of anchor passages.

    * simple → answer fully contained in ONE passage (gold = 1)
    * hard   → answer requires combining MANY passages across the document
               (gold >= hard_min_gold, no upper bound)

    Output format is identical to the neighbour generator (keeps gold_chunk_ids).
    """
    enabled: bool = False
    questions_per_unit: int = 1
    # difficulty mix (sampled per unit)
    difficulty_weights: dict[str, float] = field(default_factory=lambda: {
        "simple": 0.5,
        "hard": 0.5,
    })
    simple_max_gold: int = 1       # passages a "simple" question may rest on
    hard_min_gold: int = 2         # minimum anchors for a "hard" question
    hard_max_gold: int = 0         # 0 = unlimited anchor passages
    # reuse the same user-phrasing styles as the neighbour generator
    styles: dict[str, float] = field(default_factory=lambda: {
        "simple_user": 0.45,
        "novice": 0.20,
        "expert": 0.20,
        "search_query": 0.15,
    })
    style_seed: int = 29
    units: DocUnitConfig = field(default_factory=DocUnitConfig)


@dataclass
class VerifyConfig:
    enabled: bool = True
    run_groundedness: bool = True
    run_minimality: bool = True
    judge: LLMConfig = field(default_factory=LLMConfig)  # may differ from generator
    drop_if_single_chunk_sufficient: bool = True
    require_standalone: bool = True
    require_specific: bool = True


@dataclass
class CollectConfig:
    """Collect-all-positives validation (the final step).

    Each verified question is decomposed into atomic *clues*; you retrieve
    top-k passages per clue across the whole corpus; an entailment judge then
    keeps every passage that actually states the clue's fact. All such passages
    (plus the original gold) become the question's positive set, so
    near-duplicate sources are not mislabelled as negatives.
    """
    enabled: bool = False
    top_k: int = 20                  # passages to retrieve per clue (spec for you)
    require_original_gold: bool = True   # original gold always counts as positive
    max_positives_per_clue: int = 0  # 0 = unlimited
    # clue generation uses the main `llm`; entailment uses this judge (often a
    # strong, cheap, deterministic model).
    judge: "LLMConfig" = field(default_factory=lambda: LLMConfig(temperature=0.0))


@dataclass
class RetrieveConfig:
    """Embedding index over the corpus + dense retrieval, used to fill the
    collect-all-positives retrieval round automatically (no manual step)."""
    enabled: bool = True
    backend: str = "gigachat"      # "gigachat" | "mock"
    model: str = "GigaEmbeddings-3B-2025-09"
    base_url: str = "https://gigachat.devices.sberbank.ru/api/v1/embeddings"
    # Auth: a static bearer token if you have one, else OAuth client-credentials.
    token_env: str = "GIGACHAT_TOKEN"          # static "Bearer <token>"
    oauth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    auth_key_env: str = "GIGACHAT_AUTH_KEY"    # base64(client_id:secret) for OAuth Bearer
    scope: str = "GIGACHAT_API_PERS"
    oauth_refresh_interval: float = 600.0      # re-fetch OAuth token every N seconds (0 = expiry only)
    verify_ssl: bool = True        # GigaChat needs the Russian Min-Digit CA; see README
    ca_bundle: str = ""            # path to a CA bundle (.pem) if verify_ssl
    # GigaEmbeddings is asymmetric: queries get an instruction prefix, passages don't.
    query_instruction: str = (
        "Instruct: Given a web search query, retrieve relevant passages "
        "that answer the query\nQuery: ")
    embed_with_title: bool = True  # prepend the passage title before embedding
    batch_size: int = 32
    max_input_chars: int = 8192    # truncate a single passage if the API rejects it (0 = off)
    max_concurrency: int = 8
    min_request_interval: float = 0.0   # min seconds between embedding API calls (0 = off)
    request_timeout: float = 60.0
    max_retries: int = 5
    top_k: int = 0                 # 0 = use each request's top_k (= collect.top_k)
    index_dir: str = ""            # default: <out_dir>/index ; embeddings are cached here
    rebuild_index: bool = False    # force re-embedding the corpus


@dataclass
class NegativesConfig:
    enabled: bool = False
    embedding_model: str = "intfloat/multilingual-e5-large"
    top_k: int = 5
    exclude_same_file: bool = False  # if True, only cross-document negatives
    device: str = "cpu"
    batch_size: int = 64


@dataclass
class PathsConfig:
    chunks: str = "data/chunks.jsonl"
    index: str = ""                # corpus override; when set, used instead of chunks
    out_dir: str = "out"

    @property
    def corpus(self) -> str:
        """Chunked corpus for windows, embedding index, and retrieval."""
        return self.index or self.chunks

    @property
    def windows(self) -> str:
        return os.path.join(self.out_dir, "windows.jsonl")

    @property
    def docunits(self) -> str:
        return os.path.join(self.out_dir, "docunits.jsonl")

    @property
    def candidates(self) -> str:
        return os.path.join(self.out_dir, "candidates.jsonl")

    @property
    def verified(self) -> str:
        return os.path.join(self.out_dir, "verified.jsonl")

    @property
    def clues(self) -> str:
        return os.path.join(self.out_dir, "clues.jsonl")

    @property
    def retrieval_requests(self) -> str:
        # WHAT I GIVE YOU: one query per clue to retrieve top-k passages for.
        return os.path.join(self.out_dir, "retrieval_requests.jsonl")

    @property
    def retrieval_results(self) -> str:
        # WHAT YOU RETURN: top-k passages per clue (see README format).
        return os.path.join(self.out_dir, "retrieval_results.jsonl")

    @property
    def collected(self) -> str:
        return os.path.join(self.out_dir, "collected.jsonl")

    @property
    def dataset(self) -> str:
        return os.path.join(self.out_dir, "dataset.jsonl")

    def items_source(self) -> str:
        """Most-downstream item file that exists: collected > verified.
        Used by negatives/finalize so they pick up expanded positives."""
        return self.collected if os.path.exists(self.collected) else self.verified


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    docgen: DocGenConfig = field(default_factory=DocGenConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    collect: CollectConfig = field(default_factory=CollectConfig)
    retrieve: RetrieveConfig = field(default_factory=RetrieveConfig)
    negatives: NegativesConfig = field(default_factory=NegativesConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    log_level: str = "INFO"

    # ---- loading -------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        cfg = cls()
        for f in fields(cls):
            if f.name not in d or d[f.name] is None:
                continue
            val = d[f.name]
            cur = getattr(cfg, f.name)
            if hasattr(cur, "__dataclass_fields__") and isinstance(val, dict):
                setattr(cfg, f.name, _merge_dataclass(cur, val))
            else:
                setattr(cfg, f.name, val)
        cfg._resolve_secrets()
        return cfg

    @classmethod
    def load(cls, path: str | None) -> "Config":
        data: dict[str, Any] = {}
        if path:
            import yaml  # local import so PyYAML isn't required for mock/tests
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def _resolve_secrets(self) -> None:
        for llm in (self.llm, self.verify.judge, self.collect.judge):
            if not llm.api_key:
                llm.api_key = os.environ.get(llm.api_key_env, "") or llm.default_dummy_key
            env_url = os.environ.get("ARQG_BASE_URL")
            if env_url:
                llm.base_url = env_url


def _merge_dataclass(obj: Any, overrides: dict[str, Any]) -> Any:
    for f in fields(obj):
        if f.name not in overrides or overrides[f.name] is None:
            continue
        val = overrides[f.name]
        cur = getattr(obj, f.name)
        if hasattr(cur, "__dataclass_fields__") and isinstance(val, dict):
            setattr(obj, f.name, _merge_dataclass(cur, val))
        else:
            setattr(obj, f.name, val)
    return obj
