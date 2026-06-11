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
class VerifyConfig:
    enabled: bool = True
    run_groundedness: bool = True
    run_minimality: bool = True
    judge: LLMConfig = field(default_factory=LLMConfig)  # may differ from generator
    drop_if_single_chunk_sufficient: bool = True
    require_standalone: bool = True
    require_specific: bool = True


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
    out_dir: str = "out"

    @property
    def windows(self) -> str:
        return os.path.join(self.out_dir, "windows.jsonl")

    @property
    def candidates(self) -> str:
        return os.path.join(self.out_dir, "candidates.jsonl")

    @property
    def verified(self) -> str:
        return os.path.join(self.out_dir, "verified.jsonl")

    @property
    def dataset(self) -> str:
        return os.path.join(self.out_dir, "dataset.jsonl")


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
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
        for llm in (self.llm, self.verify.judge):
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
