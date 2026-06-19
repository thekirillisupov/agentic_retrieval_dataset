"""Embedding clients for dense retrieval.

* ``gigachat`` — Sber GigaEmbeddings via the GigaChat REST API. Supports either a
                 static bearer token or OAuth client-credentials (auto-refreshed).
                 Queries get GigaEmbeddings' instruction prefix; passages don't.
* ``mock``     — deterministic offline embeddings (hashed bag-of-words) so the
                 retrieve stage and tests run with no network.

All clients return L2-normalised float32 vectors, so a dot product is cosine
similarity.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import time
import uuid

import numpy as np

from .config import RetrieveConfig
from .utils import log


class EmbeddingsError(RuntimeError):
    pass


def make_embedder(cfg: RetrieveConfig) -> "BaseEmbedder":
    backend = cfg.backend.lower()
    if backend == "gigachat":
        return GigaChatEmbedder(cfg)
    if backend == "mock":
        return MockEmbedder(cfg)
    raise ValueError(f"unknown embeddings backend: {cfg.backend!r}")


def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype="float32")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class BaseEmbedder:
    def __init__(self, cfg: RetrieveConfig):
        self.cfg = cfg
        self._sem = asyncio.Semaphore(cfg.max_concurrency)

    def _format(self, texts: list[str], kind: str) -> list[str]:
        if kind == "query":
            return [self.cfg.query_instruction + t for t in texts]
        return texts

    async def _embed_batch(self, inputs: list[str]) -> list[list[float]]:
        raise NotImplementedError

    async def embed(self, texts: list[str], kind: str) -> np.ndarray:
        """Embed texts (kind: 'query' | 'passage'); returns normalised vectors."""
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        inputs = self._format(texts, kind)
        bs = max(1, self.cfg.batch_size)
        batches = [inputs[i:i + bs] for i in range(0, len(inputs), bs)]
        results: list[list[list[float]]] = [None] * len(batches)  # type: ignore

        async def run(i: int, batch: list[str]) -> None:
            async with self._sem:
                results[i] = await self._embed_batch(batch)

        await asyncio.gather(*(run(i, b) for i, b in enumerate(batches)))
        vecs = [v for batch in results for v in batch]
        return _normalize(np.asarray(vecs, dtype="float32"))

    async def aclose(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# GigaChat / GigaEmbeddings
# --------------------------------------------------------------------------- #
class GigaChatEmbedder(BaseEmbedder):
    def __init__(self, cfg: RetrieveConfig):
        super().__init__(cfg)
        try:
            import httpx
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install httpx — required for the gigachat embedder") from e
        verify: object = cfg.verify_ssl
        if cfg.verify_ssl and cfg.ca_bundle:
            verify = cfg.ca_bundle
        self._httpx = httpx
        self._client = httpx.AsyncClient(
            verify=verify, timeout=httpx.Timeout(cfg.request_timeout))
        self._static_token = os.environ.get(cfg.token_env, "").strip()
        self._oauth_token = ""
        self._oauth_exp = 0.0
        self._auth_lock = asyncio.Lock()

    async def _token(self) -> str:
        if self._static_token:
            return self._static_token
        # OAuth client-credentials with refresh ~30s before expiry
        async with self._auth_lock:
            if self._oauth_token and time.time() < self._oauth_exp - 30:
                return self._oauth_token
            auth_key = os.environ.get(self.cfg.auth_key_env, "").strip()
            if not auth_key:
                raise EmbeddingsError(
                    f"no token: set ${self.cfg.token_env} (static) or "
                    f"${self.cfg.auth_key_env} (OAuth Basic key)")
            resp = await self._client.post(
                self.cfg.oauth_url,
                headers={
                    "Authorization": f"Basic {auth_key}",
                    "RqUID": str(uuid.uuid4()),
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={"scope": self.cfg.scope},
            )
            resp.raise_for_status()
            data = resp.json()
            self._oauth_token = data["access_token"]
            # expires_at is epoch milliseconds
            self._oauth_exp = float(data.get("expires_at", 0)) / 1000.0 or (time.time() + 1500)
            return self._oauth_token

    async def _embed_batch(self, inputs: list[str]) -> list[list[float]]:
        last: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                token = await self._token()
                resp = await self._client.post(
                    self.cfg.base_url,
                    headers={"Authorization": f"Bearer {token}",
                             "Content-Type": "application/json"},
                    json={"model": self.cfg.model, "input": inputs},
                )
                if resp.status_code == 401 and not self._static_token:
                    self._oauth_token = ""        # force refresh and retry
                    raise EmbeddingsError("401 unauthorized — refreshing token")
                resp.raise_for_status()
                data = resp.json()["data"]
                data = sorted(data, key=lambda d: d.get("index", 0))
                return [d["embedding"] for d in data]
            except Exception as e:  # noqa: BLE001
                last = e
                delay = min(2 ** attempt, 30)
                log.warning("embeddings call failed (%d/%d): %s — retry in %ds",
                            attempt + 1, self.cfg.max_retries, e, delay)
                await asyncio.sleep(delay)
        raise EmbeddingsError(f"embeddings failed after {self.cfg.max_retries} tries: {last}")

    async def aclose(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------------- #
# Mock (offline, deterministic)
# --------------------------------------------------------------------------- #
class MockEmbedder(BaseEmbedder):
    """Hashed bag-of-words embeddings. Identical text -> identical vector, so
    overlapping texts score high — enough to exercise indexing/ranking offline."""

    DIM = 256

    def _format(self, texts: list[str], kind: str) -> list[str]:
        return texts   # no instruction prefix, so query==passage text matches exactly

    async def _embed_batch(self, inputs: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in inputs]

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.DIM, dtype="float32")
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            v[h % self.DIM] += 1.0
        return v.tolist()


def _tokens(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]
