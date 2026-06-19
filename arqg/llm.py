"""Unified async LLM client.

One interface, three backends:

* ``openai``    — any OpenAI-compatible endpoint. This covers BOTH a hosted API
                  *and* a locally served vLLM instance (`vllm serve ...` exposes
                  exactly this API), so the pipeline code never branches on which
                  you use — only ``base_url`` / ``model`` change.
* ``gateway``   — direct HTTP POST with optional mTLS client certs and a custom
                  JSON body (for corporate gateways that are not OpenAI-compatible).
* ``anthropic`` — native Anthropic Messages API.
* ``mock``      — deterministic offline backend for tests / dry runs (no network).

All backends share concurrency limiting, retries with exponential backoff, and a
robust JSON extractor (models love to wrap JSON in prose or ```code fences```).
"""
from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Any, Callable

from .config import LLMConfig
from .utils import log

JsonObj = dict[str, Any]


class LLMError(RuntimeError):
    pass


def make_client(cfg: LLMConfig) -> "BaseLLM":
    backend = cfg.backend.lower()
    if backend in ("openai", "vllm", "openai-compatible"):
        return OpenAICompatLLM(cfg)
    if backend in ("gateway", "qwen_gateway"):
        return GatewayLLM(cfg)
    if backend == "anthropic":
        return AnthropicLLM(cfg)
    if backend == "mock":
        return MockLLM(cfg)
    raise ValueError(f"unknown LLM backend: {cfg.backend!r}")


class BaseLLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._sem = asyncio.Semaphore(cfg.max_concurrency)

    async def _raw_complete(self, system: str, user: str, **kw: Any) -> str:
        raise NotImplementedError

    async def complete(self, system: str, user: str, **kw: Any) -> str:
        """Concurrency-limited, retried text completion."""
        last: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                async with self._sem:
                    return await asyncio.wait_for(
                        self._raw_complete(system, user, **kw),
                        timeout=self.cfg.request_timeout,
                    )
            except Exception as e:  # noqa: BLE001 - backends raise varied types
                last = e
                delay = min(2 ** attempt, 30) + random.uniform(0, 1)
                log.warning("LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                            attempt + 1, self.cfg.max_retries, e, delay)
                await asyncio.sleep(delay)
        raise LLMError(f"LLM call failed after {self.cfg.max_retries} attempts: {last}")

    async def complete_json(self, system: str, user: str, **kw: Any) -> JsonObj:
        """Complete and parse a JSON object, with one corrective re-ask."""
        text = await self.complete(system, user, **kw)
        obj = extract_json(text)
        if obj is not None:
            return obj
        # one corrective attempt: restate the requirement tersely
        fix = (user + "\n\nВерни СТРОГО валидный JSON-объект и ничего больше.")
        text = await self.complete(system, fix, **kw)
        obj = extract_json(text)
        if obj is None:
            raise LLMError(f"could not parse JSON from model output: {text[:400]!r}")
        return obj

    async def aclose(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# OpenAI-compatible (hosted API *or* vLLM)
# --------------------------------------------------------------------------- #
class OpenAICompatLLM(BaseLLM):
    def __init__(self, cfg: LLMConfig):
        super().__init__(cfg)
        try:
            from openai import AsyncOpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install openai — required for the openai backend") from e
        self._client = AsyncOpenAI(
            base_url=cfg.base_url or None,
            api_key=cfg.api_key or cfg.default_dummy_key,
            max_retries=0,  # we handle retries ourselves
        )

    async def _raw_complete(self, system: str, user: str, **kw: Any) -> str:
        # response_format=json_object is honoured by OpenAI and recent vLLM;
        # harmless guidance otherwise.
        want_json = kw.pop("json", False)
        extra: dict[str, Any] = {}
        if want_json:
            extra["response_format"] = {"type": "json_object"}
        resp = await self._client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=kw.pop("temperature", self.cfg.temperature),
            max_tokens=kw.pop("max_tokens", self.cfg.max_tokens),
            **extra,
        )
        return resp.choices[0].message.content or ""

    async def complete_json(self, system: str, user: str, **kw: Any) -> JsonObj:
        kw.setdefault("json", True)
        return await super().complete_json(system, user, **kw)

    async def aclose(self) -> None:
        await self._client.close()


# --------------------------------------------------------------------------- #
# Gateway (mTLS / custom HTTP — not OpenAI SDK)
# --------------------------------------------------------------------------- #
def parse_chat_completion(data: dict[str, Any]) -> str:
    """Extract assistant text from an OpenAI-shaped chat completion JSON."""
    try:
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        if content:
            return content
        # Some Qwen gateways put text in reasoning_content when thinking is on.
        return msg.get("reasoning_content") or ""
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"unexpected gateway response shape: {data!r}") from e


class GatewayLLM(BaseLLM):
    """POST JSON directly to ``base_url`` with optional client-cert auth.

    Matches corporate gateways where curl works but the OpenAI SDK does not
    (mTLS, non-/v1 paths, Qwen-specific body fields via ``extra_body``).
    """

    def __init__(self, cfg: LLMConfig):
        super().__init__(cfg)
        if not cfg.base_url:
            raise ValueError("gateway backend requires llm.base_url (full POST URL)")
        try:
            import httpx
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install httpx — required for the gateway backend") from e
        cert: str | tuple[str, str] | None = None
        if cfg.cert_file and cfg.key_file:
            cert = (cfg.cert_file, cfg.key_file)
        elif cfg.cert_file:
            cert = cfg.cert_file
        self._client = httpx.AsyncClient(
            cert=cert,
            verify=cfg.verify_ssl,
            timeout=httpx.Timeout(cfg.request_timeout),
        )

    async def _raw_complete(self, system: str, user: str, **kw: Any) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        body: dict[str, Any] = {
            "messages": messages,
            "max_tokens": kw.pop("max_tokens", self.cfg.max_tokens),
            "temperature": kw.pop("temperature", self.cfg.temperature),
        }
        body.update(self.cfg.extra_body)

        resp = await self._client.post(self.cfg.base_url, json=body)
        if resp.status_code >= 400:
            raise LLMError(f"gateway HTTP {resp.status_code}: {resp.text[:500]}")
        return parse_chat_completion(resp.json())

    async def aclose(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------------- #
# Anthropic native
# --------------------------------------------------------------------------- #
class AnthropicLLM(BaseLLM):
    def __init__(self, cfg: LLMConfig):
        super().__init__(cfg)
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install anthropic — required for the anthropic backend") from e
        self._client = AsyncAnthropic(api_key=cfg.api_key, max_retries=0)

    async def _raw_complete(self, system: str, user: str, **kw: Any) -> str:
        kw.pop("json", None)
        resp = await self._client.messages.create(
            model=self.cfg.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=kw.pop("temperature", self.cfg.temperature),
            max_tokens=kw.pop("max_tokens", self.cfg.max_tokens),
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    async def aclose(self) -> None:
        await self._client.close()


# --------------------------------------------------------------------------- #
# Mock (offline, deterministic)
# --------------------------------------------------------------------------- #
class MockLLM(BaseLLM):
    """Returns canned, schema-valid JSON. Driven by an optional handler so tests
    can simulate generation and judging without a network."""

    def __init__(self, cfg: LLMConfig, handler: Callable[[str, str], JsonObj] | None = None):
        super().__init__(cfg)
        self.handler = handler

    async def _raw_complete(self, system: str, user: str, **kw: Any) -> str:
        if self.handler is not None:
            return json.dumps(self.handler(system, user), ensure_ascii=False)
        return json.dumps(_default_mock(system, user), ensure_ascii=False)


def _default_mock(system: str, user: str) -> JsonObj:
    # ids appear in the prompt as "[CHUNK <id>]"; capture real ids only
    # (skip the literal "<id>" placeholder that appears in the instructions).
    ids = re.findall(r"\[CHUNK ([^\]]+?)\]", user)
    ids = [i for i in ids if "::" in i]
    if '"supports"' in user:                 # clue-entailment judge
        return {"supports": True, "notes": "mock"}
    if '"clues"' in user:                    # clue decomposition
        return {"clues": [{"clue": f"mock факт по {cid}", "source_gold_ids": [cid]}
                          for cid in (ids or ["?::0"])]}
    if "necessary_chunk_ids" in user:        # minimality judge
        return {
            "necessary_chunk_ids": ids[:2],
            "answerable": True,
            "single_chunk_sufficient": False,
            "notes": "mock",
        }
    if '"supported"' in user or "ПРЕДЛОЖЕННЫЙ ОТВЕТ" in user:  # groundedness judge
        return {
            "supported": True,
            "answer_correct": True,
            "standalone": True,
            "specific": True,
            "answerable_from_world_knowledge": False,
            "notes": "mock",
        }
    return {
        "reasoning": "mock multi-hop spanning two chunks",
        "question": "Тестовый вопрос, объединяющий сведения из нескольких фрагментов?",
        "answer": "Тестовый ответ.",
        "required_chunk_ids": ids[:2],
        "question_type": "multi_hop",
        "self_contained": True,
    }


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> JsonObj | None:
    """Best-effort: parse a JSON object out of arbitrary model text."""
    if not text:
        return None
    text = text.strip()
    # direct
    obj = _try_load(text)
    if obj is not None:
        return obj
    # fenced
    m = _FENCE_RE.search(text)
    if m:
        obj = _try_load(m.group(1))
        if obj is not None:
            return obj
    # first balanced {...}
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        obj = _try_load(text[start:i + 1])
                        if obj is not None:
                            return obj
                        break
        start = text.find("{", start + 1)
    return None


def _try_load(s: str) -> JsonObj | None:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None
