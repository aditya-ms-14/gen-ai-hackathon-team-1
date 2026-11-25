"""Provider clients: Gemini (REST), Groq and LM Studio (both OpenAI-compatible).

Each provider exposes one method, ``complete(...) -> str``, which returns the
model's raw text and raises ProviderError on any failure. Retry and fallback
decisions live in the router, not here.
"""

from __future__ import annotations

import json
import logging
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from .config import RouterConfig
from .errors import ProviderError

log = logging.getLogger("llm_router")


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        from datetime import datetime, timezone

        return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return None


class BaseProvider:
    name = "base"

    def __init__(self, model: str, timeout: float, client: httpx.Client | None = None) -> None:
        self.model = model
        self.timeout = timeout
        self._client = client or httpx.Client()

    def available(self) -> bool:
        """False when the provider is not configured (e.g. no API key)."""
        return True

    def complete(
        self,
        system: str,
        user: str,
        *,
        schema: dict | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        raise NotImplementedError

    # -- shared HTTP handling -------------------------------------------------
    def _request(self, method: str, url: str, headers: dict, payload: dict | None = None) -> Any:
        try:
            resp = self._client.request(method, url, headers=headers, json=payload, timeout=self.timeout)
        except httpx.TimeoutException:
            raise ProviderError(self.name, f"no response within {self.timeout:.0f}s", kind="timeout", retryable=True)
        except httpx.ConnectError as exc:
            raise ProviderError(self.name, f"cannot connect to {url} ({exc})", kind="unreachable", retryable=False)
        except httpx.HTTPError as exc:
            raise ProviderError(self.name, f"transport error: {exc}", kind="network", retryable=True)

        status = resp.status_code
        if status == 429:
            raise ProviderError(
                self.name, "rate limited / quota exceeded", kind="rate_limit", retryable=False,
                status=status, retry_after=_retry_after(resp),
            )
        if status in (401, 403):
            raise ProviderError(self.name, f"auth failed: {resp.text[:200]}", kind="auth", retryable=False, status=status)
        if status >= 500:
            raise ProviderError(self.name, f"server error {status}: {resp.text[:200]}", kind="server", retryable=True, status=status)
        if status >= 400:
            body = resp.text[:400]
            # Groq returns 400 json_validate_failed when the model produced invalid JSON:
            # that is a bad generation, worth retrying, not a bad request.
            if "json_validate_failed" in body:
                raise ProviderError(self.name, "model produced invalid JSON", kind="bad_output", retryable=True, status=status)
            raise ProviderError(self.name, f"HTTP {status}: {body}", kind="bad_request", retryable=False, status=status)
        try:
            return resp.json()
        except ValueError:
            raise ProviderError(self.name, "response was not JSON", kind="bad_output", retryable=True, status=status)


class GeminiProvider(BaseProvider):
    """Google Gemini via the generateContent REST endpoint."""

    name = "gemini"

    def __init__(self, cfg: RouterConfig, client: httpx.Client | None = None) -> None:
        super().__init__(cfg.gemini_model, cfg.gemini_timeout, client)
        self.api_key = cfg.gemini_api_key
        self.base_url = cfg.gemini_base_url.rstrip("/")
        self.thinking_budget = cfg.gemini_thinking_budget
        # Native schema enforcement; switched off automatically if the model rejects it.
        self.use_native_schema = True

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system, user, *, schema=None, temperature=0.0, max_tokens=4096) -> str:
        gen: dict[str, Any] = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if schema is not None:
            gen["responseMimeType"] = "application/json"
            if self.use_native_schema:
                gen["responseJsonSchema"] = schema
        if self.thinking_budget is not None:
            gen["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        url = f"{self.base_url}/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        try:
            data = self._request("POST", url, headers, payload)
        except ProviderError as exc:
            # Some models / API versions reject responseJsonSchema: retry once without it
            # (the schema is still in the prompt and output is still validated).
            if exc.kind == "bad_request" and "responseJsonSchema" in gen:
                log.warning("gemini rejected native JSON schema, retrying with prompt-only schema: %s", exc.message)
                self.use_native_schema = False
                return self.complete(system, user, schema=schema, temperature=temperature, max_tokens=max_tokens)
            raise

        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise ProviderError(self.name, f"no output ({reason})", kind="bad_output", retryable=False)
        cand = candidates[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            raise ProviderError(
                self.name, f"empty output (finishReason={cand.get('finishReason')})", kind="bad_output", retryable=True
            )
        return text


class OpenAICompatProvider(BaseProvider):
    """Any OpenAI-compatible /chat/completions server (Groq, LM Studio)."""

    # "json_object": provider only guarantees valid JSON, schema is in the prompt.
    # "json_schema": provider enforces the schema itself (LM Studio supports this).
    json_mode = "json_object"

    def __init__(self, model: str, timeout: float, base_url: str, api_key: str = "", client: httpx.Client | None = None):
        super().__init__(model, timeout, client)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _model_id(self) -> str:
        return self.model

    def complete(self, system, user, *, schema=None, temperature=0.0, max_tokens=4096) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        payload: dict[str, Any] = {
            "model": self._model_id(),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if schema is not None:
            if self.json_mode == "json_schema":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "strict": True, "schema": schema},
                }
            else:
                payload["response_format"] = {"type": "json_object"}

        data = self._request("POST", f"{self.base_url}/chat/completions", self._headers(), payload)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise ProviderError(self.name, f"unexpected response shape: {json.dumps(data)[:200]}", kind="bad_output", retryable=True)
        if not text.strip():
            raise ProviderError(self.name, "empty output", kind="bad_output", retryable=True)
        return text


class GroqProvider(OpenAICompatProvider):
    name = "groq"
    json_mode = "json_object"

    def __init__(self, cfg: RouterConfig, client: httpx.Client | None = None) -> None:
        super().__init__(cfg.groq_model, cfg.groq_timeout, cfg.groq_base_url, cfg.groq_api_key, client)

    def available(self) -> bool:
        return bool(self.api_key)


class LMStudioProvider(OpenAICompatProvider):
    """Local LM Studio server (Developer tab -> Start Server, default port 1234)."""

    name = "lmstudio"
    json_mode = "json_schema"

    def __init__(self, cfg: RouterConfig, client: httpx.Client | None = None) -> None:
        super().__init__(cfg.lmstudio_model, cfg.lmstudio_timeout, cfg.lmstudio_base_url, "", client)

    def _model_id(self) -> str:
        if not self.model:
            # No model configured: use whatever is loaded in LM Studio.
            data = self._request("GET", f"{self.base_url}/models", self._headers())
            models = [m.get("id") for m in data.get("data", []) if m.get("id")]
            chat_models = [m for m in models if "embed" not in m.lower()]
            if not chat_models:
                raise ProviderError(self.name, "no model loaded in LM Studio", kind="unreachable", retryable=False)
            self.model = chat_models[0]
            log.info("lmstudio: using loaded model %s", self.model)
        return self.model


PROVIDERS = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "lmstudio": LMStudioProvider,
}


def build_providers(cfg: RouterConfig, client: httpx.Client | None = None) -> list[BaseProvider]:
    out = []
    for name in cfg.provider_order:
        if name not in PROVIDERS:
            raise ValueError(f"unknown provider {name!r}; choose from {sorted(PROVIDERS)}")
        out.append(PROVIDERS[name](cfg, client))
    return out
