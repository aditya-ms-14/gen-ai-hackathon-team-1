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

from .config import RouterConfig, ipv4_localhost
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


def _enforceable(schema: dict | None) -> bool:
    """Providers can enforce a schema whose root is one concrete type. A union such as
    {"type": ["object", "array"]} (used for "any JSON") is only checked by the router afterwards."""
    return isinstance(schema, dict) and schema.get("type") in ("object", "array")


class BaseProvider:
    name = "base"

    def __init__(self, model: str, timeout: float, client: httpx.Client | None = None,
                 connect_timeout: float = 10.0) -> None:
        self.model = model
        self.timeout = timeout
        self.connect_timeout = min(connect_timeout, timeout)
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
    def _request(self, method: str, url: str, headers: dict, payload: dict | None = None,
                 timeout: float | None = None) -> Any:
        total = timeout or self.timeout
        try:
            resp = self._client.request(
                method, url, headers=headers, json=payload,
                timeout=httpx.Timeout(total, connect=min(self.connect_timeout, total)),
            )
        except httpx.ConnectTimeout:
            # Nothing accepted the connection: retrying would only repeat the wait.
            raise ProviderError(self.name, f"cannot connect to {url} (timed out)", kind="unreachable", retryable=False)
        except httpx.TimeoutException:
            raise ProviderError(self.name, f"no response within {total:.0f}s", kind="timeout", retryable=True)
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
        super().__init__(cfg.gemini_model, cfg.gemini_timeout, client, cfg.connect_timeout)
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
            if self.use_native_schema and _enforceable(schema):
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

    def __init__(self, model: str, timeout: float, base_url: str, api_key: str = "",
                 client: httpx.Client | None = None, connect_timeout: float = 10.0):
        super().__init__(model, timeout, client, connect_timeout)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        # response_format modes this server refused (HTTP 400/422): remembered, never sent again.
        self._rejected_modes: set[str] = set()
        self._merge_system = False  # server rejected the system role (some chat templates, e.g. Gemma)

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _model_id(self) -> str:
        return self.model

    def _modes(self, schema: dict | None) -> list[str | None]:
        """response_format modes to try, best first; None = plain prompting (schema is in the prompt)."""
        if schema is None:
            return [None]
        order: list[str | None] = []
        if self.json_mode == "json_schema" and _enforceable(schema):
            order.append("json_schema")
        order += ["json_object", None]
        return [m for m in order if m is None or m not in self._rejected_modes]

    def complete(self, system, user, *, schema=None, temperature=0.0, max_tokens=4096) -> str:
        model = self._model_id()
        modes = self._modes(schema)
        for n, mode in enumerate(modes):
            if system and not self._merge_system:
                messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            else:
                messages = [{"role": "user", "content": f"{system}\n\n{user}" if system else user}]
            payload: dict[str, Any] = {
                "model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
            }
            if mode == "json_schema":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "strict": True, "schema": schema},
                }
            elif mode == "json_object":
                payload["response_format"] = {"type": "json_object"}
            try:
                data = self._request("POST", f"{self.base_url}/chat/completions", self._headers(), payload)
            except ProviderError as exc:
                if exc.kind != "bad_request":
                    raise
                low = exc.message.lower()
                if system and not self._merge_system and "system" in low and ("role" in low or "not supported" in low):
                    log.info("%s rejected the system role, merging it into the user message", self.name)
                    self._merge_system = True
                    return self.complete(system, user, schema=schema, temperature=temperature, max_tokens=max_tokens)
                if mode is not None and n < len(modes) - 1:
                    log.info("%s rejected response_format=%s: %s", self.name, mode, exc.message)
                    self._rejected_modes.add(mode)
                    continue
                raise
            break
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
        super().__init__(cfg.groq_model, cfg.groq_timeout, cfg.groq_base_url, cfg.groq_api_key, client,
                         cfg.connect_timeout)

    def available(self) -> bool:
        return bool(self.api_key)


class LMStudioProvider(OpenAICompatProvider):
    """Local LM Studio server (Developer tab -> Start Server, default port 1234)."""

    name = "lmstudio"
    json_mode = "json_schema"
    PROBE_TIMEOUT = 3.0  # listing models is instant on a running server

    def __init__(self, cfg: RouterConfig, client: httpx.Client | None = None) -> None:
        # trust_env=False: never send local traffic through HTTP(S)_PROXY from the environment.
        super().__init__(cfg.lmstudio_model, cfg.lmstudio_timeout, ipv4_localhost(cfg.lmstudio_base_url), "",
                         client or httpx.Client(trust_env=False), cfg.lmstudio_connect_timeout)
        self._auto_model = not cfg.lmstudio_model

    def complete(self, system, user, *, schema=None, temperature=0.0, max_tokens=4096) -> str:
        try:
            return super().complete(system, user, schema=schema, temperature=temperature, max_tokens=max_tokens)
        except ProviderError as exc:
            if self._auto_model and exc.kind in ("bad_request", "unreachable"):
                self.model = ""  # the loaded model may have changed: detect again next time
            raise

    def _model_id(self) -> str:
        if not self.model:
            self.model = self._detect_model()
            log.info("lmstudio: using loaded model %s", self.model)
        return self.model

    def _detect_model(self) -> str:
        """The chat model LM Studio has loaded: prefers its REST API (knows type and loaded state),
        falls back to /v1/models minus embedding models. Raises ProviderError if there is none."""
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        try:
            data = self._request("GET", f"{root}/api/v0/models", self._headers(), timeout=self.PROBE_TIMEOUT)
            chat = [m for m in data.get("data", []) if m.get("type") in ("llm", "vlm") and m.get("id")]
            loaded = [m for m in chat if m.get("state") == "loaded"]
            if loaded or chat:
                return (loaded or chat)[0]["id"]
        except ProviderError as exc:
            if exc.kind in ("unreachable", "timeout", "network"):
                raise ProviderError(self.name, exc.message, kind="unreachable", retryable=False)
            # older LM Studio / another OpenAI-compatible server: use /v1/models
        except (AttributeError, TypeError):
            pass  # unexpected JSON shape
        data = self._request("GET", f"{self.base_url}/models", self._headers(), timeout=self.PROBE_TIMEOUT)
        ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
        chat_ids = [i for i in ids if "embed" not in str(i).lower()]
        if not chat_ids:
            raise ProviderError(self.name, "no chat model loaded in LM Studio", kind="unreachable", retryable=False)
        return chat_ids[0]


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
