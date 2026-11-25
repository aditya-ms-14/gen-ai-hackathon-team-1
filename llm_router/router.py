"""LLM Router: Gemini -> Groq -> LM Studio -> regex fallback.

Usage
-----
    from llm_router import LLMRouter

    router = LLMRouter.from_env()
    res = router.call_json(
        task="parse_claim",
        system="Extract sustainability claims.",
        user=report_text,
        schema=CLAIM_SCHEMA,
        fallback=parse_claims_regex,   # optional, used only if every LLM fails
    )
    res.data        # dict, already validated against CLAIM_SCHEMA
    res.provider    # "gemini" | "groq" | "lmstudio" | "regex"

Per provider:
  * timeout / 5xx / network      -> retry with exponential backoff
  * invalid or off-schema JSON   -> retry with a "fix your JSON" note
  * 429 / auth error / unreachable -> move on now, skip this provider for a cool-down
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from .config import RouterConfig
from .errors import AllProvidersFailed, ProviderError
from .json_utils import JSONExtractError, extract_json, schema_errors
from .providers import BaseProvider, build_providers

log = logging.getLogger("llm_router")

JSON_INSTRUCTION = (
    "Respond with a single JSON object only: no markdown, no code fences, no commentary. "
    "It must validate against this JSON Schema:\n{schema}"
)


@dataclass
class Attempt:
    provider: str
    model: str
    status: str            # "ok", "skipped", or a ProviderError kind
    message: str = ""
    latency_ms: int = 0


@dataclass
class RouterResult:
    data: Any                       # parsed + validated JSON (call_json) or text (call_text)
    raw_text: str
    provider: str                   # which provider answered ("regex" = no LLM)
    model: str
    task: str
    attempts: list[Attempt] = field(default_factory=list)
    latency_ms: int = 0

    @property
    def used_fallback(self) -> bool:
        """True when the answer did not come from the first provider tried."""
        tried = [a for a in self.attempts if a.status != "skipped"]
        return bool(tried) and tried[0].provider != self.provider or self.provider == "regex"

    @property
    def is_llm(self) -> bool:
        return self.provider != "regex"


class LLMRouter:
    def __init__(
        self,
        providers: list[BaseProvider],
        config: RouterConfig | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not providers:
            raise ValueError("LLMRouter needs at least one provider")
        self.providers = providers
        self.cfg = config or RouterConfig()
        self._sleep = sleep
        self._clock = clock
        self._cooldown_until: dict[str, float] = {}

    @classmethod
    def from_env(cls, dotenv_path: str | None = ".env", client: httpx.Client | None = None) -> "LLMRouter":
        cfg = RouterConfig.from_env(dotenv_path)
        return cls(build_providers(cfg, client), cfg)

    # -- public API -------------------------------------------------------------
    def call_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        task: str = "generic",
        fallback: Callable[[str], Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> RouterResult:
        """Get JSON that validates against ``schema``. Raises AllProvidersFailed."""
        sys_prompt = (system.rstrip() + "\n\n" if system else "") + JSON_INSTRUCTION.format(
            schema=json.dumps(schema, separators=(",", ":"))
        )
        return self._run(task, sys_prompt, user, schema, fallback, temperature, max_tokens)

    def call_text(
        self,
        *,
        system: str,
        user: str,
        task: str = "generic",
        fallback: Callable[[str], str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> RouterResult:
        """Free-text completion (e.g. narrating a verdict). Raises AllProvidersFailed."""
        return self._run(task, system, user, None, fallback, temperature, max_tokens)

    def status(self) -> list[dict]:
        """Which providers are configured / cooling down (handy for a UI badge)."""
        now = self._clock()
        return [
            {
                "provider": p.name,
                "model": p.model or "(auto)",
                "configured": p.available(),
                "cooldown_s": max(0, round(self._cooldown_until.get(p.name, 0) - now)),
            }
            for p in self.providers
        ]

    # -- internals --------------------------------------------------------------
    def _run(self, task, system, user, schema, fallback, temperature, max_tokens) -> RouterResult:
        temperature = self.cfg.temperature if temperature is None else temperature
        max_tokens = self.cfg.max_tokens if max_tokens is None else max_tokens
        attempts: list[Attempt] = []
        start = self._clock()

        for provider in self.providers:
            if not provider.available():
                attempts.append(Attempt(provider.name, provider.model, "skipped", "not configured (no API key)"))
                continue
            wait = self._cooldown_until.get(provider.name, 0) - self._clock()
            if wait > 0:
                attempts.append(Attempt(provider.name, provider.model, "skipped", f"cooling down after failure, {wait:.0f}s left"))
                continue

            prompt = user
            for try_no in range(self.cfg.max_retries + 1):
                t0 = self._clock()
                try:
                    raw = provider.complete(system, prompt, schema=schema, temperature=temperature, max_tokens=max_tokens)
                    data = self._parse(provider.name, raw, schema)
                except ProviderError as exc:
                    attempts.append(Attempt(provider.name, provider.model, exc.kind, exc.message, self._ms(t0)))
                    log.warning("[%s] %s attempt %d failed: %s", task, provider.name, try_no + 1, exc)
                    if exc.kind in ("rate_limit", "unreachable", "auth"):
                        # Won't fix itself within this call: skip the provider for a while
                        # so a "Verify All" batch doesn't pay the failure cost on every claim.
                        self._cooldown_until[provider.name] = self._clock() + max(
                            self.cfg.rate_limit_cooldown, exc.retry_after or 0
                        )
                        break
                    if not exc.retryable or try_no == self.cfg.max_retries:
                        break
                    if exc.kind == "bad_output" and schema is not None:
                        prompt = self._repair_prompt(user, exc.message)  # retry at once, no backoff
                    else:
                        self._sleep(min(self.cfg.backoff_max, self.cfg.backoff_base * 2**try_no))
                    continue

                attempts.append(Attempt(provider.name, provider.model, "ok", "", self._ms(t0)))
                log.info("[%s] answered by %s (%s)", task, provider.name, provider.model)
                return RouterResult(data, raw, provider.name, provider.model, task, attempts, self._ms(start))

        if fallback is not None:
            t0 = self._clock()
            try:
                data = fallback(user)
                if schema is not None:
                    errs = schema_errors(data, schema)
                    if errs:
                        raise ValueError("fallback output does not match schema: " + "; ".join(errs[:3]))
            except Exception as exc:  # noqa: BLE001 - fallback is user code
                attempts.append(Attempt("regex", "", "error", str(exc), self._ms(t0)))
            else:
                attempts.append(Attempt("regex", "", "ok", "", self._ms(t0)))
                log.warning("[%s] all LLMs failed, used regex fallback", task)
                raw = data if isinstance(data, str) else json.dumps(data)
                return RouterResult(data, raw, "regex", "", task, attempts, self._ms(start))

        raise AllProvidersFailed(attempts)

    @staticmethod
    def _parse(provider: str, raw: str, schema: dict | None) -> Any:
        if schema is None:
            return raw.strip()
        try:
            data = extract_json(raw)
        except JSONExtractError as exc:
            raise ProviderError(provider, str(exc), kind="bad_output", retryable=True)
        errs = schema_errors(data, schema)
        if errs:
            raise ProviderError(provider, "schema mismatch: " + "; ".join(errs[:5]), kind="bad_output", retryable=True)
        return data

    @staticmethod
    def _repair_prompt(user: str, problem: str) -> str:
        return (
            f"{user}\n\n---\nYour previous answer was rejected: {problem}\n"
            "Return ONLY a corrected JSON object that matches the schema."
        )

    def _ms(self, t0: float) -> int:
        return int((self._clock() - t0) * 1000)
