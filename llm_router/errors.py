"""Error types shared by the providers and the router."""

from __future__ import annotations

import re

# API keys must never reach logs, exceptions or the UI, even if a provider echoes one back.
_SECRETS = re.compile(r"AIza[0-9A-Za-z_\-]{20,}|gsk_[0-9A-Za-z]{20,}|sk-[0-9A-Za-z_\-]{20,}|(?i:bearer)\s+[0-9A-Za-z._\-]{12,}")


def redact(text: str) -> str:
    return _SECRETS.sub("***", text)


class ProviderError(Exception):
    """A single provider call failed.

    kind is one of:
      rate_limit  - HTTP 429 / quota exhausted       -> move to next provider, cool down
      auth        - bad / missing key (401, 403)     -> move to next provider
      unreachable - nothing listening (LM Studio off)-> move to next provider
      timeout     - request took too long            -> retry with backoff
      network     - other transport error            -> retry with backoff
      server      - HTTP 5xx                         -> retry with backoff
      bad_request - other HTTP 4xx                   -> move to next provider
      bad_output  - empty / blocked / non-JSON reply -> retry (repair prompt)
    """

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        kind: str,
        retryable: bool,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        message = redact(message)
        super().__init__(f"[{provider}] {kind}: {message}")
        self.provider = provider
        self.message = message
        self.kind = kind
        self.retryable = retryable
        self.status = status
        self.retry_after = retry_after


class AllProvidersFailed(Exception):
    """Every provider (and the regex fallback, if given) failed."""

    def __init__(self, attempts: list) -> None:
        lines = [f"  - {a.provider}: {a.status} {a.message}" for a in attempts]
        super().__init__("All LLM providers failed:\n" + "\n".join(lines))
        self.attempts = attempts
