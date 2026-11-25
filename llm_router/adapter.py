"""Adapter between the team router (router.LLMRouter) and the interface the auditor pipeline uses.

    complete(system, user)                              -> str | None    plain text (narration)
    complete_json(system, user, schema=None, name=...)  -> dict | None   JSON (claim parsing, passage reading)
    status()                                            -> dict[str, str]  one line per provider (sidebar)
    last_provider, stats                                                 shown in the UI

Every failure becomes None, so callers keep their deterministic fallbacks and the demo never dies.
A JSON array reply comes back as {"items": [...]}.

``make_router()`` never raises: if the router cannot be built it returns a router whose
calls return None and whose ``status()`` says why.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .errors import redact
from .router import LLMRouter

log = logging.getLogger("llm_router")


class TeamRouter:
    """Gives ``LLMRouter`` the complete / complete_json / status interface the pipeline uses."""

    ANY_JSON = {"type": ["object", "array"]}

    def __init__(self, inner: LLMRouter) -> None:
        self.inner = inner

    # -- UI state (kept by the router itself) -------------------------------------------
    @property
    def last_provider(self) -> Optional[str]:
        return self.inner.last_provider

    @property
    def stats(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self.inner.stats.items()}

    # -- calls ------------------------------------------------------------------------------
    def complete(self, system: str, user: str) -> Optional[str]:
        try:
            result = self.inner.call_text(system=system, user=user, task="narrate")
        except Exception as exc:  # AllProvidersFailed, or anything unexpected
            log.warning("complete failed: %s", redact(str(exc)))
            return None
        return result.data if isinstance(result.data, str) and result.data.strip() else None

    def complete_json(self, system: str, user: str, schema: Optional[dict] = None,
                      name: str = "result") -> Optional[dict]:
        try:
            result = self.inner.call_json(system=system, user=user, schema=schema or self.ANY_JSON, task=name)
        except Exception as exc:
            log.warning("complete_json(%s) failed: %s", name, redact(str(exc)))
            return None
        data = {"items": result.data} if isinstance(result.data, list) else result.data
        return data if isinstance(data, dict) else None

    def status(self) -> dict[str, str]:
        out = {}
        for s in self.inner.status():
            state = s["model"] if s["configured"] else "not configured"
            if s.get("calls"):
                state += f" · {s['calls']} calls, {s['failures']} failed"
            if s.get("cooldown_s"):
                state += f" · paused {s['cooldown_s']}s"
            out[s["provider"]] = state
        return out


class NullRouter:
    """Stand-in when the router could not be built: every call returns None."""

    last_provider: Optional[str] = None
    stats: dict[str, dict[str, int]] = {}

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def complete(self, system: str, user: str) -> Optional[str]:
        return None

    def complete_json(self, system: str, user: str, schema: Optional[dict] = None,
                      name: str = "result") -> Optional[dict]:
        return None

    def status(self) -> dict[str, str]:
        return {"llm": f"unavailable: {self.reason}"}


def make_router() -> Any:
    """A TeamRouter built from .env / environment variables. Never raises."""
    try:
        return TeamRouter(LLMRouter.from_env())
    except Exception as exc:  # bad LLM_PROVIDER_ORDER, unreadable .env, missing dependency...
        reason = redact(f"{type(exc).__name__}: {exc}")[:200]
        log.warning("LLM router unavailable: %s", reason)
        return NullRouter(reason)
