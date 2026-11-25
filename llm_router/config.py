"""Router configuration, read from environment variables (or a .env file)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: str | os.PathLike = ".env") -> None:
    """Minimal .env loader (no extra dependency). Existing env vars win."""
    p = Path(path)
    if not p.is_file():
        return
    # utf-8-sig: Notepad on Windows saves a BOM, which would otherwise glue itself to the first key.
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if value[:1] in ("'", '"') and value.count(value[0]) >= 2:
            value = value[1 : value.index(value[0], 1)]
        else:
            value = value.split(" #", 1)[0].strip()  # inline comment on an unquoted value
        if key:
            os.environ.setdefault(key, value)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name)) if _env(name) else default
    except ValueError:  # a typo in .env must not take the whole app down
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name)) if _env(name) else default
    except ValueError:
        return default


def native_gemini_url(url: str) -> str:
    """The native REST base. .env.example used to ship the OpenAI-compatible '.../v1beta/openai/' URL."""
    url = url.strip().rstrip("/")
    return url[: -len("/openai")] if url.endswith("/openai") else url


def ipv4_localhost(url: str) -> str:
    """'localhost' makes Windows try ::1 first and wait seconds for the refusal; LM Studio listens on IPv4."""
    return re.sub(r"//localhost(?=[:/]|$)", "//127.0.0.1", url.strip(), count=1)


@dataclass
class RouterConfig:
    # Order the router tries providers in. LM Studio is the local fallback.
    provider_order: list[str] = field(default_factory=lambda: ["gemini", "groq", "lmstudio"])

    # --- Gemini (primary, cloud) ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_timeout: float = 30.0
    connect_timeout: float = 10.0   # cloud providers: fail fast when the network is down
    # Thinking tokens count against max output tokens on 2.5 models.
    # Set to 0 to disable thinking on flash models (faster, cheaper). Empty = model default.
    gemini_thinking_budget: int | None = None

    # --- Groq (second, cloud) ---
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_timeout: float = 30.0

    # --- LM Studio (local fallback, OpenAI-compatible server) ---
    # 127.0.0.1, not localhost: on Windows localhost tries IPv6 first and takes seconds to fail.
    lmstudio_base_url: str = "http://127.0.0.1:1234/v1"
    # Model id exactly as LM Studio shows it. Empty = use the first loaded model.
    lmstudio_model: str = ""
    lmstudio_timeout: float = 120.0  # small local models on CPU are slow
    # A running server accepts a local connection in milliseconds; Windows needs ~2 s to refuse one.
    lmstudio_connect_timeout: float = 1.5

    # --- Retry policy (per provider) ---
    max_retries: int = 2            # extra attempts after the first one
    timeout_retries: int = 1        # a provider that timed out gets at most this many retries (retrying
                                    # a 120 s timeout twice would stall one call for 6 minutes)
    failure_cooldown: float = 30.0  # skip a provider this long after it exhausted its retries (5xx / timeout)
    backoff_base: float = 1.0       # seconds; doubles each retry
    backoff_max: float = 8.0
    rate_limit_cooldown: float = 60.0  # skip a provider this long after a 429 / bad key / not running
    max_cooldown: float = 600.0        # cap on any cooldown (a Retry-After can be hours)

    max_tokens: int = 4096
    temperature: float = 0.0

    @classmethod
    def from_env(cls, dotenv_path: str | os.PathLike | None = ".env") -> "RouterConfig":
        if dotenv_path:
            load_dotenv(dotenv_path)
        order = [p.strip().lower() for p in _env("LLM_PROVIDER_ORDER", "gemini,groq,lmstudio").split(",") if p.strip()]
        thinking = _env("GEMINI_THINKING_BUDGET")
        d = cls()
        shared_timeout = _env_float("LLM_TIMEOUT_SECONDS", d.gemini_timeout)  # same knob the app's .env uses
        return cls(
            provider_order=order,
            gemini_api_key=_env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"),
            gemini_model=_env("GEMINI_MODEL", d.gemini_model),
            gemini_base_url=native_gemini_url(_env("GEMINI_BASE_URL", d.gemini_base_url)),
            gemini_timeout=_env_float("GEMINI_TIMEOUT", shared_timeout),
            gemini_thinking_budget=_env_int("GEMINI_THINKING_BUDGET", 0) if thinking else None,
            groq_api_key=_env("GROQ_API_KEY"),
            groq_model=_env("GROQ_MODEL", d.groq_model),
            groq_base_url=_env("GROQ_BASE_URL", d.groq_base_url),
            groq_timeout=_env_float("GROQ_TIMEOUT", shared_timeout),
            lmstudio_base_url=ipv4_localhost(_env("LMSTUDIO_BASE_URL", d.lmstudio_base_url)),
            lmstudio_model=_env("LMSTUDIO_MODEL", d.lmstudio_model),
            lmstudio_timeout=_env_float("LMSTUDIO_TIMEOUT", d.lmstudio_timeout),
            max_retries=_env_int("LLM_MAX_RETRIES", d.max_retries),
            timeout_retries=_env_int("LLM_TIMEOUT_RETRIES", d.timeout_retries),
            failure_cooldown=_env_float("LLM_FAILURE_COOLDOWN", d.failure_cooldown),
            backoff_base=_env_float("LLM_BACKOFF_BASE", d.backoff_base),
            backoff_max=_env_float("LLM_BACKOFF_MAX", d.backoff_max),
            rate_limit_cooldown=_env_float("LLM_RATE_LIMIT_COOLDOWN", d.rate_limit_cooldown),
            max_tokens=_env_int("LLM_MAX_TOKENS", d.max_tokens),
            temperature=_env_float("LLM_TEMPERATURE", d.temperature),
        )
