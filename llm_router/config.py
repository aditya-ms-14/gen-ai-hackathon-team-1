"""Router configuration, read from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: str | os.PathLike = ".env") -> None:
    """Minimal .env loader (no extra dependency). Existing env vars win."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return int(raw) if raw else default


@dataclass
class RouterConfig:
    # Order the router tries providers in. LM Studio is the local fallback.
    provider_order: list[str] = field(default_factory=lambda: ["gemini", "groq", "lmstudio"])

    # --- Gemini (primary, cloud) ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_timeout: float = 30.0
    # Thinking tokens count against max output tokens on 2.5 models.
    # Set to 0 to disable thinking on flash models (faster, cheaper). Empty = model default.
    gemini_thinking_budget: int | None = None

    # --- Groq (second, cloud) ---
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_timeout: float = 30.0

    # --- LM Studio (local fallback, OpenAI-compatible server) ---
    lmstudio_base_url: str = "http://localhost:1234/v1"
    # Model id exactly as LM Studio shows it. Empty = use the first loaded model.
    lmstudio_model: str = ""
    lmstudio_timeout: float = 120.0  # small local models on CPU are slow

    # --- Retry policy (per provider) ---
    max_retries: int = 2            # extra attempts after the first one
    backoff_base: float = 1.0       # seconds; doubles each retry
    backoff_max: float = 8.0
    rate_limit_cooldown: float = 60.0  # skip a provider this long after a 429

    max_tokens: int = 4096
    temperature: float = 0.0

    @classmethod
    def from_env(cls, dotenv_path: str | os.PathLike | None = ".env") -> "RouterConfig":
        if dotenv_path:
            load_dotenv(dotenv_path)
        order = [p.strip().lower() for p in _env("LLM_PROVIDER_ORDER", "gemini,groq,lmstudio").split(",") if p.strip()]
        thinking = _env("GEMINI_THINKING_BUDGET")
        d = cls()
        return cls(
            provider_order=order,
            gemini_api_key=_env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"),
            gemini_model=_env("GEMINI_MODEL", d.gemini_model),
            gemini_base_url=_env("GEMINI_BASE_URL", d.gemini_base_url),
            gemini_timeout=_env_float("GEMINI_TIMEOUT", d.gemini_timeout),
            gemini_thinking_budget=int(thinking) if thinking else None,
            groq_api_key=_env("GROQ_API_KEY"),
            groq_model=_env("GROQ_MODEL", d.groq_model),
            groq_base_url=_env("GROQ_BASE_URL", d.groq_base_url),
            groq_timeout=_env_float("GROQ_TIMEOUT", d.groq_timeout),
            lmstudio_base_url=_env("LMSTUDIO_BASE_URL", d.lmstudio_base_url),
            lmstudio_model=_env("LMSTUDIO_MODEL", d.lmstudio_model),
            lmstudio_timeout=_env_float("LMSTUDIO_TIMEOUT", d.lmstudio_timeout),
            max_retries=_env_int("LLM_MAX_RETRIES", d.max_retries),
            backoff_base=_env_float("LLM_BACKOFF_BASE", d.backoff_base),
            backoff_max=_env_float("LLM_BACKOFF_MAX", d.backoff_max),
            rate_limit_cooldown=_env_float("LLM_RATE_LIMIT_COOLDOWN", d.rate_limit_cooldown),
            max_tokens=_env_int("LLM_MAX_TOKENS", d.max_tokens),
            temperature=_env_float("LLM_TEMPERATURE", d.temperature),
        )
