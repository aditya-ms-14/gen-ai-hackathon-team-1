"""LLM Router layer: Gemini (primary) -> Groq -> LM Studio (local) -> regex fallback."""

from .config import RouterConfig
from .errors import AllProvidersFailed, ProviderError
from .regex_fallback import parse_claims_regex
from .router import Attempt, LLMRouter, RouterResult

__all__ = [
    "LLMRouter",
    "RouterConfig",
    "RouterResult",
    "Attempt",
    "ProviderError",
    "AllProvidersFailed",
    "parse_claims_regex",
]
