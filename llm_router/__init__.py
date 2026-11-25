"""LLM Router layer: Gemini (primary) -> Groq -> LM Studio (local) -> regex fallback.

    from llm_router import make_router      # what the auditor uses: complete / complete_json / status
    from llm_router import LLMRouter        # the router itself: call_json / call_text / status
"""

from .adapter import NullRouter, TeamRouter, make_router
from .claims import has_vague_term, parse_pages, parse_report
from .config import RouterConfig
from .errors import AllProvidersFailed, ProviderError
from .regex_fallback import parse_claims_regex
from .router import Attempt, LLMRouter, RouterResult

__all__ = [
    "make_router",
    "TeamRouter",
    "NullRouter",
    "LLMRouter",
    "RouterConfig",
    "RouterResult",
    "Attempt",
    "ProviderError",
    "AllProvidersFailed",
    "parse_claims_regex",
    "parse_pages",
    "parse_report",
    "has_vague_term",
]
