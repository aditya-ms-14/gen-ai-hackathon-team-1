"""Pull JSON out of model text and validate it against a JSON Schema."""

from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import Draft202012Validator

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


class JSONExtractError(ValueError):
    pass


def strip_reasoning(text: str) -> str:
    """Drop <think>...</think> blocks that reasoning models (Qwen3, R1) put before the answer."""
    text = _THINK.sub("", text)
    if "</think>" in text:  # some chat templates swallow the opening tag
        text = text.split("</think>", 1)[1]
    return text.strip()


def _is_wanted(value: Any) -> bool:
    """An object, or a list of objects. Skips scalars and stray '[1]' footnote markers in chatter."""
    return isinstance(value, dict) or (isinstance(value, list) and all(isinstance(v, dict) for v in value))


def extract_json(text: str) -> Any:
    """Return the first JSON object (or array of objects) found in ``text``.

    Handles plain JSON, ```json fenced blocks, JSON surrounded by chatter, <think> blocks
    and trailing commas (all common with small local models). If nothing else is found,
    falls back to the first array of any kind. Raises JSONExtractError.
    """
    text = strip_reasoning(text or "")
    decoder = json.JSONDecoder()
    fallback: Any = None
    found_fallback = False
    for candidate in [text] + [m.group(1) for m in _FENCE.finditer(text)]:
        for i, ch in enumerate(candidate):
            if ch not in "{[":
                continue
            tail = candidate[i:]
            for attempt in (tail, _TRAILING_COMMA.sub(r"\1", tail)):
                try:
                    value, _ = decoder.raw_decode(attempt)
                except ValueError:
                    continue
                if _is_wanted(value):
                    return value
                if not found_fallback and isinstance(value, list):
                    fallback, found_fallback = value, True
                break
    if found_fallback:
        return fallback
    raise JSONExtractError("no valid JSON found in model output")


def schema_errors(data: Any, schema: dict) -> list[str]:
    """Human-readable schema violations (empty list = valid)."""
    validator = Draft202012Validator(schema)
    errors = []
    for err in sorted(validator.iter_errors(data), key=lambda e: list(e.path)):
        where = "/".join(str(p) for p in err.path) or "<root>"
        errors.append(f"{where}: {err.message}")
    return errors[:10]
