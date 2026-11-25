"""Pull JSON out of model text and validate it against a JSON Schema."""

from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import Draft202012Validator

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class JSONExtractError(ValueError):
    pass


def extract_json(text: str) -> Any:
    """Return the first JSON object/array found in ``text``.

    Handles plain JSON, ```json fenced blocks and JSON surrounded by chatter
    (common with small local models).
    """
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass

    for block in _FENCE.findall(text):
        try:
            return json.loads(block.strip())
        except ValueError:
            continue

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(text, i)
                return obj
            except ValueError:
                continue
    raise JSONExtractError("no valid JSON found in model output")


def schema_errors(data: Any, schema: dict) -> list[str]:
    """Human-readable schema violations (empty list = valid)."""
    validator = Draft202012Validator(schema)
    errors = []
    for err in sorted(validator.iter_errors(data), key=lambda e: list(e.path)):
        where = "/".join(str(p) for p in err.path) or "<root>"
        errors.append(f"{where}: {err.message}")
    return errors[:10]
