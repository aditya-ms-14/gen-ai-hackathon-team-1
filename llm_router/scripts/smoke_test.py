"""Live check of the LLM router on a real machine.

    python llm_router/scripts/smoke_test.py            # test each provider, then the full chain
    python llm_router/scripts/smoke_test.py --only lmstudio

Reads keys from .env (see .env.example). Providers without a key are skipped; LM Studio is
skipped when its server is not running.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from llm_router import AllProvidersFailed, LLMRouter, RouterConfig, parse_claims_regex  # noqa: E402
from llm_router.providers import build_providers  # noqa: E402

CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "indicator": {"type": "string"},
                    "value": {"type": "number"},
                    "unit": {"type": "string"},
                    "year": {"type": ["integer", "null"]},
                },
                "required": ["indicator", "value", "unit", "year"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}
SYSTEM = "You extract quantitative sustainability claims from report text."
TEXT = "In 2023 we cut Scope 1 emissions by 18% and sourced 1,250 MWh of renewable electricity."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["gemini", "groq", "lmstudio"], help="test a single provider")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    cfg = RouterConfig.from_env()
    names = [args.only] if args.only else cfg.provider_order
    ok = True

    print("== Each provider on its own ==")
    for name in names:
        cfg_one = RouterConfig.from_env()
        cfg_one.provider_order = [name]
        cfg_one.max_retries = 0
        router = LLMRouter(build_providers(cfg_one), cfg_one)
        if not router.providers[0].available():
            print(f"  {name:9} SKIPPED (no API key in .env)")
            continue
        try:
            res = router.call_json(system=SYSTEM, user=TEXT, schema=CLAIM_SCHEMA, task="smoke")
            print(f"  {name:9} OK   {res.model}  {res.latency_ms} ms  -> {json.dumps(res.data)[:120]}")
        except AllProvidersFailed as exc:
            a = exc.attempts[-1]
            if a.status == "unreachable":  # e.g. LM Studio server not started: not a failure of the router
                print(f"  {name:9} SKIPPED ({a.message[:100]})")
                continue
            ok = False
            print(f"  {name:9} FAIL {a.status}: {a.message[:150]}")

    if not args.only:
        print("\n== Full chain (with regex fallback) ==")
        router = LLMRouter.from_env()
        res = router.call_json(system=SYSTEM, user=TEXT, schema={"type": "object"}, task="smoke", fallback=parse_claims_regex)
        print(f"  answered by: {res.provider} ({res.model or 'no LLM'})")
        for a in res.attempts:
            print(f"    {a.provider:9} {a.status:11} {a.latency_ms:>6} ms  {a.message[:80]}")

        print("\n== Text call (narrate verdict) ==")
        try:
            res = router.call_text(system="Answer in one sentence.", user="Say hello to the auditors.", task="smoke")
            print(f"  {res.provider}: {res.data[:150]}")
        except AllProvidersFailed as exc:
            ok = False
            print(f"  FAIL: {exc}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
