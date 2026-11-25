"""Offline tests for the LLM router: no API keys, no network, no models needed."""

from __future__ import annotations

import json

import httpx
import pytest

from llm_router import AllProvidersFailed, LLMRouter, ProviderError, RouterConfig, parse_claims_regex
from llm_router.json_utils import extract_json
from llm_router.providers import GeminiProvider, GroqProvider, LMStudioProvider, build_providers

SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "number"}, "unit": {"type": "string"}},
    "required": ["value", "unit"],
}


class FakeProvider:
    """Scripted provider: each item is a str (reply) or a ProviderError (raised)."""

    def __init__(self, name, script, available=True):
        self.name, self.model, self.script, self._available = name, f"{name}-model", list(script), available
        self.calls = []

    def available(self):
        return self._available

    def complete(self, system, user, *, schema=None, temperature=0.0, max_tokens=4096):
        self.calls.append(user)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def err(name, kind, retryable, retry_after=None):
    return ProviderError(name, kind, kind=kind, retryable=retryable, retry_after=retry_after)


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.sleeps = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def make_router(*providers, **cfg):
    clock = Clock()
    router = LLMRouter(list(providers), RouterConfig(**cfg), sleep=clock.sleep, clock=clock)
    return router, clock


# -- routing ---------------------------------------------------------------------

def test_primary_answers():
    g = FakeProvider("gemini", ['{"value": 12.5, "unit": "%"}'])
    router, _ = make_router(g, FakeProvider("groq", []))
    res = router.call_json(system="s", user="u", schema=SCHEMA, task="parse_claim")
    assert res.provider == "gemini" and res.data == {"value": 12.5, "unit": "%"}
    assert not res.used_fallback and res.is_llm


def test_rate_limit_moves_to_next_and_cools_down():
    g = FakeProvider("gemini", [err("gemini", "rate_limit", False, retry_after=5)])
    q = FakeProvider("groq", ['{"value": 1, "unit": "t"}', '{"value": 2, "unit": "t"}'])
    router, clock = make_router(g, q, rate_limit_cooldown=60)
    assert router.call_json(system="", user="u", schema=SCHEMA).provider == "groq"
    assert clock.sleeps == []  # 429 does not wait, it falls through immediately
    # second call within cool-down skips gemini entirely
    res = router.call_json(system="", user="u", schema=SCHEMA)
    assert res.provider == "groq" and len(g.calls) == 1
    assert res.attempts[0].status == "skipped"
    # after cool-down gemini is tried again
    clock.t += 61
    g.script.append('{"value": 3, "unit": "t"}')
    assert router.call_json(system="", user="u", schema=SCHEMA).provider == "gemini"


def test_timeouts_retry_with_backoff_then_fall_back():
    g = FakeProvider("gemini", [err("gemini", "timeout", True)] * 3)
    lm = FakeProvider("lmstudio", ['{"value": 4, "unit": "MWh"}'])
    router, clock = make_router(g, lm, max_retries=2, backoff_base=1)
    res = router.call_json(system="", user="u", schema=SCHEMA)
    assert res.provider == "lmstudio" and res.used_fallback
    assert clock.sleeps == [1, 2]
    assert [a.status for a in res.attempts] == ["timeout", "timeout", "timeout", "ok"]


def test_server_error_recovers_on_retry():
    g = FakeProvider("gemini", [err("gemini", "server", True), '{"value": 1, "unit": "x"}'])
    router, _ = make_router(g)
    assert router.call_json(system="", user="u", schema=SCHEMA).provider == "gemini"


def test_bad_json_gets_repair_prompt():
    g = FakeProvider("gemini", ["Sure! here you go", '```json\n{"value": 7, "unit": "kg"}\n```'])
    router, clock = make_router(g)
    res = router.call_json(system="", user="report", schema=SCHEMA)
    assert res.data == {"value": 7, "unit": "kg"}
    assert "previous answer was rejected" in g.calls[1]
    assert clock.sleeps == []


def test_schema_mismatch_is_retried_then_next_provider():
    g = FakeProvider("gemini", ['{"value": "lots"}'] * 3)
    q = FakeProvider("groq", ['{"value": 9, "unit": "t"}'])
    router, _ = make_router(g, q)
    res = router.call_json(system="", user="u", schema=SCHEMA)
    assert res.provider == "groq"
    assert all(a.status == "bad_output" for a in res.attempts[:3])


def test_unconfigured_and_unreachable_are_skipped_fast():
    g = FakeProvider("gemini", [], available=False)
    q = FakeProvider("groq", [err("groq", "auth", False)])
    lm = FakeProvider("lmstudio", [err("lmstudio", "unreachable", False)])
    router, clock = make_router(g, q, lm)
    with pytest.raises(AllProvidersFailed) as exc:
        router.call_json(system="", user="u", schema=SCHEMA)
    assert [a.status for a in exc.value.attempts] == ["skipped", "auth", "unreachable"]
    assert clock.sleeps == []


def test_unreachable_provider_is_not_retried_on_next_call():
    lm = FakeProvider("lmstudio", [err("lmstudio", "unreachable", False)])
    router, _ = make_router(lm)
    for _ in range(3):
        res = router.call_json(system="", user="x 5 t", schema={"type": "object"}, fallback=parse_claims_regex)
        assert res.provider == "regex"
    assert len(lm.calls) == 1


def test_regex_fallback_when_all_llms_fail():
    g = FakeProvider("gemini", [err("gemini", "rate_limit", False)])
    lm = FakeProvider("lmstudio", [err("lmstudio", "unreachable", False)])
    router, _ = make_router(g, lm)
    res = router.call_json(
        system="", user="Scope 1 emissions fell 12% in 2023.", schema={"type": "object"},
        task="parse_claim", fallback=parse_claims_regex,
    )
    assert res.provider == "regex" and not res.is_llm and res.used_fallback
    assert res.data["claims"][0]["value"] == 12.0


def test_fallback_output_must_match_schema():
    g = FakeProvider("gemini", [err("gemini", "auth", False)])
    router, _ = make_router(g)
    with pytest.raises(AllProvidersFailed):
        router.call_json(system="", user="u", schema=SCHEMA, fallback=lambda _: {"nope": 1})


def test_call_text():
    g = FakeProvider("gemini", ["  The claim is supported.  "])
    router, _ = make_router(g)
    assert router.call_text(system="", user="narrate").data == "The claim is supported."


# -- JSON extraction -------------------------------------------------------------

@pytest.mark.parametrize("text", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    'Here is the result: {"a": 1} hope it helps',
])
def test_extract_json(text):
    assert extract_json(text) == {"a": 1}


# -- regex fallback --------------------------------------------------------------

def test_regex_parser_basic():
    out = parse_claims_regex(
        "We reduced Scope 1 emissions by 18.5% between 2019 and 2023. "
        "Renewable electricity reached 1,250 MWh in FY2024. We aim for net zero by 2040."
    )
    c = out["claims"]
    assert c[0]["value"] == 18.5 and c[0]["unit"] == "%" and c[0]["direction"] == "decrease"
    assert c[0]["is_percent_change"] and c[0]["year"] == 2023 and c[0]["baseline_year"] == 2019
    assert c[1]["value"] == 1250 and c[1]["unit"] == "MWh" and c[1]["year"] == 2024
    assert not any(x["value"] == 2040 for x in c)  # bare years are not metrics


# -- provider wire formats (mocked HTTP) ---------------------------------------

def mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_gemini_request_and_parse():
    seen = {}

    def handler(req):
        seen["url"], seen["key"], seen["body"] = str(req.url), req.headers["x-goog-api-key"], json.loads(req.content)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"value":1,"unit":"t"}'}]}}]})

    p = GeminiProvider(RouterConfig(gemini_api_key="k", gemini_model="gemini-x"), mock_client(handler))
    assert p.complete("sys", "hi", schema=SCHEMA) == '{"value":1,"unit":"t"}'
    assert seen["url"].endswith("/models/gemini-x:generateContent") and seen["key"] == "k"
    assert seen["body"]["generationConfig"]["responseJsonSchema"] == SCHEMA
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "sys"


def test_gemini_drops_native_schema_if_rejected():
    bodies = []

    def handler(req):
        body = json.loads(req.content)
        bodies.append(body)
        if "responseJsonSchema" in body["generationConfig"]:
            return httpx.Response(400, json={"error": {"message": "Unknown name responseJsonSchema"}})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "{}"}]}}]})

    p = GeminiProvider(RouterConfig(gemini_api_key="k"), mock_client(handler))
    assert p.complete("", "hi", schema=SCHEMA) == "{}"
    assert len(bodies) == 2 and p.use_native_schema is False


def test_gemini_429_maps_to_rate_limit():
    p = GeminiProvider(RouterConfig(gemini_api_key="k"), mock_client(lambda r: httpx.Response(429, headers={"retry-after": "7"})))
    with pytest.raises(ProviderError) as exc:
        p.complete("", "hi")
    assert exc.value.kind == "rate_limit" and exc.value.retry_after == 7


def test_groq_uses_json_object_and_bearer():
    seen = {}

    def handler(req):
        seen["auth"], seen["body"] = req.headers["authorization"], json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    p = GroqProvider(RouterConfig(groq_api_key="gk"), mock_client(handler))
    assert p.complete("", "hi", schema=SCHEMA) == "{}"
    assert seen["auth"] == "Bearer gk" and seen["body"]["response_format"] == {"type": "json_object"}


def test_lmstudio_autodetects_model_and_uses_json_schema():
    seen = {}

    def handler(req):
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "text-embedding-nomic"}, {"id": "qwen2.5-3b-instruct"}]})
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    p = LMStudioProvider(RouterConfig(), mock_client(handler))
    p.complete("", "hi", schema=SCHEMA)
    assert seen["body"]["model"] == "qwen2.5-3b-instruct"
    assert seen["body"]["response_format"]["json_schema"]["schema"] == SCHEMA
    assert "authorization" not in {k.lower() for k in seen.get("headers", {})}


def test_lmstudio_not_running_is_unreachable():
    def handler(req):
        raise httpx.ConnectError("refused")

    p = LMStudioProvider(RouterConfig(lmstudio_model="m"), mock_client(handler))
    with pytest.raises(ProviderError) as exc:
        p.complete("", "hi")
    assert exc.value.kind == "unreachable" and not exc.value.retryable


def test_build_providers_order_and_unknown():
    names = [p.name for p in build_providers(RouterConfig(provider_order=["groq", "lmstudio"]))]
    assert names == ["groq", "lmstudio"]
    with pytest.raises(ValueError):
        build_providers(RouterConfig(provider_order=["openai"]))
