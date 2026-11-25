"""Adapter (TeamRouter / make_router) and claim parser. No network: scripted fake providers only."""
import pytest

import llm_router
from llm_router import LLMRouter, NullRouter, ProviderError, RouterConfig, TeamRouter, make_router
from llm_router.claims import parse_claim_regex, parse_pages, parse_report
from auditor.schemas import Direction, Metric


# ---- fakes ----------------------------------------------------------------------
class FakeProvider:
    """Scripted provider: each item is a str (reply) or an Exception (raised)."""

    def __init__(self, name, *script, available=True):
        self.name, self.model, self.script, self._available = name, f"{name}-model", list(script), available
        self.calls = []

    def available(self):
        return self._available

    def complete(self, system, user, *, schema=None, temperature=0.0, max_tokens=4096):
        self.calls.append((system, user, schema))
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return item


def down(name, kind="unreachable"):
    return ProviderError(name, "down", kind=kind, retryable=False)


def team(*providers):
    return TeamRouter(LLMRouter(list(providers), RouterConfig(max_retries=0), sleep=lambda s: None))


class FakeLLM:
    """Stands in for the router in the claim parser: returns canned complete_json results."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def complete_json(self, system, user, schema=None, name="result"):
        self.calls.append(user)
        return self.results.pop(0) if self.results else None


# ---- adapter ----------------------------------------------------------------------
def test_complete_falls_through_providers_and_records_stats():
    g, lm = FakeProvider("gemini", down("gemini", "rate_limit")), FakeProvider("lmstudio", "Emissions fell 30%.")
    router = team(g, lm)
    assert router.complete("sys", "user") == "Emissions fell 30%."
    assert router.last_provider == "lmstudio"
    assert router.stats == {"gemini": {"calls": 1, "failures": 1}, "lmstudio": {"calls": 1, "failures": 0}}
    status = router.status()
    assert "1 calls, 1 failed" in status["gemini"] and "paused" in status["gemini"]
    assert status["lmstudio"].startswith("lmstudio-model")


def test_everything_down_returns_none_never_raises():
    router = team(FakeProvider("gemini", down("gemini", "auth")), FakeProvider("lmstudio", down("lmstudio")))
    assert router.complete("s", "u") is None
    assert router.complete_json("s", "u") is None
    assert router.last_provider is None


def test_complete_json_wraps_lists_and_accepts_chatty_replies():
    router = team(FakeProvider("gemini", 'Sure! ```json\n[{"a": 1}, {"a": 2}]\n```'))
    assert router.complete_json("s", "u") == {"items": [{"a": 1}, {"a": 2}]}
    router = team(FakeProvider("gemini", '<think>x</think> {"claims": []}'))
    assert router.complete_json("s", "u", schema={"type": "object"}, name="claims") == {"claims": []}


def test_complete_json_rejects_off_schema_replies_and_uses_next_provider():
    schema = {"type": "object", "properties": {"n": {"type": "number"}}, "required": ["n"]}
    router = team(FakeProvider("gemini", '{"n": "many"}'), FakeProvider("lmstudio", '{"n": 3}'))
    assert router.complete_json("s", "u", schema=schema) == {"n": 3}
    assert router.last_provider == "lmstudio"


def test_empty_text_reply_is_none():
    assert team(FakeProvider("gemini", "<think>only thinking</think>")).complete("s", "u") is None


def test_make_router_builds_the_team_router_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini,groq,lmstudio")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    router = make_router()
    assert isinstance(router, TeamRouter)
    assert list(router.status()) == ["gemini", "groq", "lmstudio"]
    assert router.status()["gemini"] == "not configured"


def test_make_router_never_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "openai")  # unknown provider
    router = make_router()
    assert isinstance(router, NullRouter)
    assert router.complete("s", "u") is None and router.complete_json("s", "u") is None
    assert router.last_provider is None and router.stats == {}
    assert "unknown provider" in router.status()["llm"]

    def boom(cls, *a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(LLMRouter, "from_env", classmethod(boom))
    assert "kaboom" in make_router().status()["llm"]


def test_public_exports():
    for name in ("make_router", "TeamRouter", "LLMRouter", "RouterConfig", "parse_claims_regex", "AllProvidersFailed"):
        assert hasattr(llm_router, name)
    assert not hasattr(llm_router, "builtin")


# ---- regex claim parser -------------------------------------------------------------
def rx(text, year=2025):
    c = parse_claim_regex("X", text, year)
    return (c.indicator, c.metric, c.direction, c.value, c.unit, c.baseline_year, c.target_year, c.entity)


PC, SH, AB, VG = Metric.PCT_CHANGE, Metric.SHARE, Metric.ABSOLUTE, Metric.VAGUE
DOWN, UP, FLAT = Direction.DECREASE, Direction.INCREASE, Direction.NONE


@pytest.mark.parametrize("text, expected", [
    # problem-statement sample style
    ("Carbon emissions reduced by 40%.", ("ghg_emissions", PC, DOWN, 40, "%", 2024, 2025, None)),
    ("Renewable energy usage reached 75%.", ("renewable_share", SH, FLAT, 75, "%", None, 2025, None)),
    ("Biodiversity improved by 30%.", ("biodiversity", PC, UP, 30, "%", 2024, 2025, None)),
    ("Our products are eco-friendly and fully sustainable.", (None, VG, FLAT, None, None, None, 2025, None)),
    # demo report style, with entities
    ("Germany's CO2 emissions fell by 30% between 2010 and 2022.",
     ("co2_emissions", PC, DOWN, 30, "%", 2010, 2022, "DEU")),
    ("Global CO2 emissions rose 1% in 2023.", ("co2_emissions", PC, UP, 1, "%", 2022, 2023, "WLD")),
    ("EU greenhouse gas emissions fell 32% since 1990.", ("ghg_emissions", PC, DOWN, 32, "%", 1990, 2025, "EUU")),
    ("Our global emissions fell 10%.", ("ghg_emissions", PC, DOWN, 10, "%", 2024, 2025, None)),
    # percentage points
    ("Renewable electricity rose by 12 percentage points between 2015 and 2022.",
     ("renewable_electricity_share", PC, UP, 12, "pp", 2015, 2022, None)),
    ("Renewable energy share up 5 pp since 2015.", ("renewable_share", PC, UP, 5, "pp", 2015, 2025, None)),
    # from X% to Y% on a share indicator
    ("Renewable electricity grew from 40% in 2015 to 55% in 2022.",
     ("renewable_electricity_share", SH, UP, 55, "%", 2015, 2022, None)),
    ("Forest cover fell from 32% to 29% in 2020.", ("forest_area_share", SH, DOWN, 29, "%", None, 2020, None)),
    # words instead of numbers
    ("We halved our water use since 2015.", ("water_consumption", PC, DOWN, 50, "%", 2015, 2025, None)),
    ("Energy use doubled between 2000 and 2020.", ("energy_consumption", PC, UP, 100, "%", 2000, 2020, None)),
    ("Waste was cut by a third compared to 2019 levels.", ("waste", PC, DOWN, 33.3, "%", 2019, 2025, None)),
    # baseline phrasing
    ("In 2024, GHG emissions were 25% lower relative to 2015.",
     ("ghg_emissions", PC, DOWN, 25, "%", 2015, 2024, None)),
    ("Scope 1 emissions in 2024 were 30% lower than in 2019.", ("ghg_emissions", PC, DOWN, 30, "%", 2019, 2024, None)),
    # future target stays parseable
    ("We will cut carbon emissions by 50% by 2030 compared to 2019.",
     ("ghg_emissions", PC, DOWN, 50, "%", 2019, 2030, None)),
    # absolute amounts, thousands separators, scale words
    ("Water consumption fell to 12,500 kL in 2024.", ("water_consumption", AB, DOWN, 12500, "kL", None, 2024, None)),
    ("Waste generated was 1.2 million tonnes in 2023.", ("waste", AB, FLAT, 1_200_000, "tonnes", None, 2023, None)),
    ("Germany emitted 640 Mt CO2e in 2022.", ("ghg_emissions", AB, FLAT, 640, "Mt CO2e", None, 2022, "DEU")),
    ("Total emissions were 1.2 million tonnes of CO2e in 2024.",
     ("ghg_emissions", AB, FLAT, 1.2, "Mt CO2e", None, 2024, None)),
    ("Freshwater withdrawals in India reached 40 billion m3 in 2020.",
     ("water_withdrawal", AB, FLAT, 40, "billion m3", None, 2020, "IND")),
    ("Electricity consumption was 2.1 TWh in 2023.", ("energy_consumption", AB, FLAT, 2.1, "TWh", None, 2023, None)),
    ("Waste fell to 2000 tonnes in 2024.", ("waste", AB, DOWN, 2000, "tonnes", None, 2024, None)),  # 2000 is not a year
    ("Emissions fell from 1,000 tCO2e in 2019 to 700 tCO2e in 2024.",
     ("ghg_emissions", AB, DOWN, 700, "tCO2e", 2019, 2024, None)),
])
def test_regex_parser(text, expected):
    assert rx(text) == expected


def test_uncheckable_numbers_are_not_mistaken_for_levels():
    change = parse_claim_regex("X", "Energy use was reduced by 500 MWh.", 2025)
    assert change.metric == AB and change.value is None and change.unit == "MWh"
    intensity = parse_claim_regex("X", "Emissions intensity was 0.5 tCO2e per tonne of product.", 2025)
    assert intensity.metric == AB and intensity.value is None  # not vague, so no greenwash flag
    share_of = parse_claim_regex("X", "45% of our water is recycled.", 2025)
    assert share_of.indicator is None and share_of.metric == SH


def test_parse_report_keeps_sample_behaviour():
    text = ("HYPOTHETICAL COMPANY · 2025 SUSTAINABILITY REPORT\n\n"
            "C01 Carbon emissions reduced by 40%.\nC03 Renewable energy usage reached 75%.\n"
            "C06 Our products are eco-friendly and fully sustainable.\n")
    claims = parse_report(text)
    assert [c.claim_id for c in claims] == ["C01", "C03", "C06"]
    assert all(c.page is None for c in claims)
    assert claims[0].target_year == 2025 and claims[2].metric == VG


# ---- parse_pages ------------------------------------------------------------------------
PAGES = {
    1: "ACME Corp 2025 Sustainability Report\nContents\nClimate ........ 3\nWater ........ 5\n2",
    3: ("Climate action\nOur Scope 1 and 2 emissions fell by 30%\ncompared to 2019 levels. We are proud of\n"
        "our people and our culture.\nRenewable electricity reached 62% of supply in 2024.\n- 3 -"),
    5: ("Water stewardship\nWater consump-\ntion was reduced by 12% in 2024.\n"
        "• Our packaging is fully eco-friendly\n"
        "Our Scope 1 and 2 emissions fell by 30% compared to 2019 levels.\nPage 5 of 40"),
}


def test_parse_pages_sets_pages_skips_headings_and_dedupes():
    claims = parse_pages(PAGES)
    got = [(c.claim_id, c.page, c.text) for c in claims]
    assert got == [
        ("C01", 3, "Our Scope 1 and 2 emissions fell by 30% compared to 2019 levels."),
        ("C02", 3, "Renewable electricity reached 62% of supply in 2024."),
        ("C03", 5, "Water consumption was reduced by 12% in 2024."),
        ("C04", 5, "Our packaging is fully eco-friendly"),
    ]
    c01, c02, c03, c04 = claims
    assert (c01.value, c01.baseline_year, c01.target_year) == (30, 2019, 2025)  # report year from page 1
    assert (c02.metric, c02.value, c02.target_year) == (SH, 62, 2024)
    assert (c03.indicator, c03.baseline_year, c03.target_year) == ("water_consumption", 2023, 2024)
    assert c04.metric == VG


def test_parse_pages_keeps_explicit_ids_and_wrapped_lines():
    pages = {1: "C01 Germany's CO2 emissions fell by 30% between\n2010 and 2022.\nC02 Global forest area\n"
                "fell from 31.9% to 31.2% between 2000 and 2020."}
    c01, c02 = parse_pages(pages)
    assert (c01.claim_id, c01.entity, c01.indicator, c01.metric, c01.direction, c01.value) == \
           ("C01", "DEU", "co2_emissions", PC, DOWN, 30)
    assert (c01.baseline_year, c01.target_year, c01.page) == (2010, 2022, 1)
    assert (c02.claim_id, c02.entity, c02.metric, c02.value, c02.baseline_year, c02.target_year) == \
           ("C02", "WLD", SH, 31.2, 2000, 2020)


# ---- LLM batch extraction -------------------------------------------------------------------
PAGE = ("Highlights\nGermany's CO2 emissions fell by 30% between 2010 and 2022. "
        "Water consumption was reduced by 12% in 2024.")


def test_llm_batch_keeps_verbatim_claims_and_drops_invented_ones():
    llm = FakeLLM({"claims": [
        {"text": "Germany's CO2 emissions fell by 30% between 2010 and 2022", "indicator": "co2_emissions",
         "metric": "pct_change", "direction": "decrease", "value": 30, "unit": "percent",
         "baseline_year": 2010, "target_year": 2022, "entity": "Germany"},
        {"text": "Our Scope 3 emissions fell by 45% in 2024.", "indicator": "ghg_emissions",
         "metric": "pct_change", "direction": "decrease", "value": 45, "unit": "%", "target_year": 2024},
    ]})
    claims = parse_pages({7: PAGE}, report_year=2025, llm=llm)
    assert len(llm.calls) == 1  # one call per page
    assert [c.text for c in claims] == ["Germany's CO2 emissions fell by 30% between 2010 and 2022.",
                                        "Water consumption was reduced by 12% in 2024."]  # LLM missed it: regex
    first = claims[0]
    assert (first.entity, first.unit, first.value, first.page, first.claim_id) == ("DEU", "%", 30, 7, "C01")
    assert not any("45" in c.text for c in claims)


def test_llm_numbers_not_in_the_text_fall_back_to_regex():
    llm = FakeLLM({"claims": [{"text": "Water consumption was reduced by 12% in 2024.", "metric": "pct_change",
                               "indicator": "water_consumption", "direction": "decrease", "value": 21,
                               "unit": "%", "baseline_year": 2019, "target_year": 2024}]})
    claim = parse_pages({1: "Water consumption was reduced by 12% in 2024."}, 2025, llm)[0]
    assert (claim.value, claim.baseline_year, claim.target_year) == (12, 2023, 2024)


def test_llm_none_or_garbage_falls_back_to_regex():
    expected = [(c.claim_id, c.text, c.value, c.entity) for c in parse_pages({1: PAGE}, 2025)]
    for result in (None, {"claims": "oops"}, {"unexpected": 1}, {"claims": [{"no_text": True}]}):
        llm = FakeLLM(result)
        assert [(c.claim_id, c.text, c.value, c.entity) for c in parse_pages({1: PAGE}, 2025, llm)] == expected


def test_llm_that_raises_never_breaks_parsing():
    class Broken:
        def complete_json(self, *args, **kwargs):
            raise RuntimeError("boom")

    assert [c.text for c in parse_report(PAGE, 2025, Broken())] == [c.text for c in parse_report(PAGE, 2025)]


def test_llm_scaled_units_are_canonicalised_and_explicit_ids_kept():
    llm = FakeLLM({"claims": [{"text": "C04 Waste generated was 1.2 million tonnes in 2023.", "indicator": "waste",
                               "metric": "absolute", "direction": "none", "value": 1.2, "unit": "million tonnes",
                               "target_year": 2023, "entity": None}]})
    claim = parse_report("C04 Waste generated was 1.2 million tonnes in 2023.", 2025, llm)[0]
    assert (claim.claim_id, claim.value, claim.unit, claim.entity) == ("C04", 1_200_000, "tonnes", None)


def test_sentence_split_survives_abbreviations_and_decimals():
    from llm_router.claims import split_sentences

    text = ("Emissions fell 20% vs. 2019, e.g. in the U.S. Plants cut 1.2 million tonnes of waste. "
            "U.S. emissions fell 3% (see Fig. 3). Done!")
    assert split_sentences(text) == ["Emissions fell 20% vs. 2019, e.g. in the U.S.",
                                     "Plants cut 1.2 million tonnes of waste.",
                                     "U.S. emissions fell 3% (see Fig. 3).", "Done!"]


def test_wrapped_pdf_line_ending_in_preposition_is_not_a_heading():
    # PDF text wraps mid-sentence; "fell by" + "12% ..." must stay one claim, not a heading + fragment.
    page = ("Our manufacturing sites improved efficiency across the year and our\n"
            "Scope 1 emissions fell by\n"
            "12% from 2022 to 2024, driven by electrification.\n\n"
            "Operations\n"
            "Water consumption was reduced by 18% compared with 2023.\n")
    claims = parse_pages({2: page}, report_year=2024)
    texts = [c.text for c in claims]
    assert any("fell by 12% from 2022 to 2024" in t for t in texts), texts
    assert not any(t.strip() == "Operations" for t in texts)
    assert all(c.page == 2 for c in claims)
