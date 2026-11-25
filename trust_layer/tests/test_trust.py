"""Trust layer: units, verifier maths, Citation Guard, confidence & greenwash, template text."""
import pytest

from trust_layer import verifier
from trust_layer.explain import entity_name, template_explanation
from trust_layer.guard import apply_guard, numbers_ok, stated_verdicts, verdict_ok
from auditor.schemas import (Claim, Direction, Evidence, ExtractionMethod, Fact, Metric, SourceType, Status,
                             Verdict)
from trust_layer.scoring import confidence, greenwash_flags, score
from trust_layer.units import fmt_number, is_pp, normalize
from trust_layer.verifier import verify

WDI_URL = "https://data.worldbank.org/indicator/EN.GHG.CO2.MT.CE.AR5?locations=DE"
SERIES = "EN.GHG.CO2.MT.CE.AR5"


def fact(value, year, unit="%", indicator="renewable_share", source="esg_2025.csv", **kw):
    kw.setdefault("quote", f"{indicator},{unit},{value}")
    return Fact(fact_id=f"{source}-{year}-{value}", indicator=indicator, value=value, unit=unit, year=year,
                source=source, page=1, locator=f"row {year - 2020}", **kw)


def wdi(value, year=2020, unit="Mt CO2e", entity="DEU"):
    return fact(value, year, unit=unit, indicator="ghg_emissions", source="World Bank WDI", entity=entity,
                url=WDI_URL, source_type=SourceType.OFFICIAL, quote=f"{entity},{year},{value}")


def claim(value, unit="%", metric=Metric.PCT_CHANGE, direction=Direction.INCREASE, indicator="renewable_share",
          baseline=2024, target=2025, entity=None, text="claim"):
    return Claim(claim_id="T", text=text, indicator=indicator, metric=metric, direction=direction, value=value,
                 unit=unit, baseline_year=baseline, target_year=target, entity=entity)


def run(c, facts):
    return verify(c, Evidence(claim_id=c.claim_id, facts=facts))


SHARE_60_75 = [fact(60, 2024), fact(75, 2025)]


# ---- 1. units -------------------------------------------------------------------------------------

@pytest.mark.parametrize("unit, expected", [
    ("Mt CO2e", (1e6, "tCO2e")), ("MtCO2e", (1e6, "tCO2e")), ("mt co2e", (1e6, "tCO2e")),
    ("Mt CO₂e", (1e6, "tCO2e")), ("Mt CO2eq", (1e6, "tCO2e")), ("Mt CO2-eq", (1e6, "tCO2e")),
    ("Mt CO2 equivalent", (1e6, "tCO2e")), ("kt CO2e", (1e3, "tCO2e")), ("Gg CO2e", (1e3, "tCO2e")),
    ("kt of CO2 equivalent", (1e3, "tCO2e")), ("million tonnes of CO2 equivalent", (1e6, "tCO2e")),
    ("Mt CO2e/yr", (1e6, "tCO2e")), ("Mt", (1e6, "t")), ("million tonnes", (1e6, "t")),
    ("billion m3", (1e9, "kL")), ("billion cubic meters", (1e9, "kL")), ("km3", (1e9, "kL")), ("m3", (1, "kL")),
    ("m³", (1, "kL")), ("TWh", (1e6, "MWh")), ("GWh", (1e3, "MWh")), ("PJ", (1e6 / 3.6, "MWh")),
    ("EJ", (1e9 / 3.6, "MWh")), ("% of total", (1, "%")), ("% of land area", (1, "%")),
    ("% of total final energy consumption", (1, "%")), ("percent", (1, "%")), ("pp", (1, "pp")),
    ("percentage points", (1, "pp")), ("ML", (1e3, "kL")), ("tCO2e", (1, "tCO2e")),
])
def test_dataset_units(unit, expected):
    value, base = normalize(1, unit)
    assert base == expected[1]
    assert value == pytest.approx(expected[0])


def test_unknown_units_unchanged_and_bare_mass_is_co2e_for_emissions():
    assert normalize(5, "widgets") == (5, "widgets")
    assert normalize(5, "CO2e") == (5, "CO2e")
    assert normalize(700, "Mt", "ghg_emissions") == (7e8, "tCO2e")
    assert normalize(700, "Mt", "waste") == (7e8, "t")
    assert is_pp("percentage points") and not is_pp("%")


def test_fmt_number_never_uses_exponents():
    assert fmt_number(7e8) == "700,000,000"
    assert fmt_number(1000) == "1000"
    assert fmt_number(674.7531) == "674.75"
    assert fmt_number(-30.0) == "-30"


# ---- 2. verifier ----------------------------------------------------------------------------------

def test_tolerance_is_one_point():
    assert verifier.TOL_PP == 1.0


def test_percentage_points_compare_level_difference():
    v = run(claim(15, unit="pp"), SHARE_60_75)
    assert v.status == Status.SUPPORTED and v.computed_value == 15.0 and "pp" in v.calculation
    assert run(claim(25, unit="pp"), SHARE_60_75).status == Status.CONTRADICTED
    assert run(claim(15.9, unit="percentage points"), SHARE_60_75).status == Status.SUPPORTED
    assert run(claim(15, unit="pp", direction=Direction.DECREASE), SHARE_60_75).status == Status.CONTRADICTED


def test_percentage_points_need_two_percent_levels():
    no_base = run(claim(15, unit="pp", metric=Metric.SHARE, baseline=None), SHARE_60_75)
    assert no_base.status == Status.UNSUPPORTED
    tonnes = [fact(60, 2024, unit="t"), fact(75, 2025, unit="t")]
    assert run(claim(15, unit="pp"), tonnes).status == Status.UNSUPPORTED


def test_share_percent_change_flags_pp_ambiguity():
    v = run(claim(15), SHARE_60_75)  # 60% -> 75% is +25% relative, +15 pp
    assert v.status == Status.CONTRADICTED and v.computed_value == 25.0
    assert "ambiguous_pp" in v.flags and "percentage points" in v.calculation
    clean = run(claim(25), SHARE_60_75)
    assert clean.status == Status.SUPPORTED and "ambiguous_pp" not in clean.flags
    emissions = [fact(60, 2024, unit="tCO2e", indicator="ghg_emissions"),
                 fact(75, 2025, unit="tCO2e", indicator="ghg_emissions")]
    assert "ambiguous_pp" not in run(claim(15, indicator="ghg_emissions"), emissions).flags


def test_calculation_names_the_entity():
    facts = [wdi(800, 2019), wdi(700, 2020)]
    v = run(claim(12.5, direction=Direction.DECREASE, indicator="ghg_emissions", baseline=2019, target=2020,
                  entity="DEU"), facts)
    assert v.status == Status.SUPPORTED and v.calculation.startswith("Germany: ")
    assert entity_name("WLD") == "World" and entity_name("EUU") == "European Union"
    assert entity_name("KOR") == "South Korea" and entity_name(None) == "the reporting organisation"


def test_facts_about_another_entity_are_not_used():
    v = run(claim(700, unit="Mt", metric=Metric.ABSOLUTE, indicator="ghg_emissions", target=2020, entity="DEU"),
            [wdi(700, entity="WLD")])
    assert v.status == Status.UNSUPPORTED and "entity_mismatch" in v.flags and "World" in v.calculation


def test_absolute_claims_across_units():
    c = claim(700, unit="Mt", metric=Metric.ABSOLUTE, indicator="ghg_emissions", target=2020, entity="DEU")
    v = run(c, [wdi(700_000, unit="kt CO2e")])
    assert v.status == Status.SUPPORTED and v.computed_value == 700.0
    assert "700,000 kt CO2e" in v.calculation
    assert run(c.model_copy(update={"value": 800}), [wdi(700_000, unit="kt CO2e")]).status == Status.CONTRADICTED
    water = c.model_copy(update={"unit": "billion m3"})
    assert run(water, [wdi(700, unit="Mt CO2e")]).status == Status.UNSUPPORTED


def test_absolute_change_claim_in_amount_units():
    facts = [fact(1000, 2024, unit="tCO2e", indicator="ghg_emissions"),
             fact(700, 2025, unit="tCO2e", indicator="ghg_emissions")]
    c = claim(300, unit="tCO2e", direction=Direction.DECREASE, indicator="ghg_emissions")
    v = run(c, facts)
    assert v.status == Status.SUPPORTED and v.computed_value == -300.0


def test_same_value_in_two_units_is_not_a_conflict_but_different_values_are():
    c = claim(700, unit="Mt CO2e", metric=Metric.ABSOLUTE, indicator="ghg_emissions", target=2020, entity="DEU")
    agree = [wdi(700), fact(700_000, 2020, unit="kt CO2e", indicator="ghg_emissions", entity="DEU",
                            source="UNFCCC", source_type=SourceType.OFFICIAL)]
    assert run(c, agree).status == Status.SUPPORTED
    disagree = [wdi(700), fact(650, 2020, unit="Mt CO2e", indicator="ghg_emissions", entity="DEU", source="UNFCCC")]
    v = run(c, disagree)
    assert v.status == Status.UNSUPPORTED and v.conflict and "conflict" in v.flags


# ---- 3. guard -------------------------------------------------------------------------------------

def official_verdict(status=Status.SUPPORTED, explanation=""):
    c = claim(675, unit="Mt", metric=Metric.ABSOLUTE, indicator="ghg_emissions", target=2020, entity="DEU")
    v = run(c, [wdi(674.7531)])
    return v.model_copy(update={"status": status, "explanation": explanation})


def test_numbers_tolerate_separators_percent_and_rounding():
    v = official_verdict()
    for text in ["Germany emitted 674.75 Mt in 2020.", "about 674.8 Mt", "roughly 675 Mt CO2e",
                 "674,753,100 tCO2e is 674.7531 million tonnes", "675 Mt (2020)"]:
        assert numbers_ok(text, v), text
    for text in ["Germany emitted 676 Mt.", "674.6 Mt", "down 55%"]:
        assert not numbers_ok(text, v), text


def test_numbers_ignore_sources_urls_series_codes_and_entities():
    v = official_verdict()
    text = (f"Source: World Bank WDI, {SERIES} · DEU · 2020 ({WDI_URL}); Germany emitted 675 Mt. "
            "Scope 1 and 2 emissions, see EN.ATM.CO2E.KT also.")
    assert numbers_ok(text, v)
    named = v.model_copy(update={"evidence": [v.evidence[0].model_copy(update={"source": "UN Data 2023 v4"})]})
    assert numbers_ok("UN Data 2023 v4 shows 675 Mt", named)


def test_quotes_are_refound_in_official_snapshots():
    v = official_verdict(explanation="Supported: Germany emitted 675 Mt in 2020.")
    ok = apply_guard(v, {"World Bank WDI": {1: "country,year,value\nDEU,2020,674.7531\n"}})
    assert ok.status == Status.SUPPORTED and "guard_failed" not in ok.flags
    assert ok.explanation == v.explanation
    bad = apply_guard(v, {"World Bank WDI": {1: "DEU,2020,999"}})
    assert bad.status == Status.UNSUPPORTED and {"guard_failed", "narration_replaced"} <= set(bad.flags)
    assert "cannot be verified" in bad.explanation


def test_stated_verdict_words():
    assert stated_verdicts("The claim is supported by the data.") == {Status.SUPPORTED}
    assert stated_verdicts("The claim is not supported.") == {Status.UNSUPPORTED}
    assert stated_verdicts("Unsupported: no 2025 figure.") == {Status.UNSUPPORTED}
    assert stated_verdicts("The data contradicts the claim.") == {Status.CONTRADICTED}
    assert stated_verdicts("The evidence does not contradict the claim.") == set()
    assert stated_verdicts("Emissions fell 30%.") == set()


def test_wrong_verdict_word_is_replaced():
    pages = {"World Bank WDI": {1: "DEU,2020,674.7531"}}
    v = official_verdict(status=Status.CONTRADICTED, explanation="The claim is supported: 675 Mt in 2020.")
    out = apply_guard(v, pages)
    assert "narration_replaced" in out.flags and out.status == Status.CONTRADICTED
    assert "supported" not in out.explanation.lower() and verdict_ok(out.explanation, out)
    same = official_verdict(status=Status.SUPPORTED, explanation="Supported: Germany emitted 675 Mt in 2020.")
    assert "narration_replaced" not in apply_guard(same, pages).flags


# ---- 4. scoring -----------------------------------------------------------------------------------

def scored(text, value=None, metric=Metric.VAGUE):
    c = Claim(claim_id="G", text=text, value=value, metric=metric, unit="%" if value else None)
    return score(Verdict(claim=c, status=Status.UNSUPPORTED))


@pytest.mark.parametrize("text, value, expected", [
    ("Our products are eco-friendly.", None, {"greenwash", "greenwash:vague"}),
    ("We are carbon neutral.", None, {"greenwash", "greenwash:no_target_year"}),
    ("We will be net-zero.", None, {"greenwash", "greenwash:no_target_year"}),
    ("Climate neutral operations, achieved through offsets.", None, {"greenwash:no_target_year", "greenwash:offsets"}),
    ("Emissions are compensated with carbon credits.", None, {"greenwash", "greenwash:offsets"}),
    ("We are the greenest airline.", None, {"greenwash", "greenwash:superlative"}),
    ("The most sustainable and industry-leading packaging.", None, {"greenwash:superlative"}),
])
def test_greenwash_rules_fire(text, value, expected):
    flags = set(scored(text, value).flags)
    assert expected <= flags and "greenwash" in flags


@pytest.mark.parametrize("text, value, metric, absent", [
    ("We will reach net zero by 2040, cutting emissions 40%.", 40, Metric.PCT_CHANGE, "greenwash:no_target_year"),
    ("Industry-leading 40% cut in emissions.", 40, Metric.PCT_CHANGE, "greenwash:superlative"),
    ("Efficiency gains, leading to 12% lower energy use.", 12, Metric.PCT_CHANGE, "greenwash:superlative"),
    ("Greenhouse gas emissions fell 10%.", 10, Metric.PCT_CHANGE, "greenwash"),
    ("We follow best practices on water, cutting use 5%.", 5, Metric.PCT_CHANGE, "greenwash:superlative"),
])
def test_greenwash_rules_stay_quiet(text, value, metric, absent):
    assert absent not in scored(text, value, metric).flags


def checked(facts, method=ExtractionMethod.TABLE):
    facts = [f.model_copy(update={"method": method}) for f in facts]
    return Verdict(claim=claim(25), status=Status.SUPPORTED, evidence=facts)


def test_confidence_rewards_agreeing_sources_and_penalises_text():
    single = checked(SHARE_60_75)
    assert confidence(single) == 0.8
    both = SHARE_60_75 + [fact(60.0, 2024, source="audit.xlsx"), fact(75.0, 2025, source="audit.xlsx")]
    assert confidence(checked(both)) == 0.9
    half = SHARE_60_75 + [fact(75.0, 2025, source="audit.xlsx")]  # only 2025 corroborated
    assert confidence(checked(half)) == 0.8
    assert confidence(checked(SHARE_60_75, ExtractionMethod.TEXT)) == 0.5
    assert confidence(Verdict(claim=claim(25), status=Status.UNSUPPORTED, evidence=SHARE_60_75)) == 0.0


# ---- 5. template explanations ---------------------------------------------------------------------

def test_template_names_entity_source_and_url():
    text = template_explanation(official_verdict())
    assert "Germany" in text and "World Bank WDI" in text and WDI_URL in text and "matches" in text


def _all_kinds_of_verdicts():
    emissions = [fact(1000, 2024, unit="tCO2e", indicator="ghg_emissions"),
                 fact(700, 2025, unit="tCO2e", indicator="ghg_emissions")]
    big = [wdi(1_234_567.891, 2019, unit="kt CO2e"), wdi(1_100_000.5, 2020, unit="kt CO2e")]
    cases = [
        (claim(40, direction=Direction.DECREASE, indicator="ghg_emissions"), emissions),
        (claim(30, direction=Direction.DECREASE, indicator="ghg_emissions"), emissions),
        (claim(300, unit="tCO2e", direction=Direction.DECREASE, indicator="ghg_emissions"), emissions),
        (claim(15), SHARE_60_75), (claim(25), SHARE_60_75), (claim(15, unit="pp"), SHARE_60_75),
        (claim(75, metric=Metric.SHARE, baseline=None), SHARE_60_75),
        (claim(675, unit="Mt", metric=Metric.ABSOLUTE, indicator="ghg_emissions", target=2020, entity="DEU"),
         [wdi(674.7531)]),
        (claim(1.1, unit="billion tonnes", metric=Metric.ABSOLUTE, indicator="ghg_emissions", target=2020,
               entity="DEU"), big),
        (claim(10.9, direction=Direction.DECREASE, indicator="ghg_emissions", baseline=2019, target=2020,
               entity="DEU"), big),
        (claim(700, unit="Mt", metric=Metric.ABSOLUTE, indicator="ghg_emissions", target=2020, entity="DEU"),
         [wdi(700, entity="WLD")]),
        (claim(30, direction=Direction.DECREASE, indicator="ghg_emissions", target=2026), emissions),
        (claim(700, unit="Mt CO2e", metric=Metric.ABSOLUTE, indicator="ghg_emissions", target=2020, entity="DEU"),
         [wdi(700), fact(650, 2020, unit="Mt CO2e", indicator="ghg_emissions", entity="DEU", source="UN 2024.csv")]),
        (Claim(claim_id="V", text="We are green."), []),
    ]
    return [run(c, facts) for c, facts in cases]


def test_every_template_explanation_passes_the_guard():
    for v in _all_kinds_of_verdicts():
        text = template_explanation(v)
        assert numbers_ok(text, v), text
        assert verdict_ok(text, v), text
        pages: dict[str, dict[int, str]] = {}
        for f in v.evidence:
            pages.setdefault(f.source, {1: ""})[1] += f.quote + "\n"
        out = apply_guard(v.model_copy(update={"explanation": text}), pages)
        assert "narration_replaced" not in out.flags and out.status == v.status, text
