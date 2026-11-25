"""Verifier & Calculator: deterministic verdicts, no LLM. Owner: Sibi.

Supported    computed value within tolerance of the claim
Contradicted evidence covers the claim but the numbers differ
Unsupported  a needed year/indicator is missing, units can't be reconciled,
             or sources conflict (flagged, never silently picked)

Claim kinds
- "%" change      relative change (target - baseline) / baseline × 100, ±TOL_PP
- "pp" change     level difference target - baseline of two % levels, ±TOL_PP
- absolute change claim unit is an amount ("cut by 300 t"): target - baseline, ±TOL_REL
- share level     recorded % level, ±TOL_PP
- absolute level  recorded amount in any convertible unit, ±TOL_REL

Ambiguity: a "%" change on a share indicator (renewables 60% -> 75%) is computed as a
relative change (+25%). If reading the "%" as percentage points (+15 pp) would match
while the relative reading does not, the verdict keeps the relative status and gets
flag "ambiguous_pp", so a reviewer sees that the report may have meant points.
"""
from __future__ import annotations

from typing import Optional

from trust_layer.explain import entity_name
from auditor.indicators import is_share, label
from auditor.schemas import Claim, Direction, Evidence, Fact, Metric, Status, Verdict
from trust_layer.units import base_unit, fmt_number, is_pp, normalize

TOL_PP = 1.0        # percentage points, for % changes and shares (the demo answer key depends on it)
TOL_REL = 0.02      # 2 % relative, for absolute amounts
CONFLICT_REL = 0.005


def verify(claim: Claim, evidence: Evidence) -> Verdict:
    if claim.metric == Metric.VAGUE or claim.value is None or claim.indicator is None:
        return _unsupported(claim, "No checkable number or indicator in the claim.")

    # Facts and claims match only when their entity codes are equal (entities.py contract).
    facts = [f for f in evidence.facts if f.entity == claim.entity]
    other = [f for f in evidence.facts if f.entity != claim.entity]
    flags = ["entity_mismatch"] if other else []
    if other and not facts:
        about = ", ".join(sorted({entity_name(f.entity) for f in other}))
        return _unsupported(claim, f"Evidence is about {about}, not {entity_name(claim.entity)}.",
                            evidence=other, flags=flags)

    by_year: dict[int, list[Fact]] = {}
    for fact in facts:
        by_year.setdefault(fact.year, []).append(fact)

    claim_base = base_unit(claim.unit, claim.indicator)
    if is_pp(claim.unit):
        v = _difference(claim, by_year, points=True)
    elif claim.metric == Metric.PCT_CHANGE and claim_base not in (None, "", "%"):
        v = _difference(claim, by_year, points=False)
    elif claim.metric == Metric.PCT_CHANGE:
        v = _pct_change(claim, by_year)
    elif claim.metric == Metric.SHARE:
        v = _level(claim, by_year, tolerance=TOL_PP, relative=False)
    else:
        v = _level(claim, by_year, tolerance=TOL_REL, relative=True)
    if flags:
        v.flags = flags + [f for f in v.flags if f not in flags]
    return v


# ---- helpers ------------------------------------------------------------------

def _prefix(claim: Claim) -> str:
    return f"{entity_name(claim.entity)}: " if claim.entity else ""


def _what(claim: Claim) -> str:
    where = f" for {entity_name(claim.entity)}" if claim.entity else ""
    return f"{label(claim.indicator)}{where}"


def _pick(facts: list[Fact], indicator: Optional[str]) -> tuple[Optional[float], Optional[str], bool]:
    """One normalised value for a year, or conflict=True if sources disagree."""
    values = [normalize(f.value, f.unit, indicator) for f in facts]
    units = {u for _, u in values}
    nums = [v for v, _ in values]
    if len(units) > 1 or (max(nums) - min(nums)) > CONFLICT_REL * max(abs(max(nums)), 1e-9):
        return None, None, True
    return nums[0], units.pop(), False


def _signed(claim: Claim) -> float:
    return -claim.value if claim.direction == Direction.DECREASE else claim.value


def _matches(computed: float, claim: Claim, tolerance: float, relative: bool = False) -> bool:
    """Signed comparison when the claim states a direction, magnitude only otherwise."""
    if claim.direction == Direction.NONE:
        computed, claimed = abs(computed), abs(claim.value)
    else:
        claimed = _signed(claim)
    diff = abs(computed - claimed)
    if relative:
        diff /= max(abs(claimed), 1e-9)
    return diff <= tolerance


def _two_years(claim: Claim, by_year: dict[int, list[Fact]]):
    """(base, target, unit, used facts) or an Unsupported verdict."""
    b, t = claim.baseline_year, claim.target_year
    every = [f for fs in by_year.values() for f in fs]
    if b is None or t is None or b not in by_year or t not in by_year:
        missing = [str(y) for y in (b, t) if y is not None and y not in by_year]
        need = ", ".join(missing) if missing else "a baseline and a target year"
        return _unsupported(claim, f"No {_what(claim)} data for {need}.", evidence=every)
    base, base_unit_, c1 = _pick(by_year[b], claim.indicator)
    tgt, tgt_unit, c2 = _pick(by_year[t], claim.indicator)
    used = by_year[b] + by_year[t]
    if c1 or c2:
        return _conflict(claim, used)
    if base_unit_ != tgt_unit:
        return _unsupported(claim, f"Baseline unit {base_unit_} differs from target unit {tgt_unit}.", evidence=used)
    return base, tgt, tgt_unit, used


def _pct_change(claim: Claim, by_year: dict[int, list[Fact]]) -> Verdict:
    got = _two_years(claim, by_year)
    if isinstance(got, Verdict):
        return got
    base, tgt, unit, used = got
    b, t = claim.baseline_year, claim.target_year
    if not base:
        return _unsupported(claim, f"Baseline {b} is zero, so a % change is undefined.", evidence=used)

    computed = (tgt - base) / base * 100
    ok = _matches(computed, claim, TOL_PP)
    status = Status.SUPPORTED if ok else Status.CONTRADICTED
    # Show the numbers as the source wrote them (658.09 Mt CO2e, not 658,088,600 t); the ratio is the same.
    sb, st_, scale = base, tgt, ""
    rb = next((f for f in used if f.year == b), None)
    rt = next((f for f in used if f.year == t), None)
    if rb and rt and rb.unit.strip().lower() == rt.unit.strip().lower() and rb.value != base:
        sb, st_, scale = rb.value, rt.value, f", in {rb.unit}"
    calc = (f"{_prefix(claim)}({fmt_number(st_)} - {fmt_number(sb)}) / {fmt_number(sb)} × 100 "
            f"= {computed:+.1f}% ({b} → {t}{scale})")
    flags: list[str] = []
    if is_share(claim.indicator) and unit == "%":
        points = tgt - base
        if not ok and _matches(points, claim, TOL_PP):
            flags.append("ambiguous_pp")
            calc += (f"; read as percentage points: {fmt_number(tgt)} - {fmt_number(base)} "
                     f"= {points:+.1f} pp, which would match")
    return Verdict(claim=claim, status=status, computed_value=round(computed, 2), calculation=calc,
                   evidence=used, flags=flags)


def _difference(claim: Claim, by_year: dict[int, list[Fact]], points: bool) -> Verdict:
    """target - baseline: percentage points of two % levels, or an absolute amount."""
    got = _two_years(claim, by_year)
    if isinstance(got, Verdict):
        return got
    base, tgt, unit, used = got
    b, t = claim.baseline_year, claim.target_year
    if points:
        if unit not in ("%", "", None):
            return _unsupported(claim, f"Percentage points need % levels, but the evidence is in {unit}.",
                                evidence=used)
        computed, claim_unit, ok_args = tgt - base, "pp", (TOL_PP, False)
    else:
        claimed_base = base_unit(claim.unit, claim.indicator)
        if claimed_base != unit:
            return _unsupported(claim, f"Claim unit {claim.unit} can't be compared with {unit}.", evidence=used)
        factor = normalize(1.0, claim.unit, claim.indicator)[0]
        base, tgt = base / factor, tgt / factor  # show in the claim's own unit
        computed, claim_unit, ok_args = tgt - base, f" {claim.unit}", (TOL_REL, True)
    status = Status.SUPPORTED if _matches(computed, claim, *ok_args) else Status.CONTRADICTED
    calc = (f"{_prefix(claim)}{fmt_number(tgt)} - {fmt_number(base)} = "
            f"{'+' if computed >= 0 else ''}{fmt_number(round(computed, 2))}{claim_unit} ({b} → {t})")
    return Verdict(claim=claim, status=status, computed_value=round(computed, 2), calculation=calc, evidence=used)


def _level(claim: Claim, by_year: dict[int, list[Fact]], tolerance: float, relative: bool) -> Verdict:
    t = claim.target_year
    if t not in by_year:
        return _unsupported(claim, f"No {_what(claim)} data for {t}.",
                            evidence=[f for fs in by_year.values() for f in fs])
    facts = by_year[t]
    actual, unit, conflict = _pick(facts, claim.indicator)
    if conflict:
        return _conflict(claim, facts)
    claimed, claimed_unit = normalize(claim.value, claim.unit, claim.indicator)
    if relative and claimed_unit != unit:
        return _unsupported(claim, f"Claim unit {claim.unit} can't be compared with {facts[0].unit}.", evidence=facts)
    if not relative and claimed_unit and unit and claimed_unit != unit:
        return _unsupported(claim, f"Claim unit {claim.unit} can't be compared with {facts[0].unit}.", evidence=facts)

    diff = abs(actual - claimed) / max(abs(actual), 1e-9) if relative else abs(actual - claimed)
    status = Status.SUPPORTED if diff <= tolerance else Status.CONTRADICTED
    factor = claimed / claim.value if claim.value else 1.0
    shown = actual / factor if factor else actual  # recorded value in the claim's own unit
    sep = "" if (claim.unit or "").strip() in ("%", "") else " "
    unit_txt = f"{sep}{claim.unit or ''}"
    calc = f"{_prefix(claim)}claimed {fmt_number(claim.value)}{unit_txt} vs recorded {fmt_number(shown)}{unit_txt}"
    raw = facts[0]
    if fmt_number(raw.value) != fmt_number(shown):  # evidence written in another scale: show it as written
        calc += f" (= {fmt_number(raw.value)} {raw.unit})"
    calc += f" ({t})"
    return Verdict(claim=claim, status=status, computed_value=round(shown, 2), calculation=calc, evidence=facts)


def _unsupported(claim: Claim, why: str, evidence: Optional[list[Fact]] = None,
                 flags: Optional[list[str]] = None) -> Verdict:
    return Verdict(claim=claim, status=Status.UNSUPPORTED, calculation=why, evidence=evidence or [],
                   flags=list(flags or []))


def _conflict(claim: Claim, facts: list[Fact]) -> Verdict:
    values = "; ".join(f"{fmt_number(f.value)} {f.unit} ({f.source}, {f.year})" for f in facts)
    return Verdict(claim=claim, status=Status.UNSUPPORTED, conflict=True, flags=["conflict"],
                   calculation=f"{_prefix(claim)}Sources disagree: {values}", evidence=facts)
