"""Compare LLM models on the evaluation datasets (Review 1 feedback: "compare different LLM models").

Each mode forces ONE provider (LLM_PROVIDER_ORDER=<provider>), so a fallback never hides a model's
failures:
    baseline  no LLM: regex claim parser + template explanations
    gemini    Gemini only          (needs GEMINI_API_KEY in .env)
    groq      Groq only            (needs GROQ_API_KEY in .env)
    lmstudio  local LM Studio only (needs its server on 127.0.0.1:1234)

Calls are spaced out to stay under free-tier rate limits, so the comparison measures the models,
not the quotas. Per mode it reports: claims found (extraction recall), extra claims, verdict
accuracy, flag accuracy, explanations rejected by the Citation Guard (invented numbers or wrong
verdict words: a hallucination measure), LLM calls / failures, mean LLM latency and total time.

Run from the repo root:
    python scripts/compare_llms.py                      # every mode that is available
    python scripts/compare_llms.py --modes baseline,groq
Writes docs/llm_comparison.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as ev  # noqa: E402  (scripts/evaluate.py: datasets, answer-key matching, metrics)
import auditor.pipeline as pipeline  # noqa: E402
from llm_router import make_router  # noqa: E402
from llm_router.config import load_dotenv  # noqa: E402

MODES = ["baseline", "gemini", "groq", "lmstudio"]  # or provider:model, e.g. groq:openai/gpt-oss-20b
GAP_SECONDS = {"gemini": 6.5, "groq": 2.2, "lmstudio": 0.0}  # free tiers: ~10 and ~30 requests per minute
MODEL_ENV = {"gemini": "GEMINI_MODEL", "groq": "GROQ_MODEL", "lmstudio": "LMSTUDIO_MODEL"}
CORE = ("PS sample", "Country demo", "Company (Greenfield)")
OUT = ROOT / "docs" / "llm_comparison.md"
DATASETS: list = []  # set in main()


class Throttled:
    """Wraps the pipeline's router: spaces calls out and times the LLM itself (not the waiting)."""

    def __init__(self, inner, gap: float) -> None:
        self.inner, self.gap = inner, gap
        self._last = 0.0
        self.calls = 0
        self.answered = 0  # calls that got a usable reply (None = failed, skipped in cooldown, or bad JSON)
        self.llm_seconds = 0.0

    def _call(self, fn, *args):
        wait = self.gap - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        t0 = time.perf_counter()
        result = None
        try:
            result = fn(*args)
            return result
        finally:
            self.llm_seconds += time.perf_counter() - t0
            self.calls += 1
            self.answered += result is not None
            self._last = time.monotonic()

    def complete(self, system, user):
        return self._call(self.inner.complete, system, user)

    def complete_json(self, system, user, schema=None, name="result"):
        return self._call(self.inner.complete_json, system, user, schema, name)

    def status(self):
        return self.inner.status()

    @property
    def last_provider(self):
        return getattr(self.inner, "last_provider", None)

    @property
    def stats(self):
        return getattr(self.inner, "stats", {}) or {}


def available(mode: str) -> tuple[bool, str]:
    mode = mode.split(":", 1)[0]
    if mode == "baseline":
        return True, "no LLM"
    if mode == "gemini":
        return bool(os.environ.get("GEMINI_API_KEY", "").strip()), "GEMINI_API_KEY"
    if mode == "groq":
        return bool(os.environ.get("GROQ_API_KEY", "").strip()), "GROQ_API_KEY"
    base = os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1").replace("localhost", "127.0.0.1")
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/models", timeout=2) as r:
            return bool(json.load(r).get("data")), "LM Studio server with a model loaded"
    except Exception:
        return False, "LM Studio server with a model loaded"


def run_mode(mode: str) -> dict:
    """All datasets with one mode. Returns per-dataset metrics plus LLM call totals."""
    routers: list[Throttled] = []
    last_results: list = []
    provider, _, model = mode.partition(":")
    if provider != "baseline":
        os.environ["LLM_PROVIDER_ORDER"] = provider
        if model:
            os.environ[MODEL_ENV[provider]] = model

        def factory():
            r = Throttled(make_router(), GAP_SECONDS[provider])
            routers.append(r)
            return r

        pipeline.make_router = factory
    original_run_audit = ev.run_audit

    def capture(*args, **kwargs):
        res = original_run_audit(*args, **kwargs)
        last_results.append(res)
        return res

    ev.run_audit = capture
    try:
        per_ds = []
        for ds in DATASETS:
            m = ev.evaluate(ds, use_llm=(provider != "baseline"))
            res = last_results[-1]
            m["narration_rejected"] = sum("narration_replaced" in v.flags for v in res.verdicts)
            m["n_verdicts"] = len(res.verdicts)
            per_ds.append(m)
    finally:
        ev.run_audit = original_run_audit
        pipeline.make_router = make_router

    calls = sum(r.calls for r in routers)
    answered = sum(r.answered for r in routers)
    llm_secs = sum(r.llm_seconds for r in routers)
    return dict(mode=mode, datasets=per_ds, calls=calls, answered=answered,
                mean_llm_s=(llm_secs / calls) if calls else None)


def summarise(r: dict) -> dict:
    ds = r["datasets"]
    n_key = sum(d["n_key"] for d in ds)
    found = sum(d["n_found"] for d in ds)
    correct = sum(round(d["acc"] * d["n_key"]) for d in ds)
    flag_ok = sum(d["flags"][0] for d in ds)
    flag_total = sum(d["flags"][1] for d in ds)
    extra = sum(max(0, d["n_extracted"] - d["n_found"]) for d in ds)
    return dict(mode=r["mode"], n_key=n_key, found=found, extra=extra, correct=correct,
                flags=f"{flag_ok}/{flag_total}", rejected=sum(d["narration_rejected"] for d in ds),
                verdicts=sum(d["n_verdicts"] for d in ds), calls=r["calls"], answered=r["answered"],
                mean_llm_s=r["mean_llm_s"], secs=sum(d["secs"] for d in ds),
                guard_fail=sum(d["guard_fail"] for d in ds))


def render(rows: list[dict], skipped: list[str], results: list[dict]) -> str:
    out = [
        "# LLM model comparison: PS 99 Citation Auditor", "",
        f"- Date: {datetime.now():%Y-%m-%d %H:%M}",
        "- Command: `python scripts/compare_llms.py`",
        f"- Same claims for every mode ({', '.join(d['name'] for d in DATASETS)}); one provider and model forced per mode, no fallback.",
        "- Calls spaced to stay under free-tier rate limits; 'mean LLM s/call' excludes that waiting.",
    ]
    if skipped:
        out.append(f"- Not run (not configured): {', '.join(skipped)}")
    out += ["", "| Mode | Claims found | Extra claims | Verdict accuracy | Flags | Explanations rejected by guard | LLM calls answered | Mean LLM s/call | Total time (s) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for s in rows:
        rej = "n/a (templates)" if s["mode"] == "baseline" else f"{s['rejected']}/{s['verdicts']}"
        calls = "-" if s["mode"] == "baseline" else f"{s['answered']}/{s['calls']}"
        lat = "-" if s["mean_llm_s"] is None else f"{s['mean_llm_s']:.2f}"
        out.append(f"| {s['mode']} | {s['found']}/{s['n_key']} | {s['extra']} | {s['correct']}/{s['n_key']} "
                   f"({100 * s['correct'] / s['n_key']:.1f}%) | {s['flags']} | {rej} | {calls} | {lat} | {s['secs']:.1f} |")
    out += ["", "## Per dataset", "", "| Mode | Dataset | Found | Accuracy | Flags | Rejected explanations |", "|---|---|---|---|---|---|"]
    for r in results:
        for d in r["datasets"]:
            out.append(f"| {r['mode']} | {d['name']} | {d['n_found']}/{d['n_key']} | {d['acc'] * 100:.1f}% | "
                       f"{d['flags'][0]}/{d['flags'][1]} | {d['narration_rejected']} |")
    out += ["", "## Mismatches", ""]
    any_err = False
    for r in results:
        for d in r["datasets"]:
            for e in d["errors"]:
                any_err = True
                out.append(f"- **{r['mode']} · {d['name']} · {e['id']}**: expected {e['exp']}, got {e['got']}. {e['text']}")
    if not any_err:
        out.append("None: every mode reproduced every expected verdict.")
    out += ["", "## Notes", "",
            "- Verdicts are computed in Python in every mode; a model can only change which claims are found and how they are parsed, and the wording of explanations.",
            "- A rejected explanation means the Citation Guard found a number or verdict word not backed by the evidence and replaced it with the template text.",
            "- 34 claims is a small benchmark; differences of one or two claims are within noise."]
    return "\n".join(out) + "\n"


def main() -> int:
    global OUT, DATASETS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modes", default=",".join(MODES), help="comma list; provider or provider:model")
    ap.add_argument("--datasets", choices=["core", "all"], default="core",
                    help="core = PS sample, country demo, company report (34 claims); all = every dataset in evaluate.py")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    os.chdir(ROOT)
    load_dotenv(ROOT / ".env")
    OUT = Path(args.out)
    DATASETS = [d for d in ev.DATASETS if args.datasets == "all" or d["name"] in CORE]
    results, rows, skipped = [], [], []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        ok, need = available(mode)
        if not ok:
            skipped.append(f"{mode} (needs {need})")
            print(f"skip {mode}: needs {need}")
            continue
        print(f"running {mode} ...", flush=True)
        r = run_mode(mode)
        results.append(r)
        rows.append(summarise(r))
        s = rows[-1]
        print(f"  found {s['found']}/{s['n_key']}  accuracy {s['correct']}/{s['n_key']}  rejected {s['rejected']}  "
              f"answered {s['answered']}/{s['calls']}  {s['secs']:.1f}s", flush=True)
    OUT.write_text(render(rows, skipped, results), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
