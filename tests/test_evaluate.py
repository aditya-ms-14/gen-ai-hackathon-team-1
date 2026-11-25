"""The evaluation script runs and reproduces the PS sample answer key."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("evaluate_script", ROOT / "scripts" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_script"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_ps_sample_is_100_percent():
    mod = _load()
    ps = next(d for d in mod.DATASETS if d["name"] == "PS sample")
    r = mod.evaluate(ps, use_llm=False)
    assert r["n_found"] == r["n_key"] == 6
    assert r["acc"] == 1.0
    assert r["errors"] == []
    assert r["guard_fail"] == 0


def test_render_runs():
    mod = _load()
    ps = next(d for d in mod.DATASETS if d["name"] == "PS sample")
    r = mod.evaluate(ps, use_llm=False)
    assert "PS sample" in mod.render([r], "t")
