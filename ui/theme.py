"""Design palette (warm paper light + dark) and CSS injection for Streamlit. Owner: UI.

The same colour tokens drive the Streamlit page (ui/theme.css, as --ps-* vars)
and the audit component iframe (ui/audit_component.html, as the design's vars).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

UI_DIR = Path(__file__).resolve().parent
FONTS_URL = ("https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;"
             "0,9..144,700;1,9..144,400&family=JetBrains+Mono:wght@400;600&display=swap")

PALETTES: dict[str, dict[str, str]] = {
    "light": {"bg": "#f4efe6", "card": "#fbf8f2", "ink": "#1d2320", "mute": "#6b706a", "line": "#d9d2c3",
              "ok": "#2d6a4f", "okbg": "#dcebdf", "bad": "#b3261e", "badbg": "#f6dcd8",
              "warn": "#9a6a00", "warnbg": "#f6e8c3", "acc": "#1d2320", "hl": "#efe4c4"},
    "dark": {"bg": "#141816", "card": "#1c211e", "ink": "#ece7da", "mute": "#9aa096", "line": "#323a35",
             "ok": "#7fd1a0", "okbg": "#1f3a2b", "bad": "#ff8b80", "badbg": "#401f1b",
             "warn": "#e8bb55", "warnbg": "#3b3014", "acc": "#ece7da", "hl": "#2e2b1a"},
}


def css_vars(kind: str, prefix: str = "--ps-") -> str:
    return ";".join(f"{prefix}{k}:{v}" for k, v in PALETTES[kind].items())


def theme_type() -> Optional[str]:
    """'light' / 'dark' as the browser reports it, None when unknown (first run, tests)."""
    try:
        import streamlit as st

        kind = st.context.theme.type
    except Exception:
        return None
    return kind if kind in PALETTES else None


def page_css(kind: Optional[str]) -> str:
    """Full <style> block for the Streamlit page."""
    if kind in PALETTES:
        root = f":root{{{css_vars(kind)}}}"
    else:  # unknown: follow the OS, like Streamlit does when both [theme.light] and [theme.dark] exist
        root = f":root{{{css_vars('light')}}}@media (prefers-color-scheme:dark){{:root{{{css_vars('dark')}}}}}"
    body = (UI_DIR / "theme.css").read_text(encoding="utf-8")
    return f"<style>@import url('{FONTS_URL}');{root}\n{body}</style>"


def inject(kind: Optional[str] = None) -> None:
    import streamlit as st

    st.html(page_css(kind))
