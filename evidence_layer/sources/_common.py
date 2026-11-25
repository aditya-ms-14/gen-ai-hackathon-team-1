"""Shared plumbing for the official-data connectors. Owner: Aditya (official data sources).

Every connector turns a published series into `Row`s, one per (series, entity, year).
A row's canonical CSV line ("EN.GHG.CO2.MT.CE.AR5,DEU,2010,814.962,Mt CO2e") is
both the line written to the curated snapshot in data/official/ and the Fact quote,
so the Citation Guard can re-find every quote in `snapshot_pages()`.

Lookup order per series: live API (with a file cache in data/runtime/cache/) and,
when that fails or we are offline, the curated snapshot. Nothing here raises to callers.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from auditor.schemas import ExtractionMethod, Fact, SourceType

log = logging.getLogger("evidence_layer.sources")

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = ROOT / "evidence_layer" / "data" / "official"  # curated snapshots ship with the evidence layer
CACHE_DIR = ROOT / "data" / "runtime" / "cache"
HEADER = "series,entity,year,value,unit"
TIMEOUT = (5, 15)            # connect, read (seconds); never more than 15 s per request
CACHE_TTL = 7 * 24 * 3600    # a week: official series change a few times a year
BREAKER_SECONDS = 300        # after a network failure, skip that source's API for 5 minutes
USER_AGENT = "ps99-citation-auditor/0.1 (hackathon demo)"
AGGREGATES = ("WLD", "EUU")

_offline = os.getenv("AUDITOR_OFFLINE", "").strip() in ("1", "true", "yes")
_seen: dict[str, set[str]] = {}          # source name -> quote lines handed out this process
_seen_lock = threading.Lock()


def set_offline(flag: bool) -> None:
    """True = never touch the network; only the curated snapshots are used (tests, demos)."""
    global _offline
    _offline = bool(flag)


def is_offline() -> bool:
    return _offline


def fmt(value: float) -> str:
    """Deterministic number text used in snapshot lines and quotes (max 4 decimals)."""
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


@dataclass(frozen=True)
class Row:
    series: str
    entity: str
    year: int
    value: float
    unit: str

    @classmethod
    def make(cls, series: str, entity: str, year, value, unit: str) -> Optional["Row"]:
        try:
            num = round(float(value), 4)
            return cls(series, entity, int(float(year)), float(fmt(num)), unit)
        except (TypeError, ValueError):
            return None

    def line(self) -> str:
        return f"{self.series},{self.entity},{self.year},{fmt(self.value)},{self.unit}"


# ---------------------------------------------------------------- snapshots

def snapshot_path(slug: str) -> Path:
    return OFFICIAL_DIR / f"{slug}.csv"


_snap_cache: dict[str, tuple[float, list[Row]]] = {}


def snapshot_rows(slug: str) -> list[Row]:
    path = snapshot_path(slug)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []
    hit = _snap_cache.get(slug)
    if hit and hit[0] == mtime:
        return hit[1]
    rows: list[Row] = []
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            for rec in csv.DictReader(fh):
                row = Row.make(rec["series"], rec["entity"], rec["year"], rec["value"], rec["unit"])
                if row:
                    rows.append(row)
    except Exception as exc:  # a damaged snapshot must not break the audit
        log.warning("snapshot %s unreadable: %s", path.name, exc)
        return []
    _snap_cache[slug] = (mtime, rows)
    return rows


def write_snapshot(slug: str, rows: list[Row]) -> Path:
    OFFICIAL_DIR.mkdir(parents=True, exist_ok=True)
    lines = sorted({r.line() for r in rows})
    path = snapshot_path(slug)
    path.write_text(HEADER + "\n" + "\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    _snap_cache.pop(slug, None)
    return path


def remember(source: str, line: str) -> None:
    with _seen_lock:
        _seen.setdefault(source, set()).add(line)


def page_text(source: str, slug: str) -> str:
    """Deterministic page 1 text: header + sorted canonical lines (snapshot + served)."""
    lines = {r.line() for r in snapshot_rows(slug)}
    with _seen_lock:
        lines |= _seen.get(source, set())
    return f"{source}\n{HEADER}\n" + "\n".join(sorted(lines)) + "\n"


# ---------------------------------------------------------------- HTTP + cache

def _cache_file(slug: str, key: str, ext: str = "json") -> Path:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / slug / f"{digest}.{ext}"


def _fresh(path: Path) -> bool:
    try:
        return time.time() - path.stat().st_mtime < CACHE_TTL
    except OSError:
        return False


def get_json(slug: str, url: str, params: Optional[dict] = None, headers: Optional[dict] = None):
    """GET JSON with a file cache. Stale cache is used if the network fails."""
    req = requests.Request("GET", url, params=params).prepare()
    path = _cache_file(slug, req.url or url)
    if _fresh(path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT,
                            headers={"User-Agent": USER_AGENT, "Accept": "application/json", **(headers or {})})
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def download(slug: str, url: str, name: str) -> Path:
    """Stream a (large) file into the cache; reuse it while fresh."""
    path = CACHE_DIR / slug / name
    if _fresh(path):
        return path
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        if path.exists():
            return path
        raise
    return path


# ---------------------------------------------------------------- connector base

class Connector:
    """One official source. Subclasses fill the class attributes and `live_rows` / `url`."""

    name: str = ""             # Fact.source, e.g. "World Bank WDI"
    slug: str = ""             # data/official/<slug>.csv
    home: str = ""             # landing page of the dataset
    endpoint: str = ""         # API base, for the manifest
    SERIES: dict[str, list[str]] = {}   # canonical indicator -> series codes, in preference order
    UNITS: dict[str, str] = {}          # series code -> unit as published

    def __init__(self) -> None:
        self._down_until = 0.0

    # --- to override
    def supports(self, entity: str) -> bool:
        return True

    def live_rows(self, series: str, entity: str) -> list[Row]:
        raise NotImplementedError

    def url(self, series: str, entity: str) -> str:
        return self.home

    # --- shared
    def series_for(self, indicator: str) -> list[str]:
        return list(self.SERIES.get(indicator, []))

    def rows(self, series: str, entity: str) -> list[Row]:
        if not is_offline() and time.time() >= self._down_until:
            try:
                live = [r for r in self.live_rows(series, entity) if r.entity == entity]
                if live:
                    return live
            except (requests.ConnectionError, requests.Timeout) as exc:
                log.warning("%s offline (%s); using snapshot", self.name, type(exc).__name__)
                self._down_until = time.time() + BREAKER_SECONDS
            except Exception as exc:
                log.info("%s live lookup failed for %s/%s: %s", self.name, series, entity, exc)
        return [r for r in snapshot_rows(self.slug) if r.series == series and r.entity == entity]

    def fetch(self, indicator: str, entity: str, years: list[int]) -> list[Fact]:
        wanted = sorted(set(years))
        best: list[Row] = []
        for series in self.series_for(indicator):
            by_year: dict[int, Row] = {}
            for r in self.rows(series, entity):
                by_year.setdefault(r.year, r)
            hit = [by_year[y] for y in wanted if y in by_year]
            if len(hit) > len(best):
                best = hit
            if len(hit) == len(wanted):
                break
        return [self.to_fact(indicator, r) for r in best]

    def to_fact(self, indicator: str, r: Row) -> Fact:
        quote = r.line()
        remember(self.name, quote)
        return Fact(
            fact_id=f"{self.slug}:{r.series}:{r.entity}:{r.year}",
            indicator=indicator, value=r.value, unit=r.unit, year=r.year,
            source=self.name, page=1, locator=f"{r.series} · {r.entity} · {r.year}",
            quote=quote, method=ExtractionMethod.TABLE, source_type=SourceType.OFFICIAL,
            entity=r.entity, url=self.url(r.series, r.entity),
        )
