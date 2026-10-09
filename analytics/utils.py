"""Shared constants and small, well-tested helpers used across the analytics layer."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Mapping

import numpy as np
import pandas as pd

TOP_N = 50
OFF_CHART_RANK = TOP_N + 1  # virtual rank used for "not on the Top 50"

REQUIRED_COLUMNS = [
    "date", "position", "song", "artist", "popularity", "duration_ms",
    "album_type", "total_tracks", "is_explicit", "album_cover_url",
]
ALLOWED_ALBUM_TYPES = ("single", "album", "compilation")

# Documented weights (see Methodology page). Each set sums to 1.0.
MOMENTUM_WEIGHTS: dict[str, float] = {
    "rank_jump": 0.20,
    "peak_improvement": 0.20,
    "popularity_lift": 0.20,
    "recovery_speed": 0.15,
    "retention_snapshots": 0.25,
}
FANDOM_WEIGHTS: dict[str, float] = {
    "reentry_count": 0.20,
    "momentum": 0.25,
    "popularity_lift": 0.10,
    "rank_recovery": 0.10,
    "recovery_speed": 0.10,
    "retention": 0.10,
    "sharpness": 0.15,
}
SHRINKAGE_K_SONG = 3.0
SHRINKAGE_K_ARTIST = 5.0

_NULL_TOKENS = {"", "nan", "none", "null", "n/a", "na", "-", "--", "<na>"}


def normalize_text(value) -> str:
    """Unicode-normalise (NFKC), collapse whitespace and strip. Null-like tokens become ''."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return "" if text.lower() in _NULL_TOKENS else text


def make_song_key(song, artist) -> str:
    """Case-insensitive normalised ``song||artist`` key (title alone is NOT unique)."""
    return f"{normalize_text(song).casefold()}||{normalize_text(artist).casefold()}"


def make_song_id(song, artist) -> str:
    """Stable 10-char identifier derived from the normalised song+artist key."""
    return "S" + hashlib.md5(make_song_key(song, artist).encode("utf-8")).hexdigest()[:9]


def percentile_normalize(values: pd.Series) -> pd.Series:
    """Mid-rank percentile in (0,1); ties share the average rank. NaN stays NaN."""
    s = pd.to_numeric(values, errors="coerce")
    n = s.notna().sum()
    if n == 0:
        return s.astype(float)
    return (s.rank(method="average") - 0.5) / n


def minmax_scale(values: pd.Series, lo: float = 0.0, hi: float = 100.0) -> pd.Series:
    """Linearly rescale to [lo, hi]. A constant series maps to the midpoint."""
    s = pd.to_numeric(values, errors="coerce").astype(float)
    vmin, vmax = s.min(), s.max()
    if not np.isfinite(vmin) or vmax == vmin:
        return s.where(s.isna(), (lo + hi) / 2)
    return lo + (s - vmin) / (vmax - vmin) * (hi - lo)


def weighted_composite(components: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    """Row-wise weighted mean of already-normalised components.

    Missing components are dropped and the remaining weights re-normalised, so a row is
    never penalised for a metric that is undefined for it. All-missing rows return NaN.
    """
    cols = [c for c in weights if c in components.columns]
    w = np.array([weights[c] for c in cols], dtype=float)
    vals = components[cols].to_numpy(dtype=float)
    mask = ~np.isnan(vals)
    num = np.nansum(vals * w, axis=1)
    den = (mask * w).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return pd.Series(out, index=components.index)


def shrink_mean(mean, n, global_mean, k):
    """Empirical-Bayes style shrinkage toward the global mean: (n*mean + k*global)/(n+k)."""
    mean = pd.to_numeric(mean, errors="coerce")
    n = pd.to_numeric(n, errors="coerce").fillna(0)
    shrunk = (n * mean + k * global_mean) / (n + k)
    return shrunk.where(mean.notna(), np.nan)


def cliffs_delta(a, b) -> float:
    """Cliff's delta effect size in [-1, 1] (positive => a tends to exceed b)."""
    a = np.asarray(pd.Series(a).dropna(), dtype=float)
    b = np.asarray(pd.Series(b).dropna(), dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def effect_label(delta: float) -> str:
    d = abs(delta)
    if np.isnan(d):
        return "n/a"
    return "negligible" if d < 0.147 else "small" if d < 0.33 else "medium" if d < 0.474 else "large"


def confidence_label(n_events) -> str:
    """Sample-size context indicator for an entity's re-entry-event count."""
    n = 0 if pd.isna(n_events) else int(n_events)
    return "Low (n<3)" if n < 3 else "Medium (3-9)" if n < 10 else "High (10+)"


def safe_div(a, b, default=np.nan):
    return a / b if b not in (0, None) and not pd.isna(b) else default
