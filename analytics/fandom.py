"""Fandom Intensity Proxy (song and artist level). A behavioural proxy, NOT fan counts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import (FANDOM_WEIGHTS, SHRINKAGE_K_ARTIST, SHRINKAGE_K_SONG, confidence_label, minmax_scale,
                    percentile_normalize, shrink_mean, weighted_composite)

BAND_ORDER = ["Low", "Moderate", "High", "Very High", "Extreme"]
BAND_QUANTILES = [0.30, 0.60, 0.80, 0.95]  # data-driven cut points on the score distribution

_METRICS = {  # event column -> entity column
    "momentum_score": "mean_momentum", "popularity_lift": "mean_pop_lift",
    "rank_recovery_total": "mean_rank_recovery", "recovery_speed": "mean_recovery_speed",
    "retention_snapshots": "mean_retention", "retention_days": "mean_retention_days",
}


def comeback_sharpness(events: pd.DataFrame, top_half: int = 25) -> pd.Series:
    """Share of an entity's re-entries that land in the upper half of the chart (<= rank 25)."""
    return (events["reentry_position"] <= top_half).groupby(events["entity"]).mean()


def normalize_fandom_score(components: pd.DataFrame, weights: dict | None = None,
                           n_events: pd.Series | None = None, k: float = 0.0) -> pd.Series:
    """Weighted composite of component percentiles, sample-size shrunk, min-max scaled to 0-100.

    When ``n_events`` and ``k`` are given, the composite is shrunk toward the median entity:
    (n*composite + k*median) / (n + k), so a single lucky comeback cannot top the ranking.
    """
    w = weights or FANDOM_WEIGHTS
    comp = pd.DataFrame({c: percentile_normalize(components[c]) for c in w}, index=components.index)
    composite = weighted_composite(comp, w)
    if n_events is not None and k > 0:
        composite = (n_events * composite + k * composite.median()) / (n_events + k)
    return minmax_scale(composite, 0, 100)


def assign_bands(scores: pd.Series) -> tuple[pd.Series, dict]:
    """Quantile-based intensity bands (Low/Moderate/High/Very High/Extreme)."""
    valid = scores.dropna()
    if len(valid) < 5:
        return pd.Series(np.where(scores.notna(), "Moderate", None), index=scores.index), {}
    cuts = [float(valid.quantile(q)) for q in BAND_QUANTILES]
    bins = [-np.inf] + cuts + [np.inf]
    bands = pd.cut(scores, bins=bins, labels=BAND_ORDER, include_lowest=True, right=False)
    return bands.astype(object).where(scores.notna(), None), dict(zip(["P30", "P60", "P80", "P95"], cuts))


def aggregate_events(events: pd.DataFrame, key: str) -> pd.DataFrame:
    """Plain per-entity aggregates of re-entry events (no scoring). Reused by the filter layer
    so that song/artist statistics always describe exactly the selected events."""
    ev = events.copy()
    ev["entity"] = ev[key]
    g = ev.groupby("entity")
    out = pd.DataFrame({"reentry_count": g.size()})
    for src, dst in _METRICS.items():
        out[dst] = g[src].mean()
    out["median_momentum"] = g["momentum_score"].median()
    out["median_retention_days"] = g["retention_days"].median()
    out["sharpness"] = comeback_sharpness(ev)
    out["mean_reentry_gap_days"] = g["gap_days"].mean()
    out["censored_events"] = g["is_right_censored"].sum().astype(int)
    out["confidence"] = out["reentry_count"].map(confidence_label)
    return out


def _entity_table(events: pd.DataFrame, key: str, k: float) -> pd.DataFrame:
    out = aggregate_events(events, key)
    for src, dst in _METRICS.items():
        out[dst + "_shrunk"] = shrink_mean(out[dst], out["reentry_count"], events[src].mean(), k)
    out["sharpness_shrunk"] = shrink_mean(out["sharpness"], out["reentry_count"], events["reentry_position"].le(25).mean(), k)
    comp = pd.DataFrame({
        "reentry_count": out["reentry_count"], "momentum": out["mean_momentum"],
        "popularity_lift": out["mean_pop_lift"], "rank_recovery": out["mean_rank_recovery"],
        "recovery_speed": out["mean_recovery_speed"], "retention": out["mean_retention"],
        "sharpness": out["sharpness"]})
    out["fandom_score"] = normalize_fandom_score(comp, n_events=out["reentry_count"], k=k)
    out["fandom_band"], _ = assign_bands(out["fandom_score"])
    return out


def compute_song_fandom(events: pd.DataFrame, song_summary: pd.DataFrame, appearances: pd.DataFrame,
                        snapshots: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Song-level table with re-entry stats + fandom proxy. Songs w/o re-entries get no score."""
    re = events[events["is_reentry"]]
    ent = _entity_table(re, "song_id", SHRINKAGE_K_SONG) if len(re) else pd.DataFrame()
    attrs = (appearances.sort_values("date").groupby("song_id").agg(
        song=("song", "first"), artist=("artist", "first"), song_label=("song_label", "first"),
        album_type=("album_type", lambda s: s.mode().iloc[0]), is_explicit=("is_explicit", "mean"),
        duration_minutes=("duration_minutes", "median"), total_tracks=("total_tracks", "median"),
        mean_popularity=("popularity", "mean")))
    attrs["is_explicit"] = attrs["is_explicit"] >= 0.5
    songs = song_summary.set_index("song_id").drop(columns=["reentry_count"], errors="ignore").join(attrs)
    songs = songs.join(ent, how="left")
    songs["reentry_count"] = songs["reentry_count"].fillna(0).astype(int)
    last = snapshots["date"].max()
    songs["exposure_days"] = (last - songs["first_observed"]).dt.days + 1
    songs["reentries_per_100d"] = songs["reentry_count"] / songs["exposure_days"] * 100
    songs["confidence"] = songs["reentry_count"].map(confidence_label)
    songs["fandom_band"] = songs["fandom_band"].where(songs["fandom_score"].notna(), "No re-entries")
    _, cuts = assign_bands(songs["fandom_score"])
    return songs.reset_index(), cuts


def compute_artist_fandom(events: pd.DataFrame, songs: pd.DataFrame, appearances: pd.DataFrame
                          ) -> tuple[pd.DataFrame, dict]:
    """Artist-level analytics + fandom proxy with shrinkage to limit small-sample bias."""
    re = events[events["is_reentry"]]
    ent = _entity_table(re, "artist", SHRINKAGE_K_ARTIST) if len(re) else pd.DataFrame()
    base = appearances.groupby("artist").agg(total_appearances=("position", "size"),
                                             unique_songs=("song_id", "nunique"),
                                             best_position=("position", "min"))
    swr = songs[songs["reentry_count"] > 0].groupby("artist").size().rename("songs_with_reentries")
    art = base.join(swr).join(ent, how="left")
    art["songs_with_reentries"] = art["songs_with_reentries"].fillna(0).astype(int)
    art["reentry_count"] = art["reentry_count"].fillna(0).astype(int)
    art["reentries_per_song"] = art["reentry_count"] / art["unique_songs"]
    art["confidence"] = art["reentry_count"].map(confidence_label)
    art["fandom_band"] = art["fandom_band"].where(art["fandom_score"].notna(), "No re-entries")
    _, cuts = assign_bands(art["fandom_score"])
    return art.reset_index(), cuts


def explain_score(row: pd.Series, kind: str = "song") -> str:
    """Plain-language explanation generated from the entity's computed numbers."""
    if pd.isna(row.get("fandom_score")):
        return "No re-entries observed, so no behavioural fandom proxy is computed."
    return (f"{row.get('song_label', row.get('artist'))} scores {row['fandom_score']:.0f}/100 "
            f"({row['fandom_band']}) from {int(row['reentry_count'])} re-entr"
            f"{'y' if row['reentry_count'] == 1 else 'ies'}, mean momentum {row['mean_momentum']:.1f}, "
            f"mean retention {row['mean_retention']:.1f} snapshots and "
            f"{row['sharpness'] * 100:.0f}% of comebacks landing in the top half. "
            f"Confidence: {row['confidence']}. This is a behavioural chart proxy, not a fan count.")
