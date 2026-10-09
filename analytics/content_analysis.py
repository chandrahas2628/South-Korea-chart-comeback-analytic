"""Content-attribute analysis: single/album, explicit/clean, album size, duration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .utils import cliffs_delta, effect_label

SIZE_LABELS = ["1-3 tracks", "4-7 tracks", "8-12 tracks", "13+ tracks"]
MIN_GROUP_SONGS = 10     # below this a group is flagged as a small sample
MIN_GROUP_EVENTS = 10


def add_content_bins(songs: pd.DataFrame) -> pd.DataFrame:
    """Add readable album-size and duration bins (duration bins are data-driven quartiles)."""
    out = songs.copy()
    out["album_size_bin"] = pd.cut(out["total_tracks"], [0, 3, 7, 12, 10_000], labels=SIZE_LABELS)
    out["explicit_label"] = np.where(out["is_explicit"], "Explicit", "Clean")
    try:
        out["duration_bin"] = pd.qcut(out["duration_minutes"], 4, duplicates="drop")
        out["duration_bin"] = out["duration_bin"].map(lambda i: f"{i.left:.2f}-{i.right:.2f} min")
    except ValueError:
        out["duration_bin"] = "all"
    return out


def attach_song_attributes(events: pd.DataFrame, songs: pd.DataFrame) -> pd.DataFrame:
    cols = ["song_id", "album_size_bin", "duration_bin", "explicit_label", "fandom_score"]
    cols = [c for c in cols if c in songs.columns]
    ev = events.drop(columns=[c for c in cols if c != "song_id" and c in events.columns])
    return ev.merge(songs[cols].rename(columns={"fandom_score": "song_fandom_score"}), on="song_id", how="left")


def group_summary(songs: pd.DataFrame, events: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per-group re-entry rate and momentum statistics (song attributes applied to events)."""
    re = events[events["is_reentry"]]
    s = songs.groupby(by, observed=True).agg(
        n_songs=("song_id", "size"), songs_with_reentry=("reentry_count", lambda x: int((x > 0).sum())),
        avg_reentries_per_song=("reentry_count", "mean"), total_reentries=("reentry_count", "sum"),
        exposure_days=("exposure_days", "sum"), mean_song_fandom=("fandom_score", "mean"))
    s["reentry_rate"] = s["songs_with_reentry"] / s["n_songs"]
    s["reentries_per_100_exposure_days"] = s["total_reentries"] / s["exposure_days"] * 100
    if by in re.columns:
        e = re.groupby(by, observed=True).agg(
            n_events=("reentry_id", "size"), mean_momentum=("momentum_score", "mean"),
            median_momentum=("momentum_score", "median"), mean_retention=("retention_snapshots", "mean"),
            mean_retention_days=("retention_days", "mean"), median_retention_days=("retention_days", "median"),
            mean_pop_lift=("popularity_lift", "mean"), mean_recovery_speed=("recovery_speed", "mean"))
        s = s.join(e)
    s = s.reset_index()
    s["low_sample"] = (s["n_songs"] < MIN_GROUP_SONGS) | (s.get("n_events", pd.Series(0, index=s.index)).fillna(0) < MIN_GROUP_EVENTS)
    return s.sort_values("n_songs", ascending=False)


def compare_groups(events: pd.DataFrame, by: str, a, b, metric: str = "momentum_score") -> dict:
    """Mann-Whitney U + Cliff's delta between two groups of re-entry events."""
    re = events[events["is_reentry"]]
    xa = re.loc[re[by] == a, metric].dropna()
    xb = re.loc[re[by] == b, metric].dropna()
    res = {"group_a": a, "group_b": b, "metric": metric, "n_a": len(xa), "n_b": len(xb),
           "median_a": float(xa.median()) if len(xa) else np.nan, "median_b": float(xb.median()) if len(xb) else np.nan,
           "mean_a": float(xa.mean()) if len(xa) else np.nan, "mean_b": float(xb.mean()) if len(xb) else np.nan}
    if len(xa) >= 5 and len(xb) >= 5:
        res["p_value"] = float(stats.mannwhitneyu(xa, xb, alternative="two-sided").pvalue)
        res["cliffs_delta"] = cliffs_delta(xa, xb)
        res["effect"] = effect_label(res["cliffs_delta"])
    else:
        res.update({"p_value": np.nan, "cliffs_delta": np.nan, "effect": "insufficient sample"})
    return res


def numeric_associations(songs: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlations between content attributes and comeback outcomes."""
    re = events[events["is_reentry"]].merge(
        songs[["song_id", "duration_minutes", "total_tracks"]].rename(
            columns={"duration_minutes": "song_duration", "total_tracks": "song_total_tracks"}), on="song_id", how="left")
    rows = []
    for attr, col in (("Duration (min)", "song_duration"), ("Album size (tracks)", "song_total_tracks")):
        for outcome in ("momentum_score", "retention_days", "retention_snapshots", "popularity_lift", "recovery_speed"):
            d = re[[col, outcome]].dropna()
            if len(d) >= 10 and d[col].nunique() > 1 and d[outcome].nunique() > 1:
                r = stats.spearmanr(d[col], d[outcome])
                rows.append({"attribute": attr, "outcome": outcome, "level": "event", "n": len(d),
                             "spearman_rho": float(r.statistic), "p_value": float(r.pvalue)})
    for attr, col in (("Duration (min)", "duration_minutes"), ("Album size (tracks)", "total_tracks")):
        d = songs[[col, "reentry_count"]].dropna()
        if len(d) >= 10 and d[col].nunique() > 1 and d["reentry_count"].nunique() > 1:
            r = stats.spearmanr(d[col], d["reentry_count"])
            rows.append({"attribute": attr, "outcome": "reentry_count", "level": "song", "n": len(d),
                         "spearman_rho": float(r.statistic), "p_value": float(r.pvalue)})
    return pd.DataFrame(rows, columns=["attribute", "outcome", "level", "n", "spearman_rho", "p_value"])


def correlation_matrix(events: pd.DataFrame) -> pd.DataFrame:
    cols = ["momentum_score", "rank_jump", "peak_improvement", "popularity_lift", "recovery_speed",
            "retention_snapshots", "retention_days", "days_to_peak", "volatility", "decay_speed", "gap_days"]
    re = events[events["is_reentry"]]
    return re[[c for c in cols if c in re.columns]].corr(method="spearman")


def kruskal_groups(events: pd.DataFrame, by: str, metric: str = "momentum_score", min_n: int = 5) -> dict:
    """Kruskal-Wallis H test across groups with at least ``min_n`` events (multi-group comparison)."""
    re = events[events["is_reentry"]]
    groups = [g[metric].dropna().to_numpy() for _, g in re.groupby(by, observed=True) if g[metric].notna().sum() >= min_n]
    out = {"metric": metric, "n_groups": len(groups), "n": int(sum(len(g) for g in groups))}
    if len(groups) >= 2:
        h = stats.kruskal(*groups)
        out.update({"H": float(h.statistic), "p_value": float(h.pvalue)})
    else:
        out.update({"H": np.nan, "p_value": np.nan})
    return out


def compare_rates(songs: pd.DataFrame, by: str, a, b) -> dict:
    """Fisher exact test on the share of songs with >=1 re-entry in two groups (exposure-unadjusted)."""
    ga, gb = songs[songs[by] == a], songs[songs[by] == b]
    ka, kb = int((ga["reentry_count"] > 0).sum()), int((gb["reentry_count"] > 0).sum())
    res = {"group_a": a, "group_b": b, "n_a": len(ga), "n_b": len(gb), "rate_a": ka / len(ga) if len(ga) else np.nan,
           "rate_b": kb / len(gb) if len(gb) else np.nan}
    if len(ga) >= 5 and len(gb) >= 5:
        odds, p = stats.fisher_exact([[ka, len(ga) - ka], [kb, len(gb) - kb]])
        res.update({"odds_ratio": float(odds), "p_value": float(p)})
    else:
        res.update({"odds_ratio": np.nan, "p_value": np.nan})
    return res
