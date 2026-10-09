"""Comeback momentum metrics, composite score, sustainability patterns and trajectories.

Terminology (see Methodology page)
----------------------------------
snapshot        : an observed playlist date (valid when it holds a complete Top-N list)
calendar day    : any calendar date; dates with no snapshot are *calendar gaps*
retention_days  : calendar days from the comeback date to the stint's last observed
                  appearance, inclusive. Calendar gaps inside the span are NOT assumed to be
                  observed presence (reported as ``unobserved_days_in_stint``).
retention_snapshots : number of observed snapshots in which the song is listed during the stint.
right-censored  : stint still on chart at the final observed snapshot -> retention is a lower bound.
left-censored   : song already on chart at the first valid snapshot -> true entry unobserved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .utils import MOMENTUM_WEIGHTS, OFF_CHART_RANK, percentile_normalize, weighted_composite

COMPONENTS = list(MOMENTUM_WEIGHTS)
PATTERNS = ["Explosive / Short-lived", "Strong / Sustainable", "Moderate / Stable", "Weak / Short-lived"]


def stint_metrics(dates, positions, popularity, baseline_pos, baseline_pop) -> dict:
    """Metrics for ONE stint (arrays are chronological appearances within the stint).

    baseline_pos : previous exit position (or off-chart rank for a first entry).
    baseline_pop : popularity at previous exit (NaN for a first entry).
    """
    dates = pd.DatetimeIndex(dates)
    p = np.asarray(positions, dtype=float)
    q = np.asarray(popularity, dtype=float)
    entry_pos = p[0]
    peak_i = int(np.argmin(p))                       # first time the best rank is reached
    peak_rank = p[peak_i]
    days_to_peak = int((dates[peak_i] - dates[0]).days)
    end_i = len(p) - 1
    days_peak_to_exit = int((dates[end_i] - dates[peak_i]).days)
    lifespan = int((dates[end_i] - dates[0]).days) + 1
    diffs = np.diff(p)
    has_base_pop = not pd.isna(baseline_pop)
    return {
        "rank_jump": float(baseline_pos - entry_pos),
        "peak_rank": float(peak_rank),
        "peak_date": dates[peak_i],
        "peak_improvement": float(entry_pos - peak_rank),
        "popularity_lift": float(np.max(q) - baseline_pop) if has_base_pop else np.nan,
        "popularity_change_at_entry": float(q[0] - baseline_pop) if has_base_pop else np.nan,
        "peak_popularity": float(np.max(q)),
        "days_to_peak": days_to_peak,
        "snapshots_to_peak": peak_i,
        "retention_snapshots": int(len(p)),
        "retention_days": lifespan,
        "lifespan_days": lifespan,
        "rank_recovery_total": float(baseline_pos - peak_rank),
        "recovery_speed": float((baseline_pos - peak_rank) / (days_to_peak + 1)),
        "decay_speed": float((p[end_i] - peak_rank) / days_peak_to_exit) if days_peak_to_exit > 0 else np.nan,
        "volatility": float(np.std(diffs, ddof=1)) if len(diffs) >= 2 else np.nan,
        "mean_rank": float(np.mean(p)),
        "days_peak_to_exit": days_peak_to_exit,
    }


def score_momentum(df: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """Momentum Intensity (0-100): weighted mean of component percentile ranks.

    Percentiles are computed *within stint type* (re-entries vs first entries) because the
    two populations have different baselines (previous exit rank vs the off-chart rank 51);
    pooling them would bias re-entry scores downward. Components are percentile-ranked so
    no unit or outlier dominates; undefined components are dropped and weights re-normalised.
    """
    w = weights or MOMENTUM_WEIGHTS
    groups = df["stint_type"] if "stint_type" in df.columns else pd.Series("all", index=df.index)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, idx in groups.groupby(groups).groups.items():
        sub = df.loc[idx]
        comp = pd.DataFrame({c: percentile_normalize(sub[c]) for c in w if c in sub.columns}, index=sub.index)
        out.loc[idx] = weighted_composite(comp, w) * 100.0
    return out


def _classify_group(sub: pd.DataFrame):
    scored = sub["momentum_score"].notna()
    out = pd.Series("Not scored", index=sub.index, dtype=object)
    if scored.sum() < 3:
        return out, {}
    hi = float(sub.loc[scored, "momentum_score"].quantile(0.67))
    lo = float(sub.loc[scored, "momentum_score"].quantile(0.33))
    ret_med = float(sub.loc[scored, "retention_snapshots"].median())
    s, r = sub["momentum_score"], sub["retention_snapshots"]
    long_ret = r >= ret_med
    cond = [scored & (s >= hi) & long_ret, scored & (s >= hi) & ~long_ret, scored & (s <= lo) & ~long_ret]
    labels = np.select(cond, [PATTERNS[1], PATTERNS[0], PATTERNS[3]], default="Moderate / Stable")
    out = pd.Series(np.where(scored, labels, "Not scored"), index=sub.index)
    return out, {"high_momentum_threshold": hi, "low_momentum_threshold": lo, "retention_median": ret_med}


def classify_patterns(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Sustainability pattern from data-driven quantiles, computed within stint type.

    high momentum  : score >= 67th percentile      low momentum : score <= 33rd percentile
    long retention : retention_snapshots >= group median
    Returned thresholds are those of re-entries (the primary population); first-entry
    thresholds are returned with a ``first_entry_`` prefix.
    """
    out = pd.Series("Not scored", index=df.index, dtype=object)
    thr: dict = {}
    if "stint_type" not in df.columns:
        return _classify_group(df)
    for typ, idx in df["stint_type"].groupby(df["stint_type"]).groups.items():
        if typ == "initial_observed":
            continue
        res, t = _classify_group(df.loc[idx])
        out.loc[idx] = res
        prefix = "" if typ == "re_entry" else f"{typ}_"
        thr.update({prefix + k: v for k, v in t.items()})
    return out, thr


def _span_counts(dates_sorted: np.ndarray, start, end) -> int:
    return int(np.searchsorted(dates_sorted, end, side="right") - np.searchsorted(dates_sorted, start, side="left"))


def compute_momentum(stints: pd.DataFrame, appearances: pd.DataFrame, weights: dict | None = None,
                     snapshots: pd.DataFrame | None = None):
    """Momentum metrics for every stint; returns (events_df, pattern_thresholds).

    ``snapshots`` (observed dates + ``is_valid_snapshot``) enables observed-exit dates and
    calendar-gap accounting inside each stint. Without it, appearance dates are used as the
    observation calendar (only suitable for tests).
    'initial_observed' stints (already on the chart when observation began) get metrics for
    context but NO momentum score: their true entry is unobserved (left-censored).
    """
    if snapshots is not None:
        valid = np.sort(snapshots.loc[snapshots["is_valid_snapshot"], "date"].to_numpy(dtype="datetime64[ns]"))
        obs = np.sort(snapshots["date"].to_numpy(dtype="datetime64[ns]"))
    else:
        valid = obs = np.sort(appearances["date"].unique().astype("datetime64[ns]"))
    app = appearances.sort_values(["song_id", "stint_no", "date"])
    groups = app.groupby(["song_id", "stint_no"], sort=False).indices
    recs = []
    for r in stints.itertuples(index=False):
        g = app.iloc[groups[(r.song_id, r.stint_no)]]
        base_pos = r.previous_exit_position if r.stint_no > 0 else OFF_CHART_RANK
        base_pop = r.previous_exit_popularity if r.stint_no > 0 else np.nan
        m = stint_metrics(g["date"], g["position"], g["popularity"], base_pos, base_pop)
        start, end = g["date"].iloc[0].to_datetime64(), g["date"].iloc[-1].to_datetime64()
        app_valid = int(np.isin(g["date"].to_numpy(dtype="datetime64[ns]"), valid).sum())
        valid_span = _span_counts(valid, start, end)
        obs_span = _span_counts(obs, start, end)
        nxt = int(np.searchsorted(valid, end, side="right"))
        exit_obs = pd.Timestamp(valid[nxt]) if nxt < len(valid) else pd.NaT
        m.update({
            "reentry_id": r.reentry_id, "baseline_position": float(base_pos), "baseline_popularity": base_pop,
            "stint_span_snapshots": valid_span,
            "dropout_snapshots": max(valid_span - app_valid, 0),
            "unobserved_days_in_stint": max(m["retention_days"] - obs_span, 0),
            "exit_observed_date": exit_obs,
            "exit_uncertainty_days": (int((exit_obs - pd.Timestamp(end)).days) - 1) if pd.notna(exit_obs) else np.nan,
        })
        recs.append(m)
    mom = pd.DataFrame(recs)
    out = stints.merge(mom, on="reentry_id", how="left")
    out["retention_is_lower_bound"] = out["is_right_censored"]
    out = out.sort_values(["song_id", "stint_no"])
    out["previous_exit_observed_date"] = out.groupby("song_id")["exit_observed_date"].shift(1)
    scorable = out["stint_type"].ne("initial_observed")
    out["momentum_score"] = np.nan
    out.loc[scorable, "momentum_score"] = score_momentum(out[scorable], weights).to_numpy()
    out["pattern"], thr = classify_patterns(out)
    out["is_reentry"] = out["stint_no"] > 0
    out.rename(columns={"start_date": "event_date", "entry_position": "event_position"}, inplace=True)
    out["reentry_date"] = out["event_date"].where(out["is_reentry"])
    out["reentry_position"] = out["event_position"].where(out["is_reentry"])
    num = out.select_dtypes("number").columns
    out[num] = out[num].replace([np.inf, -np.inf], np.nan)          # never let inf reach charts
    return out.sort_values(["event_date", "song_id"]).reset_index(drop=True), thr


def weight_sensitivity(events: pd.DataFrame, n_draws: int = 200, seed: int = 11) -> dict:
    """Robustness of the momentum ranking to the choice of weights (re-entry events).

    * equal weights vs documented weights (Spearman, top-10 overlap)
    * 200 random weight vectors drawn around the documented weights (Dirichlet, concentration 50)
    """
    ev = events[events["is_reentry"] & events["momentum_score"].notna()] if "is_reentry" in events else \
        events[events["momentum_score"].notna()]
    empty = {"spearman_vs_equal_weights": np.nan, "n": int(len(ev)), "perturbation_median_rho": np.nan,
             "perturbation_min_rho": np.nan, "top10_overlap_vs_equal": np.nan, "n_draws": 0}
    if len(ev) < 10:
        return empty
    base = score_momentum(ev)
    eq = score_momentum(ev, {c: 1 / len(COMPONENTS) for c in COMPONENTS})
    rho_eq = float(stats.spearmanr(base, eq).statistic)
    k = min(10, len(ev))
    overlap = len(set(base.nlargest(k).index) & set(eq.nlargest(k).index)) / k
    rng = np.random.default_rng(seed)
    w0 = np.array([MOMENTUM_WEIGHTS[c] for c in COMPONENTS])
    rhos = []
    for w in rng.dirichlet(w0 * 50, size=n_draws):
        alt = score_momentum(ev, dict(zip(COMPONENTS, w)))
        rhos.append(float(stats.spearmanr(base, alt).statistic))
    return {"spearman_vs_equal_weights": rho_eq, "n": int(len(ev)), "perturbation_median_rho": float(np.median(rhos)),
            "perturbation_min_rho": float(np.min(rhos)), "top10_overlap_vs_equal": float(overlap), "n_draws": n_draws}


def lifecycle_stages(ev: pd.Series) -> list[dict]:
    """PRE-COMEBACK -> EXIT -> RE-ENTRY -> SURGE -> PEAK -> DECAY -> EXIT, from computed values."""
    fmt = lambda d: "—" if pd.isna(d) else pd.Timestamp(d).strftime("%d %b %Y")
    stages = []
    if ev["is_reentry"]:
        stages.append({"stage": "PRE-COMEBACK", "date": fmt(ev["previous_exit_date"]),
                       "rank": f"#{int(ev['previous_exit_position'])}", "detail": "last appearance before leaving"})
        stages.append({"stage": "EXIT", "date": fmt(ev["previous_exit_observed_date"]), "rank": "off chart",
                       "detail": f"{int(ev['absent_snapshots'])} observed absent snapshot(s); "
                                 f"{int(ev['unobserved_calendar_days'])} unobserved calendar day(s)"})
    else:
        stages.append({"stage": "PRE-COMEBACK", "date": "—", "rank": "off chart",
                       "detail": "first entry: no earlier appearance observed"})
    stages += [
        {"stage": "RE-ENTRY" if ev["is_reentry"] else "ENTRY", "date": fmt(ev["event_date"]),
         "rank": f"#{int(ev['event_position'])}", "detail": f"rank jump {ev['rank_jump']:+.0f} vs baseline"},
        {"stage": "SURGE", "date": f"{int(ev['days_to_peak'])} day(s)", "rank": f"{ev['peak_improvement']:+.0f} places",
         "detail": "entry → peak"},
        {"stage": "PEAK", "date": fmt(ev["peak_date"]), "rank": f"#{int(ev['peak_rank'])}",
         "detail": f"popularity peak {ev['peak_popularity']:.0f}"},
        {"stage": "DECAY", "date": f"{int(ev['days_peak_to_exit'])} day(s)",
         "rank": "n/a" if pd.isna(ev["decay_speed"]) else f"{ev['decay_speed']:.2f} /day", "detail": "peak → last appearance"},
        {"stage": "EXIT" if not ev["is_right_censored"] else "STILL ON CHART",
         "date": fmt(ev["exit_observed_date"]) if not ev["is_right_censored"] else fmt(ev["end_date"]),
         "rank": f"#{int(ev['exit_position'])} last",
         "detail": ("retention is a lower bound (right-censored)" if ev["is_right_censored"]
                    else f"{int(ev['retention_days'])} day(s) / {int(ev['retention_snapshots'])} snapshot(s) retained")},
    ]
    return stages


def event_trajectory(appearances: pd.DataFrame, snapshots: pd.DataFrame, song_id: str, stint_no: int,
                     pre_days: int = 14, post_days: int = 14) -> pd.DataFrame:
    """Rank/popularity trajectory around one stint over VALID snapshots.

    Valid snapshots where the song is absent are returned with ``on_chart=False`` and
    rank = off-chart rank (51) so exits are visible. Calendar gaps are not invented.
    """
    s = appearances[appearances["song_id"] == song_id].sort_values("date")
    st = s[s["stint_no"] == stint_no]
    start, end = st["date"].min(), st["date"].max()
    valid = snapshots.loc[snapshots["is_valid_snapshot"], "date"].sort_values()
    before = valid[valid < start].tail(pre_days)
    after = valid[valid > end].head(post_days)
    lo = before.min() if len(before) else start
    hi = after.max() if len(after) else end
    win = valid[(valid >= lo) & (valid <= hi)]
    traj = pd.DataFrame({"date": win.to_numpy()}).merge(
        s[["date", "position", "popularity", "stint_no"]], on="date", how="left")
    traj["on_chart"] = traj["position"].notna()
    traj["rank_plot"] = traj["position"].fillna(OFF_CHART_RANK)
    traj["in_selected_stint"] = traj["stint_no"].eq(stint_no)
    return traj
