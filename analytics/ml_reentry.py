"""Optional, time-respecting model: P(song re-enters within H days | currently exited).

Leakage control: every feature is computed from data on or before the cutoff date T; the
target looks only at (T, T+H]. Train/test are split chronologically with an H-day embargo.
The result carries a `reliable` flag; if the data do not support the target, the reason is
returned and the dashboard documents it instead of presenting forecasts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["days_since_exit", "absent_snapshots", "prior_reentries", "appearances", "best_position",
            "exit_position", "mean_prior_retention", "mean_popularity", "exit_popularity", "artist_prior_reentries",
            "days_since_first", "duration_minutes", "total_tracks", "is_explicit", "is_single"]


def _to_ns64(values) -> np.ndarray:
    """Return ``values`` as a tz-naive ``datetime64[ns]`` array (scalars become 1-element arrays).

    pandas 2.x/3.x can hold datetimes at different resolutions (s/ms/us/ns) and ``Timestamp`` scalars are
    not numpy scalars, so mixing them inside ``np.searchsorted`` is version dependent. Every date used in a
    ``searchsorted`` call goes through this helper so both sides share one dtype and the comparison is exact.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(values) if not hasattr(values, "dtype") else values))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.as_unit("ns").to_numpy()


def build_samples(stints: pd.DataFrame, appearances: pd.DataFrame, snapshots: pd.DataFrame,
                  min_absent: int, horizon: int = 30, step_days: int = 7, cutoffs=None) -> pd.DataFrame:
    """One row per (cutoff date, song with a confirmed exit at that date)."""
    valid = snapshots.loc[snapshots["is_valid_snapshot"], "date"].sort_values().reset_index(drop=True)
    last_date = valid.iloc[-1]
    valid_ns = _to_ns64(valid)
    if cutoffs is None:
        cutoffs = valid.iloc[::step_days]
        cutoffs = cutoffs[cutoffs + pd.Timedelta(days=horizon) <= last_date]
    else:
        cutoffs = pd.Series(pd.to_datetime(list(cutoffs)))
    attrs = appearances.groupby("song_id").agg(
        artist=("artist", "first"), duration_minutes=("duration_minutes", "median"),
        total_tracks=("total_tracks", "median"), is_explicit=("is_explicit", "max"),
        is_single=("album_type", lambda s: float((s == "single").mean() >= 0.5)))
    rows = []
    st = stints.sort_values(["song_id", "stint_no"])
    for T in cutoffs:
        past_st = st[st["start_date"] <= T]
        last = past_st.groupby("song_id").tail(1).set_index("song_id")
        exited = last[last["end_date"] < T].copy()
        # one consistent representation (datetime64[ns]) on both sides of every searchsorted call
        n_abs = np.searchsorted(valid_ns, _to_ns64([T])[0], side="right") - np.searchsorted(
            valid_ns, _to_ns64(exited["end_date"]), side="right")
        exited = exited[n_abs >= min_absent].copy()
        exited["absent_snapshots"] = n_abs[n_abs >= min_absent]
        if exited.empty:
            continue
        past_app = appearances[(appearances["date"] <= T) & appearances["song_id"].isin(exited.index)]
        ag = past_app.groupby("song_id").agg(appearances=("position", "size"), best_position=("position", "min"),
                                             mean_popularity=("popularity", "mean"),
                                             first_date=("date", "min"))
        hist = past_st[past_st["song_id"].isin(exited.index)].groupby("song_id").agg(
            n_stints=("stint_no", "size"), mean_prior_retention=("stint_appearances", "mean"))
        art_re = past_st[past_st["stint_no"] > 0].merge(attrs[["artist"]], left_on="song_id", right_index=True,
                                                         suffixes=("_s", "")).groupby("artist").size()
        s = exited.join(ag).join(hist).join(attrs, rsuffix="_a")
        s["cutoff"] = T
        s["days_since_exit"] = (T - s["end_date"]).dt.days
        s["prior_reentries"] = s["n_stints"] - 1
        s["exit_position"] = s["exit_position"]
        s["exit_popularity"] = s["exit_popularity"]
        s["artist_prior_reentries"] = s["artist"].map(art_re).fillna(0)
        s["days_since_first"] = (T - s["first_date"]).dt.days
        future = st[(st["stint_no"] > 0) & (st["start_date"] > T) & (st["start_date"] <= T + pd.Timedelta(days=horizon))]
        s["target"] = s.index.isin(future["song_id"]).astype(int)
        rows.append(s.reset_index()[["song_id", "cutoff", "target"] + FEATURES])
    if not rows:
        return pd.DataFrame(columns=["song_id", "cutoff", "target"] + FEATURES)
    out = pd.concat(rows, ignore_index=True)
    out["is_explicit"] = out["is_explicit"].astype(float)
    return out


def _bootstrap_auc(y, p, groups, n=300, seed=7):
    """95% interval for ROC-AUC from a *song-clustered* bootstrap (repeated cutoffs of one song are not independent)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    members = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(members), len(members))
        b = np.concatenate([members[i] for i in pick])
        if y[b].min() != y[b].max():
            vals.append(roc_auc_score(y[b], p[b]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def temporal_split(samples: pd.DataFrame, horizon: int, train_frac: float = 0.6):
    """Chronological split with an embargo of ``horizon`` days between train cutoffs and test cutoffs."""
    cuts = np.sort(samples["cutoff"].unique())
    split = pd.Timestamp(cuts[int(len(cuts) * train_frac)])
    embargo_end = split + pd.Timedelta(days=horizon)
    return samples[samples["cutoff"] < split], samples[samples["cutoff"] >= embargo_end], split, embargo_end


def train_and_evaluate(samples: pd.DataFrame, horizon: int = 30, train_frac: float = 0.6, min_pos: int = 30) -> dict:
    """Chronological train/test with embargo; returns metrics and a reliability verdict."""
    res = {"horizon_days": horizon, "n_samples": int(len(samples)), "supported": False, "reliable": False, "reason": ""}
    if len(samples) < 200:
        res["reason"] = "Too few (cutoff, exited song) samples to model."
        return res
    train, test, split, embargo = temporal_split(samples, horizon, train_frac)
    res.update({"split_date": pd.Timestamp(split), "test_start": embargo, "n_train": len(train), "n_test": len(test),
                "pos_train": int(train["target"].sum()), "pos_test": int(test["target"].sum()),
                "prevalence_test": float(test["target"].mean()) if len(test) else np.nan})
    if train["target"].sum() < min_pos or test["target"].sum() < min_pos or test["target"].nunique() < 2:
        res["reason"] = f"Fewer than {min_pos} positive events in train or test period; target not reliably learnable."
        return res
    res["supported"] = True
    Xtr, ytr, Xte, yte = train[FEATURES], train["target"].to_numpy(), test[FEATURES], test["target"].to_numpy()
    models = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    }
    metrics, preds = [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        preds[name] = p
        lo, hi = _bootstrap_auc(yte, p, test["song_id"].to_numpy())
        top = max(1, int(0.1 * len(p)))
        order = np.argsort(-p)[:top]
        metrics.append({"model": name, "roc_auc": roc_auc_score(yte, p), "auc_ci_low": lo, "auc_ci_high": hi,
                        "pr_auc": average_precision_score(yte, p), "brier": brier_score_loss(yte, np.clip(p, 0, 1)),
                        "precision_top10pct": float(yte[order].mean()),
                        "lift_top10pct": float(yte[order].mean() / yte.mean())})
    base_p = -test["days_since_exit"].to_numpy(dtype=float)      # trivial baseline: recently exited songs return sooner
    lo, hi = _bootstrap_auc(yte, base_p, test["song_id"].to_numpy())
    top = max(1, int(0.1 * len(base_p)))
    metrics.append({"model": "Baseline: recency only", "roc_auc": roc_auc_score(yte, base_p), "auc_ci_low": lo, "auc_ci_high": hi,
                    "pr_auc": average_precision_score(yte, base_p), "brier": np.nan,
                    "precision_top10pct": float(yte[np.argsort(-base_p)[:top]].mean()),
                    "lift_top10pct": float(yte[np.argsort(-base_p)[:top]].mean() / yte.mean())})
    mt = pd.DataFrame(metrics)
    res["baseline_auc"] = float(mt.loc[mt["model"].str.startswith("Baseline"), "roc_auc"].iloc[0])
    mt = mt
    best = mt[~mt["model"].str.startswith("Baseline")].sort_values("roc_auc", ascending=False).iloc[0]
    res["metrics"] = mt
    res["best_model"] = best["model"]
    res["beats_baseline"] = bool(best["roc_auc"] >= res["baseline_auc"] + 0.01)
    clears_bar = bool(best["auc_ci_low"] > 0.60 and best["pr_auc"] > 1.2 * res["prevalence_test"])
    res["reliable"] = bool(clears_bar and res["beats_baseline"])
    if res["reliable"]:
        res["reason"] = "Model beats the base rate (AUC interval above 0.60) and improves on the recency-only baseline by >= 0.01 AUC."
    elif clears_bar:
        res["reason"] = (f"Discriminates well (AUC {best['roc_auc']:.3f}) but does not improve on a trivial recency-only baseline "
                         f"(AUC {res['baseline_auc']:.3f}); the signal is almost entirely 'recently exited songs return sooner'. "
                         "Treat as a descriptive risk ranking, not a validated forecast.")
    else:
        res["reason"] = ("Model does not clear the reliability bar (95% AUC lower bound > 0.60 and PR-AUC > 1.2x base rate); "
                         "forecasts are not presented as dependable.")
    final = models[best["model"]]
    final.fit(samples[FEATURES], samples["target"])
    res["model"] = final
    return res


def score_current(model, stints, appearances, snapshots, min_absent: int) -> pd.DataFrame:
    """Score songs currently exited as of the last valid snapshot (features use data up to that date)."""
    last = snapshots.loc[snapshots["is_valid_snapshot"], "date"].max()
    s = build_samples(stints, appearances, snapshots, min_absent, cutoffs=[last])
    if s.empty:
        return s
    s["reentry_probability"] = model.predict_proba(s[FEATURES])[:, 1]
    meta = appearances.groupby("song_id").agg(song_label=("song_label", "first"), artist=("artist", "first"))
    return (s.merge(meta, left_on="song_id", right_index=True)
            .sort_values("reentry_probability", ascending=False)
            [["song_label", "artist", "reentry_probability", "days_since_exit", "prior_reentries", "exit_position"]]
            .reset_index(drop=True))
