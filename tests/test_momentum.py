import numpy as np
import pandas as pd
import pytest

from analytics.momentum import classify_patterns, compute_momentum, score_momentum, stint_metrics, weight_sensitivity
from analytics.reentry import compute_reentries
from conftest import make_rows, make_snapshots

D = pd.date_range("2024-01-01", periods=20, freq="D")


def test_known_metrics_for_a_comeback():
    # re-entry rank 38 -> 30 -> 12 -> 20 -> 35 ; baseline exit rank 48, baseline popularity 50
    m = stint_metrics(D[:5], [38, 30, 12, 20, 35], [52, 55, 60, 58, 54], baseline_pos=48, baseline_pop=50)
    assert m["rank_jump"] == 10 and m["peak_rank"] == 12 and m["peak_improvement"] == 26
    assert m["days_to_peak"] == 2 and m["snapshots_to_peak"] == 2
    assert m["popularity_lift"] == 10                       # max 60 - baseline 50
    assert m["retention_snapshots"] == 5 and m["lifespan_days"] == 5
    assert m["recovery_speed"] == pytest.approx((48 - 12) / 3)
    assert m["decay_speed"] == pytest.approx((35 - 12) / 2)
    assert m["volatility"] == pytest.approx(np.std(np.diff([38, 30, 12, 20, 35]), ddof=1))


def test_rank_does_not_improve_after_reentry():
    m = stint_metrics(D[:3], [30, 31, 33], [50, 50, 50], baseline_pos=40, baseline_pop=50)
    assert m["peak_improvement"] == 0 and m["peak_rank"] == 30 and m["days_to_peak"] == 0
    assert m["recovery_speed"] == 10.0                      # (40-30)/(0+1)


def test_popularity_unchanged_gives_zero_lift():
    m = stint_metrics(D[:3], [30, 25, 20], [50, 50, 50], baseline_pos=40, baseline_pop=50)
    assert m["popularity_lift"] == 0


def test_very_short_retention_edge_case():
    m = stint_metrics(D[:1], [45], [40], baseline_pos=49, baseline_pop=41)
    assert m["retention_snapshots"] == 1 and m["lifespan_days"] == 1
    assert np.isnan(m["volatility"]) and np.isnan(m["decay_speed"])
    assert m["popularity_lift"] == -1                       # popularity fell vs baseline


def test_first_entry_has_no_popularity_baseline():
    m = stint_metrics(D[:2], [20, 10], [50, 60], baseline_pos=51, baseline_pop=np.nan)
    assert np.isnan(m["popularity_lift"]) and m["rank_jump"] == 31


def _events():
    app = {"a": [(D[0], 40), (D[1], 41), (D[5], 20), (D[6], 10), (D[7], 5), (D[8], 8), (D[9], 15),
                 (D[12], 49)],                             # a: initial stint, re-entry (strong), re-entry (weak 1-day)
           "b": [(D[2], 30), (D[3], 25), (D[8], 47), (D[9], 48)]}
    rows = make_rows(app)
    snaps = make_snapshots(D)
    r = compute_reentries(rows, snaps, 2)
    return compute_momentum(r.stints, r.appearances)


def test_compute_momentum_pipeline_scores_and_flags():
    ev, thr = _events()
    assert ev["stint_type"].eq("initial_observed").sum() == 1
    init = ev[ev["stint_type"] == "initial_observed"]
    assert init["momentum_score"].isna().all()               # left-censored => not scored
    scored = ev[ev["momentum_score"].notna()]
    assert scored["momentum_score"].between(0, 100).all()
    assert ev["pattern"].isin(["Explosive / Short-lived", "Strong / Sustainable", "Moderate / Stable",
                               "Weak / Short-lived", "Not scored"]).all()
    re = ev[ev["is_reentry"]]
    assert re["retention_snapshots"].min() >= 1
    strong = re.sort_values("peak_improvement").iloc[-1]
    assert strong["peak_rank"] <= 20


def test_score_orders_stronger_comeback_higher():
    df = pd.DataFrame({"rank_jump": [1, 30], "peak_improvement": [0, 30], "popularity_lift": [0, 20],
                       "recovery_speed": [0.1, 10], "retention_snapshots": [1, 40]})
    s = score_momentum(df)
    assert s.iloc[1] > s.iloc[0] and s.between(0, 100).all()


def test_missing_component_reweights_instead_of_penalising():
    df = pd.DataFrame({"rank_jump": [1, 30, 10], "peak_improvement": [0, 30, 5], "popularity_lift": [np.nan] * 3,
                       "recovery_speed": [0.1, 10, 1], "retention_snapshots": [1, 40, 5]})
    s = score_momentum(df)
    assert s.notna().all() and s.iloc[1] > s.iloc[2] > s.iloc[0]


def test_classify_patterns_uses_quantiles():
    df = pd.DataFrame({"momentum_score": np.linspace(10, 90, 30), "retention_snapshots": [1] * 15 + [20] * 15})
    cls, thr = classify_patterns(df)
    assert thr["high_momentum_threshold"] > thr["low_momentum_threshold"]
    assert cls.iloc[-1] == "Strong / Sustainable" and cls.iloc[0] == "Weak / Short-lived"
    assert (cls == "Moderate / Stable").any()


def test_weight_sensitivity_returns_rank_correlation():
    ev, _ = _events()
    out = weight_sensitivity(ev)
    assert -1 <= out["spearman_vs_equal_weights"] <= 1 or np.isnan(out["spearman_vs_equal_weights"])
