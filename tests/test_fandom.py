import numpy as np
import pandas as pd
import pytest

from analytics.fandom import _entity_table, assign_bands, normalize_fandom_score
from analytics.utils import cliffs_delta, minmax_scale, percentile_normalize, shrink_mean, weighted_composite


def _components(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({k: rng.random(n) for k in
                         ["reentry_count", "momentum", "popularity_lift", "rank_recovery", "recovery_speed", "retention", "sharpness"]})


def test_score_is_normalised_0_to_100():
    s = normalize_fandom_score(_components())
    assert s.min() == pytest.approx(0) and s.max() == pytest.approx(100)


def test_constant_inputs_do_not_crash():
    c = pd.DataFrame({k: [1.0] * 5 for k in ["reentry_count", "momentum", "popularity_lift", "rank_recovery",
                                             "recovery_speed", "retention", "sharpness"]})
    s = normalize_fandom_score(c)
    assert s.notna().all() and s.between(0, 100).all()


def test_percentile_and_minmax_helpers():
    p = percentile_normalize(pd.Series([10, 20, 30, np.nan]))
    assert p.iloc[0] < p.iloc[1] < p.iloc[2] and p.isna().iloc[3] and p.dropna().between(0, 1).all()
    assert minmax_scale(pd.Series([2, 4, 6])).tolist() == [0, 50, 100]


def test_weighted_composite_ignores_missing():
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [0.0, 1.0]})
    out = weighted_composite(df, {"a": 0.5, "b": 0.5})
    assert out.iloc[0] == pytest.approx(0.5) and out.iloc[1] == pytest.approx(1.0)


def test_shrinkage_pulls_small_samples_to_global_mean():
    m = shrink_mean(pd.Series([100.0, 100.0]), pd.Series([1, 100]), global_mean=50, k=5)
    assert m.iloc[0] < 60 and m.iloc[1] > 95


def _events(entities):
    rows = []
    for ent, (n, mom) in entities.items():
        for i in range(n):
            rows.append({"song_id": ent, "momentum_score": mom, "popularity_lift": 5.0, "rank_recovery_total": 10.0,
                         "recovery_speed": 2.0, "retention_snapshots": 10, "retention_days": 12, "is_right_censored": False,
                         "reentry_position": 20, "gap_days": 10})
    return pd.DataFrame(rows)


def test_single_lucky_comeback_does_not_outrank_a_long_record():
    ents = {f"s{i}": (5, 50 + i) for i in range(10)}
    ents["lucky"] = (1, 95)              # one very strong comeback
    ents["proven"] = (12, 70)            # many strong comebacks
    t = _entity_table(_events(ents), "song_id", k=3.0)
    assert t.loc["proven", "fandom_score"] > t.loc["lucky", "fandom_score"]
    assert t.loc["lucky", "confidence"].startswith("Low") and t.loc["proven", "confidence"].startswith("High")


def test_bands_are_quantile_based_and_ordered():
    s = pd.Series(np.arange(100, dtype=float))
    bands, cuts = assign_bands(s)
    assert bands.iloc[0] == "Low" and bands.iloc[-1] == "Extreme"
    assert cuts["P30"] < cuts["P60"] < cuts["P80"] < cuts["P95"]


def test_cliffs_delta_bounds():
    assert cliffs_delta([5, 6, 7], [1, 2, 3]) == 1.0 and cliffs_delta([1, 2], [1, 2]) == 0.0
