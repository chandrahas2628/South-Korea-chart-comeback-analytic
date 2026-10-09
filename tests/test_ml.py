import numpy as np
import pandas as pd
import pytest

from analytics.ml_reentry import FEATURES, build_samples, temporal_split, train_and_evaluate
from analytics.pipeline import run_pipeline
from analytics.reentry import compute_reentries

MIN_ABSENT, H = 3, 30


@pytest.fixture(scope="module")
def res():
    return run_pipeline("data/raw/Atlantic_South_Korea.csv", MIN_ABSENT)


@pytest.fixture(scope="module")
def samples(res):
    return build_samples(res.reentry.stints, res.appearances, res.cleaned.snapshots, MIN_ABSENT, H)


def test_temporal_split_respects_embargo(samples):
    train, test, split, embargo = temporal_split(samples, H)
    assert train["cutoff"].max() < split <= embargo - pd.Timedelta(days=H)
    assert test["cutoff"].min() >= embargo
    assert (test["cutoff"].min() - train["cutoff"].max()).days >= H        # no horizon overlap between train and test


def test_features_do_not_use_future_information(res, samples):
    """Re-compute stints on data truncated at the cutoff: features for exited songs must be identical."""
    cutoff = pd.Timestamp(sorted(samples["cutoff"].unique())[30])
    clean = res.cleaned.clean[res.cleaned.clean["date"] <= cutoff]
    snaps = res.cleaned.snapshots[res.cleaned.snapshots["date"] <= cutoff]
    trunc = compute_reentries(clean, snaps, MIN_ABSENT)
    a = build_samples(trunc.stints, trunc.appearances, snaps, MIN_ABSENT, cutoffs=[cutoff]).set_index("song_id")
    b = samples[samples["cutoff"] == cutoff].set_index("song_id")
    common = a.index.intersection(b.index)
    assert len(common) > 50 and set(b.index) == set(a.index)
    pd.testing.assert_frame_equal(a.loc[common, FEATURES].sort_index(), b.loc[common, FEATURES].sort_index(), check_dtype=False)


def test_target_only_looks_forward(res, samples):
    row = samples[samples["target"] == 1].iloc[0]
    st = res.reentry.stints
    nxt = st[(st["song_id"] == row["song_id"]) & (st["stint_no"] > 0) & (st["start_date"] > row["cutoff"])
             & (st["start_date"] <= row["cutoff"] + pd.Timedelta(days=H))]
    assert len(nxt) >= 1


def test_evaluation_reports_baseline_prevalence_and_honest_reliability(samples):
    out = train_and_evaluate(samples, H)
    assert out["supported"] and 0 < out["prevalence_test"] < 0.2
    assert any(m.startswith("Baseline") for m in out["metrics"]["model"])
    assert out["reliable"] == (out["beats_baseline"] and out["metrics"].iloc[:2]["auc_ci_low"].max() > 0.6
                               and out["metrics"].iloc[:2]["pr_auc"].max() > 1.2 * out["prevalence_test"])


def test_too_few_samples_is_declared_unsupported():
    out = train_and_evaluate(pd.DataFrame({"cutoff": pd.to_datetime(["2024-01-01"]), "target": [0]}), H)
    assert out["supported"] is False and out["reason"]
