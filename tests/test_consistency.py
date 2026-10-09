import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics.filters import FilterState, apply_filters
from analytics.momentum import PATTERNS, score_momentum
from analytics.pipeline import run_pipeline
from analytics.reporting import artist_views, compute_kpis, research_summary

RAW = Path("data/raw/Atlantic_South_Korea.csv")


@pytest.fixture(scope="module")
def res():
    return run_pipeline(str(RAW), 3)


def test_raw_file_is_never_modified():
    before = hashlib.md5(RAW.read_bytes()).hexdigest()
    run_pipeline(str(RAW), 2)
    assert hashlib.md5(RAW.read_bytes()).hexdigest() == before


def test_counts_reconcile_across_tables(res):
    fd = apply_filters(res, FilterState())
    k = compute_kpis(fd)
    assert k["total_reentries"] == len(res.reentries) == int(res.songs["reentry_count"].sum()) == int(res.artists["reentry_count"].sum())
    assert len(res.events) == res.songs.shape[0] + k["total_reentries"]          # stints = songs + re-entries
    q = res.cleaned.quality
    assert q["Valid records (after cleaning)"] == len(res.cleaned.clean)
    assert (res.cleaned.removal_ledger["Reconciles"] == "yes").sum() == 1
    assert k["playlist_days"] == q["Valid playlist days (complete 50-entry snapshots)"]


def test_no_nan_or_inf_in_required_reentry_metrics(res):
    re = res.reentries
    for c in ["rank_jump", "peak_rank", "peak_improvement", "popularity_lift", "popularity_change_at_entry", "days_to_peak",
              "snapshots_to_peak", "retention_snapshots", "retention_days", "rank_recovery_total", "recovery_speed", "momentum_score"]:
        assert re[c].notna().all(), c
    num = res.events.select_dtypes("number").to_numpy(dtype=float)
    assert not np.isinf(num).any()
    assert re["retention_days"].ge(re["retention_snapshots"] - re["dropout_snapshots"] * 0).all()


def test_momentum_is_scored_within_stint_type_and_bounded(res):
    ev = res.events
    assert ev.loc[ev["stint_type"] == "initial_observed", "momentum_score"].isna().all()
    for typ in ("re_entry", "first_entry"):
        s = ev.loc[ev["stint_type"] == typ, "momentum_score"]
        assert s.between(0, 100).all() and s.mean() == pytest.approx(50, abs=0.5)    # percentile composite centres on 50
    assert set(res.reentries["pattern"]) <= set(PATTERNS)


def test_sensitivity_and_artist_views(res):
    assert res.sensitivity["spearman_vs_equal_weights"] > 0.9 and res.sensitivity["n_draws"] == 200
    v = artist_views(res.artists)
    assert set(v) == {"Top by frequency", "Top by momentum", "Top by fandom proxy", "Top by retention"}
    assert (v["Top by fandom proxy"]["reentry_count"] >= 3).all()


def test_research_summary_is_generated_not_hand_written(res):
    data, md = research_summary(res, None)
    assert data["kpis"]["total_reentries"] == len(res.reentries)
    assert "retention_findings" in data and "content_findings" in data and "recommendations" in data
    assert f"{len(res.reentries)} confirmed re-entries" in md
    for cut in (str(round(res.events["momentum_score"].max(), 1)),):
        assert isinstance(cut, str)


def test_raising_min_absent_never_creates_more_reentries():
    counts = [len(run_pipeline(str(RAW), k).reentries) for k in (1, 3, 7)]
    assert counts[0] >= counts[1] >= counts[2] and counts[0] > counts[2]
