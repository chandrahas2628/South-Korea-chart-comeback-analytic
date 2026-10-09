import numpy as np
import pandas as pd
import pytest

from analytics.momentum import compute_momentum, lifecycle_stages, stint_metrics
from analytics.reentry import compute_reentries
from conftest import make_rows, make_snapshots

D = pd.date_range("2024-01-01", periods=30, freq="D")


def build(app, snap_dates=None, min_absent=2):
    rows = make_rows(app)
    snaps = make_snapshots(snap_dates if snap_dates is not None else D)
    r = compute_reentries(rows, snaps, min_absent)
    ev, _ = compute_momentum(r.stints, r.appearances, snapshots=snaps)
    return ev


def test_retention_days_counts_calendar_span_inclusive():
    ev = build({"x": [(D[0], 40), (D[5], 30), (D[6], 28), (D[7], 29), (D[8], 33)]})
    e = ev[ev["is_reentry"]].iloc[0]
    assert e["retention_snapshots"] == 4 and e["retention_days"] == 4       # D5..D8 inclusive
    assert e["exit_observed_date"] == D[9]                                    # first valid absence after the stint
    assert not e["is_right_censored"] and e["exit_uncertainty_days"] == 0


def test_missing_calendar_dates_inside_stint_are_not_treated_as_observed():
    snaps = [d for i, d in enumerate(D) if i not in (7,)]                    # D7 has no snapshot
    ev = build({"x": [(D[0], 40), (D[5], 30), (D[6], 28), (D[8], 33)]}, snaps)
    e = ev[ev["is_reentry"]].iloc[0]
    assert e["retention_snapshots"] == 3                                      # only 3 observed appearances
    assert e["retention_days"] == 4                                           # calendar span D5..D8
    assert e["unobserved_days_in_stint"] == 1                                 # D7 not pretended to be observed
    assert e["dropout_snapshots"] == 0                                        # D7 is not an observed absence


def test_observed_dropout_shorter_than_threshold_stays_in_stint_and_is_counted():
    # D0, then 4 absences, back D5; one observed absence (D6) inside the comeback is < threshold 2 => same stint
    ev = build({"x": [(D[0], 40), (D[5], 30), (D[7], 28), (D[8], 33)]}, min_absent=2)
    re = ev[ev["is_reentry"]]
    assert len(re) == 1                                                       # D7 does NOT create a second re-entry
    e = re.iloc[0]
    assert e["retention_snapshots"] == 3 and e["retention_days"] == 4 and e["dropout_snapshots"] == 1


def test_one_snapshot_dropout_is_a_reentry_only_when_threshold_is_one():
    app = {"x": [(D[0], 40), (D[2], 38)]}                                     # exactly one observed absence (D1)
    assert len(build(app, min_absent=1).query("is_reentry")) == 1
    assert len(build(app, min_absent=2).query("is_reentry")) == 0


def test_right_censored_stint_is_flagged_and_has_no_exit_date():
    ev = build({"x": [(D[0], 40), (D[20], 30), (D[29], 25)]}, min_absent=2)
    last = ev[ev["is_reentry"]].sort_values("event_date").iloc[-1]
    assert bool(last["is_right_censored"]) and bool(last["retention_is_lower_bound"])
    assert pd.isna(last["exit_observed_date"])


def test_exit_uncertainty_when_calendar_gap_follows_last_appearance():
    snaps = [d for i, d in enumerate(D) if i not in (7, 8)]
    ev = build({"x": [(D[0], 40), (D[5], 30), (D[6], 28)]}, snaps)
    e = ev[ev["is_reentry"]].iloc[0]
    assert e["exit_observed_date"] == D[9] and e["exit_uncertainty_days"] == 2
    assert e["retention_days"] == 2                                           # conservative: only observed presence


def test_first_entry_retention_and_stage_flow():
    ev = build({"x": [(D[3], 40), (D[4], 30)], "y": [(D[0], 10)]})
    fe = ev[ev["stint_type"] == "first_entry"].iloc[0]
    assert fe["retention_days"] == 2 and np.isnan(fe["popularity_lift"])
    stages = [s["stage"] for s in lifecycle_stages(fe)]
    assert stages[0] == "PRE-COMEBACK" and "ENTRY" in stages and stages[-1] in ("EXIT", "STILL ON CHART")


def test_lifecycle_for_reentry_has_all_stages():
    ev = build({"x": [(D[0], 40), (D[5], 30), (D[6], 20), (D[7], 25)]})
    e = ev[ev["is_reentry"]].iloc[0]
    assert [s["stage"] for s in lifecycle_stages(e)] == ["PRE-COMEBACK", "EXIT", "RE-ENTRY", "SURGE", "PEAK", "DECAY", "EXIT"]


def test_no_infinite_values_in_momentum_output():
    ev = build({"x": [(D[0], 40), (D[5], 30), (D[6], 20)], "y": [(D[1], 50), (D[9], 50)]})
    num = ev.select_dtypes("number")
    assert not np.isinf(num.to_numpy(dtype=float)).any()


def test_negative_rank_jump_and_zero_day_recovery_are_valid():
    m = stint_metrics(pd.date_range("2024-01-01", periods=2), [49, 49], [50, 50], baseline_pos=40, baseline_pop=50)
    assert m["rank_jump"] == -9 and m["recovery_speed"] == -9.0 and m["days_to_peak"] == 0 and np.isfinite(m["recovery_speed"])
