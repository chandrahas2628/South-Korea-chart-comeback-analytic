import pandas as pd
import pytest

from analytics.reentry import compute_reentries
from conftest import make_rows, make_snapshots


def run(app, dates, min_absent=1, snaps=None):
    rows = make_rows(app)
    s = snaps if snaps is not None else make_snapshots(dates)
    return compute_reentries(rows, s, min_absent)


def test_song_appears_once_has_no_reentry(days):
    r = run({"x": [(days[3], 20)]}, days)
    assert r.events.empty and r.song_summary["reentry_count"].iloc[0] == 0 and len(r.stints) == 1


def test_continuous_song_never_reenters(days):
    r = run({"x": [(d, 10) for d in days]}, days)
    assert r.events.empty and r.stints["stint_appearances"].iloc[0] == 12


def test_single_reentry_gap_and_absence(days):
    # present d0-d2, absent d3-d5 (observed), back d6
    app = {"x": [(days[i], 30 + i) for i in (0, 1, 2)] + [(days[6], 22)]}
    r = run(app, days)
    e = r.events.iloc[0]
    assert len(r.events) == 1
    assert e["gap_days"] == 4 and e["absent_snapshots"] == 3 and e["unobserved_calendar_days"] == 0
    assert e["previous_exit_position"] == 32 and e["reentry_position"] == 22
    assert e["previous_appearance_count"] == 3 and e["reentry_date"] == days[6]


def test_multiple_reentries_counted_in_order(days):
    app = {"x": [(days[0], 40), (days[3], 35), (days[6], 30), (days[9], 25)]}
    r = run(app, days)
    assert len(r.events) == 3
    assert r.events["stint_no"].tolist() == [1, 2, 3]
    assert r.song_summary["reentry_count"].iloc[0] == 3
    assert r.events["previous_appearance_count"].tolist() == [1, 2, 3]


def test_missing_calendar_dates_do_not_create_reentry(days):
    # Snapshots missing for d3..d5; song seen d2 and d6 -> no OBSERVED absence -> not a re-entry
    snaps = make_snapshots([d for i, d in enumerate(days) if i not in (3, 4, 5)])
    r = run({"x": [(days[2], 20), (days[6], 20)]}, days, snaps=snaps)
    assert r.events.empty
    assert r.stints["stint_appearances"].iloc[0] == 2


def test_gap_with_partial_observation_counts_only_observed_absence(days):
    # d3 observed (absent), d4,d5 missing; song d2 and d6 -> 1 observed absence, 2 unobserved days
    snaps = make_snapshots([d for i, d in enumerate(days) if i not in (4, 5)])
    r = run({"x": [(days[2], 20), (days[6], 18)]}, days, snaps=snaps)
    e = r.events.iloc[0]
    assert e["absent_snapshots"] == 1 and e["unobserved_calendar_days"] == 2 and e["gap_days"] == 4


def test_incomplete_snapshot_is_not_proof_of_absence(days):
    snaps = make_snapshots(days)
    snaps.loc[3, ["is_valid_snapshot", "status"]] = [False, "incomplete"]
    r = run({"x": [(days[2], 20), (days[4], 20)]}, days, snaps=snaps)
    assert r.events.empty                      # only an incomplete day lies between


def test_min_absent_threshold(days):
    app = {"x": [(days[0], 45), (days[2], 44)]}      # one observed absence (d1)
    assert len(run(app, days, min_absent=1).events) == 1
    assert run(app, days, min_absent=2).events.empty
    with pytest.raises(ValueError):
        run(app, days, min_absent=0)


def test_initial_stint_is_left_censored_and_unscored_type(days):
    r = run({"x": [(days[0], 5), (days[1], 6)], "y": [(days[4], 9)]}, days)
    t = r.stints.set_index("song_id")["stint_type"]
    assert t["id_x"] == "initial_observed" and t["id_y"] == "first_entry"
    assert bool(r.stints.set_index("song_id").loc["id_y", "is_right_censored"]) is False


def test_right_censoring_flag(days):
    r = run({"x": [(days[0], 5), (days[11], 6)]}, days)
    assert bool(r.events["is_right_censored"].iloc[0]) is True
