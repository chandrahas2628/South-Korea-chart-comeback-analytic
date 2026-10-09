import pytest

from analytics.filters import FilterState, apply_filters
from analytics.pipeline import run_pipeline
from analytics.recommendations import MIN_ARTIST_EVENTS, MIN_SONG_EVENTS, RULES, build_recommendations


@pytest.fixture(scope="module")
def res():
    return run_pipeline("data/raw/Atlantic_South_Korea.csv", 3)


def test_rules_are_data_driven_and_consistent_with_thresholds(res):
    fd = apply_filters(res, FilterState())
    rec = build_recommendations(fd)
    t, thr = rec["table"], rec["thresholds"]
    assert set(t["rule"]) <= set(RULES) and len(t) > 0
    songs = fd.songs.set_index("song_label")
    hi = t[t["rule"] == "HIGH MOMENTUM + HIGH RETENTION"]
    for _, r in hi.iterrows():
        s = songs.loc[r["entity"]]
        assert s["mean_momentum"] >= thr["high momentum (P75 of mean momentum)"] - 1e-9
        assert s["mean_retention_days"] >= thr["high retention (P75, days)"] - 1e-9
        assert s["reentry_count"] >= MIN_SONG_EVENTS
    multi = t[t["rule"] == "MULTIPLE RE-ENTRIES"]
    assert (multi["n_events"] >= thr["multiple re-entries (>= max(3, P90))"]).all()
    art = t[t["rule"] == "HIGH ARTIST FANDOM PROXY"]
    assert (art["n_events"] >= MIN_ARTIST_EVENTS).all()


def test_evidence_contains_actual_numbers_incl_retention_days(res):
    t = build_recommendations(apply_filters(res, FilterState()))["table"]
    ev = t[t["level"] == "Song"]["evidence"].iloc[0]
    assert "re-entries" in ev and "mean momentum" in ev and "days" in ev


def test_recommendations_change_with_filters_and_degrade_gracefully(res):
    full = build_recommendations(apply_filters(res, FilterState()))
    sub = build_recommendations(apply_filters(res, FilterState(date_range=("2025-01-01", "2025-02-15"))))
    assert full["thresholds"] != sub["thresholds"] or len(full["table"]) != len(sub["table"])
    none = build_recommendations(apply_filters(res, FilterState(explicit="Explicit", min_reentries=40)))
    assert none["table"].empty and none["notes"]


def test_no_hardcoded_names_in_source():
    import pathlib
    src = pathlib.Path("analytics/recommendations.py").read_text()
    for name in ("Jimin", "NewJeans", "DAY6", "Lim Young Woong"):
        assert name not in src
