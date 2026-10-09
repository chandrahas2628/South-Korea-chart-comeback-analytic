import pytest

from analytics.filters import FilterState, apply_filters
from analytics.pipeline import run_pipeline


@pytest.fixture(scope="module")
def res():
    return run_pipeline("data/raw/Atlantic_South_Korea.csv", 3)


def test_no_filter_returns_everything(res):
    fd = apply_filters(res, FilterState())
    assert len(fd.reentries) == len(res.reentries) and not fd.active


def test_artist_filter_restricts_rows_events_songs(res):
    artist = res.reentries["artist"].value_counts().index[0]
    fd = apply_filters(res, FilterState(artists=[artist]))
    assert set(fd.rows["artist"]) == {artist} and set(fd.reentries["artist"]) == {artist}
    assert set(fd.songs["artist"]) == {artist} and fd.active


def test_date_range_filters_events_by_reentry_date(res):
    import pandas as pd
    fd = apply_filters(res, FilterState(date_range=("2025-01-01", "2025-03-31")))
    assert fd.reentries["reentry_date"].between("2025-01-01", "2025-03-31").all()
    assert fd.rows["date"].between("2025-01-01", "2025-03-31").all()
    assert len(fd.reentries) < len(res.reentries)


def test_min_reentries_and_momentum_range(res):
    fd = apply_filters(res, FilterState(min_reentries=3))
    assert (fd.reentries.groupby("song_id").size() >= 3).all()
    fd = apply_filters(res, FilterState(momentum_range=(55.0, 100.0)))
    assert fd.reentries["momentum_score"].min() >= 55.0 and fd.active


def test_empty_selection_gives_notice_not_error(res):
    fd = apply_filters(res, FilterState(explicit="Explicit", min_reentries=50))
    assert fd.reentries.empty and fd.notices


def test_scores_remain_globally_normalised_under_filters(res):
    artist = res.reentries["artist"].value_counts().index[0]
    fd = apply_filters(res, FilterState(artists=[artist]))
    a = fd.artists.set_index("artist").loc[artist, "fandom_score"]
    assert a == res.artists.set_index("artist").loc[artist, "fandom_score"]


def test_date_filter_propagates_to_song_and_artist_statistics(res):
    fd = apply_filters(res, FilterState(date_range=("2025-01-01", "2025-03-31")))
    assert fd.songs["reentry_count"].sum() == len(fd.reentries) == fd.artists["reentry_count"].sum()
    top = fd.songs.sort_values("reentry_count", ascending=False).iloc[0]
    sub = fd.reentries[fd.reentries["song_id"] == top["song_id"]]
    assert top["mean_momentum"] == pytest.approx(sub["momentum_score"].mean())
    assert top["mean_retention_days"] == pytest.approx(sub["retention_days"].mean())


def test_reset_equals_default_state(res):
    a = apply_filters(res, FilterState())
    b = apply_filters(res, FilterState(date_range=(res.cleaned.snapshots["date"].min(), res.cleaned.snapshots["date"].max()),
                                       momentum_range=(0.0, 100.0)))
    assert len(a.reentries) == len(b.reentries) and not b.active


def test_album_type_and_explicit_filters(res):
    fd = apply_filters(res, FilterState(album_types=["single"], explicit="Clean"))
    # attribute filters work at song level (each song's most common value) so full song histories stay intact
    assert set(fd.songs["album_type"]) == {"single"} and set(fd.songs["explicit_label"]) == {"Clean"}
    assert set(fd.rows["song_id"]) <= set(fd.songs["song_id"])
