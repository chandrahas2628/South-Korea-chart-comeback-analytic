import numpy as np
import pandas as pd
import pytest

from analytics.preprocessing import clean_playlist
from analytics.utils import make_song_id, normalize_text
from analytics.validation import (DataValidationError, check_required_columns, find_calendar_gaps, parse_dates,
                                  snapshot_summary)
from conftest import raw_day


def test_date_parsing_is_day_first_not_swapped():
    s = pd.Series(["18-05-2024", "19-05-2024", "01-06-2024"])
    parsed, fmt = parse_dates(s)
    assert fmt == "%d-%m-%Y" and parsed.iloc[2] == pd.Timestamp("2024-06-01")


def test_ambiguous_dates_resolved_by_file_order():
    # 02-01, 03-01, 04-01 ... could be Feb 1 or Jan 2; chronological file order => dd-mm
    s = pd.Series(["02-01-2024", "03-01-2024", "04-01-2024", "05-01-2024"])
    parsed, fmt = parse_dates(s)
    assert parsed.iloc[0] == pd.Timestamp("2024-01-02")


def test_unparseable_dates_raise():
    with pytest.raises(DataValidationError):
        parse_dates(pd.Series(["foo", "bar"]))


def test_missing_required_columns_raises():
    with pytest.raises(DataValidationError, match="missing required"):
        check_required_columns(pd.DataFrame({"date": [1]}))


def test_song_id_uses_song_and_artist_and_normalises():
    assert make_song_id("Hello", "A") != make_song_id("Hello", "B")          # same title, different artist
    assert make_song_id("  Hello   World ", "Artist") == make_song_id("hello world", "ARTIST")
    assert normalize_text(None) == "" and normalize_text("nan") == "" and normalize_text("  a   b ") == "a b"


def test_clean_marks_invalid_rows_and_missing_artist():
    raw = raw_day("01-01-2024")
    raw.loc[0, "position"] = 51          # out of range
    raw.loc[1, "popularity"] = 150       # out of range
    raw.loc[2, "artist"] = ""            # filled, not rejected
    res = clean_playlist(raw.astype(str))
    assert len(res.rejected) == 2
    assert set(res.rejected["reject_reason"]) == {"invalid_position", "invalid_popularity"}
    assert (res.clean["artist"] == "Unknown Artist").sum() == 1
    assert res.clean["duration_minutes"].iloc[0] == pytest.approx(200000 / 60000)


def test_duplicate_song_date_keeps_best_rank_and_is_counted():
    raw = raw_day("01-01-2024", 49)
    dup = raw.iloc[[10]].copy()
    dup["position"] = 50
    res = clean_playlist(pd.concat([raw, dup]).astype(str))
    assert res.quality["Duplicate song/date records (raw)"] == 1
    assert res.quality["Duplicate song/date rows removed (post-resolution)"] == 1
    kept = res.clean[res.clean["song"] == "Song 10"]
    assert len(kept) == 1 and kept["position"].iloc[0] == 11


def test_stacked_snapshot_is_split_and_logged():
    d1, d3 = raw_day("01-01-2024"), raw_day("03-01-2024")
    a = raw_day("02-01-2024")
    b = raw_day("02-01-2024", offset=25)         # very different snapshot
    raw = pd.concat([d1, a, b, d3]).astype(str).reset_index(drop=True)
    res = clean_playlist(raw)
    snap = res.snapshots.set_index("date").loc[pd.Timestamp("2024-01-02")]
    assert snap["n_rows"] == 50 and snap["is_valid_snapshot"]
    assert res.snapshot_resolution["rows_discarded"].iloc[0] == 50
    assert res.snapshot_resolution["chosen_block"].iloc[0] == 0      # matches its neighbours


def test_snapshot_validation_flags_incomplete_and_calendar_gaps():
    raw = pd.concat([raw_day("01-01-2024"), raw_day("02-01-2024", 40), raw_day("05-01-2024")]).astype(str)
    res = clean_playlist(raw)
    status = res.snapshots.set_index("date")["status"]
    assert status[pd.Timestamp("2024-01-02")] == "incomplete"
    assert not res.snapshots.set_index("date").loc[pd.Timestamp("2024-01-02"), "is_valid_snapshot"]
    assert res.quality["Calendar gaps (missing dates)"] == 2           # 3rd and 4th Jan
    assert res.quality["Incomplete playlist days (<50 entries)"] == 1
    assert res.quality["Valid playlist days (complete 50-entry snapshots)"] == 2


def test_find_calendar_gaps_runs():
    missing, runs = find_calendar_gaps(pd.to_datetime(["2024-01-01", "2024-01-04", "2024-01-05", "2024-01-08"]))
    assert len(missing) == 4 and len(runs) == 2 and runs["days_missing"].tolist() == [2, 2]
