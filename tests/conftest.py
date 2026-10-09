import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_snapshots(dates, valid=None):
    """Snapshot table for tests: ``valid`` lists dates that are NOT complete if given as incomplete."""
    d = pd.to_datetime(pd.Series(dates))
    df = pd.DataFrame({"date": d, "n_rows": 50, "status": "complete", "is_valid_snapshot": True})
    return df


def make_rows(appearances, artist="A", popularity=None):
    """appearances: {song_title: [(date, position[, popularity]), ...]} -> clean-like DataFrame."""
    recs = []
    for song, items in appearances.items():
        for it in items:
            date, pos = it[0], it[1]
            pop = it[2] if len(it) > 2 else 50
            recs.append({"date": pd.Timestamp(date), "position": pos, "song_id": f"id_{song}", "song": song, "artist": artist,
                         "song_label": f"{song} — {artist}", "popularity": pop, "album_type": "single", "is_explicit": False,
                         "duration_minutes": 3.0, "total_tracks": 1})
    return pd.DataFrame(recs).sort_values(["date", "position"]).reset_index(drop=True)


def raw_day(date_str, n=50, offset=0):
    return pd.DataFrame({
        "date": date_str, "position": range(1, n + 1), "song": [f"Song {i + offset}" for i in range(n)],
        "artist": [f"Artist {i % 7}" for i in range(n)], "popularity": 60, "duration_ms": 200000, "album_type": "single",
        "total_tracks": 1, "is_explicit": "FALSE", "album_cover_url": "x"})


@pytest.fixture
def days():
    return pd.date_range("2024-01-01", periods=12, freq="D")
