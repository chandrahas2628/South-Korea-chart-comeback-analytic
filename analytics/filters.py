"""Global filter state and application. Fandom scores stay globally normalised."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .fandom import aggregate_events
from .utils import confidence_label


@dataclass
class FilterState:
    date_range: tuple | None = None
    artists: list = field(default_factory=list)
    songs: list = field(default_factory=list)          # song_label values
    album_types: list = field(default_factory=list)
    explicit: str = "All"                              # All / Explicit / Clean
    min_reentries: int = 0
    momentum_range: tuple | None = None


@dataclass
class FilteredData:
    rows: pd.DataFrame
    events: pd.DataFrame
    reentries: pd.DataFrame
    songs: pd.DataFrame
    artists: pd.DataFrame
    snapshots: pd.DataFrame
    notices: list
    active: bool


def apply_filters(result, f: FilterState) -> FilteredData:
    """Apply every filter consistently to rows, events, song and artist tables."""
    rows, ev, songs, snaps = result.appearances, result.events, result.songs, result.cleaned.snapshots
    notices = []
    mom_full = (float(ev["momentum_score"].min()), float(ev["momentum_score"].max()))
    mom_active = f.momentum_range is not None and (
        f.momentum_range[0] > mom_full[0] + 1e-9 or f.momentum_range[1] < mom_full[1] - 1e-9)
    active = bool(f.artists or f.songs or f.album_types or f.explicit != "All" or f.min_reentries > 0 or mom_active)

    if f.date_range and len(f.date_range) == 2:
        d0, d1 = pd.Timestamp(f.date_range[0]), pd.Timestamp(f.date_range[1])
        rows = rows[(rows["date"] >= d0) & (rows["date"] <= d1)]
        ev = ev[(ev["event_date"] >= d0) & (ev["event_date"] <= d1)]
        snaps = snaps[(snaps["date"] >= d0) & (snaps["date"] <= d1)]
        if d0 > result.cleaned.snapshots["date"].min() or d1 < result.cleaned.snapshots["date"].max():
            active = True
    allowed = set(songs["song_id"])
    if f.artists:
        allowed &= set(songs.loc[songs["artist"].isin(f.artists), "song_id"])
    if f.songs:
        allowed &= set(songs.loc[songs["song_label"].isin(f.songs), "song_id"])
    if f.album_types:
        allowed &= set(songs.loc[songs["album_type"].isin(f.album_types), "song_id"])
    if f.explicit != "All":
        allowed &= set(songs.loc[songs["explicit_label"] == f.explicit, "song_id"])
    rows = rows[rows["song_id"].isin(allowed)]
    ev = ev[ev["song_id"].isin(allowed)]

    re = ev[ev["is_reentry"]]
    if mom_active:
        lo, hi = f.momentum_range
        keep = re["momentum_score"].between(lo, hi)
        ev = ev[~ev["is_reentry"] | ev["reentry_id"].isin(re.loc[keep, "reentry_id"])]
        re = ev[ev["is_reentry"]]
    if f.min_reentries > 0:
        counts = re.groupby("song_id").size()
        ok = set(counts[counts >= f.min_reentries].index)
        ev = ev[ev["song_id"].isin(ok)]
        re = ev[ev["is_reentry"]]
    if mom_active or f.min_reentries > 0:
        keep_songs = set(re["song_id"])
        rows = rows[rows["song_id"].isin(keep_songs)]
        allowed = keep_songs
    s = songs[songs["song_id"].isin(allowed)].copy()
    a = result.artists[result.artists["artist"].isin(set(s["artist"]))].copy()
    # Selection-level statistics: recomputed from the filtered events/rows so every leaderboard,
    # recommendation and statistic describes exactly the selection. Fandom scores/bands stay
    # normalised on the FULL dataset (so they remain comparable) and are not recomputed.
    agg_cols = [c for c in aggregate_events(re.head(1), "song_id").columns] if len(re) else []
    for tbl, key, base in ((s, "song_id", "song_id"), (a, "artist", "artist")):
        stale = [c for c in agg_cols if c in tbl.columns]
        tbl.drop(columns=stale, inplace=True)
    if len(re):
        s = s.merge(aggregate_events(re, "song_id"), left_on="song_id", right_index=True, how="left")
        a = a.merge(aggregate_events(re, "artist"), left_on="artist", right_index=True, how="left")
    else:
        for c in agg_cols:
            s[c] = pd.NA
            a[c] = pd.NA
    for tbl in (s, a):
        tbl["reentry_count"] = tbl["reentry_count"].fillna(0).astype(int)
        tbl["confidence"] = tbl["reentry_count"].map(confidence_label)
    s["reentries_in_selection"] = s["reentry_count"]
    s["appearances_in_selection"] = s["song_id"].map(rows.groupby("song_id").size()).fillna(0).astype(int)
    a["songs_in_selection"] = a["artist"].map(rows.groupby("artist")["song_id"].nunique()).fillna(0).astype(int)
    a["appearances_in_selection"] = a["artist"].map(rows.groupby("artist").size()).fillna(0).astype(int)
    a["songs_with_reentries"] = a["artist"].map(s[s["reentry_count"] > 0].groupby("artist").size()).fillna(0).astype(int)
    a["reentries_per_song"] = a["reentry_count"] / a["songs_in_selection"].replace(0, pd.NA)
    for tbl in (s, a):
        for c in tbl.columns:
            if tbl[c].dtype == object and c.startswith(("mean_", "median_", "sharpness")):
                tbl[c] = pd.to_numeric(tbl[c], errors="coerce")
    if len(re) == 0:
        notices.append("No re-entry events match the current filters; re-entry visuals are empty. "
                       "Widen the date range, lower the minimum re-entries or reset filters.")
    if len(rows) == 0:
        notices.append("No playlist rows match the current filters.")
    return FilteredData(rows, ev, re.reset_index(drop=True), s, a, snaps, notices, active)
