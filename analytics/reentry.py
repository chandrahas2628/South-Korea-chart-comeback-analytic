"""Chart re-entry engine based on OBSERVED playlist snapshots (robust to calendar gaps).

Definitions
-----------
* valid snapshot  : an observed date with a complete Top-N list. Only valid snapshots can
                    prove a song was absent.
* observed absence: a valid snapshot in which the song is not listed.
* calendar gap    : a calendar date with no snapshot at all. It is NEVER evidence of absence.
* stint           : a run of appearances not separated by >= ``min_absent_snapshots``
                    consecutive observed absences.
* re-entry        : a stint that starts after a previous stint (stint_no >= 1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import OFF_CHART_RANK


@dataclass
class ReentryResult:
    stints: pd.DataFrame          # one row per stint (initial / first_entry / re_entry)
    events: pd.DataFrame          # re-entry events only
    song_summary: pd.DataFrame    # per-song history summary
    appearances: pd.DataFrame     # clean rows with stint_no assigned
    min_absent_snapshots: int


def _stint_type(stint_no: int, left_censored: bool) -> str:
    if stint_no > 0:
        return "re_entry"
    return "initial_observed" if left_censored else "first_entry"


def compute_reentries(clean: pd.DataFrame, snapshots: pd.DataFrame, min_absent_snapshots: int = 1) -> ReentryResult:
    """Detect stints and confirmed re-entries for every song.

    A gap between two consecutive appearances (dates a < b) counts as an exit only if at
    least ``min_absent_snapshots`` VALID snapshots lie strictly between them. Dates
    without any snapshot between a and b are reported (``unobserved_calendar_days``) but
    never create a re-entry on their own.
    """
    if min_absent_snapshots < 1:
        raise ValueError("min_absent_snapshots must be >= 1")
    valid_dates = np.sort(snapshots.loc[snapshots["is_valid_snapshot"], "date"].to_numpy(dtype="datetime64[ns]"))
    obs_dates = np.sort(snapshots["date"].to_numpy(dtype="datetime64[ns]"))
    first_valid = valid_dates[0] if len(valid_dates) else None
    last_obs = obs_dates[-1] if len(obs_dates) else None

    rows = clean.sort_values(["song_id", "date"]).reset_index(drop=True)
    stint_no_col = np.zeros(len(rows), dtype=int)
    stint_rows = []

    for song_id, idx in rows.groupby("song_id", sort=False).indices.items():
        g = rows.iloc[idx]
        dates = g["date"].to_numpy(dtype="datetime64[ns]")
        pos = g["position"].to_numpy()
        pop = g["popularity"].to_numpy()
        absent = np.zeros(len(dates), dtype=int)
        unobs = np.zeros(len(dates), dtype=int)
        if len(dates) > 1:
            absent[1:] = (np.searchsorted(valid_dates, dates[1:], side="left")
                          - np.searchsorted(valid_dates, dates[:-1], side="right"))
            between_obs = (np.searchsorted(obs_dates, dates[1:], side="left")
                           - np.searchsorted(obs_dates, dates[:-1], side="right"))
            span = (dates[1:] - dates[:-1]).astype("timedelta64[D]").astype(int) - 1
            unobs[1:] = span - between_obs
        new_stint = absent >= min_absent_snapshots
        new_stint[0] = False
        sno = np.cumsum(new_stint)
        stint_no_col[idx] = sno
        cum_app = 0
        prev = None
        for s in range(sno.max() + 1):
            m = np.where(sno == s)[0]
            start_i, end_i = m[0], m[-1]
            left_cens = bool(s == 0 and first_valid is not None and dates[start_i] <= first_valid)
            rec = {
                "song_id": song_id, "stint_no": s, "stint_type": _stint_type(s, left_cens),
                "start_date": pd.Timestamp(dates[start_i]), "end_date": pd.Timestamp(dates[end_i]),
                "entry_position": int(pos[start_i]), "exit_position": int(pos[end_i]),
                "entry_popularity": float(pop[start_i]), "stint_appearances": int(len(m)),
                "previous_appearance_count": int(cum_app),
                "is_left_censored": left_cens,
                "is_right_censored": bool(last_obs is not None and dates[end_i] == last_obs),
            }
            if prev is not None:
                rec.update({
                    "previous_exit_date": prev["end_date"], "previous_exit_position": prev["exit_position"],
                    "previous_exit_popularity": prev["exit_popularity"],
                    "gap_days": int((dates[start_i] - np.datetime64(prev["end_date"])).astype("timedelta64[D]").astype(int)),
                    "absent_snapshots": int(absent[start_i]),
                    "unobserved_calendar_days": int(unobs[start_i]),
                })
            rec["exit_popularity"] = float(pop[end_i])
            stint_rows.append(rec)
            prev = rec
            cum_app += len(m)

    rows["stint_no"] = stint_no_col
    stints = pd.DataFrame(stint_rows)
    for col in ["previous_exit_date", "previous_exit_position", "previous_exit_popularity", "gap_days",
                "absent_snapshots", "unobserved_calendar_days"]:
        if col not in stints.columns:
            stints[col] = np.nan
    meta = (rows.sort_values("date").groupby("song_id")
            .agg(song=("song", "first"), artist=("artist", "first"), song_label=("song_label", "first")))
    stints = stints.merge(meta, left_on="song_id", right_index=True, how="left")
    # Attributes as observed on the stint's first (entry) row.
    entry_rows = rows.merge(stints[["song_id", "stint_no", "start_date"]],
                            left_on=["song_id", "stint_no", "date"], right_on=["song_id", "stint_no", "start_date"])
    attrs = entry_rows[["song_id", "stint_no", "album_type", "is_explicit", "duration_minutes", "total_tracks"]]
    stints = stints.merge(attrs, on=["song_id", "stint_no"], how="left")
    stints["popularity"] = stints["entry_popularity"]
    stints["reentry_id"] = stints["song_id"] + "-R" + stints["stint_no"].astype(str)
    stints = stints.sort_values(["start_date", "song_id"]).reset_index(drop=True)

    events = stints[stints["stint_no"] > 0].copy()
    events = events.rename(columns={"start_date": "reentry_date", "entry_position": "reentry_position"})
    events["start_date"] = events["reentry_date"]
    events["entry_position"] = events["reentry_position"]
    events = events.reset_index(drop=True)

    summary = song_summary(rows, stints)
    return ReentryResult(stints=stints, events=events, song_summary=summary,
                         appearances=rows.sort_values(["date", "position"]).reset_index(drop=True),
                         min_absent_snapshots=min_absent_snapshots)


def song_summary(rows: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Per-song history: first/last appearance, appearance count, re-entries, gaps."""
    g = rows.groupby("song_id")
    out = pd.DataFrame({
        "first_observed": g["date"].min(), "last_observed": g["date"].max(),
        "appearance_count": g.size(),
        "best_position": g["position"].min(), "mean_position": g["position"].mean(),
        "peak_popularity": g["popularity"].max(),
    })
    st = stints.groupby("song_id")
    out["n_stints"] = st.size()
    out["reentry_count"] = out["n_stints"] - 1
    out["mean_gap_days"] = stints[stints["stint_no"] > 0].groupby("song_id")["gap_days"].mean()
    out["max_gap_days"] = stints[stints["stint_no"] > 0].groupby("song_id")["gap_days"].max()
    out["reentry_count"] = out["reentry_count"].astype(int)
    return out.reset_index()
