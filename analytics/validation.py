"""Schema, row-level and snapshot-level validation. Nothing here mutates the raw data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import ALLOWED_ALBUM_TYPES, REQUIRED_COLUMNS, TOP_N

DATE_FORMATS = ["%d-%m-%Y", "%Y-%m-%d", "%m-%d-%Y", "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y"]
_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}


class DataValidationError(Exception):
    """Raised when the dataset cannot be used (e.g. missing required columns)."""


def check_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Dataset is missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}."
        )


def parse_dates(series: pd.Series) -> tuple[pd.Series, str]:
    """Parse dates with an *explicit* format (never silently day/month-swapped).

    Every candidate format is tried; the one parsing the most rows wins. Ties (ambiguous
    day/month values) are broken by chronological smoothness of the file order, because
    the playlist file is written in time order.
    """
    text = series.astype(str).str.strip()
    best = None
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        ok = int(parsed.notna().sum())
        order = float(parsed.dropna().diff().abs().dt.days.sum()) if ok else float("inf")
        key = (-ok, order)
        if best is None or key < best[0]:
            best = (key, parsed, fmt)
    if best is None or -best[0][0] == 0:
        raise DataValidationError("No date could be parsed from the 'date' column.")
    return best[1], best[2]


def parse_bool(series: pd.Series) -> pd.Series:
    """Map explicit flags to True/False; unrecognised values become <NA>."""
    t = series.astype(str).str.strip().str.lower()
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out[t.isin(_TRUE)] = True
    out[t.isin(_FALSE)] = False
    return out


def validate_rows(df: pd.DataFrame, top_n: int = TOP_N) -> pd.Series:
    """Return a per-row rejection reason ('' when the row is valid).

    Expects columns: date (datetime), position, popularity, duration_ms, total_tracks
    (numeric), song_clean (str), is_explicit (boolean).
    """
    reasons = pd.DataFrame(index=df.index)
    reasons["invalid_date"] = df["date"].isna()
    reasons["invalid_position"] = ~df["position"].between(1, top_n) | (df["position"] % 1 != 0)
    reasons["invalid_popularity"] = ~df["popularity"].between(0, 100)
    reasons["invalid_duration"] = ~(df["duration_ms"] > 0)
    reasons["invalid_total_tracks"] = ~(df["total_tracks"] >= 1)
    reasons["missing_song"] = df["song_clean"].eq("")
    reasons["invalid_explicit"] = df["is_explicit"].isna()
    arr = reasons.to_numpy()
    names = np.array(reasons.columns)
    return pd.Series([";".join(names[row]) for row in arr], index=df.index)


def snapshot_summary(rows: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """One row per observed date: entry count, duplicate counts and completeness status."""
    g = rows.groupby("date")
    out = pd.DataFrame({
        "n_rows": g.size(),
        "n_unique_songs": g["song_id"].nunique(),
        "n_unique_positions": g["position"].nunique(),
    })
    out["n_duplicate_song_rows"] = out["n_rows"] - out["n_unique_songs"]
    out["status"] = np.select(
        [out["n_rows"] == top_n, out["n_rows"] < top_n],
        ["complete", "incomplete"], default="overfull")
    out["is_valid_snapshot"] = (out["n_rows"] == top_n) & (out["n_unique_songs"] == top_n)
    return out.reset_index()


def find_calendar_gaps(dates) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (missing calendar dates, gap runs) between min and max observed date."""
    d = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).dropna().unique())).sort_values()
    runs_cols = ["gap_start", "gap_end", "days_missing"]
    if len(d) == 0:
        return pd.DataFrame({"date": pd.DatetimeIndex([])}), pd.DataFrame(columns=runs_cols)
    full = pd.date_range(d.min(), d.max(), freq="D")
    missing = full.difference(d)
    missing_df = pd.DataFrame({"date": missing})
    if len(missing) == 0:
        return missing_df, pd.DataFrame(columns=runs_cols)
    run_id = (pd.Series(missing).diff().dt.days != 1).cumsum().to_numpy()
    runs = (pd.DataFrame({"date": missing, "run": run_id}).groupby("run")["date"]
            .agg(gap_start="min", gap_end="max", days_missing="count").reset_index(drop=True))
    return missing_df, runs


def album_type_normalise(series: pd.Series) -> pd.Series:
    """Lower-case album types; unknown values become 'other' (kept, but visible)."""
    t = series.astype(str).str.strip().str.lower()
    return t.where(t.isin(ALLOWED_ALBUM_TYPES), "other")
