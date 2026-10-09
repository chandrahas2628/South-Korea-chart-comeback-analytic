"""Loading, cleaning and snapshot resolution. Raw data is never modified in place."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import OFF_CHART_RANK, TOP_N, make_song_id, normalize_text
from .validation import (
    DataValidationError, album_type_normalise, check_required_columns, find_calendar_gaps,
    parse_bool, parse_dates, snapshot_summary, validate_rows,
)

UNKNOWN_ARTIST = "Unknown Artist"


@dataclass
class CleanResult:
    """Everything produced by :func:`clean_playlist`."""
    clean: pd.DataFrame
    rejected: pd.DataFrame
    snapshots: pd.DataFrame
    snapshot_resolution: pd.DataFrame
    calendar_gaps: pd.DataFrame
    gap_runs: pd.DataFrame
    quality: dict = field(default_factory=dict)
    date_format: str = ""
    removal_ledger: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_raw(source) -> pd.DataFrame:
    """Read the raw CSV (path or buffer). Raises DataValidationError with a clear message."""
    try:
        df = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise DataValidationError(f"Dataset file not found: {source}") from exc
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("The dataset file is empty.") from exc
    check_required_columns(df)
    return df


def _split_blocks(group: pd.DataFrame) -> list[pd.DataFrame]:
    """Split an over-full date into stacked snapshots using file order + rank resets."""
    g = group.sort_values("source_row")
    pos = g["position"].to_numpy()
    block = np.concatenate([[0], np.cumsum(pos[1:] <= pos[:-1])])
    return [g[block == b] for b in np.unique(block)]


def _rank_map(block: pd.DataFrame) -> dict:
    return dict(zip(block["song_id"], block["position"]))


def _distance(block: pd.DataFrame, neighbour: pd.DataFrame) -> float:
    """Mean |rank difference| over the song union (absent => off-chart rank)."""
    a, b = _rank_map(block), _rank_map(neighbour)
    songs = set(a) | set(b)
    return float(np.mean([abs(a.get(s, OFF_CHART_RANK) - b.get(s, OFF_CHART_RANK)) for s in songs]))


def resolve_multi_snapshots(rows: pd.DataFrame, top_n: int = TOP_N) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve dates holding more than ``top_n`` rows.

    Rows are split into the stacked snapshots they came from (file order, rank resets).
    The retained snapshot is the one most similar to the neighbouring clean dates
    (highest song-set Jaccard overlap, mean rank distance as tie-break). Discarded rows are logged.
    If the stacked structure cannot be recovered, the date is left for the generic
    song/date de-duplication step and is flagged as unresolved.
    """
    counts = rows.groupby("date").size()
    over = counts[counts > top_n].index
    if len(over) == 0:
        return rows, pd.DataFrame(columns=[
            "date", "rows_before", "n_blocks", "chosen_block", "distances", "rows_discarded", "method"])
    clean_dates = counts[counts == top_n].index
    log, drop_idx = [], []
    for d in over:
        blocks = _split_blocks(rows[rows["date"] == d])
        ok = len(blocks) >= 2 and all(len(b) <= top_n and b["position"].is_unique for b in blocks)
        if not ok:
            log.append({"date": d, "rows_before": int(counts[d]), "n_blocks": len(blocks), "chosen_block": None,
                        "distances": "", "rows_discarded": 0, "method": "unresolved: handled by song/date de-duplication"})
            continue
        prev_d = clean_dates[clean_dates < d].max() if (clean_dates < d).any() else None
        next_d = clean_dates[clean_dates > d].min() if (clean_dates > d).any() else None
        neighbours = [rows[rows["date"] == x] for x in (prev_d, next_d) if x is not None and not pd.isna(x)]
        if neighbours:
            dists = [float(np.mean([_distance(b, n) for n in neighbours])) for b in blocks]
            jacc = [float(np.mean([len(set(b["song_id"]) & set(n["song_id"])) / len(set(b["song_id"]) | set(n["song_id"]))
                                   for n in neighbours])) for b in blocks]
            chosen = int(sorted(range(len(blocks)), key=lambda i: (-round(jacc[i], 6), dists[i]))[0])
            method = "highest song-set overlap with neighbouring complete snapshots (rank distance as tie-break)"
        else:
            dists, jacc, chosen, method = [np.nan] * len(blocks), [np.nan] * len(blocks), 0, "no neighbour available; kept first block"
        for i, b in enumerate(blocks):
            if i != chosen:
                drop_idx.extend(b.index.tolist())
        log.append({"date": d, "rows_before": int(counts[d]), "n_blocks": len(blocks), "chosen_block": chosen,
                    "distances": ", ".join(f"jaccard={j:.3f}/rankdist={x:.2f}" for j, x in zip(jacc, dists)) if neighbours else "",
                    "rows_discarded": int(sum(len(b) for i, b in enumerate(blocks) if i != chosen)),
                    "method": method})
    return rows.drop(index=drop_idx), pd.DataFrame(log)


def _dedupe_song_date(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the best (lowest) position per song/date; return removed rows for audit."""
    ordered = rows.sort_values(["date", "song_id", "position", "source_row"])
    dup_mask = ordered.duplicated(["date", "song_id"], keep="first")
    removed = ordered[dup_mask].copy()
    removed["reject_reason"] = "duplicate_song_date"
    return ordered[~dup_mask].sort_values(["date", "position"]).reset_index(drop=True), removed


def clean_playlist(raw: pd.DataFrame, top_n: int = TOP_N) -> CleanResult:
    """Validate and clean the raw playlist; build the data-quality summary."""
    check_required_columns(raw)
    df = raw.copy()
    df["source_row"] = np.arange(len(df))
    df["date"], fmt = parse_dates(df["date"])
    for c in ("position", "popularity", "duration_ms", "total_tracks"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["song_clean"] = df["song"].map(normalize_text)
    df["artist_clean"] = df["artist"].map(normalize_text)
    df["artist_missing"] = df["artist_clean"].eq("")
    df["artist_clean"] = df["artist_clean"].where(~df["artist_missing"], UNKNOWN_ARTIST)
    df["is_explicit"] = parse_bool(df["is_explicit"])
    df["album_type"] = album_type_normalise(df["album_type"])

    reasons = validate_rows(df, top_n)
    rejected = df[reasons != ""].assign(reject_reason=reasons[reasons != ""])
    valid = df[reasons == ""].copy()
    valid["is_explicit"] = valid["is_explicit"].astype(bool)
    valid["position"] = valid["position"].astype(int)
    valid["song_id"] = [make_song_id(s, a) for s, a in zip(valid["song_clean"], valid["artist_clean"])]

    n_exact_dup_raw = int(raw.duplicated().sum())
    n_song_date_dup_raw = int(df.duplicated(["date", "song_clean", "artist_clean"]).sum())

    valid, resolution = resolve_multi_snapshots(valid, top_n)
    alt_discarded = int(resolution["rows_discarded"].sum()) if len(resolution) else 0
    valid, removed_dups = _dedupe_song_date(valid)

    # Canonical display names per song_id (most frequent cleaned variant).
    disp = (valid.groupby("song_id")[["song_clean", "artist_clean"]]
            .agg(lambda s: s.mode().iloc[0]))
    valid["song"] = valid["song_id"].map(disp["song_clean"])
    valid["artist"] = valid["song_id"].map(disp["artist_clean"])
    valid["song_label"] = valid["song"] + " — " + valid["artist"]
    valid["duration_minutes"] = valid["duration_ms"] / 60000.0
    valid["year_month"] = valid["date"].dt.to_period("M").astype(str)
    valid["weekday"] = valid["date"].dt.day_name()
    valid["is_single"] = valid["album_type"].eq("single")

    keep = ["date", "position", "song_id", "song", "artist", "song_label", "popularity", "duration_ms",
            "duration_minutes", "album_type", "total_tracks", "is_explicit", "is_single", "artist_missing",
            "album_cover_url", "year_month", "weekday", "source_row"]
    clean = valid[keep].sort_values(["date", "position"]).reset_index(drop=True)

    snaps = snapshot_summary(clean, top_n)
    raw_counts = df.groupby("date").size().rename("raw_rows")
    snaps = snaps.merge(raw_counts, left_on="date", right_index=True, how="left")
    snaps["multi_snapshot_resolved"] = snaps["date"].isin(resolution["date"]) if len(resolution) else False
    gaps_df, gap_runs = find_calendar_gaps(clean["date"])

    quality = build_quality_summary(
        raw=raw, df=df, clean=clean, rejected=rejected, snaps=snaps, gaps_df=gaps_df, gap_runs=gap_runs,
        resolution=resolution, removed_dups=removed_dups, n_exact_dup_raw=n_exact_dup_raw,
        n_song_date_dup_raw=n_song_date_dup_raw, alt_discarded=alt_discarded, fmt=fmt, top_n=top_n)
    ledger = build_removal_ledger(raw, df, clean, rejected, resolution, removed_dups, alt_discarded, n_exact_dup_raw)
    return CleanResult(clean, rejected, snaps, resolution, gaps_df, gap_runs, quality, fmt, ledger)


def build_removal_ledger(raw, df, clean, rejected, resolution, removed_dups, alt_discarded, n_exact_dup_raw) -> pd.DataFrame:
    """Row-level reconciliation: raw rows -> clean rows, with the reason for every removal."""
    stacked = set(resolution["date"]) if len(resolution) else set()
    exact_in_stacked = int(df[df.duplicated(subset=[c for c in raw.columns], keep=False)
                              & df["date"].isin(stacked)].shape[0]) if stacked else 0
    rows = [
        ("Raw records", len(raw), "Rows in the untouched raw CSV."),
        ("- Rejected by validation", -len(rejected), "Invalid date, rank, popularity, duration, total_tracks, explicit flag or empty title."),
        ("- Alternate stacked-snapshot rows set aside", -alt_discarded,
         "Dates holding two stacked 50-row snapshots: the snapshot least similar to neighbouring days is set aside "
         "(still present in the raw file)."),
        ("- Duplicate song/date rows removed", -len(removed_dups), "Repeated song on the same date after snapshot resolution; best rank kept."),
        ("= Clean records", len(clean), "Rows used by every analysis."),
    ]
    out = pd.DataFrame(rows, columns=["Step", "Rows", "Why"])
    out["Reconciles"] = ""
    out.loc[out.index[-1], "Reconciles"] = "yes" if len(raw) - len(rejected) - alt_discarded - len(removed_dups) == len(clean) else "NO - investigate"
    info = pd.DataFrame([
        ("Info: exact duplicate rows in raw", n_exact_dup_raw, f"{exact_in_stacked} of the rows involved sit on stacked-snapshot dates "
                                                              "(handled by snapshot resolution, not by blind deletion).", ""),
        ("Info: rows with missing artist (filled 'Unknown Artist')", int(clean["artist_missing"].sum()), "Kept, flagged.", ""),
    ], columns=out.columns)
    return pd.concat([out, info], ignore_index=True)


def build_quality_summary(*, raw, df, clean, rejected, snaps, gaps_df, gap_runs, resolution, removed_dups,
                          n_exact_dup_raw, n_song_date_dup_raw, alt_discarded, fmt, top_n) -> dict:
    """Assemble the data-quality metrics shown in the app and exported for the paper."""
    title_artist = clean.groupby("song")["artist"].nunique()
    raw_counts = df.groupby("date").size()
    return {
        "Total raw records": int(len(raw)),
        "Valid records (after cleaning)": int(len(clean)),
        "Rejected records (failed validation)": int(len(rejected)),
        "Exact duplicate rows (raw)": n_exact_dup_raw,
        "Duplicate song/date records (raw)": n_song_date_dup_raw,
        "Alternate-snapshot rows discarded": alt_discarded,
        "Duplicate song/date rows removed (post-resolution)": int(len(removed_dups)),
        "Date format detected": fmt,
        "Observed playlist dates": int(len(snaps)),
        "Valid playlist days (complete 50-entry snapshots)": int(snaps["is_valid_snapshot"].sum()),
        "Incomplete playlist days (<50 entries)": int((snaps["status"] == "incomplete").sum()),
        "Over-full playlist days after cleaning (>50)": int((snaps["status"] == "overfull").sum()),
        "Raw dates with >50 rows (stacked snapshots)": int((raw_counts > top_n).sum()),
        "Raw dates with <50 rows": int((raw_counts < top_n).sum()),
        "Snapshot dates resolved": int(len(resolution[resolution["chosen_block"].notna()])) if len(resolution) else 0,
        "First date": clean["date"].min().date().isoformat() if len(clean) else None,
        "Last date": clean["date"].max().date().isoformat() if len(clean) else None,
        "Calendar days in range": int((clean["date"].max() - clean["date"].min()).days + 1) if len(clean) else 0,
        "Calendar gaps (missing dates)": int(len(gaps_df)),
        "Calendar gap runs": int(len(gap_runs)),
        "Unique songs (song+artist)": int(clean["song_id"].nunique()),
        "Unique song titles": int(clean["song"].nunique()),
        "Titles shared by >1 artist": int((title_artist > 1).sum()),
        "Unique artists": int(clean["artist"].nunique()),
        "Rows with missing artist (filled)": int(clean["artist_missing"].sum()),
        "Album types present": ", ".join(f"{k}: {v}" for k, v in clean["album_type"].value_counts().items()),
    }


def save_processed(result: CleanResult, out_dir: str | Path) -> Path:
    """Write the cleaned dataset and audit tables to ``out_dir`` (raw file untouched)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.clean.to_csv(out / "clean_playlist.csv", index=False)
    result.snapshots.to_csv(out / "snapshot_validation.csv", index=False)
    result.rejected.drop(columns=["song_clean", "artist_clean"], errors="ignore").to_csv(out / "rejected_rows.csv", index=False)
    result.snapshot_resolution.to_csv(out / "snapshot_resolution.csv", index=False)
    result.calendar_gaps.to_csv(out / "calendar_gaps.csv", index=False)
    return out
