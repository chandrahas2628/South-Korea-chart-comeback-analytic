"""End-to-end pipeline: raw CSV -> clean data -> re-entries -> momentum -> fandom -> content."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .content_analysis import add_content_bins, attach_song_attributes
from .fandom import compute_artist_fandom, compute_song_fandom
from .momentum import compute_momentum, weight_sensitivity
from .preprocessing import CleanResult, clean_playlist, load_raw, save_processed
from .reentry import ReentryResult, compute_reentries

DEFAULT_MIN_ABSENT = 3


@dataclass
class PipelineResult:
    cleaned: CleanResult
    reentry: ReentryResult
    events: pd.DataFrame            # all stints with momentum metrics (is_reentry flag)
    songs: pd.DataFrame
    artists: pd.DataFrame
    pattern_thresholds: dict
    song_band_cuts: dict
    artist_band_cuts: dict
    sensitivity: dict
    min_absent_snapshots: int

    @property
    def reentries(self) -> pd.DataFrame:
        return self.events[self.events["is_reentry"]].reset_index(drop=True)

    @property
    def appearances(self) -> pd.DataFrame:
        return self.reentry.appearances


def run_pipeline(source, min_absent_snapshots: int = DEFAULT_MIN_ABSENT) -> PipelineResult:
    """Run the full analytical pipeline. ``source`` is a CSV path/buffer or a raw DataFrame."""
    raw = source if isinstance(source, pd.DataFrame) else load_raw(source)
    cleaned = clean_playlist(raw)
    reentry = compute_reentries(cleaned.clean, cleaned.snapshots, min_absent_snapshots)
    events, thr = compute_momentum(reentry.stints, reentry.appearances, snapshots=cleaned.snapshots)
    songs, song_cuts = compute_song_fandom(events, reentry.song_summary, reentry.appearances, cleaned.snapshots)
    songs = add_content_bins(songs)
    artists, artist_cuts = compute_artist_fandom(events, songs, reentry.appearances)
    events = attach_song_attributes(events, songs)
    return PipelineResult(cleaned, reentry, events, songs, artists, thr, song_cuts, artist_cuts,
                          weight_sensitivity(events), min_absent_snapshots)


def export_processed(result: PipelineResult, out_dir: str | Path) -> Path:
    """Write all processed datasets to disk (used by run_pipeline.py CLI)."""
    out = save_processed(result.cleaned, out_dir)
    result.reentries.to_csv(out / "reentry_events.csv", index=False)
    result.events.to_csv(out / "momentum_events.csv", index=False)
    result.songs.to_csv(out / "song_analytics.csv", index=False)
    result.artists.to_csv(out / "artist_analytics.csv", index=False)
    return out
