"""Methodology, glossary and limitation text shared by the app, README export and research summary."""
from .utils import FANDOM_WEIGHTS, MOMENTUM_WEIGHTS, SHRINKAGE_K_ARTIST, SHRINKAGE_K_SONG


def _w(d):
    return ", ".join(f"{k.replace('_', ' ')} {v:.0%}" for k, v in d.items())


METRIC_GLOSSARY = {
    "momentum_score": "Momentum Intensity (0-100): weighted mean of percentile ranks of rank jump, peak improvement, popularity lift, recovery speed and retention, ranked within re-entries. 50 = typical re-entry. An analytical composite, not a measure of fandom.",
    "rank_jump": "Previous exit rank minus re-entry rank (positive = re-entered higher than it left; first entries use off-chart rank 51).",
    "peak_rank": "Best (lowest-number) rank reached during the comeback stint.",
    "peak_improvement": "Re-entry rank minus peak rank: places gained after re-entering.",
    "rank_recovery": "Previous exit rank minus peak rank: total places regained versus where the song left the chart.",
    "recovery_speed": "Rank recovery divided by (days to peak + 1): places regained per day.",
    "popularity_lift": "Peak popularity during the stint minus popularity at the previous exit (platform popularity score, 0-100).",
    "days_to_peak": "Calendar days from re-entry to the first day the best rank was reached.",
    "retention_days": "Calendar days from the comeback date to the last observed appearance of the stint, inclusive. Missing calendar dates inside the span are not treated as observed. Lower bound if right-censored.",
    "retention_snapshots": "Observed playlist snapshots in which the song is listed during the stint (comeback day included).",
    "decay_speed": "Rank places lost per day between the peak and the last appearance.",
    "volatility": "Standard deviation of day-to-day rank changes within the stint (needs >= 3 snapshots).",
    "fandom_score": "Fandom Intensity Proxy (0-100): behavioural chart-based composite, sample-size shrunk. NOT fandom size, fan count or fan strength.",
    "gap_days": "Calendar days between the last appearance before exit and the re-entry date.",
    "absent_snapshots": "Valid (complete) snapshots in which the song was observed to be absent before returning.",
}

DEFINITIONS = [
    ("Observed snapshot", "A playlist date present in the dataset. Valid if it lists exactly 50 unique songs."),
    ("Calendar day", "Any date between the first and last snapshot, observed or not."),
    ("Calendar gap", "A calendar day with no snapshot. It is evidence of nothing: never an exit, never presence."),
    ("Observed absence", "A valid snapshot in which the song is not listed. Only observed absences can prove an exit."),
    ("Confirmed re-entry", "A reappearance after at least N consecutive observed absences (N = sidebar threshold, default 3)."),
    ("Right-censoring", "The stint is still on the chart at the final snapshot; its retention is a lower bound."),
    ("Left-censoring", "The song is already on the chart at the first valid snapshot; its true entry is unobserved and it is not scored."),
]


def methodology_sections(min_absent: int) -> dict:
    return {
        "1. Dataset": (
            "Daily South Korea Top 50 playlist snapshots (`data/raw/Atlantic_South_Korea.csv`): date, position, song, artist, "
            "popularity, duration_ms, album_type, total_tracks, is_explicit, album_cover_url. The raw file is read-only; every "
            "derived table is generated separately into `data/processed/`. The unit of analysis is a *song* (normalised "
            "song + artist), and each continuous run on the chart is a *stint*."),
        "2. Data validation": (
            "Dates are parsed with an explicit detected format (here dd-mm-yyyy) and never silently day/month-swapped. Text is "
            "Unicode-normalised and whitespace-collapsed; empty artists become 'Unknown Artist' (flagged). Rows are rejected "
            "and logged if date, rank (integer 1-50), popularity (0-100), duration (>0), total_tracks (>=1), explicit flag or "
            "song title is invalid. Album types outside single/album/compilation are kept as 'other'. `song_id` is a hash of "
            "the case-folded normalised **song + artist** so titles shared by different artists stay separate. Duration is "
            "converted to minutes."),
        "3. Snapshot resolution": (
            "Each observed date is complete (50 unique entries), incomplete (<50) or over-full (>50). A date with more than 50 "
            "rows is treated as *stacked snapshots*: rows are split where rank resets in file order and the snapshot with the "
            "highest song-set overlap with the neighbouring complete days is kept (rank distance as tie-break); the alternative "
            "is set aside, logged in the resolution table and never deleted from the raw file. Remaining duplicate song/date "
            "pairs keep their best rank. Only complete snapshots are *valid snapshots* and can prove absence. Calendar gaps "
            "are listed and never invented."),
        "4. Re-entry definition": (
            f"For each song, consecutive appearances are compared. If at least **{min_absent} valid snapshot(s)** lie strictly "
            "between two appearances (observed absence) the later appearance starts a new *stint*: "
            "appears → leaves a valid snapshot → stays absent for the minimum number of valid observed snapshots → returns. "
            "Stint 0 is the first appearance; every later stint is a confirmed **re-entry**. Calendar dates with no snapshot "
            "are counted separately (`unobserved_calendar_days`) and can never create a re-entry. The threshold is adjustable "
            "in the sidebar: 1 counts every one-snapshot dropout (mostly rank 49-50 flicker); larger values keep only longer "
            "exits. Short dropouts shorter than the threshold stay inside the stint and are counted as `dropout_snapshots`."),
        "5. Momentum metrics": (
            "Baseline rank = last rank before the exit (51 'off-chart' for first entries). **Rank jump** = baseline - entry "
            "rank. **Peak rank** = best rank in the stint; **peak improvement** = entry rank - peak rank. **Popularity lift** = "
            "peak popularity in the stint - popularity at the previous exit (undefined for first entries; "
            "`popularity_change_at_entry` uses the entry-day value). **Days to peak** = calendar days from entry to the first best "
            "rank; **snapshots to peak** counts snapshots. **Recovery speed** = (baseline - peak rank) / (days to peak + 1), so "
            "same-day recovery is defined. **Decay speed** = (exit rank - peak rank) / days from peak to exit (undefined if the "
            "peak is the last day). **Volatility** = standard deviation of day-to-day rank changes (needs >= 3 snapshots). "
            "Negative rank jumps and zero peak improvement are valid outcomes and are kept. Infinite values are converted to missing."),
        "6. Momentum score": (
            f"Each component (rank jump, peak improvement, popularity lift, recovery speed, retention snapshots) is converted to "
            f"a percentile rank **within its stint type** (re-entries are ranked against re-entries; first entries are ranked "
            f"separately because their baseline differs). Percentiles remove unit differences and outlier dominance. The score "
            f"(0-100) is the weighted mean: {_w(MOMENTUM_WEIGHTS)}. Weights are judgement-based: retention is slightly higher "
            "because it captures sustainability, the rest share the remainder. A component undefined for an event (e.g. "
            "popularity lift for a first entry) is dropped and the remaining weights re-normalised, so the event is not penalised. "
            "Because scores are percentile-based, 50 is the typical re-entry and the dataset-wide mean is ~50 by construction. "
            "Robustness: the Methodology page reports the rank correlation with equal weights and with 200 random weight vectors "
            "drawn around the documented weights. The score is an analytical composite, not an objective measure of fandom."),
        "7. Retention": (
            "**Retention snapshots** = observed snapshots in which the song is listed during the stint. **Retention days** = "
            "calendar days from the comeback date to the last observed appearance, inclusive. Calendar gaps inside that span are "
            "not assumed to be presence (`unobserved_days_in_stint`), and the first observed absence after the stint "
            "(`exit_observed_date`) is recorded with its uncertainty (`exit_uncertainty_days`). Right-censored stints are "
            "flagged: their retention is a lower bound, not a complete lifetime."),
        "8. Fandom proxy": (
            f"Entities (song or artist) with at least one re-entry are scored on re-entry count, mean momentum, popularity "
            f"lift, rank recovery, recovery speed, retention and comeback sharpness (share of re-entries landing in ranks "
            f"1-25). Components are percentile-ranked and weighted: {_w(FANDOM_WEIGHTS)}. The composite is shrunk toward the "
            f"median entity by sample size, (n*score + k*median)/(n+k) with k={SHRINKAGE_K_SONG:g} (songs) / "
            f"{SHRINKAGE_K_ARTIST:g} (artists), so one lucky comeback cannot outrank a long record, then min-max scaled to "
            "0-100. Bands (Low/Moderate/High/Very High/Extreme) use the 30/60/80/95th percentiles. Sample size and a "
            "confidence label (Low n<3, Medium 3-9, High 10+) accompany every score. It is a **behavioural proxy from chart "
            "data, not fandom size, a fan count or actual fandom strength**."),
        "9. Content analysis": (
            "Song-level attributes use each song's most common value. Re-entry rate = share of songs with >= 1 re-entry; "
            "because exposure windows differ, re-entries per 100 exposure days are also shown. Groups with fewer than 10 songs "
            "or 10 events are flagged as small samples. Findings are associations in observed data."),
        "10. Statistical tests": (
            "Two-group comparisons of momentum/retention: Mann-Whitney U (two-sided, non-parametric) with Cliff's delta effect "
            "size (|d|<0.147 negligible, <0.33 small, <0.474 medium, else large). Re-entry rates: Fisher exact test. "
            "Multi-group comparisons (album size, duration quartiles): Kruskal-Wallis. Numeric relationships: Spearman rank "
            "correlation. Events from the same song are not independent, so p-values are indicative; tests are skipped when a "
            "group has fewer than 5 observations. Language is 'associated with', never 'causes'."),
        "11. Recommendation rules": (
            "Rules use quantiles of the *selected* songs/artists (song rules need >= 2 re-entries, artist rules >= 3): "
            "high momentum + high retention (mean momentum >= P75 and mean retention days >= P75); high momentum + low retention "
            "(>= P75 and <= P40); multiple re-entries (>= max(3, P90)); low momentum + high frequency (count >= max(3, P75) and "
            "mean momentum <= median); high artist fandom proxy (>= P80 among artists with >= 3 events); high album-track "
            "momentum (only when album tracks' median momentum exceeds singles', reported with its significance). "
            "Each recommendation lists its evidence."),
        "12. ML methodology": (
            "Experimental model of P(an exited song re-enters within 30 days). Samples are (weekly cutoff date, song with a "
            "confirmed exit at that date). Features use only data on or before the cutoff; the label looks only at the next 30 days. "
            "Train and test are split chronologically with an embargo equal to the horizon. Metrics: ROC-AUC with a song-clustered "
            "bootstrap interval, PR-AUC against the base rate, Brier score, top-decile lift, plus a trivial recency baseline. "
            "A model is labelled 'reliable' only if the AUC lower bound > 0.60 and PR-AUC > 1.2x base rate; even then it is a "
            "descriptive risk ranking. The core dashboard does not depend on it."),
        "13. Limitations": "See the limitations list below.",
    }


LIMITATIONS = [
    "The data are observed daily Top-50 playlist snapshots, i.e. observed chart behaviour only.",
    "The Fandom Intensity Proxy is a behavioural proxy; playlist position is not a direct measure of total fandom size.",
    "Correlation does not establish causation; content findings are associations.",
    "Missing calendar dates can affect time-based interpretation; they are never treated as exits or as presence.",
    "Popularity scores are platform/API-derived and may be updated with a lag or use opaque methodology.",
    "External causes (releases, tours, TV, promotions, social trends) are not observed in this dataset.",
    "A Top-50 cut-off censors information: songs just below rank 50 are invisible, and exits near rank 50 can be rank noise (hence the adjustable minimum-absence threshold).",
    "Songs on the chart at the start (left-censored) or still on it at the end (right-censored) have incomplete histories; retention for the latter is a lower bound.",
    "Momentum scores are relative (percentile-based within re-entries); 50 means a typical re-entry, not an absolute level.",
    "Events from the same song/artist are not independent; statistical p-values are indicative.",
    "One date (01-03-2025) held two stacked snapshots; the retained one was chosen by similarity to neighbouring days and the two were near-equal, so a few presence flags that day are uncertain.",
    "The re-entry forecast is experimental/descriptive and should not be read as a dependable prediction.",
]
