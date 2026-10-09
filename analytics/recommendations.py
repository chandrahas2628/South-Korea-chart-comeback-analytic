"""Rule-based strategic recommendations generated purely from computed (selection-level) metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .content_analysis import compare_groups

MIN_SONG_EVENTS = 2
MIN_ARTIST_EVENTS = 3
RULES = ["HIGH MOMENTUM + HIGH RETENTION", "HIGH MOMENTUM + LOW RETENTION", "MULTIPLE RE-ENTRIES",
         "HIGH ALBUM-TRACK MOMENTUM", "HIGH ARTIST FANDOM PROXY", "LOW MOMENTUM + HIGH RE-ENTRY FREQUENCY"]


def _evidence_song(r) -> str:
    cens = " (lower bound)" if r.get("censored_events", 0) else ""
    fan = "n/a" if pd.isna(r.get("fandom_score")) else f"{r['fandom_score']:.0f}"
    return (f"{int(r['reentry_count'])} re-entries | mean momentum {r['mean_momentum']:.1f} | "
            f"mean retention {r['mean_retention_days']:.1f} days{cens} ({r['mean_retention']:.1f} snapshots) | fandom proxy {fan}")


def _evidence_artist(r) -> str:
    return (f"{int(r['reentry_count'])} re-entries across {int(r['songs_with_reentries'])} of {int(r['unique_songs'])} songs | "
            f"mean momentum {r['mean_momentum']:.1f} | mean retention {r['mean_retention_days']:.1f} days | "
            f"fandom proxy {r['fandom_score']:.0f} | confidence {r['confidence']}")


def build_recommendations(fd, top_n: int = 8) -> dict:
    """Return {'table', 'thresholds', 'notes'} from the filtered data ``fd``.

    All thresholds are quantiles of the *selected* songs/artists; retention is measured in
    calendar days (retention_days). Song rules require >= MIN_SONG_EVENTS re-entries and
    artist rules >= MIN_ARTIST_EVENTS to avoid acting on one-off events.
    """
    cols = ["priority", "rule", "level", "entity", "artist", "evidence", "recommended_action", "n_events"]
    songs = fd.songs[fd.songs["reentry_count"] >= MIN_SONG_EVENTS].copy()
    rows, notes, thr = [], [], {}
    if len(songs) < 5:
        return {"table": pd.DataFrame(columns=cols), "thresholds": thr,
                "notes": [f"Fewer than 5 songs with >= {MIN_SONG_EVENTS} re-entries in the selection - "
                          "song-level rules need more data. Reset or widen the filters."]}
    mom_hi, mom_med = songs["mean_momentum"].quantile(0.75), songs["mean_momentum"].median()
    ret_hi, ret_lo = songs["mean_retention_days"].quantile(0.75), songs["mean_retention_days"].quantile(0.40)
    cnt_hi, cnt_q75 = max(3, songs["reentry_count"].quantile(0.90)), max(3, songs["reentry_count"].quantile(0.75))
    thr.update({"high momentum (P75 of mean momentum)": mom_hi, "median mean-momentum": mom_med,
                "high retention (P75, days)": ret_hi, "low retention (P40, days)": ret_lo,
                "multiple re-entries (>= max(3, P90))": cnt_hi, "frequent re-entries (>= max(3, P75))": cnt_q75})

    def add(df, rule, action, sort_col, level="Song", key="song_label"):
        for _, r in df.sort_values(sort_col, ascending=False).head(top_n).iterrows():
            rows.append({"priority": float(r.get("fandom_score", np.nan)), "rule": rule, "level": level, "entity": r[key],
                         "artist": r["artist"], "n_events": int(r["reentry_count"]),
                         "evidence": _evidence_song(r) if level == "Song" else _evidence_artist(r),
                         "recommended_action": action})

    add(songs[(songs["mean_momentum"] >= mom_hi) & (songs["mean_retention_days"] >= ret_hi)], RULES[0],
        "Candidate for extended promotional investment (strong and sustained comebacks).", "mean_momentum")
    add(songs[(songs["mean_momentum"] >= mom_hi) & (songs["mean_retention_days"] <= ret_lo)], RULES[1],
        "Strong short-term activation opportunity (time-boxed push around the spike).", "mean_momentum")
    add(songs[songs["reentry_count"] >= cnt_hi], RULES[2], "Strong catalogue-reactivation candidate.", "reentry_count")
    add(songs[(songs["reentry_count"] >= cnt_q75) & (songs["mean_momentum"] <= mom_med)], RULES[5],
        "Investigate recurring event-driven spikes (external events are not observed in this data).", "reentry_count")

    arts = fd.artists[(fd.artists["reentry_count"] >= MIN_ARTIST_EVENTS)].dropna(subset=["fandom_score"])
    if len(arts) >= 5:
        a_hi = arts["fandom_score"].quantile(0.80)
        thr["artist fandom proxy high (P80, >=3 events)"] = a_hi
        add(arts[arts["fandom_score"] >= a_hi], RULES[4], "Prioritise targeted fan-engagement campaigns.",
            "fandom_score", level="Artist", key="artist")
    else:
        notes.append("Artist fandom rule skipped: fewer than 5 artists with >= 3 re-entries in the selection.")

    c = compare_groups(fd.reentries, "album_type", "album", "single") if "album_type" in fd.reentries.columns else None
    if c and c["n_a"] >= 5 and c["n_b"] >= 5:
        sig = "statistically distinguishable" if c["p_value"] < 0.05 else "not statistically distinguishable"
        if c["median_a"] > c["median_b"]:
            tracks = songs[(songs["album_type"] != "single") & (songs["mean_momentum"] >= mom_hi)]
            notes.append(f"Album tracks show a higher median re-entry momentum ({c['median_a']:.1f}, n={c['n_a']}) than singles "
                         f"({c['median_b']:.1f}, n={c['n_b']}); difference {sig} (p={c['p_value']:.3f}, effect {c['effect']}). "
                         "Investigate non-single promotional opportunities" + (" - directional only." if c["p_value"] >= 0.05 else "."))
            add(tracks, RULES[3], "Investigate non-single promotional opportunities.", "mean_momentum")
        else:
            notes.append(f"Album tracks do NOT show higher median re-entry momentum ({c['median_a']:.1f}, n={c['n_a']}) than singles "
                         f"({c['median_b']:.1f}, n={c['n_b']}); the album-track rule is not triggered.")
    else:
        notes.append("Album-track rule not evaluated: fewer than 5 re-entry events in album or single group.")
    table = pd.DataFrame(rows, columns=cols)
    if len(table):
        table = table.drop_duplicates(["rule", "entity"]).reset_index(drop=True)
    return {"table": table, "thresholds": thr, "notes": notes}
