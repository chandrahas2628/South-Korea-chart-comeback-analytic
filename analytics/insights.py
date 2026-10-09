"""Automated insight generation. Every sentence is built from computed values."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .content_analysis import compare_groups, compare_rates, group_summary, numeric_associations
from .utils import OFF_CHART_RANK


def _ins(category: str, title: str, text: str) -> dict:
    return {"category": category, "title": title, "text": text}


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%d %b %Y")


def generate_insights(fd) -> list[dict]:
    """Return a list of {category, title, text} insights for the filtered data ``fd``."""
    out: list[dict] = []
    re, songs, artists = fd.reentries, fd.songs, fd.artists
    if len(re) == 0:
        return [_ins("Data", "No re-entry events in selection",
                     "No confirmed re-entries match the current filters, so no comeback insights can be generated.")]
    scored = re[re["momentum_score"].notna()]
    r = scored.loc[scored["momentum_score"].idxmax()]
    out.append(_ins("Momentum", "Highest comeback momentum",
                    f"{r['song_label']} re-entered on {_d(r['reentry_date'])} at rank {int(r['reentry_position'])} "
                    f"with the highest momentum score ({r['momentum_score']:.1f}/100), peaking at rank "
                    f"{int(r['peak_rank'])} and staying {int(r['retention_snapshots'])} snapshots."))
    s = songs.loc[songs["reentries_in_selection"].idxmax()]
    if s["reentries_in_selection"] > 0:
        out.append(_ins("Frequency", "Most frequent re-entries",
                        f"{s['song_label']} has the most re-entries in the selection ({int(s['reentries_in_selection'])}), "
                        f"returning after a mean gap of {s['mean_reentry_gap_days']:.1f} days."))
    r = re.loc[re["retention_days"].idxmax()]
    lb = " (still on chart when the data ends, so this is a lower bound)" if r["is_right_censored"] else ""
    out.append(_ins("Retention", "Longest post-comeback retention",
                    f"{r['song_label']} stayed on the Top 50 for {int(r['retention_days'])} calendar days "
                    f"({int(r['retention_snapshots'])} observed snapshots) after its {_d(r['reentry_date'])} re-entry{lb}."))
    r = re.loc[re["recovery_speed"].idxmax()]
    out.append(_ins("Recovery", "Fastest rank recovery",
                    f"{r['song_label']} recovered {r['rank_recovery_total']:.0f} rank places "
                    f"({int(r['baseline_position'])} -> {int(r['peak_rank'])}) in {int(r['days_to_peak']) + 1} day(s): "
                    f"{r['recovery_speed']:.1f} places/day."))
    sol = artists[artists["reentry_count"] >= 3].dropna(subset=["fandom_score"])
    if len(sol):
        a = sol.loc[sol["fandom_score"].idxmax()]
        out.append(_ins("Artist", "Strongest artist comeback behaviour",
                        f"Among artists with at least 3 re-entries, {a['artist']} has the highest fandom intensity proxy "
                        f"({a['fandom_score']:.0f}/100, {int(a['reentry_count'])} re-entries across "
                        f"{int(a['unique_songs'])} songs, mean momentum {a['mean_momentum']:.1f})."))
    for by, a_, b_, label in (("album_type", "single", "album", "Single vs album"),
                              ("explicit_label", "Explicit", "Clean", "Explicit vs clean")):
        if by in re.columns:
            c = compare_groups(re, by, a_, b_)
            if c["n_a"] >= 5 and c["n_b"] >= 5:
                names = {"single": "single", "album": "album-track"}
                lead, trail = (a_, b_) if c["median_a"] >= c["median_b"] else (b_, a_)
                lead, trail = names.get(lead, lead), names.get(trail, trail)
                med_l, med_t = max(c["median_a"], c["median_b"]), min(c["median_a"], c["median_b"])
                sig = "statistically distinguishable" if c["p_value"] < 0.05 else "not statistically distinguishable"
                out.append(_ins("Content", label,
                                f"Re-entries of {lead} songs have a higher median momentum ({med_l:.1f}) than {trail} songs "
                                f"({med_t:.1f}); n={c['n_a']} vs {c['n_b']}. The difference is {sig} "
                                f"(Mann-Whitney p={c['p_value']:.3f}, effect size {c['effect']}). Association only."))
            else:
                out.append(_ins("Content", label, f"Too few events (n={c['n_a']} / {c['n_b']}) for a reliable comparison."))
    for by, a_, b_, label in (("album_type", "single", "album", "single vs album"), ("explicit_label", "Explicit", "Clean", "explicit vs clean")):
        if by in songs.columns and fd.events is not None and len(songs) >= 20:
            g = group_summary(songs, fd.events, by).set_index(by)
            if a_ in g.index and b_ in g.index and g.loc[[a_, b_], "n_songs"].min() >= 5:
                r_ = compare_rates(songs, by, a_, b_)
                out.append(_ins("Content", f"Re-entry rate: {label}",
                                f"{a_} songs: {g.loc[a_, 'reentry_rate']:.0%} had a re-entry ({g.loc[a_, 'reentries_per_100_exposure_days']:.2f} per 100 exposure days, "
                                f"n={int(g.loc[a_, 'n_songs'])}); {b_} songs: {g.loc[b_, 'reentry_rate']:.0%} "
                                f"({g.loc[b_, 'reentries_per_100_exposure_days']:.2f} per 100 exposure days, n={int(g.loc[b_, 'n_songs'])}); "
                                f"Fisher exact p{'<0.001' if r_['p_value'] < 0.001 else '=' + format(r_['p_value'], '.3f')}. The share-of-songs rate does not adjust for how long songs stay on the "
                                f"chart (median appearances: {a_} {songs.loc[songs[by] == a_, 'appearance_count'].median():.0f}, "
                                f"{b_} {songs.loc[songs[by] == b_, 'appearance_count'].median():.0f}), "
                                "so read the exposure-adjusted figure as the fairer comparison. Association only."))
    assoc = numeric_associations(songs, re)
    d = assoc[(assoc["attribute"] == "Duration (min)") & (assoc["outcome"] == "momentum_score")]
    if len(d):
        row = d.iloc[0]
        direction = "higher" if row["spearman_rho"] > 0 else "lower"
        strength = "weak" if abs(row["spearman_rho"]) < 0.2 else "moderate" if abs(row["spearman_rho"]) < 0.5 else "strong"
        out.append(_ins("Content", "Duration relationship",
                        f"Longer songs are associated with {direction} momentum (Spearman rho={row['spearman_rho']:.2f}, "
                        f"p={row['p_value']:.3f}, n={int(row['n'])}); the association is {strength}."))
    q1, q3 = scored["momentum_score"].quantile([0.25, 0.75])
    hi = q3 + 1.5 * (q3 - q1)
    n_out = int((scored["momentum_score"] > hi).sum())
    lead_txt = (f"{n_out} re-entr{'y is' if n_out == 1 else 'ies are'} unusually strong (momentum above the IQR fence "
                f"of {hi:.1f}). ") if n_out else f"No re-entry exceeds the IQR outlier fence ({hi:.1f}); momentum is evenly spread. "
    out.append(_ins("Patterns", "Unusual momentum patterns",
                    f"{lead_txt}"
                    f"{(re['absent_snapshots'] >= 30).mean() * 100:.0f}% of re-entries followed an absence of 30+ observed snapshots, "
                    f"and {re['is_right_censored'].mean() * 100:.0f}% were still on chart when data ended."))
    return out
