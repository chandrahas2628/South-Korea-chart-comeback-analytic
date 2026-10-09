"""Natural-language descriptions generated from computed event metrics (never hard-coded)."""
from __future__ import annotations

import pandas as pd


def describe_event(ev: pd.Series, context: dict | None = None) -> str:
    """Interpret one stint/comeback from its metrics. ``context`` may carry population medians."""
    kind = {"re_entry": "re-entered", "first_entry": "entered", "initial_observed": "was already on the chart when observation began, at"}
    verb = kind.get(ev["stint_type"], "entered")
    parts = [f"{ev['song_label']} {verb} the Top 50 on {pd.Timestamp(ev['event_date']):%d %b %Y} at rank {int(ev['event_position'])}"]
    if ev["stint_type"] == "re_entry":
        parts[0] += (f" after {int(ev['absent_snapshots'])} observed snapshot(s) off the chart "
                     f"({int(ev['gap_days'])} calendar days since its last appearance at rank {int(ev['previous_exit_position'])})")
    if ev["peak_improvement"] > 0:
        parts.append(f"reached a peak of rank {int(ev['peak_rank'])} within {int(ev['days_to_peak'])} day(s) "
                     f"({int(ev['snapshots_to_peak'])} observed snapshot(s)), a {int(ev['peak_improvement'])}-place improvement")
    else:
        parts.append("did not improve on its entry rank after entering (peak = entry rank)")
    after = int(ev["retention_snapshots"]) - 1
    tail = (f"remained in the Top 50 for {after} additional snapshot(s) (retention {int(ev['retention_days'])} calendar day(s), "
            f"{int(ev['retention_snapshots'])} snapshots in total)") if after > 0 else \
        "left the Top 50 at the next observed snapshot (retention 1 snapshot)"
    parts.append(tail)
    text = ", ".join(parts[:-1]) + f" and {parts[-1]}." if len(parts) > 1 else parts[0] + "."
    extras = []
    if pd.notna(ev["popularity_lift"]):
        if ev["popularity_lift"] > 0:
            extras.append(f"Popularity rose by {ev['popularity_lift']:.0f} points over its pre-comeback baseline")
        elif ev["popularity_lift"] == 0:
            extras.append("Popularity did not rise above its pre-comeback baseline")
        else:
            extras.append(f"Popularity stayed {abs(ev['popularity_lift']):.0f} points below its pre-comeback baseline")
    if pd.notna(ev["momentum_score"]):
        extras.append(f"Momentum score is {ev['momentum_score']:.1f}/100 ({ev['pattern']})")
        if context and "median_momentum" in context:
            rel = "above" if ev["momentum_score"] > context["median_momentum"] else "at or below"
            extras.append(f"{rel} the median re-entry ({context['median_momentum']:.1f})")
    if pd.notna(ev["decay_speed"]):
        extras.append(f"after the peak it lost {ev['decay_speed']:.2f} rank places per day on average")
    if ev["is_right_censored"]:
        extras.append("it is still on the chart at the end of the data (right-censored), so retention is a lower bound, not a complete lifetime")
    if ev["is_left_censored"]:
        extras.append("its true entry date is unobserved (left-censored), so no momentum score is computed")
    return text + (" " + "; ".join(extras) + "." if extras else "")
