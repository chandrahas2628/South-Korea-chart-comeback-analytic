"""Page 6 — Strategic Recommendations (data-driven rules)."""
import streamlit as st

from analytics.recommendations import MIN_ARTIST_EVENTS, MIN_SONG_EVENTS, build_recommendations
from ui.components import callout, download_button, empty_state, page_header, show_notices

RULE_HELP = {
    "HIGH MOMENTUM + HIGH RETENTION": "Mean momentum ≥ P75 and mean retention days ≥ P75 → extended promotional investment.",
    "HIGH MOMENTUM + LOW RETENTION": "Mean momentum ≥ P75 and mean retention days ≤ P40 → short-term activation.",
    "MULTIPLE RE-ENTRIES": "Re-entry count ≥ max(3, P90) → catalogue reactivation.",
    "HIGH ALBUM-TRACK MOMENTUM": "Triggered only if album tracks' median momentum exceeds singles' (significance is reported).",
    "HIGH ARTIST FANDOM PROXY": "Artist proxy ≥ P80 among artists with ≥3 re-entries → fan-engagement campaigns.",
    "LOW MOMENTUM + HIGH RE-ENTRY FREQUENCY": "Re-entry count ≥ max(3, P75) and mean momentum ≤ median → investigate recurring spikes.",
}


def render(ctx):
    fd = ctx.fd
    page_header("Strategic Recommendations", "Rule-based promotion candidates generated from the current selection — no hard-coded names.")
    callout(f"Rules use quantile thresholds computed from the <b>selected</b> data and statistics recomputed for the selection. Song rules "
            f"require ≥{MIN_SONG_EVENTS} re-entries, artist rules ≥{MIN_ARTIST_EVENTS}, so single events are never acted on. "
            "Recommendations are prompts for investigation, not causal claims; external events are not observed. "
            "Retention is in calendar days (a lower bound for songs still on the chart).")
    show_notices(fd)
    rec = build_recommendations(fd)
    for n in rec["notes"]:
        st.info(n)
    t = rec["table"]
    if t.empty:
        empty_state("No recommendation rule is triggered by the current selection.")
        return
    rules = list(t["rule"].unique())
    st.caption(f"{len(t)} candidates across {len(rules)} triggered rules (of {len(RULE_HELP)} defined).")
    tabs = st.tabs(["All"] + rules)
    show_cols = {"rule": "Rule", "level": "Level", "entity": "Candidate", "artist": "Artist", "n_events": "Re-entries",
                 "evidence": "Evidence", "recommended_action": "Recommended action", "priority": "Fandom proxy"}
    with tabs[0]:
        out = t.rename(columns=show_cols)
        st.dataframe(out, width="stretch", hide_index=True, height=420,
                     column_config={"Fandom proxy": st.column_config.NumberColumn(format="%.0f")})
        download_button(out, "Download recommendations (CSV)", "recommendations.csv", "dl_rec")
    for tab, rule in zip(tabs[1:], rules):
        with tab:
            st.caption(RULE_HELP.get(rule, ""))
            st.dataframe(t[t["rule"] == rule].rename(columns=show_cols).drop(columns=["Rule"]), width="stretch", hide_index=True)
    with st.expander("Thresholds used (computed from the current selection)"):
        st.dataframe({k: [round(float(v), 2)] for k, v in rec["thresholds"].items()}, width="stretch", hide_index=True)
