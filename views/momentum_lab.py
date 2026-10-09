"""Page 3 — Momentum Lab (single comeback drill-down)."""
import pandas as pd
import streamlit as st

from analytics.documentation import METRIC_GLOSSARY
from analytics.momentum import event_trajectory, lifecycle_stages
from analytics.narrative import describe_event
from ui import charts
from ui.components import callout, empty_state, fmt, kpi_row, page_header, section, show_notices, stage_flow

TYPE_LABEL = {"re_entry": "Re-entry", "first_entry": "First entry", "initial_observed": "Initial (left-censored)"}
G = METRIC_GLOSSARY


def render(ctx):
    fd = ctx.fd
    page_header("Momentum Lab", "Inspect one comeback: exit, re-entry, surge, peak, decay and exit — with metrics derived from observed snapshots.")
    show_notices(fd)
    ev_all = fd.events
    songs = fd.songs[fd.songs["reentries_in_selection"] > 0].sort_values("reentries_in_selection", ascending=False)
    if songs.empty:
        empty_state("No songs with re-entries match the current filters.")
        return
    c1, c2, c3 = st.columns([2, 2, 1])
    song = c1.selectbox("Song", songs["song_label"].tolist(), key="ml_song",
                        help="Songs with at least one re-entry in the current selection, most re-entries first.")
    sid = songs.loc[songs["song_label"] == song, "song_id"].iloc[0]
    ev_s = ev_all[ev_all["song_id"] == sid].sort_values("event_date")
    opts = {f"{TYPE_LABEL[r.stint_type]} · {r.event_date:%d %b %Y}"
            + (f" · momentum {r.momentum_score:.0f}" if pd.notna(r.momentum_score) else ""): r.reentry_id
            for r in ev_s.itertuples()}
    default = [i for i, v in enumerate(opts.values()) if ev_s.set_index("reentry_id").loc[v, "is_reentry"]]
    label = c2.selectbox("Comeback event", list(opts), index=default[0] if default else 0, key="ml_event")
    pre = c3.number_input("Context snapshots", 5, 60, 14, 1, help="Valid snapshots shown before/after the stint.")
    ev = ev_s[ev_s["reentry_id"] == opts[label]].iloc[0]
    traj = event_trajectory(ctx.result.appearances, ctx.result.cleaned.snapshots, sid, int(ev["stint_no"]), pre, pre)

    section("Comeback lifecycle", "Derived from this event's observed snapshots.")
    stage_flow(lifecycle_stages(ev))
    if ev["is_right_censored"]:
        st.warning("Right-censored: this song is still on the chart at the last snapshot, so retention is a lower bound, "
                   "not a complete lifetime.", icon=":material/hourglass_top:")
    if ev["is_left_censored"]:
        st.warning("Left-censored: the song was already on the chart when observation began; its true entry is unobserved.",
                   icon=":material/hourglass_bottom:")
    st.write("")
    kpi_row([("Momentum score", fmt(ev["momentum_score"]), ev["pattern"], G["momentum_score"]),
             ("Entry rank", fmt(int(ev["event_position"])), f"rank jump {ev['rank_jump']:+.0f}", G["rank_jump"]),
             ("Peak rank", fmt(int(ev["peak_rank"])), f"+{int(ev['peak_improvement'])} places after entry", G["peak_rank"]),
             ("Rank recovery", fmt(ev["rank_recovery_total"], 0, " places"), "vs previous exit rank", G["rank_recovery"])])
    st.write("")
    kpi_row([("Recovery speed", fmt(ev["recovery_speed"], 2, " /day"), "places regained per day", G["recovery_speed"]),
             ("Popularity lift", fmt(ev["popularity_lift"], 0, " pts"), "peak vs pre-comeback baseline", G["popularity_lift"]),
             ("Days to peak", fmt(int(ev["days_to_peak"]), 0), f"{int(ev['snapshots_to_peak'])} snapshots", G["days_to_peak"]),
             ("Retention days", fmt(int(ev["retention_days"]), 0, "+" if ev["is_right_censored"] else ""),
              "lower bound" if ev["is_right_censored"] else "until last appearance", G["retention_days"])])
    st.write("")
    kpi_row([("Retention snapshots", fmt(int(ev["retention_snapshots"]), 0), f"{int(ev['unobserved_days_in_stint'])} unobserved calendar day(s) inside",
              G["retention_snapshots"]),
             ("Decay speed", fmt(ev["decay_speed"], 2, " places/day"), "after the peak", G["decay_speed"]),
             ("Volatility", fmt(ev["volatility"], 2), "σ of daily rank change", G["volatility"]),
             ("Dropouts inside stint", fmt(int(ev["dropout_snapshots"]), 0), "short absences below the threshold")])
    section("Interpretation", "Generated from this event's computed metrics.")
    callout(describe_event(ev, {"median_momentum": fd.reentries["momentum_score"].median()}))
    section("Trajectory")
    st.plotly_chart(charts.trajectory_figure(traj, ev), width="stretch")
    st.plotly_chart(charts.popularity_figure(traj, ev), width="stretch")
    st.caption("Dark points are valid snapshots where the song was off the Top 50 (plotted at rank 51). "
               "Calendar dates with no snapshot are not plotted and never treated as exits.")
    with st.expander("What do these metrics mean?"):
        for k_, v in G.items():
            st.markdown(f"**{k_.replace('_', ' ')}** — {v}")
    with st.expander("All metrics for this event"):
        d = ev.drop(labels=[c for c in ev.index if c.startswith("song_fandom")]).astype(str)
        st.dataframe(d.rename("value"), width="stretch")
