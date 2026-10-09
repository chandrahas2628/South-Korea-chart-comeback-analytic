"""Page 2 — Comeback Intelligence."""
import streamlit as st

from analytics.narrative import describe_event
from ui import charts
from ui.components import callout, download_button, empty_state, fmt, kpi_row, page_header, section, show_notices, table

EVENT_COLS = {"song_label": "Song", "reentry_date": "Re-entry date", "gap_days": "Gap (days)",
              "absent_snapshots": "Absent snapshots", "reentry_position": "Re-entry rank", "peak_rank": "Peak rank",
              "momentum_score": "Momentum", "retention_days": "Retention (days)", "retention_snapshots": "Retention (snapshots)",
              "is_right_censored": "Lower bound?", "pattern": "Pattern"}


def render(ctx):
    fd = ctx.fd
    page_header("Comeback Intelligence", "Which songs return, how often, how long they were away and how strong the comeback is.")
    show_notices(fd)
    re = fd.reentries
    if re.empty:
        empty_state("No re-entry events match the current filters.")
        return
    kpi_row([("Re-entry events", fmt(len(re))), ("Songs with re-entries", fmt(re["song_id"].nunique())),
             ("Median gap away", fmt(re["gap_days"].median(), 0, " days"), f"median {fmt(re['absent_snapshots'].median(), 0)} absent snapshots"),
             ("Median retention", fmt(re["retention_days"].median(), 0, " days"),
              f"{fmt(re['retention_snapshots'].median(), 0)} snapshots · {int(re['is_right_censored'].sum())} lower-bound")])
    section("Timeline", "Each bubble is a confirmed re-entry; hover for the song and metrics.")
    st.plotly_chart(charts.reentry_timeline(re), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(charts.frequency_bar(fd.songs), width="stretch")
    b.plotly_chart(charts.gap_distribution(re), width="stretch")

    section("Comeback leaderboard", "Songs ranked by re-entries in the selection. Select a row to inspect that song below.")
    lb = fd.songs[fd.songs["reentries_in_selection"] > 0].sort_values(["reentries_in_selection", "mean_momentum"], ascending=False)
    show = lb[["song_label", "reentries_in_selection", "mean_momentum", "mean_retention_days", "mean_retention", "mean_gap_days",
               "best_position", "confidence"]].rename(columns={
        "song_label": "Song", "reentries_in_selection": "Re-entries", "mean_momentum": "Mean momentum",
        "mean_retention_days": "Mean retention (days)", "mean_retention": "Mean retention (snapshots)",
        "mean_reentry_gap_days": "Mean gap (days)", "best_position": "Best rank", "confidence": "Sample size"}).reset_index(drop=True)
    sel = st.dataframe(show, width="stretch", hide_index=True, on_select="rerun", selection_mode="single-row", height=320,
                       column_config={c: st.column_config.NumberColumn(format="%.1f") for c in
                                      ["Mean momentum", "Mean retention (days)", "Mean retention (snapshots)", "Mean gap (days)"]})
    download_button(show, "Download leaderboard (CSV)", "comeback_leaderboard.csv", "dl_comeback")

    section("Re-entry event explorer", "Every confirmed re-entry in the selection (sortable). 'Lower bound?' = right-censored retention.")
    ex = re.sort_values("reentry_date", ascending=False)[list(EVENT_COLS)].rename(columns=EVENT_COLS)
    ex["Re-entry date"] = ex["Re-entry date"].dt.date
    st.dataframe(ex, width="stretch", hide_index=True, height=300,
                 column_config={"Momentum": st.column_config.NumberColumn(format="%.1f")})
    download_button(ex, "Download events (CSV)", "reentry_events_view.csv", "dl_ev_view")

    section("Song-level comeback history")
    labels = lb["song_label"].tolist()
    default = sel.selection.rows[0] if sel.selection.rows else 0
    song = st.selectbox("Song", labels, index=min(default, len(labels) - 1), key="cb_song",
                        help="Songs with at least one re-entry in the current selection.")
    sid = lb.loc[lb["song_label"] == song, "song_id"].iloc[0]
    rows = ctx.result.appearances[ctx.result.appearances["song_id"] == sid]
    evs = fd.events[fd.events["song_id"] == sid]
    st.plotly_chart(charts.song_history(rows, evs, song), width="stretch")
    hist = evs.sort_values("event_date")[["stint_type", "event_date", "event_position", "previous_exit_date",
                                          "previous_exit_position", "gap_days", "absent_snapshots", "peak_rank",
                                          "momentum_score", "retention_days", "retention_snapshots", "is_right_censored",
                                          "pattern"]].copy()
    hist["event_date"] = hist["event_date"].dt.date
    hist["previous_exit_date"] = hist["previous_exit_date"].dt.date
    table(hist.rename(columns=lambda c: c.replace("_", " ").title()))
    for _, e in evs[evs["is_reentry"]].sort_values("event_date").tail(3).iterrows():
        callout(describe_event(e, {"median_momentum": re["momentum_score"].median()}))
        st.write("")
