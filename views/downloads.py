"""Downloads & research-paper support."""
import streamlit as st

from analytics.recommendations import build_recommendations
from analytics.reporting import research_summary, summary_json
from ui.components import callout, download_button, page_header, section


def render(ctx):
    r, fd = ctx.result, ctx.fd
    page_header("Data & Research Export", "CSV exports of the analytical datasets and an auto-generated research summary.")
    scope = st.radio("Scope for event, leaderboard and recommendation exports", ["Current filters", "Full dataset"], horizontal=True,
                     help="The cleaned dataset and research summary are always exported in full.")
    full = scope == "Full dataset"
    if full:
        from analytics.filters import FilterState, apply_filters
        fd = apply_filters(r, FilterState())
    ev, allev, songs, arts = fd.reentries, fd.events, fd.songs, fd.artists
    section("Datasets", "All event exports include retention_days, retention_snapshots and the right-censoring flag.")
    cols = st.columns(3)
    with cols[0]:
        download_button(r.cleaned.clean, "Cleaned playlist dataset", "clean_playlist.csv", "dl1")
        download_button(ev, "Re-entry events", "reentry_events.csv", "dl2")
        download_button(allev, "Momentum events (all stints)", "momentum_events.csv", "dl3")
    with cols[1]:
        download_button(songs.sort_values("fandom_score", ascending=False), "Song analytics (fandom leaderboard)", "song_analytics.csv", "dl4")
        download_button(arts.sort_values("fandom_score", ascending=False), "Artist analytics", "artist_analytics.csv", "dl5")
        rec = build_recommendations(fd)["table"]
        download_button(rec, "Recommendations", "recommendations.csv", "dl_rec_x")
    with cols[2]:
        download_button(r.cleaned.snapshots, "Snapshot validation", "snapshot_validation.csv", "dl6")
        download_button(r.cleaned.removal_ledger, "Row reconciliation (raw → clean)", "row_reconciliation.csv", "dl7")
    section("Research paper support", "Generated for the FULL dataset; every number is computed.")
    data, md = research_summary(r, ctx.ml(30))
    a, b = st.columns(2)
    a.download_button("Research summary (Markdown)", md.encode("utf-8"), "research_summary.md", "text/markdown", width="stretch")
    b.download_button("Research summary (JSON)", summary_json(data).encode("utf-8"), "research_summary.json", "application/json",
                      width="stretch")
    with st.expander("Preview summary"):
        st.markdown(md)
