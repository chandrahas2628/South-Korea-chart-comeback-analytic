"""Page 1 — Executive Overview."""
import streamlit as st

from analytics.documentation import METRIC_GLOSSARY
from analytics.insights import generate_insights
from analytics.reporting import compute_kpis
from ui import charts
from ui.components import callout, empty_state, fmt, insight_card, kpi_row, page_header, section, show_notices


def render(ctx):
    fd, n = ctx.fd, ctx.result.min_absent_snapshots
    page_header("Executive Overview", "How songs regain momentum after leaving the South Korea Top 50 — "
                                      "re-entries, comeback strength, retention and fandom-driven behaviour.")
    callout(f"<b>Method in one line:</b> a re-entry requires <b>{n} observed valid absence(s)</b> — complete snapshots in which the song "
            "is missing — before it returns. Missing calendar dates never count as exits. Change the threshold in the sidebar.")
    show_notices(fd)
    if fd.rows.empty:
        empty_state()
        return
    k = compute_kpis(fd)
    st.write("")
    kpi_row([("Total playlist days", fmt(k["playlist_days"]), "valid 50-entry snapshots", "Observed complete Top-50 snapshots"),
             ("Unique songs", fmt(k["unique_songs"]), "song + artist"),
             ("Unique artists", fmt(k["unique_artists"])),
             ("Total re-entries", fmt(k["total_reentries"]), f"≥{n} observed absences")])
    st.write("")
    top = (f"{k['top_fandom_artist']}" if k["top_fandom_artist"] else "—")
    kpi_row([("Songs with re-entries", fmt(k["songs_with_reentries"]),
              f"{k['songs_with_reentries'] / max(k['unique_songs'], 1):.0%} of songs"),
             ("Average momentum", fmt(k["avg_momentum"]), "0–100 · 50 = typical re-entry", METRIC_GLOSSARY["momentum_score"]),
             ("Average retention days", fmt(k["avg_retention_days"]),
              f"median {fmt(k['median_retention_days'])} · {k['right_censored_reentries']} lower-bound", METRIC_GLOSSARY["retention_days"]),
             ("Top fandom proxy artist", top,
              f"score {k['top_fandom_score']:.0f}/100 · ≥3 re-entries" if k["top_fandom_artist"] else "needs ≥3 re-entries",
              METRIC_GLOSSARY["fandom_score"])])
    section("Key automated insights", "Generated from the current selection — every figure is computed.")
    ins = generate_insights(fd)
    c1, c2 = st.columns(2)
    for i, it in enumerate(ins[:6]):
        with (c1 if i % 2 == 0 else c2):
            insight_card(it["category"], it["title"], it["text"])
    section("Trend and distribution")
    a, b = st.columns(2)
    a.plotly_chart(charts.reentry_trend(fd.reentries), width="stretch")
    b.plotly_chart(charts.momentum_distribution(fd.reentries), width="stretch")
    section("Top comeback songs and artists", "Comeback events ranked by momentum; artists ranked by the Fandom Intensity Proxy (≥3 re-entries).")
    a, b = st.columns(2)
    ev = fd.reentries.assign(label=lambda d: d["song_label"] + " · " + d["reentry_date"].dt.strftime("%d %b %y"))
    a.plotly_chart(charts.top_bar(ev, "label", "momentum_score", "Top comeback songs by momentum", "Momentum score",
                                  hover_cols=["retention_days", "retention_snapshots"], color="#5B8DEF"), width="stretch")
    arts = fd.artists[fd.artists["reentry_count"] >= 3]
    b.plotly_chart(charts.top_bar(arts, "artist", "fandom_score", "Top comeback artists by Fandom Intensity Proxy",
                                  "Proxy score (0-100)", hover_cols=["reentry_count", "unique_songs", "confidence"],
                                  color="#34D3A5"), width="stretch")
    if arts.empty:
        st.caption("No artist reaches 3 re-entries in the current selection.")
    st.caption("Retention days are lower bounds for right-censored comebacks (still on chart when the data ends).")
