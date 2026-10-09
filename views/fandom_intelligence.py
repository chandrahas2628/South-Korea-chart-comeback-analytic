"""Page 4 — Fandom Intelligence (behavioural proxy)."""
import streamlit as st

from analytics.fandom import explain_score
from analytics.reporting import artist_views
from ui import charts
from ui.components import callout, download_button, empty_state, page_header, section, show_notices

COLS = {"reentry_count": "Re-entries", "mean_momentum": "Mean momentum", "median_momentum": "Median momentum",
        "mean_retention_days": "Retention (days)", "mean_rank_recovery": "Rank recovery", "mean_recovery_speed": "Recovery speed",
        "mean_pop_lift": "Popularity lift", "sharpness": "Sharpness", "fandom_score": "Fandom proxy",
        "fandom_band": "Band", "confidence": "Sample size"}
FMT = {"Mean momentum": "%.1f", "Median momentum": "%.1f", "Retention (days)": "%.1f", "Rank recovery": "%.1f",
       "Recovery speed": "%.2f", "Popularity lift": "%.1f", "Sharpness": "%.2f"}


def _cfg():
    out = {k: st.column_config.NumberColumn(format=v) for k, v in FMT.items()}
    out["Fandom proxy"] = st.column_config.ProgressColumn("Fandom proxy", min_value=0, max_value=100, format="%.0f")
    return out


def render(ctx):
    fd = ctx.fd
    page_header("Fandom Intelligence", "A behavioural proxy for fan-driven reactivation, built only from observable chart behaviour.")
    callout("<b>Read this first.</b> The <b>Fandom Intensity Proxy</b> is <b>not</b> fandom size, a fan count or actual fandom strength, "
            "and is not causal evidence of fan activity. It summarises re-entry frequency, momentum, popularity lift, rank recovery, "
            "recovery speed, retention and comeback sharpness (share of comebacks entering in ranks 1-25). Scores are shrunk toward "
            "the typical entity for small samples and normalised to 0-100 across the full dataset; sample sizes are shown everywhere.")
    show_notices(fd)
    if fd.reentries.empty:
        empty_state("No re-entries in the current selection.")
        return
    a, b = st.columns(2)
    min_art = a.slider("Minimum re-entries for an artist to be ranked", 1, 20, 3, key="fan_min_art",
                       help="Guards against small-sample bias: an artist with one comeback is not ranked against artists with many.")
    min_song = b.slider("Minimum re-entries for a song to be ranked", 1, 10, 2, key="fan_min_song")

    section("Artist leaderboards", "Four lenses on artist comeback behaviour. Frequency ignores the minimum; the others apply it.")
    views = artist_views(fd.artists.dropna(subset=["reentry_count"]), min_art, 15)
    tabs = st.tabs(list(views))
    for tab, (name, _) in zip(tabs, views.items()):
        with tab:
            base = fd.artists[fd.artists["reentry_count"] >= (1 if "frequency" in name else min_art)]
            key = {"Top by frequency": "reentry_count", "Top by momentum": "mean_momentum",
                   "Top by fandom proxy": "fandom_score", "Top by retention": "mean_retention_days"}[name]
            d = base.dropna(subset=[key]).sort_values(key, ascending=False)
            if d.empty:
                st.info(f"No artist has at least {min_art} re-entries in the current selection.")
                continue
            t = d[["artist", "unique_songs", "songs_with_reentries"] + list(COLS)].rename(
                columns={**COLS, "artist": "Artist", "unique_songs": "Songs", "songs_with_reentries": "Songs w/ re-entry"})
            st.dataframe(t, width="stretch", hide_index=True, height=340, column_config=_cfg())
            download_button(t, f"Download: {name} (CSV)", f"artist_{key}.csv", f"dl_art_{key}")
    section("Song leaderboard")
    sg = fd.songs[(fd.songs["reentry_count"] >= min_song)].dropna(subset=["fandom_score"]).sort_values("fandom_score", ascending=False)
    if sg.empty:
        st.info(f"No song has at least {min_song} re-entries in the current selection.")
    else:
        t = sg[["song_label"] + list(COLS)].rename(columns={**COLS, "song_label": "Song"})
        st.dataframe(t, width="stretch", hide_index=True, height=340, column_config=_cfg())
        download_button(t, "Download song leaderboard (CSV)", "song_fandom_leaderboard.csv", "dl_song_lb")
        pick = st.selectbox("Explain a song's score", sg["song_label"].tolist(), key="fan_explain")
        callout(explain_score(sg[sg["song_label"] == pick].iloc[0]))

    section("Distribution and relationships")
    a, b = st.columns(2)
    a.plotly_chart(charts.band_histogram(fd.songs, "Song Fandom Intensity Proxy distribution"), width="stretch")
    b.plotly_chart(charts.band_histogram(fd.artists, "Artist Fandom Intensity Proxy distribution"), width="stretch")
    c, d = st.columns(2)
    sc = fd.songs[fd.songs["reentry_count"] >= 1]
    c.plotly_chart(charts.scatter_xy(sc, "reentry_count", "mean_momentum", "Re-entry frequency vs mean momentum (per song)",
                                     "Re-entries", "Mean momentum", "song_label", size="mean_retention_days"), width="stretch")
    d.plotly_chart(charts.scatter_xy(fd.reentries, "retention_days", "momentum_score", "Momentum vs retention (per re-entry event)",
                                     "Retention (calendar days)", "Momentum score", "song_label", color="pattern"), width="stretch")
    e, f = st.columns(2)
    e.plotly_chart(charts.scatter_xy(sc, "reentry_count", "fandom_score", "Fandom proxy vs re-entry count (per song)",
                                     "Re-entries", "Fandom proxy (0-100)", "song_label", size="mean_retention_days"), width="stretch")
    ar = fd.artists[fd.artists["reentry_count"] >= 1]
    f.plotly_chart(charts.scatter_xy(ar, "reentry_count", "fandom_score", "Fandom proxy vs re-entry count (per artist)",
                                     "Re-entries", "Fandom proxy (0-100)", "artist", size="unique_songs"), width="stretch")
    st.caption("Bubble size = mean retention days (songs) / number of songs (artists). Dashed line = linear trend (descriptive only). "
               "Momentum includes retention as a component and the proxy includes re-entry count, so positive relationships are "
               "partly by construction.")
