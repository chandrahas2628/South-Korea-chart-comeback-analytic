"""Page 5 — Content Attribute Analysis."""
import streamlit as st

from analytics.content_analysis import (SIZE_LABELS, compare_groups, compare_rates, correlation_matrix, group_summary,
                                        kruskal_groups, numeric_associations)
from ui import charts
from ui.components import callout, download_button, empty_state, page_header, show_notices

SUMMARY_COLS = {"n_songs": "Songs", "songs_with_reentry": "Songs w/ re-entry", "reentry_rate": "Re-entry rate",
                "avg_reentries_per_song": "Re-entries/song", "reentries_per_100_exposure_days": "Re-entries/100 exposure days",
                "n_events": "Events", "mean_momentum": "Mean momentum", "median_momentum": "Median momentum",
                "mean_retention_days": "Retention (days)", "mean_retention": "Retention (snapshots)",
                "mean_pop_lift": "Popularity lift", "mean_recovery_speed": "Recovery speed", "mean_song_fandom": "Fandom proxy"}
NUM = ["Re-entries/song", "Re-entries/100 exposure days", "Mean momentum", "Median momentum", "Retention (days)",
       "Retention (snapshots)", "Popularity lift", "Recovery speed", "Fandom proxy"]


def _summary_table(df, by, label):
    cols = [c for c in SUMMARY_COLS if c in df.columns]
    t = df[[by] + cols + ["low_sample"]].rename(columns={by: label, **SUMMARY_COLS, "low_sample": "Small sample?"})
    t["Small sample?"] = t["Small sample?"].map({True: "⚠ yes", False: "no"})
    st.dataframe(t, width="stretch", hide_index=True, column_config={
        "Re-entry rate": st.column_config.NumberColumn(format="percent"),
        **{c: st.column_config.NumberColumn(format="%.2f") for c in NUM}})
    if (df["low_sample"]).any():
        st.warning("Some groups have fewer than 10 songs or 10 events — differences involving them are unreliable.", icon=":material/warning:")
    return t


def _two_group_stats(ev, songs, by, a, b):
    st.markdown("**Statistical comparison (observed differences, not causal)**")
    rows = []
    for metric, name in (("momentum_score", "Momentum"), ("retention_days", "Retention (days)")):
        c = compare_groups(ev, by, a, b, metric)
        rows.append({"Metric": name, "Group A": a, "Group B": b, "n A": c["n_a"], "n B": c["n_b"], "Median A": c["median_a"],
                     "Median B": c["median_b"], "Test": "Mann-Whitney U", "p-value": c["p_value"],
                     "Cliff's δ": c["cliffs_delta"], "Effect": c["effect"]})
    r = compare_rates(songs, by, a, b)
    rows.append({"Metric": "Re-entry rate", "Group A": a, "Group B": b, "n A": r["n_a"], "n B": r["n_b"], "Median A": r["rate_a"],
                 "Median B": r["rate_b"], "Test": "Fisher exact", "p-value": r["p_value"], "Cliff's δ": float("nan"),
                 "Effect": f"odds ratio {r['odds_ratio']:.2f}" if r["odds_ratio"] == r["odds_ratio"] else "insufficient sample"})
    st.dataframe(rows, width="stretch", hide_index=True, column_config={
        "p-value": st.column_config.NumberColumn(format="%.3f"), "Cliff's δ": st.column_config.NumberColumn(format="%.2f"),
        "Median A": st.column_config.NumberColumn(format="%.2f"), "Median B": st.column_config.NumberColumn(format="%.2f")})
    small = [x for x in rows if x["n A"] < 10 or x["n B"] < 10]
    if small:
        st.warning("At least one group has fewer than 10 observations; treat p-values and effect sizes as indicative only.",
                   icon=":material/warning:")
    st.caption("Median columns show group rates for the re-entry-rate row. A 'directional relationship' is not evidence of causation.")


def _multi_stats(ev, by):
    out = []
    for metric, name in (("momentum_score", "Momentum"), ("retention_days", "Retention (days)")):
        k = kruskal_groups(ev, by, metric)
        out.append({"Metric": name, "Groups tested": k["n_groups"], "Events": k["n"], "Test": "Kruskal-Wallis",
                    "H": k["H"], "p-value": k["p_value"]})
    st.markdown("**Statistical comparison across groups**")
    st.dataframe(out, width="stretch", hide_index=True, column_config={
        "H": st.column_config.NumberColumn(format="%.2f"), "p-value": st.column_config.NumberColumn(format="%.3f")})


def render(ctx):
    fd = ctx.fd
    page_header("Content Analysis", "How comeback behaviour is associated with release format, content rating, album size and duration.")
    callout("All relationships below are <b>associations</b> in observed chart data — they do not show that an attribute causes "
            "a comeback. Small groups are flagged. Re-entry rate is the share of songs with ≥1 re-entry; "
            "re-entries per 100 exposure days adjusts for how long songs were observable.")
    show_notices(fd)
    if fd.reentries.empty:
        empty_state("No re-entry events in the current selection.")
        return
    songs, ev = fd.songs, fd.events
    tabs = st.tabs(["Single vs Album", "Explicit vs Clean", "Album size", "Duration", "Correlations"])
    with tabs[0]:
        s = group_summary(songs, ev, "album_type")
        t = _summary_table(s, "album_type", "Album type")
        a, b = st.columns(2)
        a.plotly_chart(charts.group_bar(s, "album_type", "reentry_rate", "Re-entry rate by album type", "% of songs with a re-entry",
                                        "Album type", pct=True), width="stretch")
        b.plotly_chart(charts.group_box(fd.reentries, "album_type", "Momentum by album type", "Album type"), width="stretch")
        _two_group_stats(ev, songs, "album_type", "single", "album")
        download_button(t, "Download table (CSV)", "content_album_type.csv", "dl_ct_at")
    with tabs[1]:
        s = group_summary(songs, ev, "explicit_label")
        t = _summary_table(s, "explicit_label", "Content")
        a, b = st.columns(2)
        a.plotly_chart(charts.group_bar(s, "explicit_label", "reentry_rate", "Re-entry rate: explicit vs clean",
                                        "% of songs with a re-entry", "Content", pct=True), width="stretch")
        b.plotly_chart(charts.group_box(fd.reentries, "explicit_label", "Momentum: explicit vs clean", "Content", violin=True),
                       width="stretch")
        _two_group_stats(ev, songs, "explicit_label", "Explicit", "Clean")
        download_button(t, "Download table (CSV)", "content_explicit.csv", "dl_ct_ex")
    with tabs[2]:
        s = group_summary(songs, ev, "album_size_bin")
        s["album_size_bin"] = s["album_size_bin"].astype(str)
        s = s.set_index("album_size_bin").reindex([x for x in SIZE_LABELS if x in set(s["album_size_bin"])]).reset_index()
        t = _summary_table(s, "album_size_bin", "Album size")
        a, b = st.columns(2)
        a.plotly_chart(charts.group_bar(s, "album_size_bin", "mean_momentum", "Mean momentum by album size", "Mean momentum",
                                        "Album size"), width="stretch")
        evb = fd.reentries.assign(album_size_bin=fd.reentries["album_size_bin"].astype(str))
        b.plotly_chart(charts.group_box(evb, "album_size_bin", "Momentum distribution by album size", "Album size", order=SIZE_LABELS),
                       width="stretch")
        _multi_stats(ev.assign(album_size_bin=ev["album_size_bin"].astype(str)), "album_size_bin")
        download_button(t, "Download table (CSV)", "content_album_size.csv", "dl_ct_sz")
    with tabs[3]:
        s = group_summary(songs, ev, "duration_bin")
        s = s.sort_values("duration_bin", key=lambda c: c.str.extract(r"^([\d.]+)")[0].astype(float))
        t = _summary_table(s, "duration_bin", "Duration quartile")
        a, b = st.columns(2)
        a.plotly_chart(charts.group_bar(s, "duration_bin", "reentry_rate", "Re-entry rate by duration quartile",
                                        "% of songs with a re-entry", "Duration", pct=True), width="stretch")
        r = fd.reentries.merge(songs[["song_id", "duration_minutes"]].rename(columns={"duration_minutes": "song_duration"}), on="song_id")
        b.plotly_chart(charts.scatter_xy(r, "song_duration", "momentum_score", "Song duration vs comeback momentum",
                                         "Duration (minutes)", "Momentum score", "song_label"), width="stretch")
        _multi_stats(ev, "duration_bin")
        download_button(t, "Download table (CSV)", "content_duration.csv", "dl_ct_du")
    with tabs[4]:
        st.plotly_chart(charts.corr_heatmap(correlation_matrix(ev)), width="stretch")
        assoc = numeric_associations(songs, ev)
        if assoc.empty:
            st.info("Not enough data for attribute correlations.")
        else:
            st.markdown("**Attribute associations (Spearman rank correlation)**")
            st.dataframe(assoc, width="stretch", hide_index=True, column_config={
                "spearman_rho": st.column_config.NumberColumn(format="%.3f"), "p_value": st.column_config.NumberColumn(format="%.4f")})
            st.caption("|ρ| < 0.2 is a weak directional relationship. Correlation is not causation.")
