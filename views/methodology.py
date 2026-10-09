"""Methodology & limitations page."""
import pandas as pd
import streamlit as st

from analytics.documentation import DEFINITIONS, LIMITATIONS, METRIC_GLOSSARY, methodology_sections
from analytics.utils import FANDOM_WEIGHTS, MOMENTUM_WEIGHTS
from ui.components import callout, page_header, section


def render(ctx):
    r = ctx.result
    page_header("Methodology & Limitations", "Exactly how every metric is calculated — written for a research reviewer.")
    callout("<b>Core rule:</b> a re-entry is never 'absent yesterday, present today'. It is based on each song's <i>observed</i> "
            f"history: appears → leaves a valid snapshot → stays absent for ≥ <b>{r.min_absent_snapshots}</b> valid observed snapshots → "
            "returns. Missing calendar dates are reported as gaps, not exits.")
    section("Key definitions")
    st.dataframe(pd.DataFrame(DEFINITIONS, columns=["Term", "Meaning"]), hide_index=True, width="stretch")
    for title, body in methodology_sections(r.min_absent_snapshots).items():
        with st.expander(title, expanded=title.startswith("4.")):
            st.markdown(body)
    section("Weights")
    a, b = st.columns(2)
    a.markdown("**Momentum Intensity**")
    a.dataframe(pd.DataFrame({"Component": [k.replace("_", " ") for k in MOMENTUM_WEIGHTS], "Weight": list(MOMENTUM_WEIGHTS.values())}),
                hide_index=True, width="stretch", column_config={"Weight": st.column_config.NumberColumn(format="percent")})
    b.markdown("**Fandom Intensity Proxy**")
    b.dataframe(pd.DataFrame({"Component": [k.replace("_", " ") for k in FANDOM_WEIGHTS], "Weight": list(FANDOM_WEIGHTS.values())}),
                hide_index=True, width="stretch", column_config={"Weight": st.column_config.NumberColumn(format="percent")})
    s = r.sensitivity
    section("Momentum score robustness", "How much do rankings change if the weights change?")
    c1, c2, c3 = st.columns(3)
    c1.metric("Spearman vs equal weights", f"{s['spearman_vs_equal_weights']:.3f}", help=f"n = {s['n']} re-entry events")
    c2.metric(f"Median Spearman, {s['n_draws']} random weight sets", f"{s['perturbation_median_rho']:.3f}",
              help="Weights drawn from a Dirichlet distribution centred on the documented weights.")
    c3.metric("Top-10 overlap vs equal weights", f"{s['top10_overlap_vs_equal']:.0%}")
    st.caption(f"Minimum Spearman across random weight sets: {s['perturbation_min_rho']:.3f}. Rankings are insensitive to the exact weights; "
               "the score is an analytical composite of observed chart behaviour, not an objective measure of fandom.")
    section("Data-derived thresholds in use")
    t = {**{f"Pattern: {k}": v for k, v in r.pattern_thresholds.items()},
         **{f"Song fandom band {k}": v for k, v in r.song_band_cuts.items()},
         **{f"Artist fandom band {k}": v for k, v in r.artist_band_cuts.items()}}
    st.dataframe(pd.DataFrame({"Threshold": list(t), "Value": [round(float(v), 2) for v in t.values()]}), hide_index=True, width="stretch")
    section("Metric glossary")
    st.dataframe(pd.DataFrame({"Metric": [k.replace("_", " ") for k in METRIC_GLOSSARY], "Definition": list(METRIC_GLOSSARY.values())}),
                 hide_index=True, width="stretch")
    section("Limitations")
    for l in LIMITATIONS:
        st.markdown(f"- {l}")
