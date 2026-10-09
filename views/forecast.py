"""Optional ML page: probability of re-entry within a horizon (time-based validation). EXPERIMENTAL."""
import streamlit as st

from analytics.ml_reentry import FEATURES, score_current
from ui.components import callout, empty_state, fmt, kpi_row, page_header, section


def render(ctx):
    page_header("Re-entry Forecast (experimental)", "Descriptive risk ranking: probability an exited song re-enters within the horizon, evaluated out of time.")
    ml = ctx.ml(30)
    callout("<b>Experimental / descriptive.</b> Features use only information available on or before each cutoff date; the label looks only "
            f"at the following {ml['horizon_days']} days. Train and test are split chronologically with an embargo equal to the horizon. "
            "This page does not affect any other analysis, and is not a dependable prediction unless marked reliable.")
    if not ml["supported"]:
        st.warning(f"Model not supported by this dataset: {ml['reason']}")
        return
    kpi_row([("Training samples", fmt(ml["n_train"]), f"{ml['pos_train']} re-entries"),
             ("Test samples", fmt(ml["n_test"]), f"{ml['pos_test']} re-entries"),
             ("Base rate (test)", f"{ml['prevalence_test']:.1%}", "P(re-entry within horizon)"),
             ("Reliability", "Reliable" if ml["reliable"] else "Not validated", f"best: {ml['best_model']}")])
    st.write("")
    st.markdown(f"Train cutoffs **before {ml['split_date']:%d %b %Y}** · embargo · test cutoffs **from {ml['test_start']:%d %b %Y}**.")
    section("Out-of-time evaluation")
    m = ml["metrics"]
    st.dataframe(m, width="stretch", hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in m.columns if c != "model"})
    (st.success if ml["reliable"] else st.warning)(ml["reason"])
    st.caption("AUC intervals use a song-clustered bootstrap. Lift@top10% = hit rate among the 10% highest-scored samples ÷ base rate. "
               "The baseline ranks songs only by how recently they exited.")
    section("Currently exited songs ranked by modelled re-entry risk", "Descriptive ranking as of the last observed snapshot.")
    cur = score_current(ctx.ml_model(), ctx.result.reentry.stints, ctx.result.appearances, ctx.result.cleaned.snapshots,
                        ctx.result.min_absent_snapshots) if ml["supported"] else None
    if cur is None or cur.empty:
        empty_state("No exited songs to score.")
    else:
        if not ml["reliable"]:
            st.caption("Shown for exploration only — the model is not validated beyond the recency baseline.")
        st.dataframe(cur.head(25), width="stretch", hide_index=True,
                     column_config={"reentry_probability": st.column_config.ProgressColumn("P(re-entry)", min_value=0, max_value=1, format="%.3f")})
    with st.expander("Features used"):
        st.write(", ".join(FEATURES))
