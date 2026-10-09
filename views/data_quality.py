"""Data quality & snapshot validation page."""
import pandas as pd
import streamlit as st

from ui import charts
from ui.components import callout, download_button, kpi_row, page_header, section, table


def render(ctx):
    cr = ctx.result.cleaned
    q = cr.quality
    page_header("Data Quality", "Validation, duplicate handling and playlist snapshot checks — anomalies are surfaced, not hidden.")
    kpi_row([("Raw records", f"{q['Total raw records']:,}", "untouched CSV"),
             ("Clean records", f"{q['Valid records (after cleaning)']:,}", "used in analysis"),
             ("Rejected records", f"{q['Rejected records (failed validation)']:,}", "failed validation"),
             ("Duplicate records", f"{q['Exact duplicate rows (raw)']:,}", f"exact rows · {q['Duplicate song/date records (raw)']} song/date duplicates")])
    st.write("")
    kpi_row([("Observed playlist dates", f"{q['Observed playlist dates']:,}", f"{q['First date']} → {q['Last date']}"),
             ("Valid 50-entry snapshots", f"{q['Valid playlist days (complete 50-entry snapshots)']:,}"),
             ("Incomplete snapshots", f"{q['Incomplete playlist days (<50 entries)']}", "<50 entries after cleaning"),
             ("Over-full snapshots", f"{q['Raw dates with >50 rows (stacked snapshots)']}", f"raw dates >50 rows · {q['Over-full playlist days after cleaning (>50)']} after cleaning")])
    st.write("")
    kpi_row([("Calendar gaps", f"{q['Calendar gaps (missing dates)']}", f"{q['Calendar days in range']} calendar days in range"),
             ("Songs", f"{q['Unique songs (song+artist)']:,}", f"{q['Unique song titles']} titles"),
             ("Artists", f"{q['Unique artists']:,}"),
             ("Duplicate song/date removed", f"{q['Duplicate song/date rows removed (post-resolution)']}", "after snapshot resolution")])
    section("What was removed and why", "Row-level reconciliation from the raw file to the clean dataset.")
    table(cr.removal_ledger)
    if (cr.removal_ledger["Reconciles"] == "yes").any():
        st.success("Raw − rejected − set-aside − duplicates = clean records (reconciled).", icon=":material/check_circle:")
    if q["Raw dates with >50 rows (stacked snapshots)"]:
        callout(f"<b>{q['Raw dates with >50 rows (stacked snapshots)']} date(s)</b> held more than 50 rows (stacked snapshots). "
                f"{q['Alternate-snapshot rows discarded']} rows of the alternative snapshot were set aside (not deleted from the raw file). "
                "The kept snapshot has the highest song-set overlap with neighbouring days; when the scores are close the choice is uncertain.")
        table(cr.snapshot_resolution.assign(date=cr.snapshot_resolution["date"].dt.date))
    if q["Calendar gaps (missing dates)"]:
        st.markdown(f"**{q['Calendar gaps (missing dates)']} calendar dates have no snapshot.** They are never invented and never treated as chart exits.")
        table(cr.gap_runs.assign(gap_start=cr.gap_runs["gap_start"].dt.date, gap_end=cr.gap_runs["gap_end"].dt.date))
    section("Snapshot validation")
    st.plotly_chart(charts.snapshot_timeline(cr.snapshots, cr.calendar_gaps), width="stretch")
    a, b = st.columns(2)
    a.markdown("**Snapshot status counts (after cleaning)**")
    a.dataframe(cr.snapshots["status"].value_counts().rename_axis("status").reset_index(name="dates"), hide_index=True, width="stretch")
    b.markdown("**Full data-quality summary**")
    b.dataframe(pd.DataFrame({"Metric": list(q), "Value": [str(v) for v in q.values()]}), hide_index=True, width="stretch", height=300)
    with st.expander("Rejected rows (failed validation)"):
        if cr.rejected.empty:
            st.success("No rows failed validation.")
        else:
            table(cr.rejected.drop(columns=["song_clean", "artist_clean"], errors="ignore").head(500))
    download_button(cr.snapshots, "Download snapshot validation (CSV)", "snapshot_validation.csv", "dl_snap")
