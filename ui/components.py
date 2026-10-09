"""Reusable Streamlit UI building blocks."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st


def kpi_card(label: str, value, sub: str = "", help_text: str = "") -> str:
    return (f'<div class="kpi" title="{html.escape(help_text)}"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{html.escape(str(value))}</div><div class="sub">{html.escape(sub)}</div></div>')


def kpi_row(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        c.markdown(kpi_card(*it), unsafe_allow_html=True)


def stage_flow(stages: list[dict]) -> None:
    """Horizontal lifecycle strip: PRE-COMEBACK -> EXIT -> RE-ENTRY -> SURGE -> PEAK -> DECAY -> EXIT."""
    cells = "".join(
        f'<div style="flex:1;min-width:120px;background:#111A2E;border:1px solid #1F2A44;border-radius:10px;padding:10px 12px">'
        f'<div style="color:#5B8DEF;font-size:.68rem;letter-spacing:.07em;font-weight:650">{html.escape(st_["stage"])}</div>'
        f'<div style="font-size:1.0rem;font-weight:650;margin-top:2px">{html.escape(st_["rank"])}</div>'
        f'<div style="color:#8B97B0;font-size:.74rem">{html.escape(st_["date"])}</div>'
        f'<div style="color:#8B97B0;font-size:.70rem;margin-top:3px">{html.escape(st_["detail"])}</div></div>'
        for st_ in stages)
    st.markdown('<div style="display:flex;gap:6px;flex-wrap:wrap;align-items:stretch">' + cells + "</div>",
                unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="sec"><div class="t">{html.escape(title)}</div>'
                f'<div class="s">{html.escape(subtitle)}</div></div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f"## {title}")
    st.caption(subtitle)


def insight_card(category: str, title: str, text: str) -> None:
    st.markdown(f'<div class="insight"><div class="c">{html.escape(category)} · {html.escape(title)}</div>'
                f'{html.escape(text)}</div>', unsafe_allow_html=True)


def callout(text: str) -> None:
    st.markdown(f'<div class="callout">{text}</div>', unsafe_allow_html=True)


def empty_state(message: str = "No data for the current selection.") -> None:
    st.info(message + " Try widening the filters or use **Reset filters** in the sidebar.", icon=":material/filter_alt_off:")


def show_notices(fd) -> None:
    for n in fd.notices:
        st.warning(n, icon=":material/info:")


def fmt(v, digits: int = 1, suffix: str = "") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}{suffix}"
    return f"{v:,.{digits}f}{suffix}"


def download_button(df: pd.DataFrame, label: str, filename: str, key: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename,
                       mime="text/csv", key=key, width="stretch")


def table(df: pd.DataFrame, **kwargs) -> None:
    st.dataframe(df, width="stretch", hide_index=True, **kwargs)
