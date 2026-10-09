"""South Korea Chart Momentum Intelligence — Streamlit entry point.

Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="South Korea Chart Momentum Intelligence", page_icon=":material/query_stats:",
                   layout="wide", initial_sidebar_state="expanded")

from analytics.filters import FilterState, apply_filters
from analytics.ml_reentry import build_samples, train_and_evaluate
from analytics.pipeline import DEFAULT_MIN_ABSENT, run_pipeline
from analytics.validation import DataValidationError
from ui.theme import CSS, register_plotly_template
from views import (comeback_intelligence, content_analysis, data_quality, downloads, fandom_intelligence, forecast,
                   methodology, momentum_lab, overview, recommendations)

log = logging.getLogger("momentum_app")
DEFAULT_DATA = Path(__file__).parent / "data" / "raw" / "Atlantic_South_Korea.csv"
register_plotly_template()
st.markdown(CSS, unsafe_allow_html=True)


@dataclass
class Ctx:
    result: object
    fd: object
    ml: callable
    ml_model: callable


@st.cache_data(show_spinner="Running the analytical pipeline…")
def _pipeline_from_path(path: str, mtime: float, min_absent: int):
    return run_pipeline(path, min_absent)


@st.cache_data(show_spinner="Running the analytical pipeline on uploaded data…")
def _pipeline_from_bytes(content: bytes, min_absent: int):
    import io
    return run_pipeline(io.BytesIO(content), min_absent)


@st.cache_data(show_spinner="Validating forecast model on a later time period…")
def _ml(path_key: str, min_absent: int, horizon: int, _result):
    s = build_samples(_result.reentry.stints, _result.appearances, _result.cleaned.snapshots, min_absent, horizon)
    return train_and_evaluate(s, horizon)


def _load():
    with st.sidebar:
        st.markdown("### Chart Momentum Intelligence")
        st.caption("Atlantic Recording Corporation · South Korea Top 50")
        with st.expander("Data & re-entry definition", expanded=False):
            up = st.file_uploader("Replace dataset (CSV, same schema)", type="csv")
            min_absent = st.number_input("Min. observed absent snapshots = exit", 1, 15, DEFAULT_MIN_ABSENT, 1,
                                         help="A song must be missing from this many consecutive VALID snapshots before its "
                                              "next appearance counts as a re-entry. 1 counts every dropout (mostly rank 49-50 "
                                              "flicker).")
    try:
        if up is not None:
            content = up.getvalue()
            return _pipeline_from_bytes(content, int(min_absent)), hashlib.md5(content).hexdigest(), int(min_absent)
        return (_pipeline_from_path(str(DEFAULT_DATA), DEFAULT_DATA.stat().st_mtime, int(min_absent)),
                f"default-{DEFAULT_DATA.stat().st_mtime}", int(min_absent))
    except DataValidationError as exc:
        st.error(f"Data validation failed: {exc}")
    except FileNotFoundError:
        st.error(f"Dataset not found at {DEFAULT_DATA}. Place Atlantic_South_Korea.csv in data/raw/ or upload a CSV.")
    except Exception:  # noqa: BLE001 — logged, user sees a clean message
        log.exception("Pipeline failure")
        st.error("The analytical pipeline failed unexpectedly. Check that the file matches the expected schema; "
                 "details were written to the server log.")
    st.stop()


DEFAULTS = {"f_date": None, "f_artists": [], "f_songs": [], "f_types": [], "f_explicit": "All", "f_min": 0, "f_mom": None}


def _reset(result):
    d0, d1 = result.cleaned.snapshots["date"].min().date(), result.cleaned.snapshots["date"].max().date()
    st.session_state.update({"f_date": (d0, d1), "f_artists": [], "f_songs": [], "f_types": [], "f_explicit": "All",
                             "f_min": 0, "f_mom": (0.0, 100.0)})


def _filters(result) -> FilterState:
    snaps = result.cleaned.snapshots
    d0, d1 = snaps["date"].min().date(), snaps["date"].max().date()
    if st.session_state.get("f_date") is None:
        _reset(result)
    songs, ev = result.songs, result.events
    max_re = int(songs["reentry_count"].max())
    with st.sidebar:
        st.markdown("#### Global filters")
        st.button("Reset filters", on_click=_reset, args=(result,), width="stretch", icon=":material/restart_alt:")
        dr = st.date_input("Date range", min_value=d0, max_value=d1, key="f_date",
                           help="Filters playlist rows by date and re-entry events by their re-entry date.")
        artists = st.multiselect("Artist", songs.sort_values("appearance_count", ascending=False)["artist"].unique().tolist(), key="f_artists")
        label_pool = songs[songs["artist"].isin(artists)] if artists else songs
        sel_songs = st.multiselect("Song", sorted(label_pool["song_label"]), key="f_songs")
        types = st.multiselect("Album type", sorted(songs["album_type"].unique()), key="f_types",
                               help="Uses each song's most common album type so complete song histories are kept.")
        explicit = st.radio("Content", ["All", "Explicit", "Clean"], horizontal=True, key="f_explicit",
                            help="Uses each song's most common explicit flag.")
        min_re = st.slider("Minimum re-entries per song", 0, max(max_re, 1), key="f_min",
                           help="Keep only songs with at least this many re-entries (within the other filters).")
        mmin, mmax = float(ev["momentum_score"].min()), float(ev["momentum_score"].max())
        mom = st.slider("Momentum range (re-entry events)", 0.0, 100.0, key="f_mom", step=1.0,
                        help="Keep re-entry events whose momentum score lies in this range. Songs with no remaining events drop out.")
    date_range = tuple(dr) if isinstance(dr, (list, tuple)) and len(dr) == 2 else (d0, d1)
    if len(date_range) == 2 and not (isinstance(dr, (list, tuple)) and len(dr) == 2):
        st.sidebar.warning("Select both a start and an end date.")
    return FilterState(date_range=date_range, artists=artists, songs=sel_songs, album_types=types, explicit=explicit,
                       min_reentries=min_re, momentum_range=mom)


result, data_key, min_absent = _load()
fstate = _filters(result)
fd = apply_filters(result, fstate)
ctx = Ctx(result=result, fd=fd, ml=lambda h: _ml(data_key, min_absent, h, result),
          ml_model=lambda: _ml(data_key, min_absent, 30, result)["model"])
with st.sidebar:
    if fd.active:
        st.caption(f":material/filter_alt: Filters active — {fd.rows['song_id'].nunique():,} songs, {len(fd.reentries):,} re-entries in view.")
    st.divider()


def _page(module, title, icon, path, default=False):
    def run():
        try:
            module.render(ctx)
        except Exception:  # noqa: BLE001 — logged; no raw stack trace for end users
            log.exception("Page %s failed", title)
            st.error(f"“{title}” could not be rendered for this selection. Try resetting the filters.")
    run.__name__ = path
    return st.Page(run, title=title, icon=icon, url_path=path, default=default)


pages = {
    "Analytics": [
        _page(overview, "Executive Overview", ":material/dashboard:", "overview", True),
        _page(comeback_intelligence, "Comeback Intelligence", ":material/replay:", "comebacks"),
        _page(momentum_lab, "Momentum Lab", ":material/science:", "momentum-lab"),
        _page(fandom_intelligence, "Fandom Intelligence", ":material/groups:", "fandom"),
        _page(content_analysis, "Content Analysis", ":material/album:", "content"),
        _page(recommendations, "Strategic Recommendations", ":material/lightbulb:", "recommendations"),
    ],
    "Reference": [
        _page(forecast, "Re-entry Forecast (beta)", ":material/trending_up:", "forecast"),
        _page(data_quality, "Data Quality", ":material/fact_check:", "data-quality"),
        _page(methodology, "Methodology & Limitations", ":material/menu_book:", "methodology"),
        _page(downloads, "Data & Research Export", ":material/download:", "export"),
    ],
}
st.navigation(pages).run()
