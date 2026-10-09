"""Plotly figure builders. Pure functions: DataFrame in, Figure out."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analytics.utils import OFF_CHART_RANK
from .theme import ACCENT, AMBER, BORDER, MUTED, PATTERN_COLORS, ROSE, TEAL, TEXT, VIOLET


def _empty(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, xaxis_visible=False, yaxis_visible=False, height=320,
                      annotations=[dict(text="No data for current selection", showarrow=False, font=dict(color=MUTED))])
    return fig


def reentry_trend(re: pd.DataFrame) -> go.Figure:
    if re.empty:
        return _empty("Re-entries per month")
    m = re.assign(month=re["reentry_date"].dt.to_period("M").dt.to_timestamp()).groupby("month").size().reset_index(name="re_entries")
    fig = px.bar(m, x="month", y="re_entries", title="Chart re-entries per month",
                 labels={"month": "Month", "re_entries": "Re-entries"})
    fig.update_traces(marker_color=ACCENT, hovertemplate="%{x|%b %Y}: %{y} re-entries<extra></extra>")
    return fig


def momentum_distribution(re: pd.DataFrame) -> go.Figure:
    d = re.dropna(subset=["momentum_score"])
    if d.empty:
        return _empty("Momentum distribution")
    fig = px.histogram(d, x="momentum_score", nbins=25, title="Distribution of comeback momentum scores",
                       labels={"momentum_score": "Momentum score (0-100)"}, color_discrete_sequence=[ACCENT])
    fig.add_vline(x=d["momentum_score"].median(), line_dash="dash", line_color=AMBER,
                  annotation_text=f"median {d['momentum_score'].median():.1f}", annotation_font_color=AMBER)
    fig.update_yaxes(title="Re-entry events")
    return fig


def top_bar(df: pd.DataFrame, label: str, value: str, title: str, xtitle: str, hover_cols: list[str] | None = None,
            n: int = 10, color: str = ACCENT) -> go.Figure:
    d = df.dropna(subset=[value]).sort_values(value, ascending=False).head(n)
    if d.empty:
        return _empty(title)
    d = d.iloc[::-1]
    fig = go.Figure(go.Bar(x=d[value], y=d[label], orientation="h", marker_color=color,
                           customdata=d[hover_cols].to_numpy() if hover_cols else None,
                           hovertemplate="<b>%{y}</b><br>" + xtitle + ": %{x:.1f}" +
                           ("".join(f"<br>{c}: %{{customdata[{i}]}}" for i, c in enumerate(hover_cols)) if hover_cols else "")
                           + "<extra></extra>"))
    fig.update_layout(title=title, xaxis_title=xtitle, yaxis_title="", height=max(320, 28 * len(d) + 110),
                      yaxis=dict(automargin=True))
    return fig


def reentry_timeline(re: pd.DataFrame) -> go.Figure:
    if re.empty:
        return _empty("Re-entry timeline")
    fig = px.scatter(re, x="reentry_date", y="reentry_position", color="pattern", color_discrete_map=PATTERN_COLORS,
                     size="retention_snapshots", size_max=18, title="Re-entry timeline (bubble size = retention)",
                     labels={"reentry_date": "Re-entry date", "reentry_position": "Re-entry rank",
                             "pattern": "Pattern"},
                     hover_name="song_label",
                     hover_data={"momentum_score": ":.1f", "peak_rank": ":.0f", "gap_days": True,
                                 "retention_snapshots": True, "reentry_date": "|%d %b %Y"})
    fig.update_yaxes(autorange="reversed", title="Re-entry rank (1 = top)")
    return fig


def frequency_bar(songs: pd.DataFrame, n: int = 15) -> go.Figure:
    d = songs[songs["reentries_in_selection"] > 0].sort_values("reentries_in_selection", ascending=False).head(n)
    if d.empty:
        return _empty("Re-entry frequency")
    d = d.iloc[::-1]
    fig = go.Figure(go.Bar(x=d["reentries_in_selection"], y=d["song_label"], orientation="h", marker_color=TEAL,
                           hovertemplate="<b>%{y}</b><br>%{x} re-entries<extra></extra>"))
    fig.update_layout(title=f"Most frequent re-entries (top {len(d)} songs)", xaxis_title="Re-entries",
                      yaxis_title="", height=max(320, 26 * len(d) + 110), yaxis=dict(automargin=True))
    return fig


def gap_distribution(re: pd.DataFrame) -> go.Figure:
    if re.empty:
        return _empty("Re-entry gap distribution")
    fig = px.histogram(re, x="gap_days", nbins=40, title="Days away from the Top 50 before re-entry",
                       labels={"gap_days": "Gap (calendar days between last and return appearance)"},
                       color_discrete_sequence=[VIOLET])
    fig.add_vline(x=re["gap_days"].median(), line_dash="dash", line_color=AMBER,
                  annotation_text=f"median {re['gap_days'].median():.0f}d", annotation_font_color=AMBER)
    fig.update_yaxes(title="Re-entry events")
    return fig


def song_history(rows: pd.DataFrame, events: pd.DataFrame, song_label: str) -> go.Figure:
    s = rows.sort_values("date")
    fig = go.Figure(go.Scatter(x=s["date"], y=s["position"], mode="markers+lines", line=dict(color=ACCENT, width=1),
                               marker=dict(size=4), name="Rank",
                               hovertemplate="%{x|%d %b %Y}<br>Rank %{y}<extra></extra>", connectgaps=False))
    re = events[events["is_reentry"]]
    fig.add_trace(go.Scatter(x=re["reentry_date"], y=re["reentry_position"], mode="markers", name="Re-entry",
                             marker=dict(symbol="diamond", size=11, color=AMBER, line=dict(color=TEXT, width=1)),
                             hovertemplate="Re-entry %{x|%d %b %Y}<br>rank %{y}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="Rank (1 = top)", range=[51, 1])
    fig.update_layout(title=f"Chart history — {song_label}", xaxis_title="Date")
    return fig


def trajectory_figure(traj: pd.DataFrame, ev: pd.Series) -> go.Figure:
    """Rank trajectory with RE-ENTRY / PEAK / EXIT annotations and the selected stint shaded."""
    fig = go.Figure()
    t = traj.sort_values("date")
    fig.add_trace(go.Scatter(x=t["date"], y=t["rank_plot"], mode="lines+markers", name="Rank (51 = off chart)",
                             line=dict(color=MUTED, width=1.5, shape="linear"), marker=dict(size=5, color=np.where(
                                 t["on_chart"], ACCENT, "#3A4560")),
                             customdata=np.stack([t["position"].fillna(-1), t["popularity"].fillna(-1)], axis=1),
                             hovertemplate="%{x|%d %b %Y}<br>%{customdata[0]:.0f} (−1 = off chart)<extra></extra>"))
    sel = t[t["in_selected_stint"]]
    if len(sel):
        fig.add_trace(go.Scatter(x=sel["date"], y=sel["rank_plot"], mode="lines+markers", name="Selected comeback",
                                 line=dict(color=ACCENT, width=3), marker=dict(size=7)))
        fig.add_vrect(x0=sel["date"].min(), x1=sel["date"].max(), fillcolor=ACCENT, opacity=0.08, line_width=0)
    marks = [("RE-ENTRY", ev["event_date"], ev["event_position"], TEAL, "triangle-up"),
             ("PEAK", ev["peak_date"], ev["peak_rank"], AMBER, "star"),
             ("EXIT", ev["end_date"], ev["exit_position"], ROSE, "x")]
    for i, (name, d, y, c, sym) in enumerate(marks):
        fig.add_trace(go.Scatter(x=[d], y=[y], mode="markers+text", name=name, text=[name],
                                 textposition=["bottom center", "top center", "bottom center"][i],
                                 textfont=dict(color=c, size=12), marker=dict(size=14, color=c, symbol=sym,
                                                                              line=dict(color=TEXT, width=1)),
                                 hovertemplate=f"{name}<br>%{{x|%d %b %Y}}<br>rank %{{y:.0f}}<extra></extra>"))
    fig.update_yaxes(autorange=False, range=[OFF_CHART_RANK + 2, 0], title="Rank (1 = top; 51 = off chart)")
    fig.update_layout(title="Rank trajectory around the comeback", xaxis_title="Observed snapshot date", height=430)
    return fig


def popularity_figure(traj: pd.DataFrame, ev: pd.Series) -> go.Figure:
    t = traj.dropna(subset=["popularity"]).sort_values("date")
    if t.empty:
        return _empty("Popularity trajectory")
    fig = go.Figure(go.Scatter(x=t["date"], y=t["popularity"], mode="lines+markers", line=dict(color=VIOLET, width=2),
                               marker=dict(size=5), name="Popularity",
                               hovertemplate="%{x|%d %b %Y}<br>popularity %{y}<extra></extra>"))
    if pd.notna(ev.get("baseline_popularity")):
        fig.add_hline(y=ev["baseline_popularity"], line_dash="dot", line_color=MUTED,
                      annotation_text=f"pre-comeback baseline {ev['baseline_popularity']:.0f}", annotation_font_color=MUTED)
    fig.add_vline(x=ev["event_date"], line_dash="dash", line_color=TEAL)
    fig.update_layout(title="Popularity trajectory (platform popularity score)", xaxis_title="Observed snapshot date",
                      yaxis_title="Popularity (0-100)", height=320)
    return fig


def scatter_xy(df: pd.DataFrame, x: str, y: str, title: str, xl: str, yl: str, hover: str, size: str | None = None,
               color: str | None = None, trend: bool = True) -> go.Figure:
    d = df.dropna(subset=[x, y]).replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y])
    if size is not None:
        d = d.assign(**{size: d[size].fillna(d[size].median() if d[size].notna().any() else 1)})
    if len(d) < 3:
        return _empty(title)
    fig = px.scatter(d, x=x, y=y, size=size, color=color, hover_name=hover, title=title, opacity=0.75,
                     labels={x: xl, y: yl}, size_max=16, color_discrete_map=PATTERN_COLORS)
    if trend and d[x].nunique() > 1:
        k, b = np.polyfit(d[x], d[y], 1)
        xs = np.linspace(d[x].min(), d[x].max(), 20)
        fig.add_trace(go.Scatter(x=xs, y=k * xs + b, mode="lines", name="Linear trend",
                                 line=dict(color=AMBER, dash="dash"), hoverinfo="skip"))
    return fig


def band_histogram(df: pd.DataFrame, title: str) -> go.Figure:
    d = df.dropna(subset=["fandom_score"])
    if d.empty:
        return _empty(title)
    fig = px.histogram(d, x="fandom_score", nbins=25, title=title, color_discrete_sequence=[TEAL],
                       labels={"fandom_score": "Fandom Intensity Proxy (0-100)"})
    fig.update_yaxes(title="Count")
    return fig


def group_box(re: pd.DataFrame, by: str, title: str, label: str, violin: bool = False, order: list | None = None) -> go.Figure:
    d = re.dropna(subset=["momentum_score", by])
    if d.empty:
        return _empty(title)
    fn = px.violin if violin else px.box
    kw = dict(box=True, points="all") if violin else dict(points="outliers")
    fig = fn(d, x=by, y="momentum_score", color=by, title=title, hover_name="song_label",
             labels={by: label, "momentum_score": "Momentum score"}, category_orders={by: order} if order else None, **kw)
    fig.update_layout(showlegend=False)
    return fig


def group_bar(summary: pd.DataFrame, by: str, metric: str, title: str, ytitle: str, label: str, pct: bool = False) -> go.Figure:
    d = summary.dropna(subset=[metric])
    if d.empty:
        return _empty(title)
    vals = d[metric] * (100 if pct else 1)
    fig = go.Figure(go.Bar(x=d[by].astype(str), y=vals, marker_color=ACCENT, text=[f"n={int(n)}" for n in d["n_songs"]],
                           textposition="outside", hovertemplate="%{x}<br>" + ytitle + ": %{y:.2f}<br>%{text} songs<extra></extra>"))
    fig.update_layout(title=title, xaxis_title=label, yaxis_title=ytitle)
    return fig


def corr_heatmap(corr: pd.DataFrame) -> go.Figure:
    if corr.empty:
        return _empty("Spearman correlations")
    fig = px.imshow(corr.round(2), text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto",
                    title="Spearman correlation between comeback metrics")
    fig.update_layout(height=520)
    return fig


def snapshot_timeline(snaps: pd.DataFrame, gaps: pd.DataFrame) -> go.Figure:
    raw = snaps["raw_rows"] if "raw_rows" in snaps.columns else snaps["n_rows"]
    fig = go.Figure(go.Scatter(x=snaps["date"], y=raw, mode="markers",
                               marker=dict(size=5, color=np.where(raw == 50, ACCENT, ROSE)),
                               hovertemplate="%{x|%d %b %Y}<br>%{y} rows<extra></extra>", name="Observed snapshot"))
    if len(gaps):
        fig.add_trace(go.Scatter(x=gaps["date"], y=[0] * len(gaps), mode="markers", name="Calendar gap (no snapshot)",
                                 marker=dict(size=10, color=AMBER, symbol="x"),
                                 hovertemplate="%{x|%d %b %Y}<br>no snapshot<extra></extra>"))
    fig.add_hline(y=50, line_dash="dot", line_color=MUTED)
    fig.update_layout(title="Raw rows per observed date (50 = complete; red = anomaly) and calendar gaps",
                      xaxis_title="Date", yaxis_title="Rows", height=340)
    return fig
