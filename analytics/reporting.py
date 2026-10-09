"""KPI computation and research-paper summary export. All numbers come from computed data."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .content_analysis import compare_groups, compare_rates, kruskal_groups, numeric_associations
from .documentation import LIMITATIONS, methodology_sections
from .filters import FilterState, apply_filters
from .insights import generate_insights
from .recommendations import build_recommendations


def compute_kpis(fd) -> dict:
    """Headline KPIs for the (filtered) selection."""
    re = fd.reentries
    arts = fd.artists[fd.artists["reentry_count"] >= 3].dropna(subset=["fandom_score"])
    top_art = arts.sort_values("fandom_score", ascending=False).iloc[0] if len(arts) else None
    n = len(re)
    return {
        "playlist_days": int(fd.snapshots["is_valid_snapshot"].sum()),
        "unique_songs": int(fd.rows["song_id"].nunique()),
        "unique_artists": int(fd.rows["artist"].nunique()),
        "total_reentries": int(n),
        "songs_with_reentries": int(re["song_id"].nunique()),
        "avg_momentum": float(re["momentum_score"].mean()) if n else np.nan,
        "avg_retention_days": float(re["retention_days"].mean()) if n else np.nan,
        "median_retention_days": float(re["retention_days"].median()) if n else np.nan,
        "avg_retention_snapshots": float(re["retention_snapshots"].mean()) if n else np.nan,
        "right_censored_reentries": int(re["is_right_censored"].sum()) if n else 0,
        "top_fandom_artist": None if top_art is None else top_art["artist"],
        "top_fandom_score": None if top_art is None else float(top_art["fandom_score"]),
    }


def artist_views(artists: pd.DataFrame, min_events: int = 3, n: int = 10) -> dict:
    """Four artist leaderboards (frequency, momentum, fandom proxy, retention) with a minimum sample."""
    a = artists[artists["reentry_count"] >= min_events]
    cols = ["artist", "unique_songs", "reentry_count", "mean_momentum", "median_momentum", "mean_retention_days",
            "fandom_score", "confidence"]
    return {
        "Top by frequency": artists.sort_values("reentry_count", ascending=False).head(n)[cols],
        "Top by momentum": a.sort_values("mean_momentum", ascending=False).head(n)[cols],
        "Top by fandom proxy": a.dropna(subset=["fandom_score"]).sort_values("fandom_score", ascending=False).head(n)[cols],
        "Top by retention": a.sort_values("mean_retention_days", ascending=False).head(n)[cols],
    }


def content_findings(fd) -> list[dict]:
    """Statistical comparisons used in the paper (computed, with sample sizes)."""
    ev, songs, out = fd.events, fd.songs, []
    for by, a, b in (("album_type", "single", "album"), ("explicit_label", "Explicit", "Clean")):
        if by not in ev.columns:
            continue
        for metric in ("momentum_score", "retention_days"):
            out.append({"comparison": f"{a} vs {b}", "type": "Mann-Whitney U", **compare_groups(ev, by, a, b, metric)})
        out.append({"comparison": f"{a} vs {b}", "type": "Fisher exact (re-entry rate)", "metric": "reentry_rate",
                    **compare_rates(songs, by, a, b)})
    for by in ("album_size_bin", "duration_bin"):
        if by in ev.columns:
            for metric in ("momentum_score", "retention_days"):
                out.append({"comparison": by, "type": "Kruskal-Wallis", **kruskal_groups(ev, by, metric)})
    assoc = numeric_associations(songs, ev)
    out += [{"comparison": f"{r.attribute} vs {r.outcome}", "type": "Spearman", "n": r.n, "rho": r.spearman_rho,
             "p_value": r.p_value} for r in assoc.itertuples()]
    return out


def research_summary(result, ml: dict | None = None) -> tuple[dict, str]:
    """Build (dict, markdown) research-support summary for the FULL dataset (no filters)."""
    fd = apply_filters(result, FilterState())
    k = compute_kpis(fd)
    q = result.cleaned.quality
    re = fd.reentries
    cols = ["song_label", "reentry_date", "reentry_position", "peak_rank", "retention_days", "retention_snapshots",
            "momentum_score"]
    top_songs = (re.sort_values("momentum_score", ascending=False).head(10)[cols]
                 .assign(reentry_date=lambda d: d["reentry_date"].dt.date.astype(str)))
    longest = (re.sort_values("retention_days", ascending=False).head(5)[cols + ["is_right_censored"]]
               .assign(reentry_date=lambda d: d["reentry_date"].dt.date.astype(str)))
    views = artist_views(result.artists)
    pat = re["pattern"].value_counts().to_dict()
    recs = build_recommendations(fd)
    ins = generate_insights(fd)
    cf = content_findings(fd)
    ledger = result.cleaned.removal_ledger
    data = {
        "dataset_statistics": q, "removal_ledger": ledger.to_dict("records"), "kpis": k,
        "reentry_definition": {"min_absent_snapshots": result.min_absent_snapshots,
                               "total_stints": int(len(result.events)), "reentry_events": int(len(re))},
        "momentum_pattern_counts": pat, "pattern_thresholds": result.pattern_thresholds,
        "weight_sensitivity": result.sensitivity,
        "momentum_summary": re[["momentum_score", "retention_days", "retention_snapshots", "rank_jump", "peak_improvement",
                                "popularity_lift", "recovery_speed", "days_to_peak"]].describe().round(3).to_dict(),
        "retention_findings": {"median_retention_days": k["median_retention_days"], "mean_retention_days": k["avg_retention_days"],
                               "right_censored_lower_bound_events": k["right_censored_reentries"],
                               "longest_retention_events": longest.to_dict("records")},
        "top_comeback_songs": top_songs.to_dict("records"),
        "artist_leaderboards": {name: df.round(3).to_dict("records") for name, df in views.items()},
        "content_findings": cf, "insights": ins,
        "recommendations": recs["table"].to_dict("records"), "recommendation_notes": recs["notes"],
        "recommendation_thresholds": recs["thresholds"], "limitations": LIMITATIONS,
    }
    if ml:
        data["reentry_forecast_model_EXPERIMENTAL"] = {k_: (v if not isinstance(v, (pd.DataFrame, pd.Timestamp)) else str(v))
                                                       for k_, v in ml.items() if k_ not in ("model", "metrics")}
        if "metrics" in ml:
            data["reentry_forecast_model_EXPERIMENTAL"]["metrics"] = ml["metrics"].round(4).to_dict("records")
    sens = result.sensitivity
    md = ["# Research Support Summary — South Korea Chart Momentum Intelligence", "",
          "_Generated automatically from the processed data; no figures are hand-entered._", "",
          "## Dataset statistics", *[f"- {a}: {b}" for a, b in q.items()], "",
          "### Row reconciliation (raw → clean)", ledger.to_markdown(index=False), "",
          "## Methodology summary", *[f"**{t}.** {b}" for t, b in methodology_sections(result.min_absent_snapshots).items()], "",
          "## KPI values", *[f"- {a}: {b if not isinstance(b, float) else round(b, 2)}" for a, b in k.items()], "",
          f"Re-entry definition: >= {result.min_absent_snapshots} observed valid absences; {len(result.events)} total stints, "
          f"{len(re)} confirmed re-entries.", "",
          "## Momentum findings",
          f"- Pattern counts (re-entries): {pat}",
          f"- Thresholds: { {a: round(b, 2) for a, b in result.pattern_thresholds.items()} }",
          f"- Robustness: Spearman vs equal weights {sens['spearman_vs_equal_weights']:.3f}; across {sens['n_draws']} random "
          f"weight vectors median {sens['perturbation_median_rho']:.3f}, minimum {sens['perturbation_min_rho']:.3f}; "
          f"top-10 overlap with equal weights {sens['top10_overlap_vs_equal']:.0%}", "",
          "## Retention findings",
          f"- Median retention {k['median_retention_days']:.1f} days; mean {k['avg_retention_days']:.1f} days "
          f"({k['avg_retention_snapshots']:.1f} snapshots). {k['right_censored_reentries']} re-entries are right-censored "
          "(retention is a lower bound).", "", longest.round(2).to_markdown(index=False), "",
          "## Top comeback songs (by momentum)", top_songs.round(2).to_markdown(index=False), "",
          "## Artist findings (>= 3 re-entries for momentum/fandom/retention views)",
          *[f"### {name}\n{df.round(2).to_markdown(index=False)}\n" for name, df in views.items()],
          "## Content findings (associations, not causation)",
          pd.DataFrame(cf).round(4).to_markdown(index=False), "",
          "## Major findings", *[f"- **{i['title']}** ({i['category']}): {i['text']}" for i in ins], "",
          "## Recommendations", *[f"- {n}" for n in recs["notes"]],
          recs["table"].round(1).to_markdown(index=False) if len(recs["table"]) else "_No rule triggered._", "",
          "## Limitations", *[f"- {l}" for l in LIMITATIONS]]
    if ml:
        md += ["", "## Re-entry forecast (EXPERIMENTAL / descriptive)", f"- Supported: {ml.get('supported')}; reliable: {ml.get('reliable')}",
               f"- Test base rate: {ml.get('prevalence_test', float('nan')):.3%}", f"- {ml.get('reason')}"]
        if "metrics" in ml:
            md += [ml["metrics"].round(3).to_markdown(index=False)]
    return data, "\n".join(md)


def summary_json(data: dict) -> str:
    return json.dumps(data, indent=2, default=str)
