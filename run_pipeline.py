"""CLI: run the full pipeline and write processed datasets + research summary to data/processed/."""
import argparse
from pathlib import Path

from analytics.ml_reentry import build_samples, train_and_evaluate
from analytics.pipeline import DEFAULT_MIN_ABSENT, export_processed, run_pipeline
from analytics.reporting import research_summary, summary_json

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/raw/Atlantic_South_Korea.csv")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--min-absent", type=int, default=DEFAULT_MIN_ABSENT)
    a = ap.parse_args()
    res = run_pipeline(a.input, a.min_absent)
    out = export_processed(res, a.out)
    ml = train_and_evaluate(build_samples(res.reentry.stints, res.appearances, res.cleaned.snapshots, a.min_absent))
    data, md = research_summary(res, ml)
    (Path(out) / "research_summary.md").write_text(md, encoding="utf-8")
    (Path(out) / "research_summary.json").write_text(summary_json(data), encoding="utf-8")
    print(f"Processed files written to {out}")
    print(f"Re-entry events: {len(res.reentries)} | songs: {res.songs.shape[0]} | artists: {res.artists.shape[0]}")
