# South Korea Chart Momentum Intelligence

**Comeback Momentum, Chart Re-Entry, and Fandom Intensity Analysis of the South Korea Top 50 Playlist**
Prepared for Atlantic Recording Corporation.

An analytical intelligence platform (Streamlit + Plotly) showing **how songs regain momentum after leaving the South Korea
Top 50**:

`DATA → VALIDATION → RE-ENTRY DETECTION → MOMENTUM ANALYSIS → FANDOM PROXY → CONTENT ANALYSIS → STRATEGIC RECOMMENDATIONS`

Every number is computed from the dataset. Nothing is hard-coded; no demo data is used.

## Business problem
Which songs return, how strongly and how long they stay; how quickly rank is recovered; whether singles or album tracks,
explicit or clean songs, album size or duration are associated with comebacks; which artists show repeated fandom-driven
reactivation; and where promotional attention should go.

## Dataset
`data/raw/Atlantic_South_Korea.csv` (read-only): `date, position, song, artist, popularity, duration_ms, album_type,
total_tracks, is_explicit, album_cover_url`. Dates are `dd-mm-yyyy` (parsed explicitly). All derived tables are written to
`data/processed/`; the raw file is never modified (a test checks its hash).

Audit of the supplied file (live on the **Data Quality** page): 27,800 raw rows → 27,750 clean rows over 555 observed dates
(2024-05-18 → 2025-11-27); 4 missing calendar dates; one date (01-03-2025) holds two stacked 50-row snapshots (the less
similar one is set aside, logged and reconciled: raw − rejected − set-aside − duplicates = clean); 537 songs
(song + artist) and 193 artists.

## Methodology (full text on the in-app *Methodology & Limitations* page)
**Vocabulary:** *observed snapshot* (a playlist date in the data; valid if it lists 50 unique songs) · *calendar day* ·
*calendar gap* (no snapshot — never an exit, never presence) · *observed absence* (valid snapshot without the song) ·
*right-censored* (still on chart at the last snapshot → retention is a lower bound) · *left-censored* (already on chart at
the first snapshot → unscored).

**Re-entry definition.** appears → leaves a valid snapshot → stays absent for ≥ *N* consecutive valid observed snapshots →
returns. *N* is the sidebar threshold (default 3; 1 = every dropout, mostly rank 49–50 flicker). Missing calendar dates can
never create an exit or re-entry. With the default, 392 re-entries (929 stints = 537 first stints + 392 re-entries) occur in
166 songs; at *N* = 1 there would be 878.

**Momentum metrics** per stint: rank jump, peak rank, peak improvement, popularity lift, popularity change at entry, days /
snapshots to peak, rank recovery, recovery speed, decay speed, volatility, **retention_snapshots** (observed snapshots in the
stint) and **retention_days** (calendar days from comeback to last observed appearance, inclusive; calendar gaps inside the
span are *not* counted as observed, and right-censored stints are flagged as lower bounds).

**Momentum score (0–100):** weighted mean of percentile ranks (rank jump 20 %, peak improvement 20 %, popularity lift 20 %,
recovery speed 15 %, retention 25 %), ranked **within stint type** (re-entries vs first entries, whose baselines differ);
undefined components are dropped and weights renormalised. 50 = a typical re-entry. Robustness: Spearman ≈ 0.99 against
equal weights and across 200 random weight vectors (shown in-app). It is an analytical composite, not a measure of fandom.

**Fandom Intensity Proxy** (song and artist, 0–100): percentile-weighted composite of re-entry count, momentum, popularity
lift, rank recovery, recovery speed, retention and comeback sharpness, shrunk toward the median entity by sample size,
quantile bands (P30/60/80/95), with sample size and confidence on every score. *A behavioural proxy — not fandom size, fan
count or fandom strength.*

**Content analysis:** group summaries (re-entry rate, re-entries per 100 exposure days, momentum, retention days) with
Mann-Whitney U + Cliff's δ, Fisher exact, Kruskal-Wallis, Spearman; small groups flagged; "associated with" language only.

**Recommendations:** six quantile-based rules on the *selected* data, each with evidence (counts, momentum, retention days).

**Experimental ML forecast:** P(exited song re-enters within 30 days); chronological split with embargo, song-clustered
bootstrap CI, base rate and a recency-only baseline. On this data it discriminates well (AUC ≈ 0.90) but **does not beat the
trivial recency baseline**, so it is labelled experimental/descriptive. The core dashboard does not depend on it.

## Architecture
```
app.py                  Streamlit entry: navigation, global filters, caching, error handling
run_pipeline.py         CLI: write processed datasets + research summary to data/processed/
analytics/              UI-free analytics layer
  utils · validation · preprocessing (cleaning, snapshot resolution, row ledger)
  reentry (stints / re-entries) · momentum (metrics, score, patterns, lifecycle, trajectory, robustness)
  fandom (song & artist proxy) · content_analysis (comparisons & tests) · insights · recommendations · narrative
  filters (selection-level recomputation) · reporting (KPIs, artist views, research summary) · documentation
  ml_reentry (experimental forecast) · pipeline (orchestration)
views/                  one module per page (named views/ so Streamlit does not auto-discover a pages/ folder)
ui/                     theme, components, Plotly chart builders
tests/                  pytest suite (73 tests)
data/raw|processed/     input / generated outputs
.streamlit/config.toml  dark theme and headless server settings
Dockerfile · .dockerignore · requirements.txt
```

## Dashboard pages
Executive Overview · Comeback Intelligence · Momentum Lab · Fandom Intelligence · Content Analysis · Strategic
Recommendations · Re-entry Forecast (experimental) · Data Quality · Methodology & Limitations · Data & Research Export.

**Global filters** (date range, artist, song, album type, explicit/clean, minimum re-entries, momentum range, reset) apply to
KPIs, charts, leaderboards, statistics, recommendations and downloads. Selection-level statistics are recomputed from the
filtered events; fandom scores stay normalised on the full dataset so they remain comparable. Album-type and explicit filters
use each song's most common value so complete song histories are kept. Empty selections show explicit messages.

**Downloads:** cleaned playlist, re-entry events, momentum events, song analytics, artist analytics, recommendations,
snapshot validation, row reconciliation, research summary (Markdown + JSON) — all event exports include `retention_days`.

## Install and run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py          # optional: regenerates data/processed/* and research_summary.md/json
streamlit run app.py
```
Runs offline, uses relative paths only, needs no secrets. Another CSV with the same schema can be uploaded from the sidebar.

## Testing
```bash
python -m pytest -q
```
73 tests: date parsing, song IDs, duplicate/stacked-snapshot handling, snapshot validation, calendar gaps, re-entry detection
(one-snapshot dropout, multi-snapshot dropout, missing calendar dates, multiple re-entries, never re-enters, present at first
observation, threshold behaviour), retention days and right-censoring, momentum metric edge cases (no rank improvement, unchanged
popularity, negative rank jump, zero-day recovery, very short retention, no inf/NaN), score and robustness, fandom shrinkage,
filters (propagation to song/artist statistics, reset), recommendations, count reconciliation, raw-data immutability,
and ML temporal split / no-leakage (features recomputed on truncated data are identical).

## Docker
```bash
docker build -t chart-momentum .
docker run -p 8501:8501 chart-momentum        # http://localhost:8501
```

## Deployment
Streamlit Community Cloud: push the repo, set the entry point to `app.py` (Python 3.10+); `requirements.txt` is used
automatically. Keep `data/raw/Atlantic_South_Korea.csv` in the repository or upload a CSV from the sidebar.

## Screenshots
No screenshots are bundled because none have been captured; add them to `assets/` after running the app.

## Limitations
Observed chart behaviour only; the fandom proxy is not fan size; correlation ≠ causation; missing calendar dates affect time
interpretation (never treated as exits); playlist position is not total fandom; popularity is platform/API-derived; external
events (releases, tours, TV, promotion) are not observed; the Top-50 cut-off censors information and exits near rank 50 can be
rank noise; left/right-censored stints are incomplete (retention is a lower bound); momentum is relative (percentile-based);
events from one song/artist are not independent; the chosen 01-03-2025 snapshot was a near-tie; the ML forecast is experimental.

## Future work
Release/event calendar enrichment, survival models for time-to-re-entry and retention (censoring-aware), multi-market
comparison, scheduled data refresh, richer forecast features.
