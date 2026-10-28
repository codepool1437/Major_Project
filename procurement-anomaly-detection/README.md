# Procurement Anomaly Detection

A Streamlit dashboard and reproducible Python pipeline for identifying
unusually short customer-part purchase intervals. The detector is a
decision-support tool: alerts require human review and are not confirmed
anomalies.

## Prerequisites

- Python 3.11 or newer (the validated environment used Python 3.14)
- The six raw yearly CSV files in `data/raw/`
- Windows, macOS, or Linux

Install the pinned dashboard dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Dataset setup

Place these semicolon-delimited files in `data/raw/`:

`2024_full_anonymus_v2_final.csv` through `2029_full_anonymus_v2_final.csv`.

The pipeline uses customer ID, item ID, and request date for temporal
features. `PRICE_EXACT` is retained in the raw source but is not used to claim
cost savings.

## Reproduce the processing pipeline

Run from the project directory:

```bash
python src/inspect_data.py
python src/build_ipi.py
python src/build_temporal_features.py
```

The feature builder writes:

- `data/processed/purchase_intervals.csv`
- `data/processed/temporal_event_features.csv`
- `data/processed/model_ready_events.csv`

The original feature builder is functionally correct but can be slow on the
full dataset because of expanding historical MAD calculations.

## Reproduce scoring and evaluation

```bash
python src/score_early_reorders.py
python src/improve_early_reorder_scoring.py
python src/analyze_thresholds.py
python src/train_and_evaluate.py
python src/evaluate_severity.py
python src/inspect_early_reorders.py
```

## Reproduce detector comparisons and reports

```bash
python src/compare_temporal_detectors.py
python src/ablate_temporal_ranking.py
python src/phase3_comparative_analysis.py
python src/phase4_sdg_mapping.py
python src/audit_real_world_comparison.py
python src/robustness_experiments.py
```

The retained outputs and figures are listed in
`results/artifact_manifest.md`. The consolidated interpretation is in
`results/algorithm_experiment_report.md`.

The robustness script also audits raw-file hashes, duplicate order lines,
same-day event collapsing, weekday/date distributions, customer
concentration, year-boundary intervals, minimum-history sensitivity, a
budget-matched fixed-day baseline, and five rolling-origin folds. It writes:

- `results/minhist.csv`
- `results/rolling_all.csv`

Its injected cases are synthetic consistency checks and must not be presented
as expert-validated anomaly labels.

## Launch the dashboard

```bash
python -m streamlit run app.py
```

Open the displayed local URL, normally `http://localhost:8501`.

The dashboard reads:

- `results/improved_ranked_early_reorders.csv`
- `data/processed/temporal_event_features.csv`

It provides Overview, Alerts & Investigation, and Customer-Part Event
Inspector tabs. Sidebar filters control the alert table and synchronize the
event inspector. The event inspector can display the full pair history and
inter-purchase interval chart.

## Reproducibility and interpretation

The active dashboard rule flags events where:

`IPI_DAYS <= 0.30 * HIST_MEDIAN_IPI`

The current retained result is 13,861 alerts from 64,892 eligible events
(21.36%). Experimental metrics use a chronological split and synthetic
early-reorder injections; they are not real-world accuracy or sustainability
impact measurements. Do not claim reductions in cost, waste, inventory, or
emissions without expert-reviewed outcomes and before/after field data.
