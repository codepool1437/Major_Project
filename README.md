# Machine Learning-Based Early Detection of Temporal Anomalies in B2B Procurement

This repository contains the major-project implementation for detecting
unusually short customer-part purchase intervals in B2B procurement data.
It includes:

- a reproducible Python feature-engineering and scoring pipeline;
- threshold, detector, ranking, and SDG-evidence experiments;
- a Streamlit dashboard for investigation and event-history review; and
- retained result tables and figures used in the final report and IEEE paper.

The detector is a **decision-support system**. An alert identifies a purchase
pattern for human review; it does not prove fraud, waste, unnecessary
procurement, or an operational outcome.

## Project layout

```text
procurement-anomaly-detection/
├── app.py                         # Streamlit dashboard
├── requirements.txt               # pinned Python dependencies
├── README.md                      # detailed project instructions
├── src/                           # pipeline and experiment scripts
├── data/
│   ├── raw/                       # six yearly source CSV files
│   └── processed/                 # generated temporal features
├── results/                       # generated tables, reports, and figures
├── documentation/                 # project reviews and synopsis
└── runtime-validation-report.md   # reproducibility evidence
```

The final paper draft is
[`procurement-anomaly-detection/conference_paper.tex`](procurement-anomaly-detection/conference_paper.tex).

## Requirements

- Python 3.11 or newer. The project was validated with Python 3.14.
- Windows, macOS, or Linux.
- The six semicolon-delimited raw CSV files supplied with the project:
  `2024_full_anonymus_v2_final.csv` through
  `2029_full_anonymus_v2_final.csv`.

The raw files must be placed in
[`procurement-anomaly-detection/data/raw/`](procurement-anomaly-detection/data/raw/).
Do not rename the files unless the input-discovery logic is updated as well.

## Installation

From the repository root:

### Windows PowerShell

```powershell
cd procurement-anomaly-detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS/Linux

```bash
cd procurement-anomaly-detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned dependencies are Streamlit, pandas, NumPy, Altair, and
Matplotlib. The experiment scripts that use scikit-learn require the
corresponding scientific Python packages available in the validated project
environment.

## Run the dashboard with retained results

The repository contains retained processed data and result files, so the
dashboard can be opened without rerunning the full pipeline:

```bash
cd procurement-anomaly-detection
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally
`http://localhost:8501`.

The dashboard has three areas:

1. **Overview** — summary counts and alert-volume views.
2. **Alerts & Investigation** — filters, ranked alerts, severity, and
   selected-alert details.
3. **Customer-Part Event Inspector** — complete chronological history and
   inter-purchase interval chart for a selected customer-part pair.

The active alert rule is:

```text
IPI_DAYS <= 0.30 * HIST_MEDIAN_IPI
```

The retained dashboard result contains 13,861 alerts from 64,892 eligible
events, an alert rate of 21.36%.

## Reproduce the complete workflow

Run these commands from `procurement-anomaly-detection/` after placing the
raw files in `data/raw/`.

### 1. Inspect and build temporal features

```bash
python src/inspect_data.py
python src/build_ipi.py
python src/build_temporal_features.py
```

This creates:

- `data/processed/purchase_intervals.csv`
- `data/processed/temporal_event_features.csv`
- `data/processed/model_ready_events.csv`

### 2. Score, rank, and evaluate alerts

```bash
python src/score_early_reorders.py
python src/improve_early_reorder_scoring.py
python src/analyze_thresholds.py
python src/train_and_evaluate.py
python src/evaluate_severity.py
python src/inspect_early_reorders.py
```

### 3. Run detector, ranking, comparative, and SDG analyses

```bash
python src/compare_temporal_detectors.py
python src/ablate_temporal_ranking.py
python src/phase3_comparative_analysis.py
python src/phase4_sdg_mapping.py
python src/audit_real_world_comparison.py
python src/robustness_experiments.py
```

The retained artifact inventory is in
[`procurement-anomaly-detection/results/artifact_manifest.md`](procurement-anomaly-detection/results/artifact_manifest.md).
The consolidated interpretation is in
[`procurement-anomaly-detection/results/algorithm_experiment_report.md`](procurement-anomaly-detection/results/algorithm_experiment_report.md).

The robustness experiment additionally audits duplicate/order-line losses,
date distributions, customer concentration, cross-year intervals,
minimum-history sensitivity, a matched fixed-day baseline, and five
rolling-origin folds. It writes `results/minhist.csv` and
`results/rolling_all.csv`. Synthetic injections are consistency checks, not
expert-validated anomaly labels.

## Important result files

| File | Purpose |
|---|---|
| `results/improved_ranked_early_reorders.csv` | Ranked alert output used by the dashboard |
| `results/final_threshold_comparison.csv` | Alert-volume comparison for 50%, 40%, 30%, and 20% thresholds |
| `results/algorithm_comparison_metrics.csv` | Machine-readable detector-comparison metrics |
| `results/ablation_metrics.csv` | Ranking-component ablation results |
| `results/sdg_evidence_mapping.csv` | Evidence mapping for SDG 12 and limited SDG 9 relevance |
| `results/real_world_comparison_status.csv` | Audit of available real-world baseline/outcome evidence |
| `results/minhist.csv` | Minimum-history sensitivity results |
| `results/rolling_all.csv` | Fixed-day, injection-scenario, and rolling-origin robustness results |
| `results/figures/threshold_vs_alert_volume.png` | Threshold sensitivity figure |
| `results/figures/detector_metrics_comparison.png` | Detector comparison figure |

## Reproducibility checks

The validation report is available at
[`procurement-anomaly-detection/runtime-validation-report.md`](procurement-anomaly-detection/runtime-validation-report.md).
It records dependency installation, imports, compilation, Streamlit startup,
health-check evidence, and dashboard interaction checks.

For a quick dashboard health check:

```bash
python -m streamlit run app.py --server.headless=true --server.port=8501
```

Then visit:

```text
http://localhost:8501/_stcore/health
```

A healthy application returns HTTP 200 with the body `ok`.

## Interpretation and limitations

- The raw data contains customer, item, project, machine, date, location,
  description, and exact-price fields, but does not provide verified fraud
  labels, quantities, inventory levels, waste measurements, emissions, or
  intervention outcomes.
- Detector metrics use a chronological split and controlled synthetic
  early-reorder cases. These cases are useful for method comparison but are
  not real-world anomaly labels.
- The natural holdout is not proven normal; its alert rate should not be
  interpreted as a measured false-positive rate.
- The project therefore supports prioritised review and temporal-pattern
  analysis. It does not claim measured reductions in cost, waste, inventory,
  or emissions.
- The robustness audit found strongly non-uniform weekday counts and substantial
  customer concentration. Dates are therefore treated as an ordered timeline,
  and alert concentration should be considered when interpreting aggregate
  results.
- SDG 12 is treated as the primary relevance connection, with SDG 9 as a
  limited secondary connection to data-driven industrial systems. Field
  evaluation with expert labels and before/after operational measurements
  would be required to establish real-world impact.

## Temporary paper-figure viewer

The temporary `temporary_figures_app.py` viewer was used to inspect and
download paper figures from retained CSV results. It is not required for the
main dashboard or the reproducible pipeline and may be deleted after the
figures have been captured.

## License and data notice

This repository is an academic project. The anonymised procurement data is
provided for project use and should not be redistributed outside the
permissions under which it was supplied.
