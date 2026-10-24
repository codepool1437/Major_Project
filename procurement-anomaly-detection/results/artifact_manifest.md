# Experiment Artifact Manifest

This directory preserves the outputs from the completed pipeline, algorithm
experiments, ablation analysis, and Phase 3 comparative analysis. Existing
artifacts should not be overwritten or deleted unless a new run is explicitly
identified with a new configuration or archived separately.

## Processed data

- `../data/processed/purchase_intervals.csv`
- `../data/processed/temporal_event_features.csv`
- `../data/processed/model_ready_events.csv`

## Pipeline diagnostics and profiles

- `missing_values_report.csv`
- `ipi_summary.csv`
- `customer_item_history_profile.csv`

## Baseline and dashboard scoring outputs

- `ranked_early_reorders.csv`
- `early_reorder_summary.csv`
- `improved_ranked_early_reorders.csv`
- `improved_scoring_summary.csv`
- `threshold_analysis.csv`
- `early_reorder_review_sample.csv`

## Controlled evaluation outputs

- `model_scored_events.csv`
- `ranked_anomaly_candidates.csv`
- `controlled_evaluation_metrics.csv`
- `controlled_evaluation_cases.csv`
- `severity_experiment.csv`

## Detector comparison and ablation outputs

- `algorithm_comparison_metrics.csv`
- `algorithm_comparison_cases.csv`
- `ablation_metrics.csv`
- `ablation_cases.csv`
- `final_threshold_comparison.csv`
- `sdg_evidence_mapping.csv`
- `sdg_quantitative_evidence.csv`
- `real_world_comparison_status.csv`

## Figures and interpretation

- `figures/threshold_vs_alert_volume.png`
- `figures/detector_metrics_comparison.png`
- `algorithm_experiment_report.md`
- `../README.md`
- `../runtime-validation-report.md`

## Reproduction scripts

- `../src/build_ipi.py`
- `../src/build_temporal_features.py`
- `../src/score_early_reorders.py`
- `../src/improve_early_reorder_scoring.py`
- `../src/analyze_thresholds.py`
- `../src/train_and_evaluate.py`
- `../src/evaluate_severity.py`
- `../src/inspect_early_reorders.py`
- `../src/compare_temporal_detectors.py`
- `../src/ablate_temporal_ranking.py`
- `../src/phase3_comparative_analysis.py`
- `../src/phase4_sdg_mapping.py`
- `../src/audit_real_world_comparison.py`

## Preservation note

The files listed here are the current retained artifacts. The clean Phase 2
and Phase 3 outputs were regenerated from the current processed data and are
documented in `algorithm_experiment_report.md`. The report records the
configuration, denominators, split, synthetic construction, metrics, edge
cases, limitations, and output filenames needed to interpret these artifacts.
