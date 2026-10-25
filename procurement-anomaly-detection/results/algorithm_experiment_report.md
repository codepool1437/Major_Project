# Temporal Detector Experiments and Ablation Report

Generated: 2026-10-24

## Scope

This report records the comparison of the current median-threshold baseline,
a robust median-MAD detector, Isolation Forest, and Local Outlier Factor (LOF).
It also records an ablation of the ranking components used by the current
dashboard scoring script.

The experiments are separate from the Streamlit dashboard. They do not replace
the canonical dashboard outputs.

## Verification of the comparison implementation

Source: `src/compare_temporal_detectors.py`

- Input: `data/processed/model_ready_events.csv`
- Eligible-event filter: `HISTORY_COUNT >= 3`, nonnegative `IPI_DAYS`, and
  positive `HIST_MEDIAN_IPI`
- Eligible events: 64,892
- Chronological split: first 51,913 events for training and final 12,979
  events for the natural holdout
- Train-only imputation: Isolation Forest and LOF fill missing feature values
  using medians calculated from the training partition
- Model features: current IPI, historical median and MAD, recent median,
  historical/recent ratios, and history count
- Synthetic cases: copies of the natural holdout with
  `IPI_DAYS = injection_ratio * HIST_MEDIAN_IPI`
- Injection ratios: 0.50, 0.30, 0.20, and 0.10
- Synthetic cases do not modify historical features, history count, or the
  chronological training data
- Isolation Forest and LOF thresholds are derived from training scores
- The experiment output was checked for syntax errors, diagnostics, missing
  values, and successful execution

The synthetic intervals are now not clipped to one day. Therefore the
injection ratios are exact, including for short historical medians.

## Comparison results

The complete machine-readable output is in
`results/algorithm_comparison_metrics.csv`. Controlled precision, recall, and
F1 use the synthetic cases as positive labels and natural holdout events as a
reference group. The natural holdout is not proven normal.

### Natural alert burden

| Method | Parameter | Natural alert rate |
|---|---|---:|
| Median threshold | 50% | 27.31% |
| Median threshold | 40% | 23.49% |
| Median threshold | 30% | 19.35% |
| Median threshold | 20% | 14.95% |
| Median-MAD | robust z >= 1 | 11.96% |
| Median-MAD | robust z >= 2 | 4.92% |
| Median-MAD | robust z >= 3 | 3.26% |
| Median-MAD | robust z >= 4 | 2.60% |
| Isolation Forest | training score 95th percentile | 6.60% |
| LOF | 35 neighbors, training score 95th percentile | 8.91% |

### Detection and controlled precision at the 30% injection ratio

| Method | Parameter | Detection rate | Controlled precision |
|---|---|---:|---:|
| Median threshold | 50% | 100.00% | 78.55% |
| Median threshold | 40% | 100.00% | 80.98% |
| Median threshold | 30% | 100.00% | 83.78% |
| Median threshold | 20% | 0.00% | 0.00% |
| Median-MAD | robust z >= 1 | 31.93% | 72.75% |
| Median-MAD | robust z >= 2 | 13.48% | 73.28% |
| Median-MAD | robust z >= 3 | 8.67% | 72.67% |
| Median-MAD | robust z >= 4 | 6.66% | 71.96% |
| Isolation Forest | training score 95th percentile | 7.24% | 52.31% |
| LOF | 35 neighbors, training score 95th percentile | 12.03% | 57.45% |

## Ablation analysis

Source: `src/ablate_temporal_ranking.py`

The alert rule is fixed at the current 30% median threshold for every
ablation. Only the ranking score changes:

- `threshold_only`: rank by interval ratio
- `earlyness_only`: rank by the monotonic earlyness score
- `robust_only`: rank by the capped MAD-based variability component
- `combined_70_30`: current 70% earlyness plus 30% robust variability score

All methods therefore produce the same alert count and natural alert rate.
The ablation tests whether the additional components improve prioritization
within the alert set.

### Ablation results

| Injection ratio | Ranking method | Natural alert rate | Detection rate | Recall@150 | Median flagged synthetic rank |
|---:|---|---:|---:|---:|---:|
| 0.50 | All methods | 19.35% | 0.00% | 0.00% | Not available |
| 0.30 | Threshold only | 19.35% | 100.00% | 0.00% | 8,977 |
| 0.30 | Earlyness only | 19.35% | 100.00% | 0.00% | 8,977 |
| 0.30 | Robust only | 19.35% | 100.00% | 0.92% | 8,311 |
| 0.30 | Combined 70/30 | 19.35% | 100.00% | 0.00% | 8,959 |
| 0.20 | Threshold only | 19.35% | 100.00% | 0.00% | 8,417 |
| 0.20 | Earlyness only | 19.35% | 100.00% | 0.00% | 8,430 |
| 0.20 | Robust only | 19.35% | 100.00% | 1.05% | 7,897 |
| 0.20 | Combined 70/30 | 19.35% | 100.00% | 0.00% | 8,553 |
| 0.10 | Threshold only | 19.35% | 100.00% | 0.00% | 7,828 |
| 0.10 | Earlyness only | 19.35% | 100.00% | 0.00% | 7,831 |
| 0.10 | Robust only | 19.35% | 100.00% | 1.05% | 7,646 |
| 0.10 | Combined 70/30 | 19.35% | 100.00% | 0.57% | 7,964 |

The 50% injection cases are not detected by the 30% alert rule because a
50%-of-median interval is not at or below 30% of the historical median. This
is expected and is not a failure of the implementation.

### Ablation interpretation

- The added components do not change alert membership; the fixed threshold
  determines all alert counts.
- Earlyness-only ranking is monotonic with interval ratio and adds no
  measurable information beyond threshold ranking.
- The robust component improves synthetic prioritization modestly in the
  tested top-150 measure.
- The current 70/30 combined score does not outperform robust-only ranking in
  this controlled ablation.
- The evidence does not justify claiming that the combined score is superior
  to the basic threshold or robust-only ranking.

## Recommended paper wording

The 30% median-threshold rule should remain the primary interpretable
detector. The median-MAD score is a useful robust comparison and may improve
prioritization under a fixed review budget. Isolation Forest and LOF were
included as unsupervised benchmarks but did not outperform the threshold
baseline in the controlled evaluation.

The ranking ablation indicates that the threshold determines alert membership,
while the robust variability component provides the only measurable ranking
benefit among the tested additions. The 70/30 combined score should be
described as a heuristic design choice, not as a validated improvement.

## Limitations

- No verified real-world anomaly labels are available.
- Natural holdout events are a reference population, not confirmed normal
  events.
- Synthetic cases are artificial interval replacements.
- The evaluation uses one chronological split.
- Isolation Forest and LOF settings were controlled comparison settings, not
  an exhaustive hyperparameter search.
- Recall@150 is a controlled prioritization measure and not production
  precision.
- Domain-expert review is still required before claiming operational accuracy.

## Artifacts

- `src/compare_temporal_detectors.py`
- `results/algorithm_comparison_metrics.csv`
- `results/algorithm_comparison_cases.csv`
- `src/ablate_temporal_ranking.py`
- `results/ablation_metrics.csv`
- `results/ablation_cases.csv`

## Phase 2 finalization

The downstream pipeline was rerun after the clean processed-data rebuild. The
following scripts completed successfully against the regenerated feature files:

- `src/score_early_reorders.py`
- `src/improve_early_reorder_scoring.py`
- `src/analyze_thresholds.py`
- `src/train_and_evaluate.py`
- `src/evaluate_severity.py`
- `src/inspect_early_reorders.py`
- `src/compare_temporal_detectors.py`
- `src/ablate_temporal_ranking.py`

Final regenerated artifacts include:

- `data/processed/temporal_event_features.csv` — 136,372 rows
- `data/processed/model_ready_events.csv` — 64,892 rows
- `results/ranked_early_reorders.csv`
- `results/early_reorder_summary.csv`
- `results/improved_ranked_early_reorders.csv`
- `results/improved_scoring_summary.csv`
- `results/threshold_analysis.csv`
- `results/controlled_evaluation_metrics.csv`
- `results/controlled_evaluation_cases.csv`
- `results/severity_experiment.csv`
- `results/early_reorder_review_sample.csv`
- `results/algorithm_comparison_metrics.csv`
- `results/algorithm_comparison_cases.csv`
- `results/ablation_metrics.csv`
- `results/ablation_cases.csv`

The regenerated threshold counts are 19,983 alerts at 50%, 17,217 at 40%,
13,861 at 30%, and 10,234 at 20%, all out of the same 64,892 eligible events.
The active improved scorer therefore remains 13,861 alerts (21.36%).

### Independent edge-case verification

The regenerated temporal features were independently checked:

- 34,664 customer-part groups
- 36,816 events with fewer than three prior intervals
- 0 zero-IPI events
- 0 negative-IPI events
- 34,664 missing historical medians on the initial event of each group
- 115 eligible events with zero historical MAD
- 0 eligible events with missing MAD
- 0 model-ready events violating the history-count, missing-value, or positive-median eligibility rules

The chronological comparison split remains 51,913 training events followed by
12,979 natural holdout events. Isolation Forest evaluation completed with
0.071192 recall and 0.125450 F1 in the controlled evaluation, compared with
0.999923 recall and 0.879835 F1 for the median baseline. These are synthetic
case metrics against a natural reference group, not real-world accuracy.

### Reproducibility note

The original `build_temporal_features.py` implementation was not used for the
final clean rebuild because its per-group expanding MAD calculation exceeded
the available execution time. The regenerated files were produced by an
equivalent indexed NumPy calculation preserving the same grouping, sort order,
past-only history definitions, and eligibility rule. The source implementation
is therefore still functionally appropriate but should be optimized before
future clean reruns are expected to use it directly.

### Phase 2 conclusion

Phase 2 is complete for the defined checklist: the processed data and
downstream outputs are regenerated, the chronological split and synthetic
construction are verified, metrics and threshold denominators are checked,
Isolation Forest is reviewed against the baseline, edge cases are recorded,
and the final parameters and output filenames are frozen above. Remaining
limitations are methodological rather than execution failures: there are no
verified real-world anomaly labels, only one chronological split was used, and
the natural reference events are not proven normal.

## Phase 3 — Comparative analysis

### Final threshold comparison

The final machine-readable table is
`results/final_threshold_comparison.csv`. Alert reduction is measured relative
to the 50% threshold, not relative to all raw events.

| Threshold | Eligible events | Flagged events | Alert percentage | Reduction in alerts | Reduction (%) |
|---:|---:|---:|---:|---:|---:|
| 50% | 64,892 | 19,983 | 30.79% | 0 | 0.00% |
| 40% | 64,892 | 17,217 | 26.53% | 2,766 | 13.84% |
| 30% | 64,892 | 13,861 | 21.36% | 6,122 | 30.64% |
| 20% | 64,892 | 10,234 | 15.77% | 9,749 | 48.79% |

The threshold plot is
`results/figures/threshold_vs_alert_volume.png`.

### Median rule versus Isolation Forest

The controlled evaluation uses the first 80% of eligible events in
chronological order for training and the final 20% as a natural reference
holdout. Synthetic positive cases are copies of the holdout with the current
IPI replaced by a fixed fraction of its historical median. The natural
holdout is not verified to be normal, so the resulting precision, recall, F1,
and false-positive rate are controlled evaluation measures rather than
real-world accuracy estimates.

| Method | Precision | Recall | F1 | False-positive rate |
|---|---:|---:|---:|---:|
| Median-based baseline | 0.7855 | 0.9999 | 0.8798 | 0.2731 |
| Isolation Forest + early-direction filter | 0.5274 | 0.0712 | 0.1254 | 0.0638 |

The metric plot is `results/figures/detector_metrics_comparison.png`.
In this tested setting, the median rule detected nearly all injected
early-reorder cases and achieved the stronger F1 score. Isolation Forest
flagged fewer reference events, but that lower alert burden cannot be
interpreted as a lower real-world false-positive rate because the reference
events are not labeled normal; it also missed most injected cases.

### Additional-baseline decision

No further baseline is necessary for Phase 3. The comparison already includes
the interpretable median-threshold family, a robust median-MAD detector,
Isolation Forest, and LOF. LOF was included as the additional unsupervised
benchmark and did not provide a stronger controlled result than the
median-threshold family. Adding another algorithm without new labels,
features, or a distinct evaluation question would increase algorithm count
without increasing evidential value.

### Precision–recall and alert-volume trade-offs

Lowering the threshold reduces the review queue from 19,983 alerts at 50% to
10,234 at 20%, but fewer alerts do not automatically mean fewer false
positives. A stricter rule can remove both ordinary events and genuine early
reorders. The 30% rule is the current operational compromise: it reduces the
50% queue by 30.64% while retaining a high controlled detection rate at the
30% synthetic injection setting. The choice should ultimately reflect review
capacity and the cost of missed anomalies, and should be revisited when
domain-reviewed labels become available.

### Comparative finding

For the current feature set, chronological split, and synthetic evaluation,
the interpretable median-based rule performed better than Isolation Forest.
It uses the customer-part history directly, is easy to explain to reviewers,
and detected 12,978 of 12,979 injected 30%-ratio cases in the controlled
evaluation. Isolation Forest had a lower alert rate and lower measured
false-positive rate against the unverified reference group, but detected only
924 of 12,979 injected cases. This result does not establish universal
superiority: the evaluation uses one split, synthetic interval replacements,
fixed model settings, and no verified real-world anomaly labels.

### Phase 3 artifacts

- `src/phase3_comparative_analysis.py`
- `results/final_threshold_comparison.csv`
- `results/figures/threshold_vs_alert_volume.png`
- `results/figures/detector_metrics_comparison.png`

Phase 3 is complete for the defined checklist.

## Phase 4 — SDG mapping with evidence

### Primary connection: SDG 12 — Responsible Consumption and Production

The primary connection is SDG 12 because the system analyzes historical
customer-part purchasing patterns and identifies unusually short replenishment
intervals for review. This is a decision-support capability, not a measured
resource-saving intervention.

### Secondary connection: SDG 9 — Industry, Innovation and Infrastructure

SDG 9 is included only as a secondary technical connection. The reproducible
feature pipeline, chronological evaluation, ranked outputs, and retained
artifacts demonstrate a data-driven industrial analytics prototype. They do
not demonstrate industrial deployment, productivity improvement, or
infrastructure innovation.

The machine-readable evidence table is
`results/sdg_evidence_mapping.csv`.

| SDG target/theme | Project capability | Supporting result | Potential benefit | Limitation |
|---|---|---|---|---|
| SDG 12: responsible and efficient resource use | Past-only IPI and robust history features for customer-part histories | 136,372 temporal events across 34,664 customer-part groups; 64,892 events have at least three prior intervals | Gives reviewers a repeatable signal for unusually early replenishment | No measured avoided waste, cost, inventory, material use, or emissions |
| SDG 12: evidence relevant to avoidable over-ordering | 30% historical-median threshold for review prioritization | 13,861 alerts (21.36%) from 64,892 eligible events; 6,122 fewer alerts than the 50% rule, a 30.64% reduction | Reduces the review queue using an auditable criterion | Alerts are not proof of over-ordering; fewer alerts do not prove fewer false positives |
| SDG 9: data-driven industrial process improvement | Reproducible temporal features, ranked alerts, controlled comparisons, and retained outputs | Median baseline controlled recall 0.9999 and F1 0.8798, with 12,978 true positives and 1 false negative on injected cases | Provides a testable analytics component for a procurement-review workflow | Prototype results do not demonstrate industrial deployment or productivity gains |

### Evidence boundary and required field evaluation

The numbers above establish technical capability: the system processed a
defined purchasing history, created past-only customer-part features, reduced
the alert queue at a selected threshold, and detected injected early-reorder
cases in a controlled experiment. They do not establish that the project has
already reduced consumption, waste, procurement cost, stock levels, or
emissions.

To support a stronger SDG 12 claim, a field evaluation would need a
pre-registered review protocol, domain-expert labels for sampled alerts and
non-alerts, baseline purchasing and inventory measures, and a comparison of
outcomes before and after adoption. Useful outcome measures would include
confirmed avoidable orders, order corrections, inventory days, write-offs,
material usage, procurement cost, and emissions proxies. The current report
does not claim any of those outcomes because they were not measured.

### Quantitative evidence table for the project report

The following table is suitable for inclusion in the paper because every
number is traceable to the current dataset or retained experiment outputs.
The change column is limited to observed dataset coverage or an internal
threshold comparison; it is not presented as a real-world sustainability
outcome.

| Indicator | Baseline/reference | Project result | Numerical change | Evidence type |
|---|---:|---:|---:|---|
| Raw procurement records analyzed | 0 | 285,941 | 285,941 records analyzed | Observed dataset coverage |
| Customer-part groups profiled | 0 | 34,664 | 34,664 groups profiled | Observed historical-pattern analysis |
| Temporal purchase events generated | 0 | 136,372 | 136,372 events generated | Reproducible pipeline output |
| Events with at least three prior intervals | 0 | 64,892 | 64,892 eligible events | Eligibility calculation |
| Alerts at 50% threshold | 19,983 | 13,861 at selected 30% threshold | 6,122 fewer alerts (30.64%) | Internal threshold comparison |
| Alert rate among eligible events | 30.79% | 21.36% | 9.43 percentage-point decrease | Internal threshold comparison |
| Controlled median-rule recall | 0 | 0.999923 | 0.999923 | Synthetic injected-case evaluation |
| Controlled median-rule F1 | 0 | 0.879835 | 0.879835 | Synthetic injected-case evaluation |

The complete machine-readable version is
`results/sdg_quantitative_evidence.csv`. The first four rows establish the
scale and historical-analysis capability of the system. The next two rows
show a measurable reduction in the review queue relative to another tested
threshold. The final two rows show controlled detector performance. None of
these rows should be renamed as cost savings, waste reduction, inventory
reduction, or emissions reduction.

### Phase 4 artifacts

- `src/phase4_sdg_mapping.py`
- `results/sdg_evidence_mapping.csv`

Phase 4 is complete for the defined checklist, with SDG claims limited to
measured technical evidence and explicitly separated from unmeasured impact.

## Real-world improvement comparison audit

The repository was checked for a genuine pre-project or operational comparison.
No defensible real-world improvement comparison can currently be calculated.
The raw files contain transaction identifiers, dates, customer-part identifiers,
descriptions, locations, and `PRICE_EXACT`, but they do not contain quantities,
inventory levels, waste/write-off outcomes, emissions, approved or rejected
orders, expert anomaly labels, manual-review decisions, or post-intervention
outcomes. There is also no prior alert-log export representing the old
operational method.

The controlled values in this report must therefore not be presented as
real-world improvement:

- `0.9999` recall and `0.8798` F1 are synthetic-case evaluation values.
- `13,861` alerts and `21.36%` are current-system screening outputs.
- The `30.64%` reduction is relative to the project's 50% threshold variant,
  not relative to a real previous operational system.
- `PRICE_EXACT` is an observed source field, not a measured cost saving.

The verified status table is
`results/real_world_comparison_status.csv`. It records each requested outcome,
the available project value, whether a real baseline or label exists, and why
the comparison is or is not defensible.

### What is required for a real improvement claim

To calculate actual improvement without fabrication, the project needs a
time-matched baseline and outcome labels. At minimum, collect the old
process's alert/review log, expert decisions for a sampled set of alerts and
non-alerts, order quantities and prices, inventory or write-off outcomes, and
the intervention period after alerts are acted upon. Then report matched
before/after or controlled-period changes such as confirmed avoidable-order
rate, review precision, missed confirmed anomalies, cost per eligible event,
inventory days, write-offs, and emissions proxies. Until those data exist,
the SDG section must remain a technical-capability mapping rather than a claim
of achieved real-world sustainability improvement.
