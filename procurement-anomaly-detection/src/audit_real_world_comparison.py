"""Audit whether real-world before/after comparison evidence exists."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def main() -> None:
    raw_files = sorted((DATA / "raw").glob("*.csv"))
    raw = pd.concat(
        [pd.read_csv(path, sep=";", low_memory=False) for path in raw_files],
        ignore_index=True,
    )
    summary = pd.read_csv(RESULTS / "improved_scoring_summary.csv")

    metrics = [
        (
            "Alert volume",
            "13,861 alerts (21.36%)",
            "No pre-project alert log",
            "Not comparable",
            "Only the project's internal 50% threshold comparison is available.",
        ),
        (
            "Confirmed useful-alert precision",
            "Controlled precision 0.7855",
            "No expert-reviewed outcome labels",
            "Not a real-world metric",
            "The controlled positives are injected cases and the reference group is not verified normal.",
        ),
        (
            "Missed real anomalies / recall",
            "Controlled recall 0.9999",
            "No real anomaly labels",
            "Not measurable",
            "Real-world recall requires confirmed anomaly cases.",
        ),
        (
            "Procurement cost",
            "Not measured",
            "No before/after cost outcome linked to alerts",
            "Unavailable",
            "PRICE_EXACT exists in source rows, but no intervention or counterfactual outcome exists.",
        ),
        (
            "Inventory, waste, or emissions",
            "Not measured",
            "No inventory, write-off, waste, quantity, or emissions fields",
            "Unavailable",
            "Cannot claim SDG impact from alert counts alone.",
        ),
    ]
    output = pd.DataFrame(
        metrics,
        columns=[
            "metric",
            "project_value",
            "real_world_baseline_or_label",
            "comparison_status",
            "evidence_note",
        ],
    )
    output.to_csv(RESULTS / "real_world_comparison_status.csv", index=False)
    print(f"Raw rows inspected: {len(raw):,}")
    print(f"Raw columns: {', '.join(raw.columns)}")
    print(f"Project summary metrics: {summary.to_dict('records')}")
    print(output.to_string(index=False))
    print(f"Saved: {RESULTS / 'real_world_comparison_status.csv'}")


if __name__ == "__main__":
    main()
