"""Create evidence-backed SDG mapping from retained project outputs."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def main() -> None:
    temporal = pd.read_csv(DATA / "processed" / "temporal_event_features.csv")
    ready = pd.read_csv(DATA / "processed" / "model_ready_events.csv")
    raw_rows = sum(
        len(pd.read_csv(path, sep=";", low_memory=False))
        for path in (DATA / "raw").glob("*.csv")
    )
    thresholds = pd.read_csv(RESULTS / "final_threshold_comparison.csv")
    controlled = pd.read_csv(RESULTS / "controlled_evaluation_metrics.csv")

    eligible = int(len(ready))
    temporal_events = int(len(temporal))
    groups = int(temporal.groupby(["CUSTOMER_ID", "ITEM_ID"]).ngroups)
    active = thresholds.loc[thresholds["threshold_percent"] == 30].iloc[0]
    baseline = thresholds.loc[thresholds["threshold_percent"] == 50].iloc[0]
    median = controlled.loc[controlled["method"] == "Median-based baseline"].iloc[0]

    rows = [
        {
            "sdg": "SDG 12",
            "target_theme": "12.2 / responsible and efficient use of resources",
            "project_capability": "Customer-part purchase histories are converted into past-only IPI and robust history features.",
            "supporting_result": f"{temporal_events:,} temporal events across {groups:,} customer-part groups; {eligible:,} events have at least three prior intervals.",
            "potential_benefit": "Gives reviewers a repeatable signal for unusually early replenishment that may support better purchasing decisions.",
            "limitation": "No field measure of avoided waste, cost, inventory, material use, or emissions is available.",
        },
        {
            "sdg": "SDG 12",
            "target_theme": "12.3 / evidence for reducing avoidable over-ordering",
            "project_capability": "The 30% median-IPI rule prioritizes materially shortened intervals for review.",
            "supporting_result": f"{int(active['flagged_events']):,} alerts ({active['alert_percentage']:.2f}%) from {eligible:,} eligible events; {int(baseline['flagged_events']) - int(active['flagged_events']):,} fewer alerts than the 50% rule ({(1 - int(active['flagged_events']) / int(baseline['flagged_events'])) * 100:.2f}% reduction).",
            "potential_benefit": "Reduces the review queue while preserving a clearly defined, auditable early-reorder criterion.",
            "limitation": "An alert is a screening signal, not proof of over-ordering or waste; fewer alerts do not prove fewer false positives.",
        },
        {
            "sdg": "SDG 9",
            "target_theme": "9.4 / data-driven industrial process improvement",
            "project_capability": "The pipeline provides reproducible temporal features, ranked alerts, controlled comparisons, and retained result artifacts.",
            "supporting_result": f"Median baseline controlled recall {median['recall']:.4f}, F1 {median['f1_score']:.4f}, with {int(median['true_positives']):,} true positives and {int(median['false_negatives']):,} false negatives on injected cases.",
            "potential_benefit": "Demonstrates a testable analytics component that could be integrated into procurement review workflows.",
            "limitation": "This is a prototype evaluation; it does not demonstrate industrial deployment, productivity gains, or infrastructure innovation.",
        },
    ]
    output = pd.DataFrame(rows)
    output.to_csv(RESULTS / "sdg_evidence_mapping.csv", index=False)
    print(output.to_string(index=False))
    print(f"Saved: {RESULTS / 'sdg_evidence_mapping.csv'}")


if __name__ == "__main__":
    main()
