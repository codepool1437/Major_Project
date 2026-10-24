"""Generate final Phase 3 comparison tables, figures, and interpretation inputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def build_threshold_table() -> pd.DataFrame:
    thresholds = pd.read_csv(RESULTS / "threshold_analysis.csv").copy()
    baseline_alerts = int(thresholds.loc[thresholds["threshold_percentage"] == 50, "flagged_events"].iloc[0])
    thresholds["alert_reduction_vs_50pct"] = baseline_alerts - thresholds["flagged_events"]
    thresholds["alert_reduction_percentage_vs_50pct"] = (
        thresholds["alert_reduction_vs_50pct"] / baseline_alerts * 100
    ).round(2)
    return thresholds[
        [
            "threshold_percentage",
            "eligible_events",
            "flagged_events",
            "flagged_percentage",
            "alert_reduction_vs_50pct",
            "alert_reduction_percentage_vs_50pct",
        ]
    ].rename(
        columns={
            "threshold_percentage": "threshold_percent",
            "flagged_percentage": "alert_percentage",
            "alert_reduction_vs_50pct": "alert_reduction",
            "alert_reduction_percentage_vs_50pct": "alert_reduction_percent",
        }
    )


def plot_threshold_volume(table: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    ax.plot(
        table["threshold_percent"],
        table["flagged_events"],
        marker="o",
        linewidth=2.4,
        color="#1f4e79",
    )
    for _, row in table.iterrows():
        ax.annotate(
            f'{int(row["flagged_events"]):,}',
            (row["threshold_percent"], row["flagged_events"]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.set_xlabel("Alert threshold (% of historical median IPI)")
    ax.set_ylabel("Flagged eligible events")
    ax.set_title("Alert volume decreases as the threshold becomes stricter")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xticks(table["threshold_percent"])
    fig.tight_layout()
    fig.savefig(FIGURES / "threshold_vs_alert_volume.png", bbox_inches="tight")
    plt.close(fig)


def plot_detector_metrics() -> None:
    metrics = pd.read_csv(RESULTS / "controlled_evaluation_metrics.csv").copy()
    labels = ["Median rule", "Isolation Forest"]
    metrics = metrics.iloc[[0, 1]]
    x = range(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    for offset, column, label, color in [
        (-width, "precision", "Precision", "#2f75b5"),
        (0, "recall", "Recall", "#70ad47"),
        (width, "f1_score", "F1 score", "#ed7d31"),
    ]:
        values = metrics[column].to_numpy()
        bars = ax.bar([i + offset for i in x], values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Controlled metric")
    ax.set_title("Controlled synthetic-case detection metrics")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "detector_metrics_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    table = build_threshold_table()
    table.to_csv(RESULTS / "final_threshold_comparison.csv", index=False)
    plot_threshold_volume(table)
    plot_detector_metrics()
    print(table.to_string(index=False))
    print(f"Saved: {RESULTS / 'final_threshold_comparison.csv'}")
    print(f"Saved figures in: {FIGURES}")


if __name__ == "__main__":
    main()
