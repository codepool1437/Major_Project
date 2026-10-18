
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_ready_events.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_FILE = RESULTS_DIR / "threshold_analysis.csv"

# Compare how strict different early-reorder thresholds are.
THRESHOLDS = [0.50, 0.40, 0.30, 0.20]

MIN_HISTORY_COUNT = 3


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():
    print("Loading model-ready events...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run src/build_temporal_features.py first."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "IPI_DAYS",
        "HIST_MEDIAN_IPI",
        "HISTORY_COUNT",
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Ensure relevant columns contain numeric values.
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Use the same basic eligibility requirements as our
    # early-reorder scoring script.
    valid_history = (
        df["HISTORY_COUNT"].ge(MIN_HISTORY_COUNT)
        & df["IPI_DAYS"].notna()
        & df["HIST_MEDIAN_IPI"].notna()
        & df["IPI_DAYS"].ge(0)
        & df["HIST_MEDIAN_IPI"].gt(0)
    )

    eligible = df.loc[valid_history].copy()

    if eligible.empty:
        raise ValueError(
            "No eligible events found. Check the input data."
        )

    print(f"Total input events: {len(df):,}")
    print(f"Eligible events: {len(eligible):,}")

    # --------------------------------------------------
    # ANALYZE EACH THRESHOLD
    # --------------------------------------------------

    results = []

    for threshold in THRESHOLDS:

        # An event qualifies when its current interval is
        # no greater than the selected fraction of its
        # historical median interval.
        flagged_mask = (
            eligible["IPI_DAYS"]
            <= threshold * eligible["HIST_MEDIAN_IPI"]
        )

        flagged_count = int(flagged_mask.sum())
        total_events = len(eligible)

        flagged_percentage = (
            100 * flagged_count / total_events
        )

        results.append(
            {
                "threshold_ratio": threshold,
                "threshold_percentage": threshold * 100,
                "rule": (
                    f"IPI_DAYS <= {threshold:.2f} "
                    "* HIST_MEDIAN_IPI"
                ),
                "eligible_events": total_events,
                "flagged_events": flagged_count,
                "flagged_percentage": round(
                    flagged_percentage, 2
                ),
                "unflagged_events": total_events - flagged_count,
            }
        )

    # --------------------------------------------------
    # SAVE AND DISPLAY RESULTS
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\n========== THRESHOLD ANALYSIS ==========")

    print(
        results_df[
            [
                "threshold_percentage",
                "eligible_events",
                "flagged_events",
                "flagged_percentage",
                "unflagged_events",
            ]
        ].to_string(index=False)
    )

    print("\nSaved threshold analysis:")
    print(OUTPUT_FILE)

    print(
        "\nInterpretation: a lower threshold is stricter "
        "because the current interval must be shorter "
        "relative to the historical median."
    )

    print(
        "These figures measure alert volume, not detection "
        "accuracy. The dataset has no verified labels "
        "confirming which events are genuine anomalies."
    )

    print("\nThreshold analysis complete.")


if __name__ == "__main__":
    main()
