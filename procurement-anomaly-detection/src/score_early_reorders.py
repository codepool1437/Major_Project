
from pathlib import Path

import numpy as np
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

OUTPUT_FILE = RESULTS_DIR / "ranked_early_reorders.csv"
SUMMARY_FILE = RESULTS_DIR / "early_reorder_summary.csv"

# An event is considered early if its interval is at most
# this fraction of its historical median.
EARLY_INTERVAL_RATIO = 0.50

# Require at least this many previous intervals.
MIN_HISTORY_COUNT = 3

# Avoid division by zero when historical MAD is zero.
EPSILON = 1.0


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
        "CUSTOMER_ID",
        "ITEM_ID",
        "REQUEST_DATE",
        "IPI_DAYS",
        "HIST_MEDIAN_IPI",
        "HIST_MAD_IPI",
        "HISTORY_COUNT",
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Convert relevant columns to numeric values.
    numeric_columns = [
        "IPI_DAYS",
        "HIST_MEDIAN_IPI",
        "HIST_MAD_IPI",
        "HISTORY_COUNT",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["REQUEST_DATE"] = pd.to_datetime(
        df["REQUEST_DATE"], errors="coerce"
    )

    # --------------------------------------------------
    # 1. VALIDATE HISTORICAL INFORMATION
    # --------------------------------------------------

    valid_history = (
        df["HISTORY_COUNT"].ge(MIN_HISTORY_COUNT)
        & df["IPI_DAYS"].notna()
        & df["HIST_MEDIAN_IPI"].notna()
        & df["HIST_MAD_IPI"].notna()
        & df["IPI_DAYS"].ge(0)
        & df["HIST_MEDIAN_IPI"].gt(0)
        & df["HIST_MAD_IPI"].ge(0)
    )

    df = df.loc[valid_history].copy()

    if df.empty:
        raise ValueError(
            "No events have valid historical information. "
            "Check model_ready_events.csv."
        )

    print(f"Events with valid history: {len(df):,}")

    # --------------------------------------------------
    # 2. CALCULATE EARLY-REORDER RATIOS
    # --------------------------------------------------

    # Ratio below 1 means the current interval is shorter
    # than the historical median.
    df["IPI_TO_HIST_MEDIAN"] = (
        df["IPI_DAYS"] / df["HIST_MEDIAN_IPI"]
    )

    # Positive values mean the current interval is shorter
    # than its historical median.
    df["DAYS_BELOW_HIST_MEDIAN"] = (
        df["HIST_MEDIAN_IPI"] - df["IPI_DAYS"]
    )

    # --------------------------------------------------
    # 3. CALCULATE ROBUST EARLY-REORDER SCORE
    # --------------------------------------------------

    # Scale MAD to be comparable to standard deviation
    # under a normal-distribution assumption.
    df["ROBUST_SCALE"] = np.maximum(
        1.4826 * df["HIST_MAD_IPI"],
        EPSILON,
    )

    # Directional score: higher values indicate an interval
    # that is unusually short compared with historical behavior.
    df["EARLY_REORDER_SCORE"] = (
        df["HIST_MEDIAN_IPI"] - df["IPI_DAYS"]
    ) / df["ROBUST_SCALE"]

    # --------------------------------------------------
    # 4. FLAG POTENTIAL EARLY REORDERS
    # --------------------------------------------------

    df["IS_EARLY_REORDER"] = (
        df["IPI_DAYS"]
        <= EARLY_INTERVAL_RATIO * df["HIST_MEDIAN_IPI"]
    )

    # Only positive scores represent intervals shorter
    # than the historical median.
    df["EARLY_REORDER_SCORE"] = df[
        "EARLY_REORDER_SCORE"
    ].clip(lower=0)

    # --------------------------------------------------
    # 5. ASSIGN INTERPRETABLE SEVERITY
    # --------------------------------------------------

    conditions = [
        df["EARLY_REORDER_SCORE"] >= 3,
        df["EARLY_REORDER_SCORE"] >= 2,
        df["EARLY_REORDER_SCORE"] >= 1,
    ]

    severity_labels = [
        "Very High",
        "High",
        "Moderate",
    ]

    df["SEVERITY"] = np.select(
        conditions,
        severity_labels,
        default="Low",
    )

    # Events that do not pass the early-interval rule
    # are not flagged, regardless of their score.
    df.loc[~df["IS_EARLY_REORDER"], "SEVERITY"] = "Not Flagged"

    # --------------------------------------------------
    # 6. ADD EXPLANATIONS
    # --------------------------------------------------

    df["ALERT_REASON"] = np.where(
        df["IS_EARLY_REORDER"],
        (
            "Reorder interval is "
            + (
                df["IPI_TO_HIST_MEDIAN"] * 100
            ).round(1).astype(str)
            + "% of the historical median."
        ),
        "Interval does not meet the early-reorder threshold.",
    )

    # --------------------------------------------------
    # 7. RANK FLAGGED EVENTS
    # --------------------------------------------------

    flagged = df.loc[df["IS_EARLY_REORDER"]].copy()

    flagged = flagged.sort_values(
        by=[
            "EARLY_REORDER_SCORE",
            "IPI_TO_HIST_MEDIAN",
        ],
        ascending=[False, True],
    ).reset_index(drop=True)

    flagged.insert(
        0,
        "ALERT_RANK",
        np.arange(1, len(flagged) + 1),
    )

    # Keep useful columns first; preserve any other existing
    # features after them for further analysis.
    preferred_columns = [
        "ALERT_RANK",
        "CUSTOMER_ID",
        "ITEM_ID",
        "REQUEST_DATE",
        "IPI_DAYS",
        "HIST_MEDIAN_IPI",
        "HIST_MAD_IPI",
        "HISTORY_COUNT",
        "IPI_TO_HIST_MEDIAN",
        "DAYS_BELOW_HIST_MEDIAN",
        "EARLY_REORDER_SCORE",
        "SEVERITY",
        "ALERT_REASON",
    ]

    remaining_columns = [
        col for col in flagged.columns
        if col not in preferred_columns
    ]

    flagged = flagged[
        preferred_columns + remaining_columns
    ]

    flagged.to_csv(OUTPUT_FILE, index=False)

    # --------------------------------------------------
    # 8. GENERATE SUMMARY
    # --------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "metric": "Eligible events",
                "value": len(df),
            },
            {
                "metric": "Potential early reorders",
                "value": len(flagged),
            },
            {
                "metric": "Flagged percentage",
                "value": round(
                    100 * len(flagged) / len(df), 2
                ),
            },
            {
                "metric": "Early interval ratio threshold",
                "value": EARLY_INTERVAL_RATIO,
            },
            {
                "metric": "Minimum historical intervals",
                "value": MIN_HISTORY_COUNT,
            },
        ]
    )

    summary.to_csv(SUMMARY_FILE, index=False)

    print("\n========== EARLY-REORDER SCORING ==========")
    print(f"Eligible events: {len(df):,}")
    print(f"Flagged events: {len(flagged):,}")
    print(
        f"Flagged percentage: "
        f"{100 * len(flagged) / len(df):.2f}%"
    )

    if not flagged.empty:
        print("\nSeverity breakdown:")
        print(
            flagged["SEVERITY"]
            .value_counts()
            .reindex(
                ["Very High", "High", "Moderate", "Low"],
                fill_value=0,
            )
            .to_string()
        )

        print("\nTop 10 potential early reorders:")
        print(
            flagged[
                [
                    "ALERT_RANK",
                    "CUSTOMER_ID",
                    "ITEM_ID",
                    "IPI_DAYS",
                    "HIST_MEDIAN_IPI",
                    "EARLY_REORDER_SCORE",
                    "SEVERITY",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )
    else:
        print("\nNo events met the current threshold.")

    print(f"\nSaved ranked alerts: {OUTPUT_FILE}")
    print(f"Saved summary: {SUMMARY_FILE}")
    print("\nScoring complete.")


if __name__ == "__main__":
    main()
