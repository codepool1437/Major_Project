
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

OUTPUT_FILE = RESULTS_DIR / "improved_ranked_early_reorders.csv"
SUMMARY_FILE = RESULTS_DIR / "improved_scoring_summary.csv"

# Provisional threshold: current interval must be <= 30%
# of the historical median.
EARLY_INTERVAL_RATIO = 0.30

MIN_HISTORY_COUNT = 3

# Small floor for the robust scale.
# This prevents division by zero but does not imply that
# a score is a probability.
MIN_SCALE_DAYS = 1.0


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

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --------------------------------------------------
    # 1. CLEAN REQUIRED VALUES
    # --------------------------------------------------

    df["REQUEST_DATE"] = pd.to_datetime(
        df["REQUEST_DATE"], errors="coerce"
    )

    numeric_columns = [
        "IPI_DAYS",
        "HIST_MEDIAN_IPI",
        "HIST_MAD_IPI",
        "HISTORY_COUNT",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = (
        df["HISTORY_COUNT"].ge(MIN_HISTORY_COUNT)
        & df["IPI_DAYS"].notna()
        & df["HIST_MEDIAN_IPI"].notna()
        & df["HIST_MAD_IPI"].notna()
        & df["IPI_DAYS"].ge(0)
        & df["HIST_MEDIAN_IPI"].gt(0)
        & df["HIST_MAD_IPI"].ge(0)
    )

    df = df.loc[valid].copy()

    if df.empty:
        raise ValueError("No events with valid historical features.")

    print(f"Eligible events: {len(df):,}")

    # --------------------------------------------------
    # 2. CALCULATE INTERVAL RATIOS
    # --------------------------------------------------

    df["IPI_TO_HIST_MEDIAN"] = (
        df["IPI_DAYS"] / df["HIST_MEDIAN_IPI"]
    )

    df["DAYS_BELOW_HIST_MEDIAN"] = (
        df["HIST_MEDIAN_IPI"] - df["IPI_DAYS"]
    ).clip(lower=0)

    # How much shorter is the current interval than the
    # historical median? Higher values mean a larger reduction.
    df["EARLYNESS_SCORE"] = (
        1 - df["IPI_TO_HIST_MEDIAN"]
    ).clip(lower=0, upper=1)

    # --------------------------------------------------
    # 3. HANDLE HISTORICAL VARIABILITY
    # --------------------------------------------------

    # MAD = 0 means historical intervals had no median
    # absolute deviation. It does not necessarily mean that
    # the data are error-free or that the event is confirmed.
    df["MAD_IS_ZERO"] = df["HIST_MAD_IPI"].eq(0)

    df["ROBUST_SCALE_DAYS"] = (
        1.4826 * df["HIST_MAD_IPI"]
    ).clip(lower=MIN_SCALE_DAYS)

    # Directional robust deviation. This is a score, not
    # a probability and not a calibrated anomaly likelihood.
    df["ROBUST_EARLY_SCORE"] = (
        df["DAYS_BELOW_HIST_MEDIAN"]
        / df["ROBUST_SCALE_DAYS"]
    )

    # Use log1p to reduce the influence of extremely large
    # robust scores when ranking alerts.
    df["CAPPED_VARIABILITY_SCORE"] = np.log1p(
        df["ROBUST_EARLY_SCORE"]
    )

    # --------------------------------------------------
    # 4. FLAG POTENTIAL EARLY REORDERS
    # --------------------------------------------------

    df["IS_EARLY_REORDER"] = (
        df["IPI_DAYS"]
        <= EARLY_INTERVAL_RATIO * df["HIST_MEDIAN_IPI"]
    )

    # --------------------------------------------------
    # 5. CREATE A BALANCED RANKING SCORE
    # --------------------------------------------------

    # Combine two distinct signals:
    # - EARLYNESS_SCORE: relative shortening of the interval
    # - CAPPED_VARIABILITY_SCORE: deviation relative to
    #   historical variability, dampened using log1p
    #
    # These weights are heuristic and should be documented
    # as design choices, not learned or validated parameters.
    df["RANKING_SCORE"] = (
        0.70 * df["EARLYNESS_SCORE"]
        + 0.30 * (
            df["CAPPED_VARIABILITY_SCORE"]
            / (1 + df["CAPPED_VARIABILITY_SCORE"])
        )
    )

    # --------------------------------------------------
    # 6. ASSIGN SEVERITY LABELS
    # --------------------------------------------------

    # Severity is based primarily on interval ratio, making
    # it less sensitive to zero MAD and extreme raw scores.
    conditions = [
        df["IPI_TO_HIST_MEDIAN"].le(0.10),
        df["IPI_TO_HIST_MEDIAN"].le(0.20),
        df["IPI_TO_HIST_MEDIAN"].le(0.30),
    ]

    severity_labels = [
        "Very High",
        "High",
        "Moderate",
    ]

    df["SEVERITY"] = np.select(
        conditions,
        severity_labels,
        default="Not Flagged",
    )

    df.loc[
        ~df["IS_EARLY_REORDER"], "SEVERITY"
    ] = "Not Flagged"

    # --------------------------------------------------
    # 7. GENERATE EXPLANATIONS
    # --------------------------------------------------

    df["ALERT_REASON"] = (
        "Current interval is "
        + (100 * df["IPI_TO_HIST_MEDIAN"]).round(1).astype(str)
        + "% of historical median; historical MAD is "
        + df["HIST_MAD_IPI"].round(2).astype(str)
        + " days."
    )

    df.loc[
        df["MAD_IS_ZERO"], "ALERT_REASON"
    ] += " Historical MAD is zero; interpret variability score cautiously."

    # --------------------------------------------------
    # 8. FILTER AND RANK ALERTS
    # --------------------------------------------------

    flagged = df.loc[df["IS_EARLY_REORDER"]].copy()

    flagged = flagged.sort_values(
        by=[
            "RANKING_SCORE",
            "EARLYNESS_SCORE",
            "IPI_TO_HIST_MEDIAN",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    flagged.insert(
        0,
        "ALERT_RANK",
        np.arange(1, len(flagged) + 1),
    )

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
        "EARLYNESS_SCORE",
        "ROBUST_EARLY_SCORE",
        "MAD_IS_ZERO",
        "RANKING_SCORE",
        "SEVERITY",
        "ALERT_REASON",
    ]

    other_columns = [
        col for col in flagged.columns
        if col not in preferred_columns
    ]

    flagged = flagged[preferred_columns + other_columns]

    # --------------------------------------------------
    # 9. SAVE OUTPUTS
    # --------------------------------------------------

    flagged.to_csv(OUTPUT_FILE, index=False)

    total_events = len(df)
    total_flagged = len(flagged)

    summary = pd.DataFrame(
        [
            {
                "metric": "Eligible events",
                "value": total_events,
            },
            {
                "metric": "Flagged events",
                "value": total_flagged,
            },
            {
                "metric": "Flagged percentage",
                "value": round(
                    100 * total_flagged / total_events, 2
                ),
            },
            {
                "metric": "Events with zero historical MAD",
                "value": int(df["MAD_IS_ZERO"].sum()),
            },
            {
                "metric": "Flagged events with zero historical MAD",
                "value": int(flagged["MAD_IS_ZERO"].sum()),
            },
            {
                "metric": "Provisional threshold",
                "value": EARLY_INTERVAL_RATIO,
            },
        ]
    )

    summary.to_csv(SUMMARY_FILE, index=False)

    # --------------------------------------------------
    # 10. PRINT RESULTS
    # --------------------------------------------------

    print("\n========== IMPROVED SCORING RESULTS ==========")
    print(f"Eligible events: {total_events:,}")
    print(f"Flagged events: {total_flagged:,}")
    print(
        f"Flagged percentage: "
        f"{100 * total_flagged / total_events:.2f}%"
    )
    print(
        "Events with zero historical MAD: "
        f"{int(df['MAD_IS_ZERO'].sum()):,}"
    )
    print(
        "Flagged events with zero historical MAD: "
        f"{int(flagged['MAD_IS_ZERO'].sum()):,}"
    )

    print("\nSeverity breakdown:")
    print(
        flagged["SEVERITY"]
        .value_counts()
        .reindex(
            ["Very High", "High", "Moderate"],
            fill_value=0,
        )
        .to_string()
    )

    print("\nTop 10 alerts:")
    print(
        flagged[
            [
                "ALERT_RANK",
                "CUSTOMER_ID",
                "ITEM_ID",
                "IPI_DAYS",
                "HIST_MEDIAN_IPI",
                "IPI_TO_HIST_MEDIAN",
                "RANKING_SCORE",
                "MAD_IS_ZERO",
                "SEVERITY",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print(f"\nSaved ranked alerts: {OUTPUT_FILE}")
    print(f"Saved summary: {SUMMARY_FILE}")

    print(
        "\nNote: scores and severity labels are heuristic. "
        "They do not establish that an event is a genuine "
        "or unnecessary purchase."
    )

    print("\nImproved scoring complete.")


if __name__ == "__main__":
    main()
