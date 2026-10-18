
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

ALERTS_FILE = RESULTS_DIR / "ranked_early_reorders.csv"

EVENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_ready_events.csv"
)

OUTPUT_FILE = RESULTS_DIR / "early_reorder_review_sample.csv"

# Review sample sizes
TOP_ALERTS_COUNT = 50
RANDOM_FLAGGED_COUNT = 50
UNFLAGGED_COUNT = 50

# Reproducible random sampling
RANDOM_SEED = 42

# Use the provisional threshold selected for further inspection.
EARLY_INTERVAL_RATIO = 0.30


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():
    print("Loading early-reorder results...")

    if not ALERTS_FILE.exists():
        raise FileNotFoundError(
            f"Alerts file not found: {ALERTS_FILE}\n"
            "Run src/score_early_reorders.py first."
        )

    if not EVENTS_FILE.exists():
        raise FileNotFoundError(
            f"Events file not found: {EVENTS_FILE}\n"
            "Run src/build_temporal_features.py first."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    alerts = pd.read_csv(ALERTS_FILE)
    events = pd.read_csv(EVENTS_FILE)

    required_alert_columns = [
        "CUSTOMER_ID",
        "ITEM_ID",
        "REQUEST_DATE",
        "IPI_DAYS",
        "HIST_MEDIAN_IPI",
        "EARLY_REORDER_SCORE",
        "SEVERITY",
    ]

    required_event_columns = [
        "CUSTOMER_ID",
        "ITEM_ID",
        "REQUEST_DATE",
        "IPI_DAYS",
        "HIST_MEDIAN_IPI",
        "HIST_MAD_IPI",
        "HISTORY_COUNT",
    ]

    missing_alerts = [
        col for col in required_alert_columns
        if col not in alerts.columns
    ]

    missing_events = [
        col for col in required_event_columns
        if col not in events.columns
    ]

    if missing_alerts:
        raise ValueError(
            f"Missing columns in ranked alerts: {missing_alerts}"
        )

    if missing_events:
        raise ValueError(
            f"Missing columns in model-ready events: {missing_events}"
        )

    # --------------------------------------------------
    # 1. PREPARE AND VALIDATE EVENT DATA
    # --------------------------------------------------

    for df in [alerts, events]:
        df["REQUEST_DATE"] = pd.to_datetime(
            df["REQUEST_DATE"], errors="coerce"
        )

        for col in [
            "IPI_DAYS",
            "HIST_MEDIAN_IPI",
            "HIST_MAD_IPI",
            "HISTORY_COUNT",
            "EARLY_REORDER_SCORE",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col], errors="coerce"
                )

    # Match the eligibility requirements of the scoring script.
    valid_history = (
        events["HISTORY_COUNT"].ge(3)
        & events["IPI_DAYS"].notna()
        & events["HIST_MEDIAN_IPI"].notna()
        & events["HIST_MAD_IPI"].notna()
        & events["IPI_DAYS"].ge(0)
        & events["HIST_MEDIAN_IPI"].gt(0)
        & events["HIST_MAD_IPI"].ge(0)
    )

    eligible = events.loc[valid_history].copy()

    # --------------------------------------------------
    # 2. IDENTIFY FLAGS USING THE PROVISIONAL 30% RULE
    # --------------------------------------------------

    flagged_mask = (
        eligible["IPI_DAYS"]
        <= EARLY_INTERVAL_RATIO * eligible["HIST_MEDIAN_IPI"]
    )

    flagged = eligible.loc[flagged_mask].copy()
    unflagged = eligible.loc[~flagged_mask].copy()

    # --------------------------------------------------
    # 3. ADD REVIEW DETAILS
    # --------------------------------------------------

    for df in [flagged, unflagged]:
        df["IPI_TO_HIST_MEDIAN"] = (
            df["IPI_DAYS"] / df["HIST_MEDIAN_IPI"]
        )

        df["DAYS_BELOW_HIST_MEDIAN"] = (
            df["HIST_MEDIAN_IPI"] - df["IPI_DAYS"]
        )

        # Recalculate the robust score consistently.
        robust_scale = (
            1.4826 * df["HIST_MAD_IPI"]
        ).clip(lower=1.0)

        df["EARLY_REORDER_SCORE"] = (
            (
                df["HIST_MEDIAN_IPI"] - df["IPI_DAYS"]
            ) / robust_scale
        ).clip(lower=0)

    # --------------------------------------------------
    # 4. SELECT TOP-SCORING ALERTS
    # --------------------------------------------------

    top_alerts = flagged.sort_values(
        by=[
            "EARLY_REORDER_SCORE",
            "IPI_TO_HIST_MEDIAN",
        ],
        ascending=[False, True],
    ).head(TOP_ALERTS_COUNT).copy()

    top_alerts["REVIEW_GROUP"] = "Top Scoring Alerts"

    # --------------------------------------------------
    # 5. SELECT RANDOM FLAGGED EVENTS
    # --------------------------------------------------

    # Exclude the top alerts from the random sample where
    # possible, so the review includes more variety.
    flagged_remaining = flagged.drop(
        index=top_alerts.index
    )

    random_flagged = flagged_remaining.sample(
        n=min(RANDOM_FLAGGED_COUNT, len(flagged_remaining)),
        random_state=RANDOM_SEED,
    ).copy()

    random_flagged["REVIEW_GROUP"] = "Random Flagged Events"

    # --------------------------------------------------
    # 6. SELECT RANDOM UNFLAGGED EVENTS
    # --------------------------------------------------

    random_unflagged = unflagged.sample(
        n=min(UNFLAGGED_COUNT, len(unflagged)),
        random_state=RANDOM_SEED,
    ).copy()

    random_unflagged["REVIEW_GROUP"] = "Random Unflagged Events"

    # --------------------------------------------------
    # 7. COMBINE REVIEW SAMPLE
    # --------------------------------------------------

    review = pd.concat(
        [
            top_alerts,
            random_flagged,
            random_unflagged,
        ],
        ignore_index=True,
    )

    review.insert(
        0,
        "REVIEW_ID",
        range(1, len(review) + 1),
    )

    # These are review aids, not confirmed anomaly labels.
    review["MANUAL_REVIEW_LABEL"] = ""
    review["REVIEW_NOTES"] = ""

    preferred_columns = [
        "REVIEW_ID",
        "REVIEW_GROUP",
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
        "MANUAL_REVIEW_LABEL",
        "REVIEW_NOTES",
    ]

    review = review[
        preferred_columns
        + [
            col for col in review.columns
            if col not in preferred_columns
        ]
    ]

    review.to_csv(OUTPUT_FILE, index=False)

    # --------------------------------------------------
    # 8. DISPLAY SUMMARY
    # --------------------------------------------------

    print("\n========== REVIEW SAMPLE ==========")
    print(f"All eligible events: {len(eligible):,}")
    print(f"Flagged at 30% threshold: {len(flagged):,}")
    print(f"Unflagged events: {len(unflagged):,}")

    print("\nReview sample breakdown:")
    print(
        review["REVIEW_GROUP"]
        .value_counts()
        .to_string()
    )

    print("\nTop 10 alerts selected for review:")
    print(
        top_alerts[
            [
                "CUSTOMER_ID",
                "ITEM_ID",
                "IPI_DAYS",
                "HIST_MEDIAN_IPI",
                "HIST_MAD_IPI",
                "EARLY_REORDER_SCORE",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print(f"\nSaved review sample: {OUTPUT_FILE}")
    print("\nInspection sample generation complete.")


if __name__ == "__main__":
    main()
