"""Compare temporal early-reorder detectors on one chronological holdout.

This experiment is deliberately separate from the dashboard scoring pipeline.
It evaluates fixed threshold rules, robust median-MAD rules, Isolation Forest,
and LOF against the same natural holdout and injected early-reorder cases.
The injected cases are controlled test cases, not real anomaly labels.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "model_ready_events.csv"
RESULTS_DIR = ROOT / "results"

FEATURE_COLS = [
    "IPI_DAYS",
    "HIST_MEDIAN_IPI",
    "HIST_MAD_IPI",
    "RECENT_MEDIAN_IPI",
    "IPI_TO_HIST_MEDIAN",
    "IPI_TO_RECENT_MEDIAN",
    "HISTORY_COUNT",
]
RATIOS = [0.50, 0.30, 0.20, 0.10]
THRESHOLDS = [0.50, 0.40, 0.30, 0.20]
MAD_THRESHOLDS = [1.0, 2.0, 3.0, 4.0]
TEST_FRACTION = 0.20
RANDOM_STATE = 42


def load_events() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}. "
            "Run src/build_temporal_features.py first."
        )

    events = pd.read_csv(INPUT_FILE, low_memory=False)
    required = [
        "REQUEST_DATE",
        "IPI_DAYS",
        "HISTORY_COUNT",
        "HIST_MEDIAN_IPI",
        "HIST_MAD_IPI",
        "RECENT_MEDIAN_IPI",
        "IPI_TO_HIST_MEDIAN",
        "IPI_TO_RECENT_MEDIAN",
    ]
    missing = [column for column in required if column not in events]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    events["REQUEST_DATE"] = pd.to_datetime(
        events["REQUEST_DATE"], errors="coerce"
    )
    events = events.replace([np.inf, -np.inf], np.nan)
    events = events.dropna(subset=required).copy()
    events = events[
        (events["HISTORY_COUNT"] >= 3)
        & (events["IPI_DAYS"] >= 0)
        & (events["HIST_MEDIAN_IPI"] > 0)
    ]
    return events.sort_values("REQUEST_DATE").reset_index(drop=True)


def feature_matrix(
    frame: pd.DataFrame, medians: pd.Series
) -> pd.DataFrame:
    matrix = frame[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    return matrix.fillna(medians).fillna(0)


def add_synthetic_cases(
    natural: pd.DataFrame, ratio: float
) -> pd.DataFrame:
    synthetic = natural.copy()
    synthetic["IPI_DAYS"] = (
        ratio * synthetic["HIST_MEDIAN_IPI"]
    )
    synthetic["IPI_TO_HIST_MEDIAN"] = (
        synthetic["IPI_DAYS"] / synthetic["HIST_MEDIAN_IPI"]
    )
    synthetic["IPI_TO_RECENT_MEDIAN"] = (
        synthetic["IPI_DAYS"]
        / synthetic["RECENT_MEDIAN_IPI"].replace(0, np.nan)
    )
    synthetic["synthetic_label"] = 1
    return synthetic


def robust_mad_prediction(
    frame: pd.DataFrame, threshold: float
) -> pd.Series:
    scale = (1.4826 * frame["HIST_MAD_IPI"]).clip(lower=1.0)
    robust_z = (
        frame["HIST_MEDIAN_IPI"] - frame["IPI_DAYS"]
    ) / scale
    return (robust_z >= threshold).astype(int)


def summarize(
    method: str,
    parameter: str,
    injection_ratio: float,
    natural_predictions: pd.Series,
    evaluation_predictions: pd.Series,
    evaluation: pd.DataFrame,
    ranking_score: pd.Series,
    ranking_ascending: bool,
    train_size: int,
    test_size: int,
) -> dict:
    natural_predictions = natural_predictions.astype(int)
    evaluation_predictions = evaluation_predictions.astype(int)
    labels = evaluation["synthetic_label"].astype(int)

    synthetic_mask = labels.eq(1)
    natural_alert_rate = float(natural_predictions.mean())
    synthetic_detection_rate = float(
        evaluation_predictions[synthetic_mask].mean()
    )
    tp = int(((evaluation_predictions == 1) & (labels == 1)).sum())
    fp = int(((evaluation_predictions == 1) & (labels == 0)).sum())
    fn = int(((evaluation_predictions == 0) & (labels == 1)).sum())
    tn = int(((evaluation_predictions == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    ranked = evaluation.assign(
        prediction=np.asarray(evaluation_predictions),
        ranking_score=np.asarray(ranking_score),
    )
    ranked = ranked.sort_values(
        "ranking_score",
        ascending=ranking_ascending,
    )
    synthetic_ranks = np.flatnonzero(
        ranked["synthetic_label"].to_numpy() == 1
    ) + 1

    return {
        "method": method,
        "parameter": parameter,
        "injection_ratio": injection_ratio,
        "train_events": train_size,
        "natural_test_events": test_size,
        "synthetic_events": int(synthetic_mask.sum()),
        "natural_alerts": int(natural_predictions.sum()),
        "natural_alert_rate": natural_alert_rate,
        "synthetic_detection_rate": synthetic_detection_rate,
        "precision_controlled": precision,
        "recall_controlled": recall,
        "f1_controlled": f1,
        "false_positives": fp,
        "true_positives": tp,
        "false_negatives": fn,
        "true_negatives": tn,
        "median_synthetic_rank": float(np.median(synthetic_ranks)),
        "recall_at_150": float(
            ranked.head(150)["synthetic_label"].sum()
            / synthetic_mask.sum()
        ),
    }


def main() -> None:
    events = load_events()
    split = int(len(events) * (1 - TEST_FRACTION))
    train = events.iloc[:split].copy()
    natural_test = events.iloc[split:].copy()
    natural_test["synthetic_label"] = 0
    train_matrix = train[FEATURE_COLS].replace(
        [np.inf, -np.inf], np.nan
    )
    train_medians = train_matrix.median()
    X_train = feature_matrix(train, train_medians)
    X_natural = feature_matrix(natural_test, train_medians)

    results = []
    details = []

    for ratio in RATIOS:
        synthetic = add_synthetic_cases(natural_test, ratio)
        evaluation = pd.concat(
            [natural_test, synthetic], ignore_index=True
        )
        X_evaluation = feature_matrix(evaluation, train_medians)

        for threshold in THRESHOLDS:
            natural_prediction = (
                natural_test["IPI_DAYS"]
                <= threshold * natural_test["HIST_MEDIAN_IPI"]
            ).astype(int)
            evaluation_prediction = (
                evaluation["IPI_DAYS"]
                <= threshold * evaluation["HIST_MEDIAN_IPI"]
            ).astype(int)
            results.append(
                summarize(
                    "median_threshold",
                    f"ratio<={threshold:.2f}",
                    ratio,
                    natural_prediction,
                    evaluation_prediction,
                    evaluation,
                    evaluation["IPI_DAYS"]
                    / evaluation["HIST_MEDIAN_IPI"],
                    True,
                    len(train),
                    len(natural_test),
                )
            )

        for threshold in MAD_THRESHOLDS:
            natural_prediction = robust_mad_prediction(
                natural_test, threshold
            )
            evaluation_prediction = robust_mad_prediction(
                evaluation, threshold
            )
            results.append(
                summarize(
                    "median_mad",
                    f"robust_z>={threshold:.1f}",
                    ratio,
                    natural_prediction,
                    evaluation_prediction,
                    evaluation,
                    (
                        evaluation["HIST_MEDIAN_IPI"]
                        - evaluation["IPI_DAYS"]
                    )
                    / (
                        1.4826 * evaluation["HIST_MAD_IPI"]
                    ).clip(lower=1.0),
                    False,
                    len(train),
                    len(natural_test),
                )
            )

        isolation_forest = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        isolation_forest.fit(X_train)
        train_scores = -isolation_forest.score_samples(X_train)
        cutoff = float(np.quantile(train_scores, 0.95))
        natural_scores = -isolation_forest.score_samples(X_natural)
        evaluation_scores = -isolation_forest.score_samples(
            X_evaluation
        )
        natural_prediction = (
            (natural_scores >= cutoff)
            & (
                natural_test["IPI_DAYS"].to_numpy()
                < natural_test["HIST_MEDIAN_IPI"].to_numpy()
            )
        ).astype(int)
        evaluation_prediction = (
            (evaluation_scores >= cutoff)
            & (
                evaluation["IPI_DAYS"].to_numpy()
                < evaluation["HIST_MEDIAN_IPI"].to_numpy()
            )
        ).astype(int)
        results.append(
            summarize(
                "isolation_forest",
                "train_score_p95",
                ratio,
                natural_prediction,
                evaluation_prediction,
                evaluation,
                pd.Series(evaluation_scores),
                False,
                len(train),
                len(natural_test),
            )
        )

        lof = LocalOutlierFactor(
            n_neighbors=35,
            contamination=0.05,
            novelty=True,
        )
        lof.fit(X_train)
        natural_scores = -lof.score_samples(X_natural)
        evaluation_scores = -lof.score_samples(X_evaluation)
        cutoff = float(np.quantile(-lof.score_samples(X_train), 0.95))
        natural_prediction = (
            (natural_scores >= cutoff)
            & (
                natural_test["IPI_DAYS"].to_numpy()
                < natural_test["HIST_MEDIAN_IPI"].to_numpy()
            )
        ).astype(int)
        evaluation_prediction = (
            (evaluation_scores >= cutoff)
            & (
                evaluation["IPI_DAYS"].to_numpy()
                < evaluation["HIST_MEDIAN_IPI"].to_numpy()
            )
        ).astype(int)
        results.append(
            summarize(
                "lof",
                "neighbors=35_train_score_p95",
                ratio,
                natural_prediction,
                evaluation_prediction,
                evaluation,
                pd.Series(evaluation_scores),
                False,
                len(train),
                len(natural_test),
            )
        )

        details.append(
            evaluation.assign(
                injection_ratio=ratio,
                median_threshold_30=(
                    evaluation["IPI_DAYS"]
                    <= 0.30 * evaluation["HIST_MEDIAN_IPI"]
                ),
                median_mad_z3=(
                    robust_mad_prediction(evaluation, 3.0)
                ),
            )
        )

    metrics = pd.DataFrame(results)
    detail_frame = pd.concat(details, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(
        RESULTS_DIR / "algorithm_comparison_metrics.csv",
        index=False,
    )
    detail_frame.to_csv(
        RESULTS_DIR / "algorithm_comparison_cases.csv",
        index=False,
    )
    print("Events:", len(events))
    print("Chronological train events:", len(train))
    print("Chronological natural test events:", len(natural_test))
    print("\nMetrics:")
    print(
        metrics[
            [
                "method",
                "parameter",
                "natural_alert_rate",
                "synthetic_detection_rate",
                "precision_controlled",
                "recall_at_150",
            ]
        ].to_string(index=False)
    )
    print("\nSaved:", RESULTS_DIR / "algorithm_comparison_metrics.csv")
    print("Saved:", RESULTS_DIR / "algorithm_comparison_cases.csv")
    print(
        "\nThese are controlled synthetic-evaluation metrics; "
        "they are not real-world anomaly accuracy."
    )


if __name__ == "__main__":
    main()
