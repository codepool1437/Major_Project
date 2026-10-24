"""Ablate the dashboard's early-reorder ranking components.

The alert rule remains fixed at the current 30% median threshold. This
experiment tests whether the ranking components add value beyond that rule.
It does not change dashboard outputs.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "model_ready_events.csv"
RESULTS_DIR = ROOT / "results"
INJECTION_RATIOS = [0.50, 0.30, 0.20, 0.10]
THRESHOLD = 0.30
TEST_FRACTION = 0.20


def robust_component(frame: pd.DataFrame) -> pd.Series:
    scale = (1.4826 * frame["HIST_MAD_IPI"]).clip(lower=1.0)
    robust_early = (
        frame["HIST_MEDIAN_IPI"] - frame["IPI_DAYS"]
    ) / scale
    capped = np.log1p(robust_early.clip(lower=0))
    return capped / (1 + capped)


def ranking_score(frame: pd.DataFrame, method: str) -> pd.Series:
    ratio = frame["IPI_DAYS"] / frame["HIST_MEDIAN_IPI"]
    earlyness = (1 - ratio).clip(lower=0, upper=1)
    robust = robust_component(frame)
    if method == "threshold_only":
        return -ratio
    if method == "earlyness_only":
        return earlyness
    if method == "robust_only":
        return robust
    if method == "combined_70_30":
        return 0.70 * earlyness + 0.30 * robust
    raise ValueError(f"Unknown ablation method: {method}")


def load_events() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    events = pd.read_csv(INPUT_FILE, low_memory=False)
    events["REQUEST_DATE"] = pd.to_datetime(
        events["REQUEST_DATE"], errors="coerce"
    )
    events = events.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "REQUEST_DATE",
            "IPI_DAYS",
            "HISTORY_COUNT",
            "HIST_MEDIAN_IPI",
            "HIST_MAD_IPI",
        ]
    )
    events = events[
        (events["HISTORY_COUNT"] >= 3)
        & (events["IPI_DAYS"] >= 0)
        & (events["HIST_MEDIAN_IPI"] > 0)
    ]
    return events.sort_values("REQUEST_DATE").reset_index(drop=True)


def evaluate(
    method: str,
    natural: pd.DataFrame,
    evaluation: pd.DataFrame,
    ratio: float,
    train_size: int,
) -> dict:
    natural = natural.copy()
    evaluation = evaluation.copy()
    natural["flagged"] = (
        natural["IPI_DAYS"]
        <= THRESHOLD * natural["HIST_MEDIAN_IPI"]
    )
    evaluation["flagged"] = (
        evaluation["IPI_DAYS"]
        <= THRESHOLD * evaluation["HIST_MEDIAN_IPI"]
    )
    natural["score"] = ranking_score(natural, method)
    evaluation["score"] = ranking_score(evaluation, method)

    flagged = evaluation[evaluation["flagged"]].sort_values(
        "score", ascending=False
    )
    synthetic = evaluation["synthetic_label"].eq(1)
    top_150 = flagged.head(150)
    top_150_recall = (
        float(top_150["synthetic_label"].sum() / synthetic.sum())
        if synthetic.sum()
        else 0.0
    )
    synthetic_flagged = evaluation.loc[
        synthetic, "flagged"
    ].mean()
    natural_alert_rate = natural["flagged"].mean()
    synthetic_ranks = np.flatnonzero(
        flagged["synthetic_label"].to_numpy() == 1
    ) + 1

    return {
        "method": method,
        "injection_ratio": ratio,
        "threshold": THRESHOLD,
        "train_events": train_size,
        "natural_test_events": len(natural),
        "natural_alerts": int(natural["flagged"].sum()),
        "natural_alert_rate": float(natural_alert_rate),
        "synthetic_events": int(synthetic.sum()),
        "synthetic_detection_rate": float(synthetic_flagged),
        "recall_at_150": top_150_recall,
        "median_flagged_synthetic_rank": float(
            np.median(synthetic_ranks)
        ),
    }


def main() -> None:
    events = load_events()
    split = int(len(events) * (1 - TEST_FRACTION))
    train = events.iloc[:split]
    natural = events.iloc[split:].copy()
    natural["synthetic_label"] = 0

    rows = []
    cases = []
    methods = [
        "threshold_only",
        "earlyness_only",
        "robust_only",
        "combined_70_30",
    ]
    for ratio in INJECTION_RATIOS:
        synthetic = natural.copy()
        synthetic["IPI_DAYS"] = (
            ratio * synthetic["HIST_MEDIAN_IPI"]
        )
        synthetic["synthetic_label"] = 1
        evaluation = pd.concat(
            [natural, synthetic], ignore_index=True
        )
        for method in methods:
            rows.append(
                evaluate(
                    method,
                    natural,
                    evaluation,
                    ratio,
                    len(train),
                )
            )
        cases.append(
            evaluation.assign(
                injection_ratio=ratio,
                threshold_flag=(
                    evaluation["IPI_DAYS"]
                    <= THRESHOLD * evaluation["HIST_MEDIAN_IPI"]
                ),
            )
        )

    metrics = pd.DataFrame(rows)
    cases_frame = pd.concat(cases, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(
        RESULTS_DIR / "ablation_metrics.csv", index=False
    )
    cases_frame.to_csv(
        RESULTS_DIR / "ablation_cases.csv", index=False
    )
    print("Ablation metrics:")
    print(metrics.to_string(index=False))
    print("\nSaved:", RESULTS_DIR / "ablation_metrics.csv")
    print("Saved:", RESULTS_DIR / "ablation_cases.csv")


if __name__ == "__main__":
    main()
