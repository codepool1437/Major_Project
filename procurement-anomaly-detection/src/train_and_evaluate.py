
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROCESSED_DIR / "model_ready_events.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)

required = [
    "CUSTOMER_ID",
    "ITEM_ID",
    "REQUEST_DATE",
    "IPI_DAYS",
    "HISTORY_COUNT",
    "HIST_MEDIAN_IPI",
    "HIST_MAD_IPI",
    "RECENT_MEDIAN_IPI",
    "IPI_TO_HIST_MEDIAN",
    "IPI_TO_RECENT_MEDIAN",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Remove rows that cannot support the first experiment.
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(
    subset=[
        "IPI_DAYS",
        "HIST_MEDIAN_IPI",
        "RECENT_MEDIAN_IPI",
        "IPI_TO_HIST_MEDIAN",
    ]
).copy()

df = df[df["HIST_MEDIAN_IPI"] > 0].copy()
df = df.sort_values("REQUEST_DATE").reset_index(drop=True)

print("Usable events:", len(df))

# --------------------------------------------------
# 1. Statistical early-reorder baseline
# --------------------------------------------------
# A conservative first rule: the current interval must be
# at most half the historical median.
# MAD is also used to inspect robust deviation.
df["ROBUST_EARLY_DEVIATION"] = (
    df["HIST_MEDIAN_IPI"] - df["IPI_DAYS"]
) / df["HIST_MAD_IPI"].replace(0, np.nan)

df["BASELINE_EARLY"] = (
    (df["IPI_DAYS"] <= 0.5 * df["HIST_MEDIAN_IPI"])
    & (df["IPI_DAYS"] < df["HIST_MEDIAN_IPI"])
)

print(
    "Baseline potential early events:",
    int(df["BASELINE_EARLY"].sum())
)

# --------------------------------------------------
# 2. Train Isolation Forest
# --------------------------------------------------
# Use only temporal features. Do not use customer/item IDs
# as numeric model features.
feature_cols = [
    "IPI_DAYS",
    "HIST_MEDIAN_IPI",
    "HIST_MAD_IPI",
    "RECENT_MEDIAN_IPI",
    "IPI_TO_HIST_MEDIAN",
    "IPI_TO_RECENT_MEDIAN",
    "HISTORY_COUNT",
]

X = df[feature_cols].replace([np.inf, -np.inf], np.nan)

# Fill feature gaps with training-data medians.
# A later improvement will use a proper time-based train/test split.
X = X.fillna(X.median()).fillna(0)

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
    n_jobs=-1,
)

model.fit(X)

# sklearn: -1 means anomaly; 1 means normal.
df["IF_PREDICTION"] = model.predict(X)

# Larger values from this transformation mean more anomalous.
df["IF_ANOMALY_SCORE"] = -model.score_samples(X)

# Isolation Forest detects unusual observations in either direction.
# For this project, only retain IF anomalies that are also unusually
# short relative to their own historical median.
df["IF_EARLY_CANDIDATE"] = (
    (df["IF_PREDICTION"] == -1)
    & (df["IPI_DAYS"] < df["HIST_MEDIAN_IPI"])
)

# Save ranked candidate events.
ranked = df.sort_values(
    "IF_ANOMALY_SCORE", ascending=False
)

ranked.to_csv(
    RESULTS_DIR / "ranked_anomaly_candidates.csv",
    index=False,
)

df.to_csv(
    RESULTS_DIR / "model_scored_events.csv",
    index=False,
)

# --------------------------------------------------
# 3. Controlled synthetic evaluation
# --------------------------------------------------
# IMPORTANT:
# This is a controlled test, not real-world ground truth.
# The test contains held-out natural events and copies with
# artificially shortened IPI values.
#
# Train/test split is chronological to reduce temporal leakage.
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

if len(train_df) < 100 or len(test_df) < 20:
    raise ValueError("Not enough events for controlled evaluation.")

X_train = train_df[feature_cols].replace(
    [np.inf, -np.inf], np.nan
)
train_medians = X_train.median()
X_train = X_train.fillna(train_medians).fillna(0)

eval_model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
    n_jobs=-1,
)
eval_model.fit(X_train)

# Natural held-out rows are not proven normal; they are only
# the reference group in this controlled experiment.
normal_like = test_df.copy()
normal_like["synthetic_label"] = 0

# Create artificial early-reorder examples by shortening IPI
# to 20% of each event's historical median.
synthetic = test_df.copy()
synthetic["IPI_DAYS"] = (
    0.20 * synthetic["HIST_MEDIAN_IPI"]
).clip(lower=1)

synthetic["IPI_TO_HIST_MEDIAN"] = (
    synthetic["IPI_DAYS"] / synthetic["HIST_MEDIAN_IPI"]
)
synthetic["IPI_TO_RECENT_MEDIAN"] = (
    synthetic["IPI_DAYS"] /
    synthetic["RECENT_MEDIAN_IPI"].replace(0, np.nan)
)
synthetic["DAYS_BELOW_HIST_MEDIAN"] = (
    synthetic["HIST_MEDIAN_IPI"] - synthetic["IPI_DAYS"]
)
synthetic["synthetic_label"] = 1

evaluation = pd.concat(
    [normal_like, synthetic],
    ignore_index=True,
)

X_eval = evaluation[feature_cols].replace(
    [np.inf, -np.inf], np.nan
)
X_eval = X_eval.fillna(train_medians).fillna(0)

evaluation["IF_PREDICTION"] = eval_model.predict(X_eval)
evaluation["IF_ANOMALY_SCORE"] = -eval_model.score_samples(X_eval)

# IF early candidate = IF anomaly AND current interval is shorter
evaluation["IF_EARLY_PREDICTION"] = (
    (evaluation["IF_PREDICTION"] == -1)
    & (
        evaluation["IPI_DAYS"]
        < evaluation["HIST_MEDIAN_IPI"]
    )
).astype(int)

# Statistical baseline on the same controlled evaluation set.
evaluation["BASELINE_PREDICTION"] = (
    evaluation["IPI_DAYS"]
    <= 0.5 * evaluation["HIST_MEDIAN_IPI"]
).astype(int)

y_true = evaluation["synthetic_label"]

def report_metrics(name, predictions):
    precision = precision_score(
        y_true, predictions, zero_division=0
    )
    recall = recall_score(
        y_true, predictions, zero_division=0
    )
    f1 = f1_score(
        y_true, predictions, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "method": name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }

metrics = pd.DataFrame([
    report_metrics(
        "Median-based baseline",
        evaluation["BASELINE_PREDICTION"],
    ),
    report_metrics(
        "Isolation Forest + early-direction filter",
        evaluation["IF_EARLY_PREDICTION"],
    ),
])

metrics.to_csv(
    RESULTS_DIR / "controlled_evaluation_metrics.csv",
    index=False,
)

evaluation.to_csv(
    RESULTS_DIR / "controlled_evaluation_cases.csv",
    index=False,
)

print("\n========== CONTROLLED EVALUATION ==========")
print(metrics.to_string(index=False))
print("\nSaved:")
print(RESULTS_DIR / "ranked_anomaly_candidates.csv")
print(RESULTS_DIR / "model_scored_events.csv")
print(RESULTS_DIR / "controlled_evaluation_metrics.csv")
print(RESULTS_DIR / "controlled_evaluation_cases.csv")
print(
    "\nReminder: metrics measure detection of injected early cases "
    "against a natural reference group, not real-world labelled anomalies."
)
