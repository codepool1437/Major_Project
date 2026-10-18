
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import recall_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "model_ready_events.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, low_memory=False)
df["REQUEST_DATE"] = pd.to_datetime(df["REQUEST_DATE"])
df = df.sort_values("REQUEST_DATE").reset_index(drop=True)

features = [
    "IPI_DAYS",
    "HIST_MEDIAN_IPI",
    "HIST_MAD_IPI",
    "RECENT_MEDIAN_IPI",
    "IPI_TO_HIST_MEDIAN",
    "IPI_TO_RECENT_MEDIAN",
    "HISTORY_COUNT",
]

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(
    subset=["IPI_DAYS", "HIST_MEDIAN_IPI",
            "RECENT_MEDIAN_IPI"]
).copy()

df = df[df["HIST_MEDIAN_IPI"] > 0].reset_index(drop=True)

# Chronological holdout
cut = int(len(df) * 0.8)
train = df.iloc[:cut].copy()
test = df.iloc[cut:].copy()

X_train = train[features].replace([np.inf, -np.inf], np.nan)
medians = X_train.median()
X_train = X_train.fillna(medians).fillna(0)

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train)

# Reference events are not assumed to be genuinely normal.
X_test = test[features].replace([np.inf, -np.inf], np.nan)
X_test = X_test.fillna(medians).fillna(0)

reference_scores = -model.score_samples(X_test)
reference_predictions = model.predict(X_test) == -1

results = []

for ratio in [0.50, 0.30, 0.20, 0.10]:
    synthetic = test.copy()

    synthetic["IPI_DAYS"] = (
        ratio * synthetic["HIST_MEDIAN_IPI"]
    ).clip(lower=1)

    synthetic["IPI_TO_HIST_MEDIAN"] = (
        synthetic["IPI_DAYS"] /
        synthetic["HIST_MEDIAN_IPI"]
    )

    synthetic["IPI_TO_RECENT_MEDIAN"] = (
        synthetic["IPI_DAYS"] /
        synthetic["RECENT_MEDIAN_IPI"].replace(0, np.nan)
    )

    synthetic["DAYS_BELOW_HIST_MEDIAN"] = (
        synthetic["HIST_MEDIAN_IPI"] - synthetic["IPI_DAYS"]
    )

    X_syn = synthetic[features].replace(
        [np.inf, -np.inf], np.nan
    )
    X_syn = X_syn.fillna(medians).fillna(0)

    syn_scores = -model.score_samples(X_syn)
    syn_predictions = model.predict(X_syn) == -1

    # Fixed Isolation Forest threshold
    detected_if = int(syn_predictions.sum())

    # Rank-based comparison: proportion of synthetic cases scoring
    # above the 95th percentile of reference-event anomaly scores.
    reference_95 = np.quantile(reference_scores, 0.95)
    rank_detected = float((syn_scores > reference_95).mean())

    # Directional statistical baseline
    baseline_detected = float(
        (
            synthetic["IPI_DAYS"]
            <= 0.5 * synthetic["HIST_MEDIAN_IPI"]
        ).mean()
    )

    results.append({
        "synthetic_ratio": ratio,
        "synthetic_events": len(synthetic),
        "baseline_detection_rate": baseline_detected,
        "if_detection_rate_fixed_threshold": float(
            syn_predictions.mean()
        ),
        "if_detection_rate_reference_95_threshold": rank_detected,
        "if_mean_anomaly_score": float(np.mean(syn_scores)),
        "reference_mean_anomaly_score": float(
            np.mean(reference_scores)
        ),
    })

out = pd.DataFrame(results)
out.to_csv(RESULTS / "severity_experiment.csv", index=False)

print(out.to_string(index=False))
print("\nSaved:", RESULTS / "severity_experiment.csv")
print(
    "\nNote: this measures sensitivity to injected cases. "
    "Reference events are not verified normal cases."
)
