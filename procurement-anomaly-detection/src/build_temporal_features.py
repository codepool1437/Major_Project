
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load all yearly files
files = sorted(RAW_DIR.glob("*.csv"))
if not files:
    raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

frames = []

for file in files:
    print("Reading:", file.name)
    data = pd.read_csv(file, sep=";", low_memory=False)
    data.columns = data.columns.str.strip()

    required = ["CUSTOMER_ID", "ITEM_ID", "REQUEST_DATE"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"{file.name}: missing columns {missing}")

    data["REQUEST_DATE"] = pd.to_datetime(
        data["REQUEST_DATE"], errors="coerce"
    )
    frames.append(data[required])

df = pd.concat(frames, ignore_index=True)

# 2. Remove records missing required fields.
# Deduplicate dates only for temporal interval analysis.
df = df.dropna(
    subset=["CUSTOMER_ID", "ITEM_ID", "REQUEST_DATE"]
)

dates = (
    df.drop_duplicates(
        ["CUSTOMER_ID", "ITEM_ID", "REQUEST_DATE"]
    )
    .sort_values(
        ["CUSTOMER_ID", "ITEM_ID", "REQUEST_DATE"]
    )
    .reset_index(drop=True)
)

# 3. Calculate the interval between consecutive purchase dates
group_cols = ["CUSTOMER_ID", "ITEM_ID"]
grouped = dates.groupby(group_cols)["REQUEST_DATE"]

dates["PREVIOUS_PURCHASE_DATE"] = grouped.shift(1)
dates["IPI_DAYS"] = (
    dates["REQUEST_DATE"] - dates["PREVIOUS_PURCHASE_DATE"]
).dt.total_seconds() / 86400

events = dates.dropna(
    subset=["PREVIOUS_PURCHASE_DATE", "IPI_DAYS"]
).copy()

events["IPI_DAYS"] = events["IPI_DAYS"].astype(float)

# 4. Historical features: current IPI is NOT included
# in the historical statistics for that event.
def add_past_only_features(group):
    group = group.sort_values("REQUEST_DATE").copy()
    intervals = group["IPI_DAYS"]

    # Shift first so the current interval is excluded
    past = intervals.shift(1)

    group["HISTORY_COUNT"] = past.expanding().count()
    group["HIST_MEDIAN_IPI"] = past.expanding().median()

    # Median absolute deviation using past intervals only
    def past_mad(values):
        values = values.dropna()
        if len(values) == 0:
            return np.nan
        median = values.median()
        return (values - median).abs().median()

    group["HIST_MAD_IPI"] = past.expanding().apply(
        past_mad, raw=False
    )

    # Recent history: up to the previous 3 intervals
    group["RECENT_MEDIAN_IPI"] = (
        past.rolling(window=3, min_periods=1).median()
    )

    # Ratios help compare customers with different cycles
    group["IPI_TO_HIST_MEDIAN"] = (
        group["IPI_DAYS"] /
        group["HIST_MEDIAN_IPI"].replace(0, np.nan)
    )

    group["IPI_TO_RECENT_MEDIAN"] = (
        group["IPI_DAYS"] /
        group["RECENT_MEDIAN_IPI"].replace(0, np.nan)
    )

    # Negative values indicate intervals below the historical median.
    # This is a feature, not yet a final anomaly decision.
    group["DAYS_BELOW_HIST_MEDIAN"] = (
        group["HIST_MEDIAN_IPI"] - group["IPI_DAYS"]
    )

    return group

features = pd.concat(
    [
        add_past_only_features(group)
        for _, group in events.groupby(group_cols, sort=False)
    ],
    ignore_index=True
)

# 5. Flag whether an event has enough history for evaluation.
# Three previous intervals is our initial minimum; we can test
# other history requirements during evaluation.
features["SUFFICIENT_HISTORY"] = (
    features["HISTORY_COUNT"] >= 3
)

# 6. Save all events and a model-ready subset
all_path = PROCESSED_DIR / "temporal_event_features.csv"
features.to_csv(all_path, index=False)

model_data = features.loc[
    features["SUFFICIENT_HISTORY"]
].copy()

model_path = PROCESSED_DIR / "model_ready_events.csv"
model_data.to_csv(model_path, index=False)

print("\n========== FEATURE BUILD RESULTS ==========")
print("All reorder events:", len(features))
print("Events with at least 3 previous intervals:",
      len(model_data))
print(
    "Pairs represented:",
    model_data[["CUSTOMER_ID", "ITEM_ID"]]
    .drop_duplicates()
    .shape[0]
)
print("Missing historical median:",
      features["HIST_MEDIAN_IPI"].isna().sum())

print("\nCurrent IPI distribution:")
print(features["IPI_DAYS"].describe().round(2))

print("\nSaved:")
print(all_path)
print(model_path)
print("\nFeature engineering complete.")
