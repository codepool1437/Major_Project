
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load all yearly CSV files
csv_files = sorted(RAW_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

frames = []

for path in csv_files:
    print(f"Loading {path.name}...")

    part = pd.read_csv(path, sep=";", low_memory=False)
    part.columns = part.columns.str.strip()

    required = [
        "CUSTOMER_ID",
        "ITEM_ID",
        "REQUEST_DATE",
        "ORDER_ID",
    ]

    missing = [c for c in required if c not in part.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    part["REQUEST_DATE"] = pd.to_datetime(
        part["REQUEST_DATE"], errors="coerce"
    )

    frames.append(part)

df = pd.concat(frames, ignore_index=True)

print("\n========== RAW DATA ==========")
print("Files:", len(csv_files))
print("Rows:", len(df))
print("Date range:", df["REQUEST_DATE"].min(),
      "to", df["REQUEST_DATE"].max())

# Exclude records without the fields needed for temporal analysis.
# Keep the original CSV files unchanged.
valid = df.dropna(
    subset=["CUSTOMER_ID", "ITEM_ID", "REQUEST_DATE"]
).copy()

# One observation per customer-item-date for interval calculation.
# Multiple order lines on the same day cannot produce a positive
# day-based interval, so they are treated as one purchase date here.
purchase_dates = (
    valid[["CUSTOMER_ID", "ITEM_ID", "REQUEST_DATE"]]
    .drop_duplicates()
    .sort_values(
        ["CUSTOMER_ID", "ITEM_ID", "REQUEST_DATE"]
    )
    .reset_index(drop=True)
)

# Previous distinct purchase date for the same customer-item pair
grouped_dates = purchase_dates.groupby(
    ["CUSTOMER_ID", "ITEM_ID"]
)["REQUEST_DATE"]

purchase_dates["PREVIOUS_PURCHASE_DATE"] = (
    grouped_dates.shift(1)
)

# Interval between consecutive purchase dates, in days
purchase_dates["IPI_DAYS"] = (
    purchase_dates["REQUEST_DATE"]
    - purchase_dates["PREVIOUS_PURCHASE_DATE"]
).dt.total_seconds() / 86400

# First purchase in a pair's observed history has no previous interval
ipi = purchase_dates.dropna(
    subset=["PREVIOUS_PURCHASE_DATE", "IPI_DAYS"]
).copy()

ipi["IPI_DAYS"] = ipi["IPI_DAYS"].astype(float)

# Save detailed interval records
ipi_path = PROCESSED_DIR / "purchase_intervals.csv"
ipi.to_csv(ipi_path, index=False)

# Save a basic summary for inspection
summary = (
    ipi.groupby(["CUSTOMER_ID", "ITEM_ID"])
    .agg(
        number_of_intervals=("IPI_DAYS", "count"),
        median_ipi_days=("IPI_DAYS", "median"),
        mean_ipi_days=("IPI_DAYS", "mean"),
        minimum_ipi_days=("IPI_DAYS", "min"),
        maximum_ipi_days=("IPI_DAYS", "max"),
    )
    .reset_index()
)

summary_path = RESULTS_DIR / "ipi_summary.csv"
summary.to_csv(summary_path, index=False)

print("\n========== IPI RESULTS ==========")
print("Distinct customer-item-date records:", len(purchase_dates))
print("Calculated intervals:", len(ipi))
print("Customer-item pairs with intervals:", len(summary))
print("\nIPI distribution:")
print(ipi["IPI_DAYS"].describe().round(2))

print("\nSaved:")
print(ipi_path)
print(summary_path)
print("\nIPI calculation complete.")
