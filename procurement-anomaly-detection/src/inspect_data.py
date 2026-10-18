
from pathlib import Path
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load all yearly CSV files in data/raw so the inspection covers the full dataset.
csv_files = list(RAW_DIR.glob("*.csv"))

if not csv_files:
    raise ValueError(
        f"Expected at least one CSV in {RAW_DIR}, but found none."
    )

# IMPORTANT: this dataset uses semicolons
frames = [
    pd.read_csv(csv_path, sep=";", low_memory=False)
    for csv_path in sorted(csv_files)
]
df = pd.concat(frames, ignore_index=True)

# Standardize column-name whitespace only
df.columns = df.columns.str.strip()

required = ["ORDER_ID", "CUSTOMER_ID", "REQUEST_DATE", "ITEM_ID"]
missing_columns = [c for c in required if c not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Parse dates, preserving invalid dates as NaT for inspection
df["REQUEST_DATE"] = pd.to_datetime(
    df["REQUEST_DATE"], errors="coerce"
)

print("========== DATASET ==========")
print("Files:", ", ".join(path.name for path in sorted(csv_files)))
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Date range:", df["REQUEST_DATE"].min(),
      "to", df["REQUEST_DATE"].max())

print("\n========== MISSING VALUES ==========")
print(df.isna().sum().to_string())

print("\n========== DUPLICATES ==========")
print("Exact duplicate rows:", df.duplicated().sum())
print(
    "Repeated order-item pairs:",
    df.duplicated(["ORDER_ID", "ITEM_ID"]).sum()
)

# Analyse purchase dates without changing the original data
valid = df.dropna(
    subset=["CUSTOMER_ID", "ITEM_ID", "REQUEST_DATE"]
).copy()

# Multiple rows for the same pair on the same date count as
# one purchase date for the initial interval analysis.
purchase_dates = valid[
    ["CUSTOMER_ID", "ITEM_ID", "REQUEST_DATE"]
].drop_duplicates()

pair_history = (
    purchase_dates.groupby(["CUSTOMER_ID", "ITEM_ID"])
    .agg(
        distinct_purchase_dates=("REQUEST_DATE", "nunique"),
        first_date=("REQUEST_DATE", "min"),
        last_date=("REQUEST_DATE", "max")
    )
    .reset_index()
)

print("\n========== REPEAT-PURCHASE HISTORY ==========")
print("Customer-item pairs:", len(pair_history))
print("Pairs with 2+ dates:",
      (pair_history["distinct_purchase_dates"] >= 2).sum())
print("Pairs with 3+ dates:",
      (pair_history["distinct_purchase_dates"] >= 3).sum())
print("Pairs with 4+ dates:",
      (pair_history["distinct_purchase_dates"] >= 4).sum())

# Save reports; do not modify or deduplicate the raw dataset.
pair_history.to_csv(
    RESULTS_DIR / "customer_item_history_profile.csv",
    index=False
)

df.isna().sum().rename("missing_count").to_csv(
    RESULTS_DIR / "missing_values_report.csv"
)

print("\nReports saved to:", RESULTS_DIR)
print("Inspection complete. Original data remains unchanged.")
