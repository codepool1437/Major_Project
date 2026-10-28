"""Supplementary robustness experiments for the revised paper.

Run after ``build_temporal_features.py``. Part 1 audits the raw files; part 2
runs minimum-history, alert-concentration, fixed-day baseline, rolling-fold,
and injection-scenario experiments. Results are written to ``results/``.
Injected cases are synthetic, not real anomaly labels.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import hashlib
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


warnings.filterwarnings("ignore")

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FEATURE_COLUMNS = [
    "IPI_DAYS",
    "HIST_MEDIAN_IPI",
    "HIST_MAD_IPI",
    "RECENT_MEDIAN_IPI",
    "IPI_TO_HIST_MEDIAN",
    "IPI_TO_RECENT_MEDIAN",
    "HISTORY_COUNT",
]


def audit_raw_files() -> None:
    frames = []
    for path in sorted(RAW.glob("*.csv")):
        frame = pd.read_csv(
            path,
            sep=";",
            low_memory=False,
            usecols=["ORDER_ID", "CUSTOMER_ID", "REQUEST_DATE", "ITEM_ID"],
        )
        frame["file"] = path.name[:4]
        frame["dt"] = pd.to_datetime(frame["REQUEST_DATE"], errors="coerce")
        file_year = int(path.name[:4])
        print(
            path.name,
            len(frame),
            frame["dt"].min().date(),
            frame["dt"].max().date(),
            "rows w/ date year != file:",
            int((frame["dt"].dt.year != file_year).sum()),
            hashlib.sha256(path.read_bytes()).hexdigest()[:12],
        )
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    key_columns = ["CUSTOMER_ID", "ITEM_ID", "dt"]
    print("rows", len(data))
    print(
        "missing keys",
        int(data[key_columns].isna().any(axis=1).sum()),
        data[key_columns].isna().sum().to_dict(),
    )
    print(
        "unique ORDER_ID",
        data["ORDER_ID"].nunique(),
        "dup order rows",
        int(data.duplicated("ORDER_ID").sum()),
    )
    print("customers", data["CUSTOMER_ID"].nunique(), "items", data["ITEM_ID"].nunique())

    valid = data.dropna(subset=["CUSTOMER_ID", "ITEM_ID", "dt"])
    distinct_events = valid.drop_duplicates(["CUSTOMER_ID", "ITEM_ID", "dt"])
    print("valid", len(valid), "distinct cust-item-date", len(distinct_events))
    print("collapsed", len(valid) - len(distinct_events))
    print("pairs", distinct_events.groupby(["CUSTOMER_ID", "ITEM_ID"]).ngroups)
    print(
        "weekday dist",
        distinct_events["dt"].dt.dayofweek.value_counts().sort_index().to_dict(),
    )
    print(
        "monthly rows min/max",
        valid.groupby(valid["dt"].dt.to_period("M")).size().agg(["min", "max"]).to_dict(),
    )
    print("overall range", valid["dt"].min(), valid["dt"].max())
    print(data.groupby("file")["dt"].agg(["min", "max"]))


def eligible_events(events: pd.DataFrame, minimum_history: int = 3) -> pd.DataFrame:
    eligible = events.dropna(subset=FEATURE_COLUMNS).copy()
    eligible = eligible[
        (eligible["HISTORY_COUNT"] >= minimum_history)
        & (eligible["IPI_DAYS"] >= 0)
        & (eligible["HIST_MEDIAN_IPI"] > 0)
    ]
    return eligible.sort_values("REQUEST_DATE").reset_index(drop=True)


def robust_prediction(data: pd.DataFrame, cutoff: float) -> np.ndarray:
    scale = (1.4826 * data["HIST_MAD_IPI"]).clip(lower=1.0)
    return (
        (data["HIST_MEDIAN_IPI"] - data["IPI_DAYS"]) / scale >= cutoff
    ).to_numpy()


def inject(data: pd.DataFrame, scenario: str, rng: np.random.Generator) -> pd.DataFrame:
    synthetic = data.copy()
    median = synthetic["HIST_MEDIAN_IPI"].to_numpy()
    if scenario == "A_0.30M":
        synthetic["IPI_DAYS"] = 0.30 * median
    elif scenario == "B_0.25M":
        synthetic["IPI_DAYS"] = 0.25 * median
    elif scenario == "B_0.35M":
        synthetic["IPI_DAYS"] = 0.35 * median
    elif scenario == "C_U(.05,.60)M":
        synthetic["IPI_DAYS"] = rng.uniform(0.05, 0.60, len(synthetic)) * median
    elif scenario == "D_0.3xIPI":
        synthetic["IPI_DAYS"] = 0.3 * synthetic["IPI_DAYS"].to_numpy()
    elif scenario == "E_1day":
        synthetic["IPI_DAYS"] = 1.0
    else:
        raise ValueError(f"Unknown injection scenario: {scenario}")
    synthetic["IPI_TO_HIST_MEDIAN"] = (
        synthetic["IPI_DAYS"] / synthetic["HIST_MEDIAN_IPI"]
    )
    synthetic["IPI_TO_RECENT_MEDIAN"] = (
        synthetic["IPI_DAYS"]
        / synthetic["RECENT_MEDIAN_IPI"].replace(0, np.nan)
    )
    return synthetic


SCENARIOS = [
    "A_0.30M",
    "B_0.25M",
    "B_0.35M",
    "C_U(.05,.60)M",
    "D_0.3xIPI",
    "E_1day",
]


def feature_matrix(data: pd.DataFrame, fill_values: pd.Series) -> pd.DataFrame:
    return (
        data[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(fill_values)
        .fillna(0)
    )


def run_split(
    train: pd.DataFrame, test: pd.DataFrame, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fill_values = (
        train[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .median()
    )
    train_matrix = feature_matrix(train, fill_values)
    train_alert_rate = (
        train["IPI_DAYS"] <= 0.3 * train["HIST_MEDIAN_IPI"]
    ).mean()
    candidate_days = np.arange(1, 400)
    candidate_rates = np.array(
        [(train["IPI_DAYS"] <= days).mean() for days in candidate_days]
    )
    matched_day = int(
        candidate_days[np.argmin(np.abs(candidate_rates - train_alert_rate))]
    )

    isolation_forest = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=seed,
        n_jobs=-1,
    ).fit(train_matrix)
    isolation_cutoff = np.quantile(
        -isolation_forest.score_samples(train_matrix), 0.95
    )
    lof = LocalOutlierFactor(
        n_neighbors=35,
        contamination=0.05,
        novelty=True,
    ).fit(train_matrix)
    lof_cutoff = np.quantile(-lof.score_samples(train_matrix), 0.95)

    def predictions(data: pd.DataFrame) -> dict[str, np.ndarray]:
        matrix = feature_matrix(data, fill_values)
        below_median = (data["IPI_DAYS"] < data["HIST_MEDIAN_IPI"]).to_numpy()
        return {
            "Median 30%": (
                data["IPI_DAYS"] <= 0.3 * data["HIST_MEDIAN_IPI"]
            ).to_numpy(),
            "Fixed-day (budget-matched)": (
                data["IPI_DAYS"] <= matched_day
            ).to_numpy(),
            "Fixed-day <= 7 d": (data["IPI_DAYS"] <= 7).to_numpy(),
            "Fixed-day <= 14 d": (data["IPI_DAYS"] <= 14).to_numpy(),
            "Median-MAD z>=1": robust_prediction(data, 1.0),
            "Isolation Forest": (
                (-isolation_forest.score_samples(matrix) >= isolation_cutoff)
                & below_median
            ),
            "LOF": ((-lof.score_samples(matrix) >= lof_cutoff) & below_median),
        }

    natural_predictions = predictions(test)
    rows = []
    for scenario in SCENARIOS:
        synthetic = inject(test, scenario, rng)
        synthetic_predictions = predictions(synthetic)
        for method, natural in natural_predictions.items():
            true_positives = int(synthetic_predictions[method].sum())
            natural_alerts = int(natural.sum())
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "natural_alert_rate": natural.mean(),
                    "detection": true_positives / len(synthetic),
                    "precision": (
                        true_positives / (true_positives + natural_alerts)
                        if true_positives + natural_alerts
                        else 0.0
                    ),
                    "d_match": matched_day,
                }
            )
    return pd.DataFrame(rows)


def run_experiments() -> None:
    events = pd.read_csv(
        PROCESSED / "temporal_event_features.csv",
        low_memory=False,
        parse_dates=["REQUEST_DATE", "PREVIOUS_PURCHASE_DATE"],
    ).replace([np.inf, -np.inf], np.nan)
    eligible = eligible_events(events)
    alert_mask = eligible["IPI_DAYS"] <= 0.3 * eligible["HIST_MEDIAN_IPI"]

    RESULTS.mkdir(exist_ok=True)
    print("all", len(events), "eligible", len(eligible), "alerts30", int(alert_mask.sum()))
    print(
        "events with HISTORY>=3:",
        int((events["HISTORY_COUNT"] >= 3).sum()),
        "of which M<=0:",
        int(
            (
                (events["HISTORY_COUNT"] >= 3)
                & (events["HIST_MEDIAN_IPI"] <= 0)
            ).sum()
        ),
    )
    print(
        "IPI==0 any:",
        int((events["IPI_DAYS"] <= 0).sum()),
        "min IPI",
        events["IPI_DAYS"].min(),
        "median M among elig",
        eligible["HIST_MEDIAN_IPI"].median(),
        "share MAD==0 among elig",
        (eligible["HIST_MAD_IPI"] == 0).mean(),
    )
    print(
        "MAD==0 share among alerts",
        (eligible.loc[alert_mask, "HIST_MAD_IPI"] == 0).mean(),
    )

    minimum_history_rows = []
    for minimum_history in [2, 3, 5, 8, 12]:
        filtered = eligible_events(events, minimum_history)
        mask = filtered["IPI_DAYS"] <= 0.3 * filtered["HIST_MEDIAN_IPI"]
        minimum_history_rows.append(
            {
                "k": minimum_history,
                "eligible": len(filtered),
                "pairs": filtered.groupby(["CUSTOMER_ID", "ITEM_ID"]).ngroups,
                "alerts": int(mask.sum()),
                "rate": mask.mean(),
            }
        )
    minimum_history = pd.DataFrame(minimum_history_rows)
    print(minimum_history)
    minimum_history.to_csv(RESULTS / "minhist.csv", index=False)

    cross_year = eligible["REQUEST_DATE"].dt.year != eligible[
        "PREVIOUS_PURCHASE_DATE"
    ].dt.year
    print(
        "cross-file share eligible",
        cross_year.mean(),
        "alert rate cross",
        alert_mask[cross_year].mean(),
        "within",
        alert_mask[~cross_year].mean(),
    )
    alerts = eligible.loc[alert_mask]
    print(
        "alert pairs",
        alerts.groupby(["CUSTOMER_ID", "ITEM_ID"]).ngroups,
        "alert customers",
        alerts["CUSTOMER_ID"].nunique(),
        "of",
        eligible["CUSTOMER_ID"].nunique(),
    )
    customer_counts = alerts["CUSTOMER_ID"].value_counts()
    print("top10 customers share", customer_counts.head(10).sum() / len(alerts))

    split = int(len(eligible) * 0.8)
    results = [run_split(eligible.iloc[:split], eligible.iloc[split:]).assign(fold="80/20")]
    bounds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for index in range(5):
        start = int(len(eligible) * bounds[index])
        end = int(len(eligible) * bounds[index + 1]) if bounds[index + 1] < 1 else len(eligible)
        fold = run_split(eligible.iloc[:start], eligible.iloc[start:end]).assign(
            fold=f"R{index + 1}"
        )
        results.append(fold)
        print(
            "fold",
            index + 1,
            start,
            end - start,
            eligible.iloc[start]["REQUEST_DATE"].date(),
            eligible.iloc[end - 1]["REQUEST_DATE"].date(),
            fold["d_match"].iloc[0],
            flush=True,
        )
    rolling = pd.concat(results, ignore_index=True)
    rolling.to_csv(RESULTS / "rolling_all.csv", index=False)
    print(results[0].loc[results[0]["scenario"] == "A_0.30M"].round(4).to_string(index=False))


if __name__ == "__main__":
    audit_raw_files()
    run_experiments()
