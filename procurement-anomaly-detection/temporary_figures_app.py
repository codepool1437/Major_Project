"""Temporary Streamlit viewer for the final paper figures.

This file is intentionally standalone and can be deleted after screenshots are
captured. It reads only the retained result CSVs; it does not alter the
pipeline or result artifacts.
"""

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold = pd.read_csv(RESULTS / "final_threshold_comparison.csv")
    detectors = pd.read_csv(RESULTS / "algorithm_comparison_metrics.csv")
    return threshold, detectors


def threshold_figure(table: pd.DataFrame) -> BytesIO:
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=180)
    ordered = table.sort_values("threshold_percent")
    bars = ax.bar(
        ordered["threshold_percent"].astype(str) + "%",
        ordered["flagged_events"],
        color="#2f75b5",
        edgecolor="#17365d",
        linewidth=0.8,
    )
    ax.bar_label(bars, labels=[f"{v:,}" for v in ordered["flagged_events"]], padding=4)
    ax.set_title("Alert volume by early-reorder threshold", fontsize=15, pad=12)
    ax.set_xlabel("Threshold (% of historical median IPI)")
    ax.set_ylabel("Flagged eligible events")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.text(
        0.01,
        0.01,
        "Source: final_threshold_comparison.csv | 64,892 eligible events",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output = BytesIO()
    fig.savefig(output, format="png", bbox_inches="tight")
    plt.close(fig)
    output.seek(0)
    return output


def detector_figure(table: pd.DataFrame) -> BytesIO:
    selected = table[table["injection_ratio"] == 0.3].copy()
    selected = selected[
        (
            (selected["method"] == "median_threshold")
            | (
                (selected["method"] == "median_mad")
                & (selected["parameter"] == "robust_z>=1.0")
            )
            | (selected["method"].isin(["isolation_forest", "lof"]))
        )
    ].copy()
    labels = {
        ("median_threshold", "ratio<=0.50"): "Median 50%",
        ("median_threshold", "ratio<=0.40"): "Median 40%",
        ("median_threshold", "ratio<=0.30"): "Median 30%",
        ("median_threshold", "ratio<=0.20"): "Median 20%",
        ("median_mad", "robust_z>=1.0"): "Median-MAD\nz >= 1",
        ("isolation_forest", "train_score_p95"): "Isolation\nForest",
        ("lof", "neighbors=35_train_score_p95"): "LOF",
    }
    selected["detector"] = [
        labels[(method, parameter)]
        for method, parameter in zip(selected["method"], selected["parameter"])
    ]

    metrics = [
        ("synthetic_detection_rate", "Synthetic detection rate", "#2f75b5"),
        ("precision_controlled", "Controlled precision", "#70ad47"),
        ("natural_alert_rate", "Natural alert rate", "#ed7d31"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), dpi=180, sharey=False)
    for ax, (column, title, color) in zip(axes, metrics):
        values = selected[column].fillna(0).mul(100)
        bars = ax.bar(selected["detector"], values, color=color)
        ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=3, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Percent")
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Detector comparison at the 30% synthetic injection ratio",
        fontsize=15,
        y=1.03,
    )
    fig.text(
        0.01,
        0.01,
        "Source: algorithm_comparison_metrics.csv, injection ratio = 0.30; natural alert rate uses the chronological holdout.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    output = BytesIO()
    fig.savefig(output, format="png", bbox_inches="tight")
    plt.close(fig)
    output.seek(0)
    return output


st.set_page_config(
    page_title="Final Paper Figures",
    page_icon="📊",
    layout="wide",
)

st.title("Final Paper Figures")
st.caption(
    "Temporary evidence viewer using the retained project results. "
    "Use the download buttons to save publication figures; delete this file afterward."
)

thresholds, detector_metrics = load_results()

tab3, tab4 = st.tabs(["Figure 3 — Threshold vs alert volume", "Figure 4 — Detector comparison"])

with tab3:
    st.subheader("Figure 3 — Threshold versus alert volume")
    st.markdown(
        "This chart uses the final threshold-sensitivity results. "
        "A stricter threshold flags fewer eligible events."
    )
    st.dataframe(
        thresholds[
            ["threshold_percent", "eligible_events", "flagged_events", "alert_percentage"]
        ].rename(
            columns={
                "threshold_percent": "Threshold (%)",
                "eligible_events": "Eligible events",
                "flagged_events": "Flagged events",
                "alert_percentage": "Alert rate (%)",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    threshold_png = threshold_figure(thresholds)
    st.image(threshold_png, use_container_width=True)
    st.download_button(
        "Download Figure 3 as threshold_vs_alert_volume.png",
        data=threshold_png.getvalue(),
        file_name="threshold_vs_alert_volume.png",
        mime="image/png",
        key="download_threshold",
    )

with tab4:
    st.subheader("Figure 4 — Detector comparison")
    st.markdown(
        "Controlled experiment at a 30% synthetic injection ratio. "
        "All four median-threshold settings are shown separately. Values are read directly from "
        "algorithm_comparison_metrics.csv; the CSV is the authoritative machine-readable result. "
        "Natural alert rate is reported separately and is not a ground-truth error rate."
    )
    selected = detector_metrics.copy()
    display = selected[
        ["method", "parameter", "synthetic_detection_rate", "precision_controlled", "natural_alert_rate"]
    ].copy()
    for column in [
        "synthetic_detection_rate",
        "precision_controlled",
        "natural_alert_rate",
    ]:
        display[column] = (display[column] * 100).round(2).astype(str) + "%"
    st.dataframe(
        display.rename(
            columns={
                "method": "Detector",
                "parameter": "Parameter",
                "synthetic_detection_rate": "Synthetic detection rate",
                "precision_controlled": "Controlled precision",
                "natural_alert_rate": "Natural alert rate",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    detector_png = detector_figure(detector_metrics)
    st.image(detector_png, use_container_width=True)
    st.download_button(
        "Download Figure 4 as detector_comparison_final.png",
        data=detector_png.getvalue(),
        file_name="detector_comparison_final.png",
        mime="image/png",
        key="download_detector",
    )
