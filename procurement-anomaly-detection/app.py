
from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_ROOT / "results"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ALERTS_FILE = RESULTS_DIR / "improved_ranked_early_reorders.csv"
# Use the full event history for the inspector.
EVENTS_FILE = PROCESSED_DIR / "temporal_event_features.csv"

# Keep the model-ready dataset separate for future scoring features.
MODEL_EVENTS_FILE = PROCESSED_DIR / "model_ready_events.csv"

st.set_page_config(
    page_title="Procurement Anomaly Detection",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@st.cache_data
def load_csv(file_path, modified_time):
    # modified_time makes Streamlit reload the file
    # if it changes on disk.
    return pd.read_csv(file_path)


def load_data():
    required_files = [ALERTS_FILE, EVENTS_FILE]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        st.error(
            "Required files were not found:\n\n"
            + "\n".join(f"- {path}" for path in missing_files)
        )
        st.info(
            "Run src/build_temporal_features.py and "
            "src/improve_early_reorder_scoring.py first."
        )
        st.stop()

    alerts = load_csv(
        str(ALERTS_FILE),
        ALERTS_FILE.stat().st_mtime,
    )

    events = load_csv(
        str(EVENTS_FILE),
        EVENTS_FILE.stat().st_mtime,
    )

    for df in [alerts, events]:
        if "REQUEST_DATE" in df.columns:
            df["REQUEST_DATE"] = pd.to_datetime(
                df["REQUEST_DATE"],
                errors="coerce",
            )

        numeric_columns = [
            "IPI_DAYS",
            "HIST_MEDIAN_IPI",
            "HIST_MAD_IPI",
            "HISTORY_COUNT",
            "IPI_TO_HIST_MEDIAN",
            "IPI_TO_RECENT_MEDIAN",
            "EARLYNESS_SCORE",
            "ROBUST_EARLY_SCORE",
            "RANKING_SCORE",
        ]

        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

    return alerts, events


alerts, events = load_data()


# --------------------------------------------------
# VALIDATE DATA
# --------------------------------------------------

required_alert_columns = [
    "CUSTOMER_ID",
    "ITEM_ID",
    "REQUEST_DATE",
    "IPI_DAYS",
    "HIST_MEDIAN_IPI",
    "RANKING_SCORE",
    "SEVERITY",
]

missing_alert_columns = [
    col for col in required_alert_columns
    if col not in alerts.columns
]

if missing_alert_columns:
    st.error(
        "The improved alert file is missing required columns: "
        + ", ".join(missing_alert_columns)
    )
    st.stop()

required_event_columns = [
    "CUSTOMER_ID",
    "ITEM_ID",
    "REQUEST_DATE",
    "IPI_DAYS",
    "HIST_MEDIAN_IPI",
]

missing_event_columns = [
    col for col in required_event_columns
    if col not in events.columns
]

if missing_event_columns:
    st.error(
        "The event file is missing required columns: "
        + ", ".join(missing_event_columns)
    )
    st.stop()


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.title("Dashboard Filters")
st.sidebar.caption("Filter potential early-reorder alerts.")

filtered = alerts.copy()

customers = sorted(
    filtered["CUSTOMER_ID"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_customers = st.sidebar.multiselect(
    "Customer",
    options=customers,
    placeholder="All customers",
)

if selected_customers:
    filtered = filtered[
        filtered["CUSTOMER_ID"].astype(str).isin(
            selected_customers
        )
    ]

items = sorted(
    filtered["ITEM_ID"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_items = st.sidebar.multiselect(
    "Item / Part",
    options=items,
    placeholder="All items",
)

if selected_items:
    filtered = filtered[
        filtered["ITEM_ID"].astype(str).isin(selected_items)
    ]

severity_options = [
    severity
    for severity in ["Very High", "High", "Moderate"]
    if severity in filtered["SEVERITY"].dropna().unique()
]

selected_severities = st.sidebar.multiselect(
    "Severity",
    options=severity_options,
    default=severity_options,
)

if selected_severities:
    filtered = filtered[
        filtered["SEVERITY"].isin(selected_severities)
    ]
else:
    filtered = filtered.iloc[0:0]

date_values = filtered["REQUEST_DATE"].dropna()

if not date_values.empty:
    min_date = date_values.min().date()
    max_date = date_values.max().date()

    date_range = st.sidebar.date_input(
        "Request date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range

        filtered = filtered[
            filtered["REQUEST_DATE"].dt.date.between(
                start_date,
                end_date,
            )
        ]


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📊 B2B Procurement Anomaly Detection")

st.markdown(
    """
    **Machine Learning-Based Early Detection of Temporal Anomalies
    in B2B Procurement Using Customer–Part Purchase Patterns**

    Explore customer–part purchasing intervals and investigate
    potential unusually early reorder events.
    """
)

st.caption(
    "Alerts are screening candidates, not confirmed unnecessary "
    "purchases or fraud. The current threshold is provisional."
)


# --------------------------------------------------
# SUMMARY METRICS
# --------------------------------------------------

total_events = len(events)
total_alerts = len(alerts)
filtered_alerts = len(filtered)

if total_events:
    overall_alert_percentage = (
        100 * total_alerts / total_events
    )
else:
    overall_alert_percentage = 0.0

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Eligible purchase events",
    f"{total_events:,}",
)

col2.metric(
    "Potential early reorders",
    f"{total_alerts:,}",
)

col3.metric(
    "Overall flagged percentage",
    f"{overall_alert_percentage:.2f}%",
)

col4.metric(
    "Alerts after filters",
    f"{filtered_alerts:,}",
)

st.divider()


# --------------------------------------------------
# CHARTS
# --------------------------------------------------

st.subheader("Alert Overview")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("**Alerts by severity**")

    if filtered.empty:
        st.info("No alerts match the current filters.")
    else:
        severity_counts = (
            filtered["SEVERITY"]
            .value_counts()
            .reindex(
                ["Very High", "High", "Moderate"],
                fill_value=0,
            )
        )

        st.bar_chart(
            severity_counts,
            x_label="Severity",
            y_label="Number of alerts",
        )

with chart_col2:
    st.markdown("**Alert score distribution**")

    if filtered.empty:
        st.info("No scores to display.")
    else:
        scores = filtered["RANKING_SCORE"].dropna()

        if scores.empty:
            st.info("Ranking scores are unavailable.")
        else:
            score_bins = pd.cut(
                scores,
                bins=10,
                duplicates="drop",
            ).value_counts().sort_index()

            score_chart = pd.DataFrame(
                {
                    "Score range": score_bins.index.astype(str),
                    "Alert count": score_bins.values,
                }
            ).set_index("Score range")

            st.bar_chart(
                score_chart,
                y_label="Number of alerts",
            )


# --------------------------------------------------
# ALERT TABLE
# --------------------------------------------------

st.divider()
st.subheader("Potential Early-Reorder Alerts")

st.write(
    "Use the filters to narrow the list. Higher ranking scores "
    "indicate stronger heuristic signals under the current scoring "
    "formula; they are not probabilities."
)

if filtered.empty:
    st.warning(
        "No alerts match the selected filters. "
        "Adjust the customer, item, severity, or date filters."
    )
else:
    sort_options = {
        "Highest ranking score": "RANKING_SCORE",
        "Shortest current interval": "IPI_DAYS",
        "Largest historical median": "HIST_MEDIAN_IPI",
        "Most recent request": "REQUEST_DATE",
    }

    sort_label = st.selectbox(
        "Sort alerts by",
        options=list(sort_options.keys()),
    )

    ascending = sort_label == "Shortest current interval"

    display_df = filtered.sort_values(
        sort_options[sort_label],
        ascending=ascending,
        na_position="last",
    )

    display_columns = [
        "CUSTOMER_ID",
        "ITEM_ID",
        "REQUEST_DATE",
        "IPI_DAYS",
        "HIST_MEDIAN_IPI",
        "HIST_MAD_IPI",
        "IPI_TO_HIST_MEDIAN",
        "RANKING_SCORE",
        "SEVERITY",
        "ALERT_REASON",
    ]

    display_columns = [
        col for col in display_columns
        if col in display_df.columns
    ]

    st.dataframe(
        display_df[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    csv_data = display_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download filtered alerts as CSV",
        data=csv_data,
        file_name="filtered_early_reorder_alerts.csv",
        mime="text/csv",
    )


# --------------------------------------------------
# SELECTED ALERT DETAILS
# --------------------------------------------------

st.divider()
st.subheader("Selected Alert Details")

if filtered.empty:
    st.info("Apply filters that return alerts to inspect their details.")
else:
    # Select an alert from the currently filtered results.
    detail_options = filtered.reset_index(drop=True).copy()

    detail_options["ALERT_LABEL"] = (
        detail_options["CUSTOMER_ID"].astype(str)
        + " | "
        + detail_options["ITEM_ID"].astype(str)
        + " | "
        + detail_options["REQUEST_DATE"].astype(str).str[:10]
        + " | Score: "
        + detail_options["RANKING_SCORE"].round(3).astype(str)
    )

    selected_label = st.selectbox(
        "Choose an alert to inspect",
        options=detail_options["ALERT_LABEL"].tolist(),
        key="selected_alert_detail",
    )

    selected_alert = detail_options.loc[
        detail_options["ALERT_LABEL"] == selected_label
    ].iloc[0]

    # Display the principal fields.
    detail_col1, detail_col2, detail_col3 = st.columns(3)

    detail_col1.metric(
        "Current interval",
        f"{selected_alert['IPI_DAYS']:.0f} days",
    )

    detail_col2.metric(
        "Historical median",
        f"{selected_alert['HIST_MEDIAN_IPI']:.0f} days",
    )

    ratio = selected_alert.get("IPI_TO_HIST_MEDIAN", float("nan"))

    detail_col3.metric(
        "Current / historical median",
        f"{ratio:.1%}" if pd.notna(ratio) else "N/A",
    )

    detail_col1, detail_col2, detail_col3 = st.columns(3)

    mad = selected_alert.get("HIST_MAD_IPI", float("nan"))
    history_count = selected_alert.get("HISTORY_COUNT", float("nan"))

    detail_col1.metric(
        "Historical MAD",
        f"{mad:.1f} days" if pd.notna(mad) else "N/A",
    )

    detail_col2.metric(
        "Previous intervals available",
        f"{history_count:.0f}" if pd.notna(history_count) else "N/A",
    )

    detail_col3.metric(
        "Ranking score",
        f"{selected_alert['RANKING_SCORE']:.4f}",
    )

    st.write(
        f"**Customer:** {selected_alert['CUSTOMER_ID']}  \n"
        f"**Item / Part:** {selected_alert['ITEM_ID']}  \n"
        f"**Request date:** {selected_alert['REQUEST_DATE']}  \n"
        f"**Severity:** {selected_alert['SEVERITY']}"
    )

    alert_reason = selected_alert.get("ALERT_REASON", "")

    if pd.notna(alert_reason) and str(alert_reason).strip():
        st.info(f"**Why this event was flagged:** {alert_reason}")

    if pd.notna(mad) and mad == 0:
        st.warning(
            "Historical MAD is zero. The past intervals show no "
            "median absolute deviation, so variability-based scores "
            "should be interpreted cautiously."
        )

    st.caption(
        "This alert is a candidate for investigation, not a confirmed "
        "unnecessary purchase. The ranking score is heuristic, "
        "not a probability."
    )


# --------------------------------------------------
# EVENT INSPECTOR
# --------------------------------------------------

st.divider()
st.subheader("Customer–Part Event Inspector")

st.write(
    "Choose a customer and item to inspect its recorded purchase "
    "events and inter-purchase intervals."
)

if events.empty:
    st.warning("No event history is available.")
else:
    # Keep the inspector synchronized with the sidebar customer/item filters.
    inspector_events = events.copy()

    if selected_customers:
        inspector_events = inspector_events[
            inspector_events["CUSTOMER_ID"].astype(str).isin(
                selected_customers
            )
        ]

    if selected_items:
        inspector_events = inspector_events[
            inspector_events["ITEM_ID"].astype(str).isin(
                selected_items
            )
        ]

    event_customers = sorted(
        inspector_events["CUSTOMER_ID"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not event_customers:
        st.info(
            "No event history matches the selected sidebar "
            "customer and item filters."
        )
    else:
        previous_customer = st.session_state.get("inspect_customer")
        if previous_customer not in event_customers:
            previous_customer = event_customers[0]

        inspect_customer = st.selectbox(
            "Select customer",
            options=event_customers,
            index=event_customers.index(previous_customer),
            key="inspect_customer",
        )

        customer_events = inspector_events[
            inspector_events["CUSTOMER_ID"].astype(str)
            == str(inspect_customer)
        ].copy()

        customer_items = sorted(
            customer_events["ITEM_ID"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if not customer_items:
            st.info("No items are available for this customer.")
        else:
            previous_item = st.session_state.get("inspect_item")
            if previous_item not in customer_items:
                previous_item = customer_items[0]

            inspect_item = st.selectbox(
                "Select item / part",
                options=customer_items,
                index=customer_items.index(previous_item),
                key="inspect_item",
            )

            pair_history = customer_events[
                customer_events["ITEM_ID"].astype(str)
                == str(inspect_item)
            ].copy()

            pair_history = pair_history.sort_values("REQUEST_DATE")

            if pair_history.empty:
                st.info("No purchase history found for this pair.")
            else:
                st.markdown(
                    f"**Recorded events for this pair:** "
                    f"{len(pair_history):,}"
                )

                st.dataframe(
                    pair_history,
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("**Inter-purchase interval history**")

                interval_history = (
                    pair_history[
                        ["REQUEST_DATE", "IPI_DAYS"]
                    ]
                    .dropna()
                    .sort_values("REQUEST_DATE")
                )

                if interval_history.empty:
                    st.info(
                        "No inter-purchase intervals are available."
                    )
                else:
                    st.line_chart(
                        interval_history.set_index(
                            "REQUEST_DATE"
                        )["IPI_DAYS"],
                        x_label="Request date",
                        y_label="Interval (days)",
                    )

                matching_alerts = alerts[
                    (alerts["CUSTOMER_ID"].astype(str) == inspect_customer)
                    & (alerts["ITEM_ID"].astype(str) == inspect_item)
                ].copy()

                st.markdown("**Flagged events for this pair**")

                if matching_alerts.empty:
                    st.info(
                        "No events for this pair appear in the current "
                        "ranked early-reorder alert file."
                    )
                else:
                    alert_columns = [
                        "REQUEST_DATE",
                        "IPI_DAYS",
                        "HIST_MEDIAN_IPI",
                        "HIST_MAD_IPI",
                        "RANKING_SCORE",
                        "SEVERITY",
                        "ALERT_REASON",
                    ]

                    alert_columns = [
                        col for col in alert_columns
                        if col in matching_alerts.columns
                    ]

                    st.dataframe(
                        matching_alerts[alert_columns],
                        use_container_width=True,
                        hide_index=True,
                    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Project prototype | Temporal anomaly screening using "
    "customer–part purchase histories. Thresholds and scores "
    "require further validation against verified labels."
)
