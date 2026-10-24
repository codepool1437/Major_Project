from pathlib import Path

import altair as alt
import numpy as np
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

st.set_page_config(
    page_title="Procurement Anomaly Detection",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# THEME / STYLING (frontend only)
# --------------------------------------------------

ACCENT = "#6C63FF"
ACCENT_2 = "#00D4FF"

SEVERITY_COLORS = {
    "Very High": "#FF4B6E",
    "High": "#FF9F43",
    "Moderate": "#FECA57",
}

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    /* ---------- App background ---------- */
    .stApp {{
        background:
            radial-gradient(1200px 600px at 10% -10%, rgba(108,99,255,0.18), transparent 60%),
            radial-gradient(1000px 500px at 100% 0%, rgba(0,212,255,0.12), transparent 55%),
            #0B0F1A;
        color: #E6E9F2;
    }}

    header[data-testid="stHeader"] {{
        background: transparent;
    }}

    .block-container {{
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #11162A 0%, #0D1120 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }}

    section[data-testid="stSidebar"] h1 {{
        font-size: 1.25rem;
        font-weight: 800;
        background: linear-gradient(90deg, {ACCENT}, {ACCENT_2});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    /* ---------- Hero ---------- */
    .hero {{
        position: relative;
        overflow: hidden;
        padding: 2rem 2.2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(108,99,255,0.35), rgba(0,212,255,0.18) 60%, rgba(17,22,42,0.9));
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 20px 50px rgba(0,0,0,0.35);
        margin-bottom: 1.2rem;
    }}
    .hero:after {{
        content: "";
        position: absolute;
        right: -60px; top: -60px;
        width: 260px; height: 260px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(0,212,255,0.35), transparent 70%);
        pointer-events: none;
    }}
    .hero-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.18);
        margin-bottom: 0.7rem;
    }}
    .hero h1 {{
        margin: 0 0 0.4rem 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF;
    }}
    .hero p {{
        margin: 0;
        max-width: 880px;
        color: #C5CAE0;
        font-size: 0.98rem;
        line-height: 1.55;
    }}
    .hero .subtitle {{
        color: {ACCENT_2};
        font-weight: 600;
        margin-bottom: 0.5rem;
    }}

    /* ---------- KPI cards ---------- */
    .kpi {{
        padding: 1.1rem 1.2rem;
        border-radius: 18px;
        background: linear-gradient(160deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        transition: transform .2s ease, border-color .2s ease;
        height: 100%;
        min-height: 150px;
        word-wrap: break-word;
        overflow-wrap: anywhere;
    }}
    .kpi:hover {{
        transform: translateY(-3px);
        border-color: rgba(108,99,255,0.6);
    }}
    .kpi .label {{
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #8E96B5;
    }}
    .kpi .info {{
        display: inline-block;
        margin-left: 4px;
        color: {ACCENT_2};
        cursor: help;
    }}
    .kpi .value {{
        font-size: 1.85rem;
        font-weight: 800;
        margin-top: 0.25rem;
        color: #FFFFFF;
    }}
    .kpi .desc {{
        margin-top: 0.35rem;
        font-size: 0.74rem;
        line-height: 1.35;
        color: #8E96B5;
    }}
    .kpi .bar {{
        height: 3px;
        border-radius: 3px;
        margin-top: 0.7rem;
        background: linear-gradient(90deg, {ACCENT}, {ACCENT_2});
    }}

    /* ---------- Section titles ---------- */
    .section-title {{
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0.2rem 0 0.2rem 0;
        color: #FFFFFF;
    }}
    .section-sub {{
        color: #8E96B5;
        font-size: 0.88rem;
        margin-bottom: 0.8rem;
    }}

    /* ---------- Tabs ---------- */
    div[data-baseweb="tab-list"] {{
        gap: 6px;
        background: rgba(255,255,255,0.04);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.06);
        overflow-x: auto;
    }}
    button[data-baseweb="tab"] {{
        height: 44px;
        padding: 0 20px;
        border-radius: 10px;
        font-weight: 600;
        color: #9AA2C0;
        background: transparent;
        white-space: nowrap;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: #FFFFFF;
        background: linear-gradient(90deg, rgba(108,99,255,0.85), rgba(0,212,255,0.55));
        box-shadow: 0 6px 18px rgba(108,99,255,0.35);
    }}
    div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {{
        display: none;
    }}

    /* ---------- Native metrics (alert details) ---------- */
    div[data-testid="stMetric"] {{
        background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.08);
        padding: 14px 16px;
        border-radius: 16px;
    }}
    div[data-testid="stMetricLabel"] p {{
        color: #8E96B5;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    div[data-testid="stMetricValue"] {{
        font-weight: 800;
    }}

    /* ---------- Bordered containers ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 18px;
        border-color: rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.025);
    }}

    /* ---------- Buttons ---------- */
    .stButton > button, .stDownloadButton > button {{
        border-radius: 12px;
        border: none;
        font-weight: 600;
        color: white;
        background: linear-gradient(90deg, {ACCENT}, {ACCENT_2});
        box-shadow: 0 6px 18px rgba(108,99,255,0.35);
        transition: transform .15s ease, box-shadow .15s ease;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(0,212,255,0.35);
        color: white;
    }}

    /* ---------- Dataframes ---------- */
    div[data-testid="stDataFrame"] {{
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }}

    /* ---------- Severity pill ---------- */
    .pill {{
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        color: #0B0F1A;
    }}

    .detail-card {{
        padding: 1rem 1.2rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        line-height: 1.9;
        overflow-wrap: anywhere;
    }}
    .detail-card b {{
        color: #8E96B5;
        font-weight: 600;
    }}

    .hint-box {{
        padding: 0.9rem 1.1rem;
        border-radius: 14px;
        border: 1px dashed rgba(0,212,255,0.45);
        background: rgba(0,212,255,0.06);
        color: #C5CAE0;
        font-size: 0.92rem;
    }}

    .footer {{
        text-align: center;
        color: #6C7494;
        font-size: 0.82rem;
        padding-top: 0.5rem;
    }}

    hr {{
        border-color: rgba(255,255,255,0.08);
    }}

    /* ---------- Responsive adjustments ---------- */
    @media (max-width: 1200px) {{
        .hero h1 {{ font-size: 1.8rem; }}
        .kpi .value {{ font-size: 1.5rem; }}
        .kpi {{ min-height: 140px; }}
    }}

    @media (max-width: 768px) {{
        .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}
        .hero {{ padding: 1.3rem 1.2rem; border-radius: 16px; }}
        .hero h1 {{ font-size: 1.45rem; }}
        .hero p {{ font-size: 0.88rem; }}
        .kpi {{ min-height: 0; margin-bottom: 0.4rem; }}
        .kpi .value {{ font-size: 1.35rem; }}
        button[data-baseweb="tab"] {{
            padding: 0 12px;
            font-size: 0.85rem;
            height: 40px;
        }}
        div[data-testid="stMetricValue"] {{ font-size: 1.3rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi_card(column, label, value, help_text=""):
    safe_help = help_text.replace('"', "&quot;")
    info_icon = (
        f'<span class="info" title="{safe_help}">ⓘ</span>'
        if help_text
        else ""
    )
    description = (
        f'<div class="desc">{help_text}</div>' if help_text else ""
    )

    column.markdown(
        f"""
        <div class="kpi" title="{safe_help}">
            <div class="label">{label}{info_icon}</div>
            <div class="value">{value}</div>
            {description}
            <div class="bar"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title, subtitle=""):
    st.markdown(
        f'<div class="section-title">{title}</div>'
        f'<div class="section-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def severity_pill(severity):
    color = SEVERITY_COLORS.get(str(severity), "#B0B6D0")
    return (
        f'<span class="pill" style="background:{color};">'
        f"{severity}</span>"
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
    "HISTORY_COUNT",
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

st.sidebar.title("🎛️ Dashboard Filters")
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

    elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:
        # A single clicked date is treated as the start of an
        # open-ended range until an end date is selected.
        partial_start_date = date_range[0]

        filtered = filtered[
            filtered["REQUEST_DATE"].dt.date >= partial_start_date
        ]

        st.sidebar.caption(
            "Showing alerts from "
            f"{partial_start_date.isoformat()} onward. "
            "Select an end date to bound the range."
        )

st.sidebar.divider()
st.sidebar.caption(
    f"🔎 **{len(filtered):,}** alerts match the current filters."
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">📊 Anomaly Screening Console</div>
        <h1>B2B Procurement Anomaly Detection</h1>
        <p class="subtitle">
            Machine Learning-Based Early Detection of Temporal Anomalies
            in B2B Procurement Using Customer–Part Purchase Patterns
        </p>
        <p>
            Explore customer–part purchasing intervals and investigate
            potential unusually early reorder events.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "⚠️ Alerts are screening candidates, not confirmed unnecessary "
    "purchases or fraud. The current threshold is provisional."
)


# --------------------------------------------------
# SUMMARY METRICS
# --------------------------------------------------

total_events = len(events)
total_alerts = len(alerts)
filtered_alerts = len(filtered)

# Eligible population: events with at least 3 previous intervals.
# This matches the population the scoring pipeline screens, so
# the overall percentage is computed against the same base.
eligible_events = int(
    (events["HISTORY_COUNT"] >= 3).sum()
)

if eligible_events:
    overall_alert_percentage = (
        100 * total_alerts / eligible_events
    )
else:
    overall_alert_percentage = 0.0

col1, col2, col3, col4, col5 = st.columns(5)

kpi_card(
    col1,
    "Eligible purchase events",
    f"{eligible_events:,}",
    "Events with at least 3 previous intervals. Only these are "
    "screened for early reorders.",
)
kpi_card(
    col2,
    "Potential early reorders",
    f"{total_alerts:,}",
    "Events flagged in the ranked alert file, before any "
    "sidebar filters.",
)
kpi_card(
    col3,
    "Overall flagged percentage",
    f"{overall_alert_percentage:.2f}%",
    "Potential early reorders divided by eligible purchase events.",
)
kpi_card(
    col4,
    "Alerts after filters",
    f"{filtered_alerts:,}",
    "Alerts matching the current sidebar filters.",
)
kpi_card(
    col5,
    "Reorder events with prior purchase",
    f"{total_events:,}",
    "All recorded events with at least 1 earlier purchase of the "
    "same customer–part pair. Larger than the eligible count.",
)

st.write("")


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab_overview, tab_alerts, tab_inspector = st.tabs(
    [
        "📈  Overview",
        "🚨  Alerts & Investigation",
        "🧭  Event Inspector",
    ]
)


# --------------------------------------------------
# TAB 1: OVERVIEW (CHARTS)
# --------------------------------------------------

def build_score_bands(all_scores, n_bands=6):
    """Fixed, labelled score bands based on the full alert file so the
    bands do not shift when sidebar filters change."""
    lo = float(all_scores.min())
    hi = float(all_scores.max())

    if not np.isfinite(lo) or not np.isfinite(hi):
        return None, None

    if hi == lo:
        hi = lo + 1.0

    edges = np.linspace(lo, hi, n_bands + 1)
    labels = [
        f"{edges[i]:.2f} – {edges[i + 1]:.2f}"
        for i in range(n_bands)
    ]

    # Guard against duplicate labels caused by rounding.
    if len(set(labels)) != len(labels):
        labels = [
            f"{edges[i]:.4f} – {edges[i + 1]:.4f}"
            for i in range(n_bands)
        ]

    return edges, labels


with tab_overview:
    section_header(
        "Alert Overview",
        "Distribution of the alerts that match your current filters.",
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        with st.container(border=True):
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
                    color=ACCENT,
                )

    with chart_col2:
        with st.container(border=True):
            st.markdown("**Alert score distribution (fixed score bands)**")

            if filtered.empty:
                st.info("No scores to display.")
            else:
                scores = filtered["RANKING_SCORE"].dropna()
                all_scores = alerts["RANKING_SCORE"].dropna()

                if scores.empty or all_scores.empty:
                    st.info("Ranking scores are unavailable.")
                else:
                    edges, band_labels = build_score_bands(all_scores)

                    if edges is None:
                        st.info("Ranking scores are unavailable.")
                    else:
                        banded = pd.cut(
                            scores,
                            bins=edges,
                            labels=band_labels,
                            include_lowest=True,
                        )

                        band_counts = (
                            banded.value_counts()
                            .reindex(band_labels, fill_value=0)
                        )

                        band_df = pd.DataFrame(
                            {
                                "Score band": band_labels,
                                "Alerts": band_counts.values,
                            }
                        )

                        band_chart = (
                            alt.Chart(band_df)
                            .mark_bar(
                                color=ACCENT_2,
                                cornerRadiusTopLeft=4,
                                cornerRadiusTopRight=4,
                            )
                            .encode(
                                x=alt.X(
                                    "Score band:N",
                                    sort=band_labels,
                                    title="Ranking score band (low → high)",
                                    axis=alt.Axis(labelAngle=-30),
                                ),
                                y=alt.Y(
                                    "Alerts:Q",
                                    title="Number of alerts",
                                ),
                                tooltip=[
                                    alt.Tooltip(
                                        "Score band:N",
                                        title="Score band",
                                    ),
                                    alt.Tooltip(
                                        "Alerts:Q",
                                        title="Alerts",
                                    ),
                                ],
                            )
                            .properties(height=300)
                            .configure_view(strokeWidth=0)
                        )

                        st.altair_chart(
                            band_chart,
                            width="stretch",
                        )

                        st.caption(
                            "Bands are equal-width slices of the full "
                            "alert-file score range, so they stay the "
                            "same when filters change. The ranking "
                            "score is a heuristic, not a probability: "
                            "higher means a stronger early-reorder "
                            "signal under the current formula."
                        )

    if not filtered.empty:
        st.write("")
        sev_cols = st.columns(3)
        for sev, sev_col in zip(
            ["Very High", "High", "Moderate"], sev_cols
        ):
            count = int((filtered["SEVERITY"] == sev).sum())
            color = SEVERITY_COLORS[sev]
            sev_col.markdown(
                f"""
                <div class="kpi" style="border-left: 4px solid {color}; min-height: 0;">
                    <div class="label">{sev} severity</div>
                    <div class="value">{count:,}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# --------------------------------------------------
# TAB 2: ALERTS & INVESTIGATION
# --------------------------------------------------

def render_alert_investigation(selected_alert):
    """Details + interval history for one alert row."""
    st.markdown("---")
    section_header(
        "Selected Alert Details",
        "Details and purchase-interval history for the alert you "
        "selected in the table above.",
    )

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

    st.write("")

    st.markdown(
        f"""
        <div class="detail-card">
            <b>Customer:</b> {selected_alert['CUSTOMER_ID']}<br>
            <b>Item / Part:</b> {selected_alert['ITEM_ID']}<br>
            <b>Request date:</b> {selected_alert['REQUEST_DATE']}<br>
            <b>Severity:</b> {severity_pill(selected_alert['SEVERITY'])}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    alert_reason = selected_alert.get("ALERT_REASON", "")

    if pd.notna(alert_reason) and str(alert_reason).strip():
        st.info(f"**Why this event was flagged:** {alert_reason}")

    if pd.notna(mad) and mad == 0:
        st.warning(
            "Historical MAD is zero. The past intervals show no "
            "median absolute deviation, so variability-based scores "
            "should be interpreted cautiously."
        )

    # ---- Interval history for this customer-part pair ----
    st.markdown("**Purchase-interval history for this customer–part pair**")

    pair_events = events[
        (events["CUSTOMER_ID"].astype(str)
         == str(selected_alert["CUSTOMER_ID"]))
        & (events["ITEM_ID"].astype(str)
           == str(selected_alert["ITEM_ID"]))
    ].copy()

    pair_intervals = (
        pair_events[["REQUEST_DATE", "IPI_DAYS"]]
        .dropna()
        .sort_values("REQUEST_DATE")
    )

    if pair_intervals.empty:
        st.info("No inter-purchase intervals are available for this pair.")
    else:
        flagged_point = pd.DataFrame(
            {
                "REQUEST_DATE": [selected_alert["REQUEST_DATE"]],
                "IPI_DAYS": [selected_alert["IPI_DAYS"]],
            }
        ).dropna()

        base = alt.Chart(pair_intervals)

        line = base.mark_line(color=ACCENT_2).encode(
            x=alt.X("REQUEST_DATE:T", title="Request date"),
            y=alt.Y("IPI_DAYS:Q", title="Interval (days)"),
        )

        points = base.mark_point(
            color=ACCENT_2, filled=True, size=45
        ).encode(
            x="REQUEST_DATE:T",
            y="IPI_DAYS:Q",
            tooltip=[
                alt.Tooltip("REQUEST_DATE:T", title="Date"),
                alt.Tooltip("IPI_DAYS:Q", title="Interval (days)"),
            ],
        )

        layers = [line, points]

        hist_median = selected_alert["HIST_MEDIAN_IPI"]
        if pd.notna(hist_median):
            median_rule = (
                alt.Chart(pd.DataFrame({"median": [hist_median]}))
                .mark_rule(color="#FECA57", strokeDash=[6, 4])
                .encode(y="median:Q")
            )
            layers.append(median_rule)

        if not flagged_point.empty:
            flagged_layer = (
                alt.Chart(flagged_point)
                .mark_point(
                    color="#FF4B6E",
                    filled=True,
                    size=170,
                    stroke="white",
                    strokeWidth=1.5,
                )
                .encode(
                    x="REQUEST_DATE:T",
                    y="IPI_DAYS:Q",
                    tooltip=[
                        alt.Tooltip(
                            "REQUEST_DATE:T",
                            title="Flagged date",
                        ),
                        alt.Tooltip(
                            "IPI_DAYS:Q",
                            title="Interval (days)",
                        ),
                    ],
                )
            )
            layers.append(flagged_layer)

        with st.container(border=True):
            st.altair_chart(
                alt.layer(*layers)
                .properties(height=300)
                .configure_view(strokeWidth=0),
                width="stretch",
            )

        st.caption(
            "Red marker = the selected alert. Dashed yellow line = "
            "historical median interval for this alert."
        )

    st.caption(
        "This alert is a candidate for investigation, not a confirmed "
        "unnecessary purchase. The ranking score is heuristic, "
        "not a probability."
    )


with tab_alerts:
    section_header(
        "Potential Early-Reorder Alerts",
        "Use the sidebar filters to narrow the list. Higher ranking "
        "scores indicate stronger heuristic signals under the current "
        "scoring formula; they are not probabilities.",
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
        ).reset_index(drop=True)

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

        st.markdown(
            '<div class="hint-box">👆 <b>Click a row</b> in the table '
            "to open its details and interval history below.</div>",
            unsafe_allow_html=True,
        )
        st.write("")

        table_event = st.dataframe(
            display_df[display_columns],
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="alerts_table",
        )

        # Export exactly the columns shown in the table above,
        # with the current filters and sorting applied.
        csv_data = display_df[display_columns].to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download filtered alerts as CSV",
            data=csv_data,
            file_name="filtered_early_reorder_alerts.csv",
            mime="text/csv",
        )

        selected_rows = table_event.selection.rows

        if selected_rows and selected_rows[0] < len(display_df):
            render_alert_investigation(
                display_df.iloc[selected_rows[0]]
            )
        else:
            st.write("")
            st.info(
                "No alert selected yet. Click a row in the table "
                "to investigate it."
            )


# --------------------------------------------------
# TAB 3: EVENT INSPECTOR
# --------------------------------------------------

with tab_inspector:
    section_header(
        "Customer–Part Event Inspector",
        "Choose a customer and item to inspect its recorded purchase "
        "events and inter-purchase intervals.",
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

            select_col1, select_col2 = st.columns(2)

            with select_col1:
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

                with select_col2:
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

                    with st.container(border=True):
                        st.dataframe(
                            pair_history,
                            width="stretch",
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
                        with st.container(border=True):
                            st.line_chart(
                                interval_history.set_index(
                                    "REQUEST_DATE"
                                )["IPI_DAYS"],
                                x_label="Request date",
                                y_label="Interval (days)",
                                color=ACCENT_2,
                            )

                    matching_alerts = alerts[
                        (alerts["CUSTOMER_ID"].astype(str) == inspect_customer)
                        & (alerts["ITEM_ID"].astype(str) == inspect_item)
                    ].copy()

                    st.markdown("**Flagged events for this pair**")

                    st.caption(
                        "These are pair-level flagged events: they are "
                        "filtered by customer and part only and do not "
                        "respond to the sidebar severity or date filters."
                    )

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
                            width="stretch",
                            hide_index=True,
                        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div class="footer">
        Project prototype | Temporal anomaly screening using
        customer–part purchase histories. Thresholds and scores
        require further validation against verified labels.
    </div>
    """,
    unsafe_allow_html=True,
)