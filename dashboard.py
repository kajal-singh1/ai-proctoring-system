import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime
import time

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Proctor Dashboard",
    page_icon="🎓",
    layout="wide"
)

LOG_FILE = "violations.csv"

st.cache_data.clear()

# ── Header ─────────────────────────────────────────────────────────
st.title("🎓 AI Proctoring System — Live Dashboard")
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.divider()

# ── Load data ──────────────────────────────────────────────────────
if not os.path.exists(LOG_FILE):
    st.warning("⚠️ No violation log found. Start a proctoring session first.")
    st.stop()

df = pd.read_csv(LOG_FILE)

if df.empty:
    st.success("✅ No violations logged yet. Session is clean.")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
df["minute"] = df["timestamp"].dt.floor("min")

# ── Top KPI cards ──────────────────────────────────────────────────
total      = len(df)
no_face    = len(df[df["violation_type"] == "NO_FACE"])
multi_face = len(df[df["violation_type"] == "MULTIPLE_FACES"])

col1, col2, col3 = st.columns(3)

col1.metric(
    label="🚨 Total Violations",
    value=total,
    delta=None
)
col2.metric(
    label="👤 No Face Detected",
    value=no_face
)
col3.metric(
    label="👥 Multiple Faces",
    value=multi_face
)

st.divider()

# ── Violation timeline chart ───────────────────────────────────────
st.subheader("📈 Violation Timeline")

timeline = (
    df.groupby(["minute", "violation_type"])
    .size()
    .reset_index(name="count")
)

fig_timeline = px.bar(
    timeline,
    x="minute",
    y="count",
    color="violation_type",
    color_discrete_map={
        "NO_FACE":        "#ef4444",   # red
        "MULTIPLE_FACES": "#f97316"    # orange
    },
    labels={"minute": "Time", "count": "Violations", "violation_type": "Type"},
    title="Violations per Minute"
)
fig_timeline.update_layout(
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font_color="white"
)
st.plotly_chart(fig_timeline, use_container_width=True)

# ── Pie chart ──────────────────────────────────────────────────────
st.subheader("🥧 Violation Breakdown")

col_pie, col_table = st.columns([1, 1])

with col_pie:
    pie_data = df["violation_type"].value_counts().reset_index()
    pie_data.columns = ["violation_type", "count"]

    fig_pie = px.pie(
        pie_data,
        names="violation_type",
        values="count",
        color="violation_type",
        color_discrete_map={
            "NO_FACE":        "#ef4444",
            "MULTIPLE_FACES": "#f97316"
        }
    )
    fig_pie.update_layout(
        paper_bgcolor="#0e1117",
        font_color="white"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Raw violation log table ────────────────────────────────────────
with col_table:
    st.subheader("📋 Violation Log")
    st.dataframe(
        df[["timestamp", "violation_type", "details"]]
        .sort_values("timestamp", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        height=300
    )

# ── Session summary ────────────────────────────────────────────────
st.divider()
st.subheader("📝 Session Summary")

first_violation = df["timestamp"].min().strftime("%H:%M:%S")
last_violation  = df["timestamp"].max().strftime("%H:%M:%S")
duration        = df["timestamp"].max() - df["timestamp"].min()

summary_col1, summary_col2, summary_col3 = st.columns(3)
summary_col1.metric("🕐 First Violation",  first_violation)
summary_col2.metric("🕐 Last Violation",   last_violation)
summary_col3.metric("⏱️ Session Span",     str(duration).split(".")[0])

# ── Risk level ─────────────────────────────────────────────────────
st.divider()
if total == 0:
    st.success("🟢 Risk Level: CLEAN — No violations detected")
elif total <= 3:
    st.warning("🟡 Risk Level: LOW — Minor violations detected")
elif total <= 8:
    st.warning("🟠 Risk Level: MEDIUM — Multiple violations detected")
else:
    st.error("🔴 Risk Level: HIGH — Frequent violations detected")

# ── Smooth auto-refresh (no flicker) ──────────────────────────────
time.sleep(3)
st.rerun()