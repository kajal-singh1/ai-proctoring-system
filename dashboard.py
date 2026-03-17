import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime
import time

st.set_page_config(
    page_title="AI Proctor Dashboard",
    page_icon="🎓",
    layout="wide"
)

LOG_FILE = "violations.csv"

st.title("🎓 AI Proctoring System — Live Dashboard")
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.divider()

if not os.path.exists(LOG_FILE):
    st.warning("⚠️ No violation log found. Start a proctoring session first.")
    st.stop()

df = pd.read_csv(LOG_FILE)

if len(df) == 0:
    st.success("✅ No violations logged yet.")
    st.stop()

# Check if session_id column exists
if "session_id" not in df.columns:
    st.warning("⚠️ Old log format detected. Please delete violations.csv and restart.")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
df["minute"]    = df["timestamp"].dt.floor("min")

# ── Session selector ───────────────────────────────────────────────
sessions = ["All Sessions"] + sorted(df["session_id"].unique().tolist(), reverse=True)
selected = st.selectbox("📁 Select Session", sessions)

if selected != "All Sessions":
    df = df[df["session_id"] == selected]

st.divider()

# ── KPI cards ──────────────────────────────────────────────────────
total      = len(df)
no_face    = len(df[df["violation_type"] == "NO_FACE"])
multi_face = len(df[df["violation_type"] == "MULTIPLE_FACES"])
gaze       = len(df[df["violation_type"].isin(["GAZE_LEFT","GAZE_RIGHT"])])
head       = len(df[df["violation_type"].isin(["HEAD_TURN","HEAD_DOWN"])])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🚨 Total",          total)
c2.metric("👤 No Face",        no_face)
c3.metric("👥 Multiple Faces", multi_face)
c4.metric("👁️ Gaze",           gaze)
c5.metric("🔄 Head",           head)

st.divider()

# ── Timeline chart ─────────────────────────────────────────────────
st.subheader("📈 Violation Timeline")
timeline = (
    df.groupby(["minute", "violation_type"])
    .size()
    .reset_index(name="count")
)
fig = px.bar(
    timeline, x="minute", y="count",
    color="violation_type",
    color_discrete_map={
        "NO_FACE":        "#ef4444",
        "MULTIPLE_FACES": "#f97316",
        "GAZE_LEFT":      "#3b82f6",
        "GAZE_RIGHT":     "#8b5cf6",
        "HEAD_TURN":      "#f59e0b",
        "HEAD_DOWN":      "#10b981"
    },
    title="Violations per Minute"
)
fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
st.plotly_chart(fig, use_container_width=True)

# ── Breakdown + log ────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🥧 Violation Breakdown")
    pie = df["violation_type"].value_counts().reset_index()
    pie.columns = ["violation_type", "count"]
    fig2 = px.pie(pie, names="violation_type", values="count",
                  color="violation_type",
                  color_discrete_map={
                      "NO_FACE":"#ef4444","MULTIPLE_FACES":"#f97316",
                      "GAZE_LEFT":"#3b82f6","GAZE_RIGHT":"#8b5cf6",
                      "HEAD_TURN":"#f59e0b","HEAD_DOWN":"#10b981"
                  })
    fig2.update_layout(paper_bgcolor="#0e1117", font_color="white")
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("📋 Violation Log")
    st.dataframe(
        df[["session_id","timestamp","violation_type","details"]]
        .sort_values("timestamp", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        height=300
    )

# ── Risk level ─────────────────────────────────────────────────────
st.divider()
if total == 0:
    st.success("🟢 Risk Level: CLEAN")
elif total <= 3:
    st.warning("🟡 Risk Level: LOW")
elif total <= 8:
    st.warning("🟠 Risk Level: MEDIUM")
else:
    st.error("🔴 Risk Level: HIGH")

# ── Auto refresh ───────────────────────────────────────────────────
time.sleep(3)
st.rerun()