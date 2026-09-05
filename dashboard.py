"""
dashboard.py
Streamlit dashboard — reads session data from SQLite and visualises it live.

Run with: streamlit run dashboard.py
Auto-refreshes every refresh_interval_sec (from config.yaml).
"""

from __future__ import annotations

import sqlite3
import time
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Config & page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Face Focus — Study Dashboard",
    page_icon="📚",
    layout="wide",
)


@st.cache_data(ttl=1)
def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


cfg = load_config()
DB_PATH     = cfg["logging"]["db_path"]
REFRESH_SEC = cfg["dashboard"]["refresh_interval_sec"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=REFRESH_SEC)
def load_data(db_path: str, session_filter: str = "all") -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM session_log ORDER BY timestamp"
    df = pd.read_sql_query(query, conn)
    conn.close()
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    if session_filter != "all":
        df = df[df["session_id"] == session_filter]
    return df


def get_sessions(db_path: str) -> list[str]:
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT DISTINCT session_id FROM session_log").fetchall()
    conn.close()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

st.title("📚 Face Focus — Study Attention Dashboard")

# Sidebar
with st.sidebar:
    st.header("Session")
    sessions = get_sessions(DB_PATH)
    if not sessions:
        st.warning("No sessions recorded yet.\nRun `python main.py` to start a session.")
        st.stop()

    session_options = ["all"] + sessions
    selected_session = st.selectbox("Select session", session_options, index=len(session_options) - 1)
    auto_refresh = st.toggle("Auto-refresh", value=True)
    st.caption(f"Refreshing every {REFRESH_SEC}s")

df = load_data(DB_PATH, selected_session)

if df.empty:
    st.info("No data for this session yet.")
    st.stop()

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

total_sec = df["timestamp"].max() - df["timestamp"].min()
total_min = total_sec / 60

focused_df = df[df["state"].isin(["READING", "WRITING"])]
focus_ratio = len(focused_df) / len(df) if len(df) > 0 else 0

distracted_df = df[df["state"].isin(["DISTRACTED", "PHONE"])]
distract_ratio = len(distracted_df) / len(df) if len(df) > 0 else 0

interventions = df["intervention_fired"].sum()
mean_attention = df["attention_filtered"].mean()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Session Duration", f"{total_min:.1f} min")
col2.metric("Mean Attention",   f"{mean_attention:.2f}")
col3.metric("Focus Ratio",      f"{focus_ratio:.0%}")
col4.metric("Distraction Ratio", f"{distract_ratio:.0%}")
col5.metric("Interventions",    int(interventions))

st.divider()

# ---------------------------------------------------------------------------
# Attention score over time
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Attention Score Over Time")
    fig_attn = go.Figure()
    fig_attn.add_trace(go.Scatter(
        x=df["datetime"], y=df["attention_filtered"],
        mode="lines", name="Filtered (EMA)",
        line=dict(color="#00c8ff", width=2),
    ))
    fig_attn.add_trace(go.Scatter(
        x=df["datetime"], y=df["attention_raw"],
        mode="lines", name="Raw",
        line=dict(color="#4488ff", width=1, dash="dot"),
        opacity=0.5,
    ))
    # Shade distracted periods
    for _, row in distracted_df.iterrows():
        fig_attn.add_vrect(
            x0=row["datetime"], x1=row["datetime"],
            fillcolor="red", opacity=0.08, line_width=0,
        )
    fig_attn.update_layout(
        yaxis=dict(range=[0, 1], title="Score"),
        xaxis_title="Time",
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_dark",
        legend=dict(orientation="h", y=1.02),
    )
    st.plotly_chart(fig_attn, use_container_width=True)

with col_right:
    st.subheader("Time in State")
    state_counts = df["state"].value_counts()
    STATE_PALETTE = {
        "READING":    "#00dc50",
        "WRITING":    "#00b0ff",
        "THINKING":   "#c8c800",
        "DISTRACTED": "#ff3c00",
        "PHONE":      "#ff0000",
        "BREAK":      "#888888",
        "UNKNOWN":    "#444444",
        "CALIBRATING":"#ffcc00",
    }
    fig_pie = px.pie(
        values=state_counts.values,
        names=state_counts.index,
        color=state_counts.index,
        color_discrete_map=STATE_PALETTE,
        hole=0.4,
    )
    fig_pie.update_layout(
        height=300, margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_dark",
        showlegend=True,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ---------------------------------------------------------------------------
# Component scores
# ---------------------------------------------------------------------------

st.subheader("Score Components")
fig_comp = go.Figure()
components = {
    "gaze_score":    ("#00ff99", "Gaze"),
    "head_score":    ("#ffaa00", "Head"),
    "context_score": ("#aa88ff", "Context"),
    "blink_score":   ("#ff66aa", "Blink"),
}
for col, (color, label) in components.items():
    if col in df.columns:
        fig_comp.add_trace(go.Scatter(
            x=df["datetime"], y=df[col],
            mode="lines", name=label,
            line=dict(color=color, width=1.5),
        ))
fig_comp.update_layout(
    yaxis=dict(range=[0, 1], title="Score"),
    xaxis_title="Time",
    height=250,
    margin=dict(l=0, r=0, t=0, b=0),
    template="plotly_dark",
    legend=dict(orientation="h", y=1.02),
)
st.plotly_chart(fig_comp, use_container_width=True)

# ---------------------------------------------------------------------------
# Gaze zone distribution & State timeline
# ---------------------------------------------------------------------------

col3a, col3b = st.columns(2)

with col3a:
    st.subheader("Gaze Zone Distribution")
    zone_counts = df["gaze_zone"].value_counts()
    fig_zone = px.bar(
        x=zone_counts.index, y=zone_counts.values,
        color=zone_counts.index,
        labels={"x": "Zone", "y": "Samples"},
        template="plotly_dark",
        height=250,
    )
    fig_zone.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    st.plotly_chart(fig_zone, use_container_width=True)

with col3b:
    st.subheader("Intervention Events")
    iv_df = df[df["intervention_fired"] == 1]
    if not iv_df.empty:
        fig_iv = go.Figure()
        fig_iv.add_trace(go.Scatter(
            x=iv_df["datetime"],
            y=iv_df["intervention_tier"],
            mode="markers",
            marker=dict(color="red", size=10, symbol="triangle-up"),
            name="Intervention",
        ))
        fig_iv.update_layout(
            yaxis=dict(tickvals=[1, 2, 3], ticktext=["Soft", "Firm", "Urgent"], title="Tier"),
            xaxis_title="Time",
            height=250,
            margin=dict(l=0, r=0, t=0, b=0),
            template="plotly_dark",
        )
        st.plotly_chart(fig_iv, use_container_width=True)
    else:
        st.info("No interventions fired in this session.")

# ---------------------------------------------------------------------------
# Raw data table
# ---------------------------------------------------------------------------

with st.expander("Raw session data"):
    st.dataframe(df.tail(200), use_container_width=True)

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

if auto_refresh:
    time.sleep(REFRESH_SEC)
    st.rerun()
