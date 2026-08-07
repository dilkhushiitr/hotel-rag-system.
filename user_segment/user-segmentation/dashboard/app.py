"""
Monitoring Dashboard (Streamlit)
─────────────────────────────────
Run: streamlit run dashboard/app.py --server.port 8501

Shows:
  - Segment distribution
  - Per-segment KPI cards
  - Score distributions
  - Anomaly list
  - Drift status
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

from src.config import DATA_PROCESSED

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "User Segmentation Dashboard",
    page_icon  = "🎯",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

PALETTE = {
    "High-Value Power Users": "#2196F3",
    "Ad Hunters":             "#4CAF50",
    "Active Casual Users":    "#FF9800",
    "Dormant Users":          "#9C27B0",
    "At-Risk Users":          "#F44336",
    "Anomaly / Fraud Suspect":"#B0B0B0",
}


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    path = DATA_PROCESSED / "user_segments.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_summary():
    path = DATA_PROCESSED / "segment_summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


df      = load_data()
summary = load_summary()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/bullseye.png", width=64)
st.sidebar.title("Segmentation Dashboard")
st.sidebar.markdown("---")

if df is not None:
    segments = ["All"] + sorted(df["segment_label"].unique().tolist())
    selected = st.sidebar.selectbox("Filter by Segment", segments)
    show_anomalies = st.sidebar.checkbox("Show anomalies only", False)
    st.sidebar.markdown(f"**Total users:** {len(df):,}")
    st.sidebar.markdown(f"**Last updated:** {df['last_updated'].max()[:10]}")
else:
    selected = "All"
    show_anomalies = False

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("🎯 User Segmentation & Ad Personalization")
st.markdown("*Powered by K-Means + DBSCAN unsupervised learning*")

if df is None:
    st.error(
        "No segment data found. "
        "Run the pipeline first:\n\n"
        "`python -m src.pipeline.batch_pipeline`"
    )
    st.stop()

# Apply filters
display_df = df.copy()
if selected != "All":
    display_df = display_df[display_df["segment_label"] == selected]
if show_anomalies:
    display_df = display_df[display_df["is_anomaly"] == 1]

# ── KPI Cards ──────────────────────────────────────────────────────────────────
st.subheader("📊 Key Metrics")
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Total Users",        f"{len(df):,}")
c2.metric("Segments",           df["segment_label"].nunique())
c3.metric("Avg Engagement",     f"{df['engagement_score'].mean():.1f}")
c4.metric("Avg Monetization",   f"{df['monetization_score'].mean():.1f}")
c5.metric("Anomalies Flagged",  f"{df['is_anomaly'].sum():,}")

st.markdown("---")

# ── Segment Distribution ───────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🥧 Segment Distribution")
    seg_counts = df["segment_label"].value_counts()
    colors = [PALETTE.get(s, "#607D8B") for s in seg_counts.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        seg_counts.values,
        labels     = seg_counts.index,
        colors     = colors,
        autopct    = "%1.1f%%",
        startangle = 140,
        pctdistance= 0.75,
    )
    for t in texts:
        t.set_fontsize(8)
    ax.set_title("User Share by Segment")
    st.pyplot(fig, use_container_width=True)

with col2:
    st.subheader("📋 Segment Summary Table")
    if summary is not None:
        st.dataframe(
            summary.style.background_gradient(subset=["avg_monetization"], cmap="Blues"),
            use_container_width=True,
            height=300,
        )
    else:
        counts = df.groupby("segment_label").agg(
            users               = ("user_id", "count"),
            avg_engagement      = ("engagement_score", "mean"),
            avg_monetization    = ("monetization_score", "mean"),
            avg_churn_risk      = ("churn_risk_score", "mean"),
        ).round(2).reset_index()
        st.dataframe(counts, use_container_width=True, height=300)

st.markdown("---")

# ── Score Distributions ────────────────────────────────────────────────────────
st.subheader("📈 Score Distributions by Segment")

score_col = st.selectbox("Select score", ["engagement_score", "monetization_score", "churn_risk_score"])

fig, ax = plt.subplots(figsize=(12, 4))
for seg, group in df.groupby("segment_label"):
    ax.hist(
        group[score_col],
        bins   = 40,
        alpha  = 0.55,
        label  = seg,
        color  = PALETTE.get(seg, "#607D8B"),
        density= True,
    )
ax.set_xlabel(score_col.replace("_", " ").title())
ax.set_ylabel("Density")
ax.set_title(f"Distribution of {score_col.replace('_', ' ').title()}")
ax.legend(fontsize=8, framealpha=0.7)
ax.grid(alpha=0.3)
st.pyplot(fig, use_container_width=True)

st.markdown("---")

# ── User Table ─────────────────────────────────────────────────────────────────
st.subheader(f"👤 User List ({len(display_df):,} users)")
st.dataframe(
    display_df[[
        "user_id", "segment_label", "kmeans_cluster",
        "is_anomaly", "engagement_score", "monetization_score",
        "churn_risk_score", "last_updated",
    ]].head(500),
    use_container_width=True,
    height=350,
)

# ── Anomaly Section ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("⚠️ Anomaly / Fraud Suspect Users")
anomalies = df[df["is_anomaly"] == 1]
if len(anomalies) == 0:
    st.success("No anomalies detected in current run. 🎉")
else:
    st.warning(f"{len(anomalies):,} users flagged as anomalies by DBSCAN.")
    st.dataframe(
        anomalies[["user_id", "segment_label", "engagement_score", "monetization_score", "churn_risk_score"]].head(100),
        use_container_width=True,
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("User Segmentation Pipeline · K-Means + DBSCAN · Refresh every 5 min")
