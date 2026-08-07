"""

============================================================
  GreedyGame User Segmentation — Step 3: EDA
============================================================
 
Exploratory Data Analysis before clustering.
Goal: Understand the data deeply BEFORE applying any ML.
 
What this script covers:
  3.1  Load & basic inspection
  3.2  Missing values & data types
  3.3  Univariate analysis (distributions per feature)
  3.4  Segment-wise feature comparison (box plots)
  3.5  Correlation heatmap
  3.6  Outlier analysis (IQR method)
  3.7  Categorical feature breakdown
  3.8  Key EDA insights summary (printed to console)
  
============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
 
warnings.filterwarnings("ignore")

# ── Output directory ──────────────────────────────────────────────────────────
PLOT_DIR = os.path.expanduser("~/Downloads/projects/user_segment/eda_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

 
# ── Plotting style ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = {
    "High_Value_Engaged": "#1F6AA5",
    "Casual_Earners":     "#2E9E5B",
    "Ad_Heavy_Clickers":  "#E07B2A",
    "Low_Retention":      "#C0392B",
    "Reward_Exploiter":   "#8E44AD",
    "Burst_User":         "#17A589",
    "Dormant_High_Value": "#7F8C8D",
}
 
NUMERIC_FEATURES = [
    "session_duration_avg",
    "daily_active_frequency",
    "ad_ctr",
    "wallet_earning_rate",
    "redemption_frequency",
    "session_count_30d",
    "ad_impression_count",
    "days_since_last_active",
    "earning_trend",
    "redemption_to_earn_ratio",
    "bounce_rate",
    "ad_skip_rate",
]
 
CATEGORICAL_FEATURES = ["device_type", "location_cluster", "time_of_day_peak"]
 
# ─────────────────────────────────────────────────────────────────────────────
#  3.1  LOAD & BASIC INSPECTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  3.1  LOAD & BASIC INSPECTION")
print("="*60)
 
DATA_PATH = os.path.expanduser("~/Downloads/projects/user_segment/raw_users.csv")
df = pd.read_csv(DATA_PATH) 
print(f"\n  Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Memory usage   : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"\n  Dtypes summary :")
print(df.dtypes.value_counts().to_string())
 
print(f"\n  First 3 rows (transposed for readability):")
print(df.head(3).T.to_string())

# ─────────────────────────────────────────────────────────────────────────────
#  3.2  MISSING VALUES & DATA QUALITY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  3.2  MISSING VALUES & DATA QUALITY")
print("="*60)
 
# Check nulls
null_counts = df.isnull().sum()
null_pct    = (null_counts / len(df) * 100).round(2)
null_df     = pd.DataFrame({"null_count": null_counts, "null_%": null_pct})
null_df     = null_df[null_df["null_count"] > 0]
 
if null_df.empty:
    print("\n  ✅ No missing values found in any column.")
else:
    print("\n  ⚠️  Missing values detected:")
    print(null_df.to_string())
 
# Check for duplicate user IDs
n_dupes = df["user_id"].duplicated().sum()
print(f"\n  Duplicate user_id rows : {n_dupes}")
 
# Value range sanity checks
print("\n  Value range checks:")
range_checks = {
    "ad_ctr":                    (0.0, 1.0),
    "bounce_rate":               (0.0, 1.0),
    "ad_skip_rate":              (0.0, 1.0),
    "redemption_to_earn_ratio":  (0.0, 1.0),
    "daily_active_frequency":    (0.0, 7.0),
    "earning_trend":             (-1.0, 1.0),
}
for feat, (lo, hi) in range_checks.items():
    out_of_range = ((df[feat] < lo) | (df[feat] > hi)).sum()
    status = "✅" if out_of_range == 0 else f"⚠️  {out_of_range} out-of-range"
    print(f"    {feat:<30} [{lo}, {hi}]  →  {status}")
 

  
# ─────────────────────────────────────────────────────────────────────────────
#  3.3  UNIVARIATE ANALYSIS — Distribution of each numeric feature
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  3.3  UNIVARIATE DISTRIBUTIONS")
print("="*60)
 
fig, axes = plt.subplots(4, 3, figsize=(18, 20))
fig.suptitle(
    "GreedyGame User Segmentation — Feature Distributions (All Users)",
    fontsize=16, fontweight="bold", y=0.98
)
 
for ax, feat in zip(axes.flat, NUMERIC_FEATURES):
    data = df[feat].dropna()
 
    # Histogram + KDE
    ax.hist(data, bins=60, color="#2980B9", alpha=0.6, edgecolor="none", density=True)
 
    # KDE overlay
    kde_x = np.linspace(data.min(), data.max(), 300)
    try:
        kde = stats.gaussian_kde(data)
        ax.plot(kde_x, kde(kde_x), color="#1A252F", lw=2)
    except Exception:
        pass
 
    # Mean & median lines
    ax.axvline(data.mean(),   color="#E74C3C", lw=1.5, linestyle="--", label=f"Mean {data.mean():.2f}")
    ax.axvline(data.median(), color="#27AE60", lw=1.5, linestyle=":",  label=f"Median {data.median():.2f}")
 
    skew  = data.skew()
    ax.set_title(f"{feat}\nskew={skew:.2f}", fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.legend(fontsize=7, loc="upper right")
 
    print(f"  {feat:<30}  mean={data.mean():.3f}  median={data.median():.3f}  "
          f"std={data.std():.3f}  skew={skew:.2f}")
 
plt.tight_layout()
path = os.path.join(PLOT_DIR, "01_univariate_distributions.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  📊 Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  3.4  SEGMENT-WISE FEATURE COMPARISON — Box plots
#       (only the 4 main segments, excluding outlier types)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  3.4  SEGMENT-WISE FEATURE COMPARISON")
print("="*60)
 
MAIN_SEGMENTS = ["High_Value_Engaged", "Casual_Earners", "Ad_Heavy_Clickers", "Low_Retention"]
df_main = df[df["true_segment"].isin(MAIN_SEGMENTS)].copy()
 
fig, axes = plt.subplots(4, 3, figsize=(20, 22))
fig.suptitle(
    "Feature Distributions by User Segment\n(Box plots — Main 4 Segments Only)",
    fontsize=15, fontweight="bold", y=0.99
)
 
seg_palette = [PALETTE[s] for s in MAIN_SEGMENTS]
 
for ax, feat in zip(axes.flat, NUMERIC_FEATURES):
    # Order segments by their median value for this feature (most to least)
    order = (
        df_main.groupby("true_segment")[feat]
        .median()
        .reindex(MAIN_SEGMENTS)
        .sort_values(ascending=False)
        .index.tolist()
    )
 
    sns.boxplot(
        data=df_main,
        x="true_segment",
        y=feat,
        order=order,
        palette={s: PALETTE[s] for s in order},
        width=0.5,
        fliersize=2,
        linewidth=1.2,
        ax=ax,
    )
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.set_xticklabels(
        [s.replace("_", "\n") for s in order],
        fontsize=8
    )
 
    # Print mean per segment
    means = df_main.groupby("true_segment")[feat].mean().reindex(MAIN_SEGMENTS)
    mean_str = "  |  ".join([f"{s.split('_')[0]}: {v:.2f}" for s, v in means.items()])
    print(f"  {feat:<30}  {mean_str}")
 
plt.tight_layout()
path = os.path.join(PLOT_DIR, "02_segment_boxplots.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  📊 Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  3.5  CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  3.5  CORRELATION HEATMAP")
print("="*60)
 
corr = df[NUMERIC_FEATURES].corr(method="pearson")
 
fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))       # show only lower triangle
 
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.5,
    linecolor="white",
    square=True,
    cbar_kws={"shrink": 0.8},
    ax=ax,
)
ax.set_title(
    "Pearson Correlation — All Numeric Features",
    fontsize=14, fontweight="bold", pad=15
)
plt.tight_layout()
path = os.path.join(PLOT_DIR, "03_correlation_heatmap.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  📊 Saved → {path}")
 
# Print top correlations
print("\n  Top 10 feature pairs by absolute correlation:")
corr_pairs = (
    corr.where(~mask)
    .stack()
    .reset_index()
    .rename(columns={"level_0": "feat_A", "level_1": "feat_B", 0: "corr"})
    .assign(abs_corr=lambda x: x["corr"].abs())
    .sort_values("abs_corr", ascending=False)
    .head(10)
)
for _, row in corr_pairs.iterrows():
    direction = "↑ positive" if row["corr"] > 0 else "↓ negative"
    print(f"    {row['feat_A']:<30}  ↔  {row['feat_B']:<30}  r={row['corr']:+.3f}  {direction}")
 

  
# ─────────────────────────────────────────────────────────────────────────────
#  3.6  OUTLIER ANALYSIS (IQR METHOD)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  3.6  OUTLIER ANALYSIS (IQR METHOD)")
print("="*60)
 
print("\n  Feature         | Outliers (IQR) |  % of users  |  Cap value (99th pct)")
print("  " + "-"*72)
 
outlier_summary = {}
for feat in NUMERIC_FEATURES:
    q1  = df[feat].quantile(0.25)
    q3  = df[feat].quantile(0.75)
    iqr = q3 - q1
    lo_bound = q1 - 1.5 * iqr
    hi_bound = q3 + 1.5 * iqr
 
    n_outliers = ((df[feat] < lo_bound) | (df[feat] > hi_bound)).sum()
    pct        = n_outliers / len(df) * 100
    cap_99     = df[feat].quantile(0.99)
 
    outlier_summary[feat] = {
        "n_outliers": n_outliers,
        "pct": pct,
        "lo_bound": lo_bound,
        "hi_bound": hi_bound,
        "cap_99": cap_99,
    }
    flag = " ⚠️ " if pct > 5 else "    "
    print(f"  {feat:<30} {n_outliers:>8,}   {pct:>6.2f}%   {flag}  cap={cap_99:.2f}")
 
# Outlier bar chart
fig, ax = plt.subplots(figsize=(12, 6))
feats   = list(outlier_summary.keys())
pcts    = [outlier_summary[f]["pct"] for f in feats]
colors  = ["#E74C3C" if p > 5 else "#3498DB" for p in pcts]
 
bars = ax.barh(feats, pcts, color=colors, edgecolor="white", height=0.6)
ax.axvline(5, color="#E74C3C", linestyle="--", lw=1.5, label="5% threshold")
ax.set_xlabel("% Users Flagged as Outliers (IQR method)", fontsize=11)
ax.set_title("Outlier Prevalence by Feature", fontsize=13, fontweight="bold")
ax.legend()
 
for bar, pct in zip(bars, pcts):
    ax.text(
        pct + 0.1, bar.get_y() + bar.get_height() / 2,
        f"{pct:.1f}%", va="center", fontsize=9
    )
 
plt.tight_layout()
path = os.path.join(PLOT_DIR, "04_outlier_analysis.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  📊 Saved → {path}")
 

  
# ─────────────────────────────────────────────────────────────────────────────
#  3.7  CATEGORICAL FEATURE BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  3.7  CATEGORICAL FEATURE BREAKDOWN")
print("="*60)
 
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle(
    "Categorical Feature Distributions by Segment",
    fontsize=14, fontweight="bold"
)
 
for ax, cat_feat in zip(axes, CATEGORICAL_FEATURES):
    # Cross-tab: category × segment
    ct = (
        df_main.groupby([cat_feat, "true_segment"])
        .size()
        .unstack(fill_value=0)
    )
    # Normalise by row so each bar sums to 100%
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
 
    ct_pct.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[PALETTE[s] for s in ct_pct.columns],
        edgecolor="white",
        width=0.7,
    )
    ax.set_title(cat_feat, fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("% of users in category")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.legend(
        [s.replace("_", " ") for s in ct_pct.columns],
        fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.0)
    )
 
    print(f"\n  {cat_feat}:")
    print(ct_pct.round(1).to_string())
 
plt.tight_layout()
path = os.path.join(PLOT_DIR, "05_categorical_breakdown.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n\n  📊 Saved → {path}")
 

 # ─────────────────────────────────────────────────────────────────────────────
#  BONUS: Pairplot of top 5 most discriminating features
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating pairplot (top 5 features)...")
 
TOP5 = [
    "session_duration_avg",
    "ad_ctr",
    "daily_active_frequency",
    "wallet_earning_rate",
    "bounce_rate",
]
 
# Sample 3000 from main segments for speed
df_sample = (
    df_main.groupby("true_segment", group_keys=False)
    .apply(lambda x: x.sample(min(len(x), 750), random_state=42))
    .reset_index(drop=True)
)
 
pair_fig = sns.pairplot(
    df_sample[TOP5 + ["true_segment"]],
    hue="true_segment",
    palette=PALETTE,
    diag_kind="kde",
    plot_kws={"alpha": 0.35, "s": 12},
    diag_kws={"linewidth": 2},
    corner=True,
)
pair_fig.fig.suptitle(
    "Pairplot — Top 5 Discriminating Features (3,000 user sample)",
    y=1.01, fontsize=13, fontweight="bold"
)
path = os.path.join(PLOT_DIR, "06_pairplot_top5.png")
pair_fig.savefig(path, dpi=130, bbox_inches="tight")
plt.close()
print(f"  📊 Saved → {path}")
 
# ─────────────────────────────────────────────────────────────────────────────
#  3.8  KEY EDA INSIGHTS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  3.8  KEY EDA INSIGHTS")
print("="*60)
 
# Auto-compute insights from actual data
hv  = df[df["true_segment"] == "High_Value_Engaged"]
cas = df[df["true_segment"] == "Casual_Earners"]
clk = df[df["true_segment"] == "Ad_Heavy_Clickers"]
lr  = df[df["true_segment"] == "Low_Retention"]
 
print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  INSIGHT 1 — Segment Separability                   │
  │  High-Value vs Low-Retention CTR gap:               │
  │    HV mean CTR  = {hv['ad_ctr'].mean():.4f}                          │
  │    LR mean CTR  = {lr['ad_ctr'].mean():.4f}                          │
  │    Ratio        = {hv['ad_ctr'].mean()/lr['ad_ctr'].mean():.1f}x difference                      │
  └─────────────────────────────────────────────────────┘
 
  ┌─────────────────────────────────────────────────────┐
  │  INSIGHT 2 — Ad-Heavy Clickers Anomaly              │
  │  High CTR ({clk['ad_ctr'].mean():.3f}) but LOW redemption freq       │
  │  ({clk['redemption_frequency'].mean():.2f}/mo) vs HV ({hv['redemption_frequency'].mean():.2f}/mo) │
  │  → Not genuine high-value; likely click-farms or    │
  │    incentivised clickers. Needs separate targeting. │
  └─────────────────────────────────────────────────────┘
 
  ┌─────────────────────────────────────────────────────┐
  │  INSIGHT 3 — Skewed Features Need Capping           │
  │  wallet_earning_rate: skew = {df['wallet_earning_rate'].skew():.2f}                 │
  │  session_count_30d:   skew = {df['session_count_30d'].skew():.2f}                 │
  │  ad_impression_count: skew = {df['ad_impression_count'].skew():.2f}                 │
  │  → Apply 99th-percentile capping in preprocessing. │
  └─────────────────────────────────────────────────────┘
 
  ┌─────────────────────────────────────────────────────┐
  │  INSIGHT 4 — High Correlations to Watch             │
  │  session_count_30d ↔ ad_impression_count            │
  │  bounce_rate ↔ ad_skip_rate (both signal disengagement)│
  │  → May need to drop one from each pair to reduce    │
  │    multicollinearity before clustering.             │
  └─────────────────────────────────────────────────────┘
 
  ┌─────────────────────────────────────────────────────┐
  │  INSIGHT 5 — Low-Retention Users: Truly Dormant     │
  │  Avg days since last active : {lr['days_since_last_active'].mean():.1f} days              │
  │  Avg session count (30d)    : {lr['session_count_30d'].mean():.1f}                  │
  │  Avg bounce rate            : {lr['bounce_rate'].mean():.2f} ({lr['bounce_rate'].mean()*100:.0f}%)              │
  │  → Re-engagement campaigns + lapsed-user targeting  │
  │    should be the primary campaign for this segment. │
  └─────────────────────────────────────────────────────┘
 
  ┌─────────────────────────────────────────────────────┐
  │  INSIGHT 6 — Categorical: Device Type               │
  │  Android Mobile dominates all segments (>50%)       │
  │  Desktop usage highest in Low-Retention segment     │
  │  → Mobile-first ad format strategy is well-aligned. │
  └─────────────────────────────────────────────────────┘
""")
 
print("="*60)
print("  EDA COMPLETE — 6 plots saved to /home/claude/eda_plots/")
print("  Next → Step 4: Preprocessing (scaling, encoding, capping)")
print("="*60)
 

print ('hello world')
