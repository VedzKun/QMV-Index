"""
app.py — QMV Index Streamlit Demo
Run: streamlit run src/dashboard/app.py
"""
import time
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, ".")
from src.api.model_store import load_model
from src.features.build_features import build_features

# ── page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QMV Index — Content Valuation",
    page_icon="📈",
    layout="wide",
)

st.title("📈 QMV Index — Quantitative Market Value")
st.caption("Real-time content valuation for Hacker News stories.")

# ── sidebar controls ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Input")
    story_id = st.number_input("HN Story ID (optional)", min_value=0, value=0, step=1)
    st.markdown("**— or paste raw snapshot data —**")
    T = st.selectbox("Prediction Horizon (T hours)", [1, 3, 6], index=1)
    st.markdown("---")
    st.markdown("**Manual Snapshots**")
    score_1h  = st.number_input("Score at 1h",  min_value=0, value=5)
    score_3h  = st.number_input("Score at 3h",  min_value=0, value=22)
    score_6h  = st.number_input("Score at 6h",  min_value=0, value=55)
    comms_1h  = st.number_input("Comments at 1h", min_value=0, value=2)
    comms_3h  = st.number_input("Comments at 3h", min_value=0, value=10)
    comms_6h  = st.number_input("Comments at 6h", min_value=0, value=18)
    title_in  = st.text_input("Story title", value="Show HN: My cool project")
    author_in = st.text_input("Author handle", value="pg")
    run_btn   = st.button("🚀 Valuate", use_container_width=True)

# ── main panel ─────────────────────────────────────────────────────────
if not run_btn:
    st.info("Fill in snapshot data on the left and click **Valuate**.")
    st.stop()

with st.spinner("Loading model and computing features…"):
    bundle = load_model(T)
    fc     = bundle["feature_cols"]

# Build a minimal stories + snapshots frame
post_time = int(time.time()) - T * 3600
snaps = [
    {"story_id": 0, "hours_since_post": 1.0, "score": score_1h, "descendants": comms_1h},
    {"story_id": 0, "hours_since_post": 3.0, "score": score_3h, "descendants": comms_3h},
    {"story_id": 0, "hours_since_post": 6.0, "score": score_6h, "descendants": comms_6h},
]
stories = pd.DataFrame([{
    "id": 0, "title": title_in, "url": "", "by": author_in,
    "time": post_time, "type": "story",
}])
snaps_df = pd.DataFrame(snaps)

feat_df  = build_features(stories, snaps_df, T)
row      = feat_df.iloc[0]
X_row    = np.ascontiguousarray([[row[c] if c in row.index else 0.0 for c in fc]], dtype=np.float64)

pred  = float(bundle["m_tree"].predict(X_row)[0])
std   = bundle["resid_std"]
lo, hi = pred - 1.96 * std, pred + 1.96 * std
from scipy.stats import percentileofscore
qmv   = float(percentileofscore(bundle["all_preds"], pred, kind="weak"))

# Attribution
m_ridge = bundle["m_ridge"]
coef  = m_ridge.named_steps["r"].coef_
sc    = m_ridge.named_steps["sc"]
z     = (X_row[0] - sc.mean_) / (sc.scale_ + 1e-8)
contribs = {feat: float(coef[i] * z[i]) for i, feat in enumerate(fc)}
top5 = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

# ── layout: KPI row ────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("QMV Index", f"{qmv:.1f} / 100")
c2.metric("Predicted log-value", f"{pred:.3f}")
c3.metric("95% CI", f"[{lo:.2f}, {hi:.2f}]")
c4.metric("Model Version", bundle["version"])

st.markdown("---")
col_left, col_right = st.columns([1.4, 1])

# ── trajectory + forecast chart ────────────────────────────────────────
with col_left:
    st.subheader("📊 Engagement Trajectory & Forecast")

    obs_hours  = [0, 1, 3, 6]
    obs_scores = [0, score_1h, score_3h, score_6h]

    # Naive linear extrapolation for the "forecast" segment
    v = (score_6h - score_3h) / 3.0 if T >= 6 else (score_3h - score_1h) / 2.0
    decay = 0.85
    fc_hours = list(range(T, 73, 3))
    fc_scores = []
    last = obs_scores[obs_hours.index(T)] if T in obs_hours else score_3h
    for h in fc_hours:
        last = last + v * decay
        v *= decay
        fc_scores.append(max(last, 0))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(obs_hours[:obs_hours.index(T)+1 if T in obs_hours else len(obs_hours)],
            obs_scores[:obs_hours.index(T)+1 if T in obs_hours else len(obs_hours)],
            "o-", color="#2ecc71", linewidth=2, label="Observed")
    ax.plot(fc_hours, fc_scores, "--", color="#3498db", linewidth=1.5, label="Forecast")
    ax.fill_between(fc_hours,
                    [s * 0.7 for s in fc_scores],
                    [s * 1.3 for s in fc_scores],
                    alpha=0.15, color="#3498db", label="Confidence band")
    ax.set_xlabel("Hours since post")
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)
    ax.set_title(f"Trajectory for: {title_in[:60]}", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

# ── driver waterfall ───────────────────────────────────────────────────
with col_right:
    st.subheader("🔍 Driver Breakdown")

    driver_df = pd.DataFrame(top5, columns=["Feature", "Contribution"])
    driver_df["Feature"] = driver_df["Feature"].str.replace("_", " ")
    driver_df = driver_df.sort_values("Contribution")

    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in driver_df["Contribution"]]
    ax2.barh(driver_df["Feature"], driver_df["Contribution"], color=colors)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_title("Top 5 Feature Contributions", fontsize=9)
    ax2.set_xlabel("Ridge contribution")
    fig2.tight_layout()
    st.pyplot(fig2)

    st.markdown("**Plain-English Summary:**")
    parts = []
    for feat, cont in top5:
        sign = "+" if cont >= 0 else ""
        parts.append(f"{sign}{cont:.1f} from **{feat.replace('_', ' ')}**")
    st.markdown(" · ".join(parts))

st.markdown("---")
st.caption("QMV Index v1.0 · Predictions use T-hour early engagement signals only · Not financial advice.")
