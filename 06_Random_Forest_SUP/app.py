"""
app.py — Pharmaceutical Syrup pH Lot Approval Dashboard
LozanoLsa · Project 06 · Random Forest · 2026

Model: Random Forest Classifier (300 trees)
Domain: Pharmaceutical Manufacturing — Syrup pH Compliance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RF · Syrup pH Compliance Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── FULL CSS INJECTION ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&family=Instrument+Serif:ital@0;1&display=swap');

:root {
    --bg:       #080c12;
    --surface:  #0e1420;
    --card:     #121922;
    --card2:    #161f2e;
    --border:   #1e2d45;
    --blue:     #3b82f6;
    --blue2:    #60a5fa;
    --teal:     #2dd4bf;
    --danger:   #f87171;
    --warn:     #fbbf24;
    --text:     #c8d8f0;
    --muted:    #4e6a8a;
    --fh: 'Syne', sans-serif;
    --fm: 'JetBrains Mono', monospace;
    --fs: 'Instrument Serif', Georgia, serif;
}

.stApp { background: var(--bg) !important; color: var(--text); font-family: var(--fh); }
.block-container { padding: 1.8rem 2.4rem 3rem !important; max-width: 1400px !important; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1rem !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--muted) !important; letter-spacing: 0.05em; }
[data-testid="stSidebar"] label { font-family: var(--fm) !important; font-size: 0.7rem !important; color: var(--text) !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; }

[data-testid="stSlider"] [role="slider"] { background: var(--blue) !important; border: 2px solid var(--blue2) !important; box-shadow: 0 0 8px rgba(59,130,246,0.5) !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { font-family: var(--fm) !important; font-size: 0.65rem !important; color: var(--blue2) !important; background: var(--card) !important; border: 1px solid var(--border) !important; padding: 1px 5px !important; border-radius: 3px !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--blue) !important; }

[data-testid="stSelectbox"] > div > div { background: var(--card) !important; border: 1px solid var(--border) !important; color: var(--text) !important; font-family: var(--fm) !important; font-size: 0.78rem !important; border-radius: 3px !important; }

[data-testid="stMetric"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-top: 2px solid var(--blue) !important; padding: 1rem 1.1rem 0.9rem !important; border-radius: 3px !important; }
[data-testid="stMetricLabel"] > div { font-family: var(--fm) !important; font-size: 0.6rem !important; text-transform: uppercase !important; letter-spacing: 0.18em !important; color: var(--muted) !important; font-weight: 400 !important; }
[data-testid="stMetricValue"] > div { font-family: var(--fm) !important; font-size: 1.7rem !important; font-weight: 600 !important; color: var(--blue2) !important; line-height: 1.1 !important; }
[data-testid="stMetricDelta"] > div { font-family: var(--fm) !important; font-size: 0.68rem !important; }

[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid var(--border) !important; gap: 0 !important; background: transparent !important; }
[data-testid="stTabs"] [role="tab"] { font-family: var(--fm) !important; font-size: 0.68rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; color: var(--muted) !important; padding: 0.5rem 1.2rem !important; border: none !important; border-radius: 0 !important; background: transparent !important; transition: all 0.2s !important; }
[data-testid="stTabs"] [role="tab"]:hover { color: var(--blue2) !important; background: rgba(59,130,246,0.06) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: var(--blue) !important; border-bottom: 2px solid var(--blue) !important; background: transparent !important; }
[data-testid="stTabsContent"] { padding-top: 1.4rem !important; }

[data-testid="stAlert"] { border-radius: 2px !important; font-family: var(--fm) !important; font-size: 0.75rem !important; letter-spacing: 0.04em !important; border: none !important; }

[data-testid="stExpander"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 2px !important; margin-bottom: 6px !important; }
[data-testid="stExpander"] summary { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--text) !important; letter-spacing: 0.06em !important; }
[data-testid="stExpander"] p { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--muted) !important; line-height: 1.7 !important; }

[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 2px !important; }
[data-testid="stDataFrame"] th { font-family: var(--fm) !important; font-size: 0.62rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; background: var(--card2) !important; color: var(--muted) !important; border-bottom: 1px solid var(--border) !important; }
[data-testid="stDataFrame"] td { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--text) !important; background: var(--card) !important; }

hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
[data-testid="stCaptionContainer"] p { font-family: var(--fm) !important; font-size: 0.62rem !important; color: var(--muted) !important; letter-spacing: 0.08em !important; }

h1, h2, h3 { font-family: var(--fh) !important; color: var(--text) !important; letter-spacing: -0.01em !important; }
h2 { font-size: 1.1rem !important; font-weight: 700 !important; }
h3 { font-size: 0.9rem !important; font-weight: 600 !important; }
p, li { font-family: var(--fh) !important; font-size: 0.88rem !important; }

.lsa-header { border-bottom: 1px solid var(--border); padding-bottom: 1.2rem; margin-bottom: 0.2rem; }
.lsa-project-tag { font-family: var(--fm); font-size: 0.6rem; color: var(--blue); text-transform: uppercase; letter-spacing: 0.22em; margin-bottom: 4px; }
.lsa-title { font-family: var(--fh); font-size: 1.85rem; font-weight: 800; color: #fff; line-height: 1.1; letter-spacing: -0.02em; }
.lsa-tagline { font-family: var(--fs); font-style: italic; font-size: 0.9rem; color: var(--muted); margin-top: 4px; }
.lsa-chip { display: inline-block; background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.3); color: var(--blue2); font-family: var(--fm); font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 2px 8px; border-radius: 2px; margin-right: 5px; }
.lsa-section { font-family: var(--fm); font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid var(--border); }
.lsa-footer { margin-top: 2.5rem; padding-top: 0.8rem; border-top: 1px solid var(--border); font-family: var(--fm); font-size: 0.58rem; color: var(--muted); letter-spacing: 0.1em; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATA_PATH    = "rf_raw_data.csv"
RANDOM_STATE = 42
FEATURES = [
    "api_pct", "citrate_buffer_pct", "citric_acid_pct", "sweetener_pct",
    "preservative_pct", "water_ph", "prev_lot_ph", "mixing_temp_c",
    "mixing_time_min", "agitation_rpm", "addition_order"
]
TARGET       = "lot_approved"
ORDER_LABELS = {1: "Buffer→Acid→API", 2: "Acid→Buffer→API", 3: "API→Buffer→Acid"}

FEAT_LABELS = {
    "api_pct":            "API (%)",
    "citrate_buffer_pct": "Citrate Buffer (%)",
    "citric_acid_pct":    "Citric Acid (%)",
    "sweetener_pct":      "Sweetener (%)",
    "preservative_pct":   "Preservative (%)",
    "water_ph":           "Purified Water pH",
    "prev_lot_ph":        "Previous Lot pH",
    "mixing_temp_c":      "Mixing Temp (°C)",
    "mixing_time_min":    "Mixing Time (min)",
    "agitation_rpm":      "Agitation (RPM)",
    "addition_order":     "Addition Order",
}

METRIC_EXPL = {
    "Accuracy":  "Out of every 100 batches, the model classifies this many correctly.",
    "Precision": "When the model predicts approval, how often the batch actually passes.",
    "Recall":    "Out of all truly approved batches, how many the model catches.",
    "F1 Score":  "Balances precision and recall — useful with class imbalance.",
    "AUC-ROC":   "How well the model separates approved from non-approved batches across all thresholds.",
}

ACTION_MAP = {
    "citric_acid_pct":    "Review citric acid charge — excess acid is the primary driver of pH dropping below 4.5",
    "citrate_buffer_pct": "Increase citrate buffer — low buffer reduces pH resistance to formulation variability",
    "agitation_rpm":      "Increase agitation speed — insufficient RPM leaves formulation incompletely homogenized",
    "mixing_time_min":    "Extend mixing time — short mixing prevents full buffer-acid equilibration",
    "water_ph":           "Check purified water quality — high water pH contributes to pH instability",
    "prev_lot_ph":        "Review previous lot pH — carryover effect is amplified by poor agitation",
    "mixing_temp_c":      "Monitor mixing temperature — above 28°C increases acid dissociation and lowers pH",
    "addition_order":     "Standardize to order 1 (Buffer→Acid→API) for maximum pH stability",
}

# ─── PLOTLY COLORS ────────────────────────────────────────────────────────────
C_BLUE   = "#3b82f6"
C_BLUE2  = "#60a5fa"
C_TEAL   = "#2dd4bf"
C_DANGER = "#f87171"
C_WARN   = "#fbbf24"
C_MUTED  = "#4e6a8a"

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono", color=C_MUTED, size=10),
    xaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", zeroline=False),
    yaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", zeroline=False),
    margin=dict(l=4, r=4, t=40, b=4),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)

def plot(fig, h=300):
    fig.update_layout(height=h, **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

# ─── DATA & MODEL ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        return pd.read_csv("https://raw.githubusercontent.com/LozanoLsa/PH_Adjustment_Syrup/main/rf_raw_data.csv")

@st.cache_resource
def train_model(df):
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                           random_state=RANDOM_STATE, stratify=y)
    rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                 min_samples_split=2, min_samples_leaf=1,
                                 random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(Xtr, ytr)
    yp, ypr = rf.predict(Xte), rf.predict_proba(Xte)[:, 1]
    metrics = {
        "Accuracy":  accuracy_score(yte, yp),
        "Precision": precision_score(yte, yp),
        "Recall":    recall_score(yte, yp),
        "F1 Score":  f1_score(yte, yp),
        "AUC-ROC":   roc_auc_score(yte, ypr),
    }
    gini_df = pd.DataFrame({
        "Feature":  FEATURES,
        "Gini":     rf.feature_importances_,
    }).sort_values("Gini", ascending=False)
    return rf, Xtr, Xte, ytr, yte, yp, ypr, metrics, gini_df

df = load_data()
rf, X_train, X_test, y_train, y_test, y_pred, y_prob, metrics, gini_df = train_model(df)
approval_rate = df[TARGET].mean()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="lsa-project-tag">LozanoLsa · Project 06</div>
    <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;
                color:#fff;margin-bottom:6px;">Syrup pH<br>Compliance Predictor</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                color:#4e6a8a;line-height:1.7;margin-bottom:12px;">
        Random Forest · 300 trees · 1,847 batches<br>11 formulation & process features
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="lsa-section">// Formulation</div>', unsafe_allow_html=True)
    api_pct       = st.slider("API (%)", 0.5, 5.0, 2.0, 0.1)
    buffer_pct    = st.slider("Citrate Buffer (%)", 1.0, 5.0, 3.0, 0.1)
    acid_pct      = st.slider("Citric Acid (%)", 0.0, 2.0, 0.8, 0.05)
    sweetener_pct = st.slider("Sweetener (%)", 5.0, 40.0, 25.0, 0.5)
    preserv_pct   = st.slider("Preservative (%)", 0.1, 1.0, 0.5, 0.05)

    st.markdown('<div class="lsa-section">// Water & Previous Lot</div>', unsafe_allow_html=True)
    water_ph = st.slider("Purified Water pH", 6.0, 7.5, 7.0, 0.05)
    prev_ph  = st.slider("Previous Lot pH", 4.0, 6.0, 5.0, 0.05)

    st.markdown('<div class="lsa-section">// Process Conditions</div>', unsafe_allow_html=True)
    mix_temp  = st.slider("Mixing Temperature (°C)", 15.0, 35.0, 25.0, 0.5)
    mix_time  = st.slider("Mixing Time (min)", 5, 60, 30)
    agit_rpm  = st.slider("Agitation (RPM)", 100, 800, 500, 10)
    add_order = st.selectbox("Addition Order",
                              [(1, "1 — Buffer → Acid → API"),
                               (2, "2 — Acid → Buffer → API"),
                               (3, "3 — API → Buffer → Acid")],
                              format_func=lambda x: x[1])[0]

    st.divider()
    st.caption("300 trees · no feature scaling · n_jobs=-1")
    st.caption("Where f(x) meets Kaizen · 2026")

# ─── PREDICT ──────────────────────────────────────────────────────────────────
def predict_batch(ap, buf, ac, sw, pr, wph, pph, mt, mtime, rpm, ao):
    row = pd.DataFrame([{
        "api_pct": ap, "citrate_buffer_pct": buf, "citric_acid_pct": ac,
        "sweetener_pct": sw, "preservative_pct": pr, "water_ph": wph,
        "prev_lot_ph": pph, "mixing_temp_c": mt, "mixing_time_min": mtime,
        "agitation_rpm": rpm, "addition_order": ao
    }])
    p = rf.predict_proba(row)[0, 1]
    return p, int(p >= 0.5)

pred_prob, pred_class = predict_batch(
    api_pct, buffer_pct, acid_pct, sweetener_pct,
    preserv_pct, water_ph, prev_ph, mix_temp,
    mix_time, agit_rpm, add_order
)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-header">
    <div class="lsa-project-tag">ML Project #06 · Random Forest · Pharmaceutical Quality</div>
    <div class="lsa-title">pH Compliance Is Not Random</div>
    <div class="lsa-tagline">The acid-buffer balance is decided at formulation. The forest knows before the batch is made.</div>
    <div style="margin-top:10px;">
        <span class="lsa-chip">RANDOM FOREST</span>
        <span class="lsa-chip">300 TREES</span>
        <span class="lsa-chip">{metrics['Accuracy']:.1%} ACCURACY</span>
        <span class="lsa-chip">AUC {metrics['AUC-ROC']:.3f}</span>
        <span class="lsa-chip">NO SCALING</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "DATA EXPLORER", "PERFORMANCE", "SIMULATOR", "RISK DRIVERS", "ACTION PLAN"
])

# ══ TAB 1 ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="lsa-section">// Dataset overview</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Batches", f"{len(df):,}")
    k2.metric("Approved",      f"{df[TARGET].sum():,}")
    k3.metric("Not Approved",  f"{(df[TARGET]==0).sum():,}")
    k4.metric("Approval Rate", f"{approval_rate:.1%}")
    st.divider()

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown('<div class="lsa-section">// Lot approval distribution</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["Not Approved", "Approved"],
            values=[(df[TARGET]==0).sum(), df[TARGET].sum()],
            marker_colors=[C_DANGER, C_BLUE],
            hole=0.52, textinfo="percent+label",
            textfont=dict(family="JetBrains Mono", size=11),
        ))
        plot(fig_pie, 270)

    with c2:
        st.markdown('<div class="lsa-section">// Final pH — specification 4.5–5.5</div>', unsafe_allow_html=True)
        fph = go.Figure()
        fph.add_trace(go.Histogram(x=df["final_ph"], nbinsx=40,
                                   marker_color=C_BLUE, opacity=0.8, name="Final pH"))
        fph.add_vline(x=4.5, line_dash="dash", line_color=C_DANGER, line_width=1.5,
                      annotation_text="Lower spec 4.5",
                      annotation_font=dict(family="JetBrains Mono", size=8, color=C_DANGER))
        fph.add_vline(x=5.5, line_dash="dash", line_color=C_DANGER, line_width=1.5,
                      annotation_text="Upper spec 5.5",
                      annotation_font=dict(family="JetBrains Mono", size=8, color=C_DANGER))
        fph.update_layout(xaxis_title="Final pH", yaxis_title="Count", showlegend=False)
        plot(fph, 270)

    st.divider()
    st.markdown('<div class="lsa-section">// Feature distributions</div>', unsafe_allow_html=True)
    fs = st.selectbox("Feature:", FEATURES, format_func=lambda x: FEAT_LABELS.get(x, x))
    c3, c4 = st.columns(2)
    with c3:
        fig_hist = go.Figure()
        for cls, color, name in [(0, C_DANGER, "Not Approved"), (1, C_BLUE, "Approved")]:
            fig_hist.add_trace(go.Histogram(
                x=df[df[TARGET]==cls][fs], name=name,
                marker_color=color, opacity=0.65, nbinsx=30,
            ))
        fig_hist.update_layout(barmode="overlay",
                               xaxis_title=FEAT_LABELS.get(fs, fs),
                               yaxis_title="Count")
        plot(fig_hist, 295)

    with c4:
        st.markdown('<div class="lsa-section">// Approval rate by addition order</div>', unsafe_allow_html=True)
        oa     = df.groupby("addition_order")[TARGET].mean()
        fig_ao = go.Figure(go.Bar(
            x=[ORDER_LABELS[k] for k in oa.index],
            y=oa.values,
            marker_color=[C_BLUE if v >= approval_rate else C_DANGER for v in oa.values],
            text=[f"{v:.1%}" for v in oa.values],
            textfont=dict(family="JetBrains Mono", size=10, color="#c8d8f0"),
            textposition="outside",
        ))
        fig_ao.add_hline(y=approval_rate, line_dash="dash", line_color=C_MUTED, line_width=1,
                         annotation_text=f"avg {approval_rate:.1%}",
                         annotation_font=dict(family="JetBrains Mono", size=8, color=C_MUTED))
        fig_ao.update_layout(yaxis_title="Approval Rate", yaxis_range=[0, 0.75],
                              yaxis_tickformat=".0%")
        plot(fig_ao, 295)

    st.divider()
    st.markdown('<div class="lsa-section">// Feature correlation matrix</div>', unsafe_allow_html=True)
    fig_corr = px.imshow(
        df[FEATURES + ["final_ph", TARGET]].corr().round(2),
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        text_auto=True, aspect="auto"
    )
    plot(fig_corr, 460)

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="lsa-section">// Model metrics</div>', unsafe_allow_html=True)
    m_cols = st.columns(5)
    for i, (name, val) in enumerate(metrics.items()):
        m_cols[i].metric(name, f"{val:.3f}")
    st.divider()

    cm_arr = confusion_matrix(y_test, y_pred)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="lsa-section">// Confusion matrix</div>', unsafe_allow_html=True)
        fig_cm = go.Figure(go.Heatmap(
            z=cm_arr,
            x=["Pred: Not Approved", "Pred: Approved"],
            y=["Actual: Not Approved", "Actual: Approved"],
            colorscale=[[0, "#0a1525"], [1, C_BLUE]],
            text=cm_arr, texttemplate="%{text}",
            textfont=dict(family="JetBrains Mono", size=18, color="#fff"),
            showscale=False,
        ))
        plot(fig_cm, 300)
        st.caption("Rows = actual class · Columns = predicted class · Diagonal = correct")

    with c2:
        st.markdown('<div class="lsa-section">// Predicted probability by true class</div>', unsafe_allow_html=True)
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Histogram(
            x=y_prob[y_test==0], name="Actual: Not Approved",
            marker_color=C_DANGER, opacity=0.65, nbinsx=30,
        ))
        fig_prob.add_trace(go.Histogram(
            x=y_prob[y_test==1], name="Actual: Approved",
            marker_color=C_BLUE, opacity=0.65, nbinsx=30,
        ))
        fig_prob.add_vline(x=0.5, line_dash="dash", line_color="#fff", line_width=1,
                           annotation_text="threshold 0.5",
                           annotation_font=dict(family="JetBrains Mono", size=9, color=C_MUTED))
        fig_prob.update_layout(barmode="overlay",
                               xaxis_title="P(Lot Approved)",
                               yaxis_title="Count")
        plot(fig_prob, 300)

    st.divider()
    st.markdown('<div class="lsa-section">// ROC curve</div>', unsafe_allow_html=True)
    fpr_r, tpr_r, _ = roc_curve(y_test, y_prob)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr_r, y=tpr_r, mode="lines",
        name=f"Random Forest (AUC = {metrics['AUC-ROC']:.3f})",
        line=dict(color=C_BLUE, width=2),
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random classifier",
        line=dict(color=C_MUTED, width=1, dash="dash"),
    ))
    fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    plot(fig_roc, 340)
    st.caption(f"Random Forest · 300 trees · max_depth=None · 70/30 stratified · random_state={RANDOM_STATE}")

    st.divider()
    st.markdown('<div class="lsa-section">// Metric explanations</div>', unsafe_allow_html=True)
    for name, expl in METRIC_EXPL.items():
        with st.expander(f"{name}  —  {metrics[name]:.3f}"):
            st.write(expl)

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    left, right = st.columns([1, 2])
    with left:
        st.markdown('<div class="lsa-section">// P(pH in specification)</div>', unsafe_allow_html=True)
        # Inverted logic: high prob = approved = teal/green, low prob = danger
        gauge_color = C_TEAL if pred_class == 1 else C_DANGER
        fg = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_prob * 100,
            number={"suffix": "%", "font": {"size": 38, "family": "JetBrains Mono", "color": "#fff"}},
            title={"text": "P(pH in Specification)", "font": {"size": 11, "family": "JetBrains Mono", "color": C_MUTED}},
            gauge={
                "axis": {"range": [0, 100],
                         "tickfont": {"size": 9, "family": "JetBrains Mono"},
                         "tickcolor": "#1e2d45"},
                "bar": {"color": gauge_color, "thickness": 0.22},
                "bgcolor": "#0e1420", "bordercolor": "#1e2d45",
                "steps": [
                    {"range": [0,  35], "color": "rgba(248,113,113,0.10)"},
                    {"range": [35, 60], "color": "rgba(251,191,36,0.08)"},
                    {"range": [60, 100], "color": "rgba(45,212,191,0.08)"},
                ],
                "threshold": {"line": {"color": "#fff", "width": 2}, "thickness": 0.75, "value": 50},
            }
        ))
        fg.update_layout(height=290, paper_bgcolor="rgba(0,0,0,0)",
                         margin=dict(t=50, b=10, l=20, r=20))
        st.plotly_chart(fg, use_container_width=True)
        if pred_class == 1:
            st.success("APPROVED — pH LIKELY IN SPECIFICATION")
        else:
            st.error("REVIEW — pH OUT-OF-SPEC RISK ELEVATED")
        st.caption(f"Historical approval rate: {approval_rate:.1%}  ·  This batch: {pred_prob:.1%}  ·  Δ {pred_prob - approval_rate:+.1%}")

    with right:
        st.markdown('<div class="lsa-section">// Gini weight × deviation from training mean</div>', unsafe_allow_html=True)
        devs    = np.array([api_pct, buffer_pct, acid_pct, sweetener_pct, preserv_pct,
                            water_ph, prev_ph, mix_temp, mix_time, agit_rpm, add_order]) \
                  - X_train.mean().values
        contrib = gini_df.set_index("Feature")["Gini"] * pd.Series(devs, index=FEATURES)
        top5    = contrib.abs().nlargest(5).index
        c5      = contrib[top5].reset_index()
        c5.columns = ["Feature", "Contribution"]
        c5["Label"] = c5["Feature"].map(lambda x: FEAT_LABELS.get(x, x.replace("_", " ").title()))
        fc2 = go.Figure(go.Bar(
            x=c5["Contribution"], y=c5["Label"], orientation="h",
            # Negative = pulling toward rejection (bad), positive = pulling toward approval (good)
            marker_color=[C_DANGER if v < 0 else C_TEAL for v in c5["Contribution"]],
            text=[f"{v:+.4f}" for v in c5["Contribution"]],
            textfont=dict(family="JetBrains Mono", size=10, color="#c8d8f0"),
            textposition="outside",
        ))
        fc2.update_layout(xaxis_title="Gini weight × deviation from training mean")
        plot(fc2, 270)
        if pred_prob <= 0.30:
            st.error("Risk: HIGH · Reformulate before manufacturing")
        elif pred_prob <= 0.55:
            st.warning("Risk: MEDIUM · Review parameters before release")
        else:
            st.success("Risk: LOW · Profile consistent with specification compliance")

    st.divider()
    st.markdown('<div class="lsa-section">// Scenario comparison</div>', unsafe_allow_html=True)
    bp  = predict_batch(2.0, 4.0, 0.8, 30.0, 0.5, 7.0, 5.0, 25.0, 40, 600, 1)[0]
    wp  = predict_batch(2.0, 1.2, 1.7, 20.0, 0.8, 6.5, 4.5, 30.0, 10, 150, 2)[0]
    cdf = pd.DataFrame([
        {"Scenario": "Best case — stable formulation + good process", "P(Approved)": f"{bp:.1%}",
         "Status": "Approved" if bp >= 0.5 else "Not Approved",
         "Δ vs current": f"{bp - pred_prob:+.1%}"},
        {"Scenario": "Current batch configuration",                   "P(Approved)": f"{pred_prob:.1%}",
         "Status": "Approved" if pred_class == 1 else "Not Approved",
         "Δ vs current": "—"},
        {"Scenario": "Worst case — high acid + poor process",         "P(Approved)": f"{wp:.1%}",
         "Status": "Approved" if wp >= 0.5 else "Not Approved",
         "Δ vs current": f"{wp - pred_prob:+.1%}"},
    ])
    st.dataframe(cdf, use_container_width=True, hide_index=True)

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="lsa-section">// Gini importance (built-in RF)</div>', unsafe_allow_html=True)
        gi_s = gini_df.sort_values("Gini", ascending=True)
        fig_gi = go.Figure(go.Bar(
            x=gi_s["Gini"],
            y=[FEAT_LABELS.get(f, f.replace("_", " ").title()) for f in gi_s["Feature"]],
            orientation="h",
            marker_color=[C_DANGER if v > 0.05 else C_BLUE for v in gi_s["Gini"]],
            text=[f"{v:.3f}" for v in gi_s["Gini"]],
            textfont=dict(family="JetBrains Mono", size=9, color="#c8d8f0"),
            textposition="outside",
        ))
        fig_gi.add_vline(x=0.05, line_dash="dash", line_color=C_WARN, line_width=1,
                         annotation_text="5% threshold",
                         annotation_font=dict(family="JetBrains Mono", size=8, color=C_WARN))
        fig_gi.update_layout(xaxis_title="Gini Importance (mean impurity reduction)")
        plot(fig_gi, 400)

    with c2:
        st.markdown('<div class="lsa-section">// Approval rate by acid-to-buffer ratio</div>', unsafe_allow_html=True)
        df3        = df.copy()
        df3["abr"] = (df3["citric_acid_pct"] / df3["citrate_buffer_pct"]).round(2)
        bins3      = pd.cut(df3["abr"], bins=8)
        ra3        = df3.groupby(bins3)[TARGET].mean().reset_index()
        ra3.columns = ["ratio", "rate"]
        ra3["label"] = ra3["ratio"].astype(str)
        fig_abr = go.Figure(go.Bar(
            x=ra3["label"], y=ra3["rate"],
            marker_color=[C_BLUE if v >= 0.5 else C_DANGER for v in ra3["rate"]],
            text=[f"{v:.1%}" for v in ra3["rate"]],
            textfont=dict(family="JetBrains Mono", size=9, color="#c8d8f0"),
            textposition="outside",
        ))
        fig_abr.add_hline(y=0.5, line_dash="dash", line_color="#fff", line_width=1,
                          annotation_text="50% threshold",
                          annotation_font=dict(family="JetBrains Mono", size=8, color=C_MUTED))
        fig_abr.update_layout(xaxis_title="Citric Acid / Citrate Buffer Ratio",
                              yaxis_title="Approval Rate", yaxis_tickformat=".0%",
                              xaxis_tickangle=30)
        plot(fig_abr, 400)
        st.caption("As acid/buffer ratio rises, approval rate drops. The inflection point is the formulation boundary.")

    st.divider()
    st.markdown('<div class="lsa-section">// Interactive feature scatter — top importance features</div>', unsafe_allow_html=True)
    t5 = gini_df.head(5)["Feature"].tolist()
    sx = st.selectbox("X axis:", t5, index=0, format_func=lambda x: FEAT_LABELS.get(x, x))
    sy = st.selectbox("Y axis:", t5, index=1, format_func=lambda x: FEAT_LABELS.get(x, x))
    fig_sc = px.scatter(
        df.sample(600, random_state=42), x=sx, y=sy, color=TARGET,
        color_discrete_map={0: C_DANGER, 1: C_BLUE}, opacity=0.5,
        labels={TARGET: "Lot Approved",
                sx: FEAT_LABELS.get(sx, sx), sy: FEAT_LABELS.get(sy, sy)},
    )
    fig_sc.update_traces(marker=dict(size=5))
    plot(fig_sc, 380)

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    # Inverted: low pred_prob = high risk of rejection
    if pred_prob <= 0.30:
        pl, hz, ac = "HIGH",   "Before manufacture",    "Reformulate — adjust acid/buffer balance and review process parameters before batch release"
    elif pred_prob <= 0.55:
        pl, hz, ac = "MEDIUM", "Pre-manufacture review", "Review borderline parameters — confirm citric acid and buffer concentrations"
    else:
        pl, hz, ac = "LOW",    "Standard monitoring",    "Proceed with standard in-process pH monitoring"

    badge_color = {"HIGH": C_DANGER, "MEDIUM": C_WARN, "LOW": C_TEAL}[pl]
    st.markdown(f"""
    <div style="background:var(--card);border:1px solid var(--border);border-left:3px solid {badge_color};
                padding:1.2rem 1.4rem;border-radius:2px;margin-bottom:1rem;">
        <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:8px;">// Action plan</div>
        <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px;">
            <div>
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;">Priority</div>
                <div style="font-family:var(--fh);font-size:1.3rem;font-weight:800;color:{badge_color};">{pl}</div>
            </div>
            <div>
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;">Est. Compliance Prob</div>
                <div style="font-family:var(--fm);font-size:1.3rem;font-weight:600;color:#fff;">{pred_prob:.1%}</div>
            </div>
            <div>
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;">Horizon</div>
                <div style="font-family:var(--fm);font-size:0.85rem;color:var(--text);">{hz}</div>
            </div>
            <div>
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;">Owner</div>
                <div style="font-family:var(--fm);font-size:0.85rem;color:var(--text);">Formulation Chemist / QA</div>
            </div>
        </div>
        <div style="margin-top:12px;font-family:var(--fm);font-size:0.72rem;color:var(--text);">
            <span style="color:var(--muted);">Action → </span>{ac}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="lsa-section">// Active risk factors</div>', unsafe_allow_html=True)
    rfs2 = []
    if acid_pct   >  1.2:  rfs2.append("citric_acid_pct")
    if buffer_pct <  2.0:  rfs2.append("citrate_buffer_pct")
    if agit_rpm   <  300:  rfs2.append("agitation_rpm")
    if mix_time   <  15:   rfs2.append("mixing_time_min")
    if water_ph   >  7.2:  rfs2.append("water_ph")
    if prev_ph    <  4.6:  rfs2.append("prev_lot_ph")
    if mix_temp   >  28:   rfs2.append("mixing_temp_c")
    if add_order  == 2:    rfs2.append("addition_order")

    if rfs2:
        for f in rfs2[:4]:
            if f in ACTION_MAP:
                label = FEAT_LABELS.get(f, f.replace("_", " ").title())
                with st.expander(f"▲  {label}  —  active risk factor"):
                    st.write(ACTION_MAP[f])
    else:
        st.success("No elevated risk factors detected — batch parameters within normal ranges.")

    st.divider()
    st.caption("This tool supports formulation and process decisions. It does not replace QA release testing, in-process pH monitoring, or regulatory batch disposition procedures.")

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lsa-footer">
    LozanoLsa · Turning Operations into Predictive Systems · Syrup pH Compliance Predictor · Project 06 · v2.0
</div>
""", unsafe_allow_html=True)
