"""
app.py — Surface Defect Inspection Dashboard
LozanoLsa · Project 02 · Gaussian Naive Bayes · 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NB · Surface Defect Predictor",
    page_icon="🔬",
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
DATA_PATH    = "surface_defect_inspection_data.csv"
RANDOM_STATE = 42
NUM_COLS = [
    "regrind_pct", "resin_temp_c", "cooling_time_s",
    "paint_viscosity", "film_thickness_um", "booth_humidity_pct",
    "pre_paint_storage_hrs", "num_handlings"
]
CAT_COLS = ["container_type", "part_protection"]
TARGET   = "surface_defect"

FEAT_LABELS = {
    "regrind_pct":                     "Regrind (%)",
    "resin_temp_c":                    "Resin Temp (°C)",
    "cooling_time_s":                  "Cooling Time (s)",
    "paint_viscosity":                 "Paint Viscosity",
    "film_thickness_um":               "Film Thickness (μm)",
    "booth_humidity_pct":              "Booth Humidity (%)",
    "pre_paint_storage_hrs":           "Pre-Paint Storage (hrs)",
    "num_handlings":                   "Num Handlings",
    "container_type_metal_rack":       "Container: Metal Rack",
    "container_type_cardboard_pallet": "Container: Cardboard Pallet",
    "part_protection_unprotected":     "Protection: Unprotected",
}

METRIC_EXPL = {
    "Accuracy":  "Out of every 100 parts, the model classifies this many correctly.",
    "Precision": "When the model flags a part as defective, how often it's actually defective.",
    "Recall":    "Out of all truly defective parts, how many the model catches. Critical — missed defects reach the customer.",
    "F1 Score":  "Balances precision and recall — important when both false positives and false negatives are costly.",
    "AUC-ROC":   "How well the model separates defective from non-defective parts across all thresholds. 1.0 = perfect.",
}

ACTION_MAP = {
    "pre_paint_storage_hrs":            "Reduce pre-paint waiting time — implement strict FIFO and buffer zone limits (target < 8 hrs)",
    "num_handlings":                    "Reduce handling steps — redesign internal flow to minimize part contact (target <= 3)",
    "part_protection_unprotected":      "Mandate part protection for all containers — especially metal racks and long-distance moves",
    "container_type_metal_rack":        "Evaluate metal rack lining or foam inserts — direct metal contact causes micro-scratches",
    "container_type_cardboard_pallet":  "Replace cardboard pallets — they introduce dust and moisture absorption",
    "booth_humidity_pct":               "Control paint booth humidity — keep below 60% to prevent adhesion failures",
    "regrind_pct":                      "Limit regrind percentage — above 25% increases surface porosity and paint adhesion issues",
    "resin_temp_c":                     "Monitor resin temperature — maintain within spec (225–245 °C) throughout the run",
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
        return pd.read_csv(
            "https://raw.githubusercontent.com/LozanoLsa/Visual_Defects_Are_Not_Random/main/surface_defect_inspection_data.csv"
        )

@st.cache_resource
def train_model(df):
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    prep  = ColumnTransformer([
        ("num", "passthrough", NUM_COLS),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), CAT_COLS)
    ])
    model = Pipeline([("prep", prep), ("clf", GaussianNB())])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1 Score":  f1_score(y_test, y_pred),
        "AUC-ROC":   roc_auc_score(y_test, y_prob),
    }
    return model, X_train, X_test, y_train, y_test, y_pred, y_prob, metrics

def cohens_d(g0, g1):
    n0, n1 = len(g0), len(g1)
    s = np.sqrt(((n0-1)*g0.std()**2 + (n1-1)*g1.std()**2) / (n0+n1-2))
    return abs((g1.mean() - g0.mean()) / s)

df = load_data()
model, X_train, X_test, y_train, y_test, y_pred, y_prob, metrics = train_model(df)
defect_rate = df[TARGET].mean()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="lsa-project-tag">LozanoLsa · Project 02</div>
    <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;
                color:#fff;margin-bottom:6px;">Surface Defect<br>Predictor</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                color:#4e6a8a;line-height:1.7;margin-bottom:12px;">
        Gaussian Naive Bayes on 1,847 parts<br>8 numeric · 2 categorical features
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="lsa-section">// Internal Logistics</div>', unsafe_allow_html=True)
    container_type        = st.selectbox("Container Type", ["plastic_box", "metal_rack", "cardboard_pallet"])
    part_protection       = st.selectbox("Part Protection", ["protected", "unprotected"])
    pre_paint_storage_hrs = st.slider("Pre-Paint Storage (hrs)", 0.0, 24.0, 4.0, 0.5)
    num_handlings         = st.slider("Number of Handlings", 1, 8, 3)

    st.markdown('<div class="lsa-section">// Paint Cabin</div>', unsafe_allow_html=True)
    paint_viscosity    = st.slider("Paint Viscosity", 18.0, 35.0, 25.0, 0.5)
    film_thickness_um  = st.slider("Film Thickness (μm)", 20.0, 45.0, 30.0, 0.5)
    booth_humidity_pct = st.slider("Booth Humidity (%)", 25.0, 80.0, 50.0, 1.0)

    st.markdown('<div class="lsa-section">// Injection</div>', unsafe_allow_html=True)
    regrind_pct    = st.slider("Regrind (%)", 0.0, 40.0, 10.0, 0.5)
    resin_temp_c   = st.slider("Resin Temperature (°C)", 215.0, 260.0, 235.0, 0.5)
    cooling_time_s = st.slider("Cooling Time (s)", 8.0, 28.0, 18.0, 0.5)

    st.divider()
    st.caption("GaussianNB · ColumnTransformer pipeline · no feature scaling needed")
    st.caption("Where f(x) meets Kaizen · 2026")

# ─── PREDICT ──────────────────────────────────────────────────────────────────
def predict_scenario(ct, pp, psh, nh, pv, ft, bh, rg, rt, cool):
    row = pd.DataFrame([{
        "regrind_pct": rg, "resin_temp_c": rt, "cooling_time_s": cool,
        "paint_viscosity": pv, "film_thickness_um": ft, "booth_humidity_pct": bh,
        "pre_paint_storage_hrs": psh, "num_handlings": nh,
        "container_type": ct, "part_protection": pp
    }])
    prob = model.predict_proba(row)[0, 1]
    return prob, int(prob >= 0.5)

pred_prob, pred_class = predict_scenario(
    container_type, part_protection, pre_paint_storage_hrs, num_handlings,
    paint_viscosity, film_thickness_um, booth_humidity_pct,
    regrind_pct, resin_temp_c, cooling_time_s
)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-header">
    <div class="lsa-project-tag">ML Project #02 · Gaussian Naive Bayes · Quality Inspection</div>
    <div class="lsa-title">Visual Defects Are Not Random</div>
    <div class="lsa-tagline">The defect appears at the paint cabin. The cause was decided three steps earlier.</div>
    <div style="margin-top:10px;">
        <span class="lsa-chip">GAUSSIAN NAIVE BAYES</span>
        <span class="lsa-chip">10 FEATURES</span>
        <span class="lsa-chip">{metrics['Accuracy']:.1%} ACCURACY</span>
        <span class="lsa-chip">AUC {metrics['AUC-ROC']:.3f}</span>
        <span class="lsa-chip">NO SCALING REQUIRED</span>
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
    k1.metric("Total Parts", f"{len(df):,}")
    k2.metric("Defective",   f"{df[TARGET].sum():,}")
    k3.metric("Passed",      f"{(df[TARGET]==0).sum():,}")
    k4.metric("Defect Rate", f"{defect_rate:.1%}")
    st.divider()

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown('<div class="lsa-section">// Class distribution</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["Pass", "Defect"],
            values=[(df[TARGET]==0).sum(), df[TARGET].sum()],
            marker_colors=[C_BLUE, C_DANGER],
            hole=0.52, textinfo="percent+label",
            textfont=dict(family="JetBrains Mono", size=11),
        ))
        plot(fig_pie, 270)

    with c2:
        st.markdown('<div class="lsa-section">// Defect rate by category</div>', unsafe_allow_html=True)
        cat_sel = st.selectbox("Category:", CAT_COLS)
        rates   = df.groupby(cat_sel)[TARGET].mean().reset_index().sort_values(TARGET)
        fig_cat = go.Figure(go.Bar(
            x=rates[TARGET],
            y=rates[cat_sel],
            orientation="h",
            marker_color=[C_DANGER if v > defect_rate else C_BLUE for v in rates[TARGET]],
            text=[f"{v:.1%}" for v in rates[TARGET]],
            textfont=dict(family="JetBrains Mono", size=10, color="#c8d8f0"),
            textposition="outside",
        ))
        fig_cat.update_layout(xaxis_tickformat=".0%")
        plot(fig_cat, 270)

    st.divider()
    st.markdown('<div class="lsa-section">// Feature distributions</div>', unsafe_allow_html=True)
    num_sel = st.selectbox("Numeric feature:", NUM_COLS,
                           format_func=lambda x: FEAT_LABELS.get(x, x))
    c3, c4 = st.columns(2)
    with c3:
        fig_hist = go.Figure()
        for cls, color, name in [(0, C_BLUE, "Pass"), (1, C_DANGER, "Defect")]:
            fig_hist.add_trace(go.Histogram(
                x=df[df[TARGET]==cls][num_sel], name=name,
                marker_color=color, opacity=0.65, nbinsx=30,
            ))
        fig_hist.update_layout(barmode="overlay",
                               xaxis_title=FEAT_LABELS.get(num_sel, num_sel),
                               yaxis_title="Count")
        plot(fig_hist, 300)

    with c4:
        fig_box = px.box(
            df, x=df[TARGET].map({0: "Pass", 1: "Defect"}), y=num_sel,
            color=df[TARGET].map({0: "Pass", 1: "Defect"}),
            color_discrete_map={"Pass": C_BLUE, "Defect": C_DANGER},
        )
        fig_box.update_layout(showlegend=False,
                              xaxis_title="",
                              yaxis_title=FEAT_LABELS.get(num_sel, num_sel))
        plot(fig_box, 300)

    st.divider()
    st.markdown('<div class="lsa-section">// Correlation with surface defect</div>', unsafe_allow_html=True)
    df_dum  = pd.get_dummies(df, drop_first=True)
    corr_t  = df_dum.corr()[[TARGET]].sort_values(TARGET, ascending=False)
    fig_corr = px.imshow(corr_t.T, color_continuous_scale="RdBu_r",
                         zmin=-0.3, zmax=0.3, text_auto=".3f", aspect="auto")
    plot(fig_corr, 240)

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
            x=["Pred: Pass", "Pred: Defect"],
            y=["Actual: Pass", "Actual: Defect"],
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
            x=y_prob[y_test==0], name="Actual: Pass",
            marker_color=C_BLUE, opacity=0.65, nbinsx=30,
        ))
        fig_prob.add_trace(go.Histogram(
            x=y_prob[y_test==1], name="Actual: Defect",
            marker_color=C_DANGER, opacity=0.65, nbinsx=30,
        ))
        fig_prob.add_vline(x=0.5, line_dash="dash", line_color="#fff", line_width=1,
                           annotation_text="threshold 0.5",
                           annotation_font=dict(family="JetBrains Mono", size=9, color=C_MUTED))
        fig_prob.update_layout(barmode="overlay",
                               xaxis_title="Predicted Defect Probability",
                               yaxis_title="Count")
        plot(fig_prob, 300)

    st.divider()
    st.markdown('<div class="lsa-section">// ROC curve</div>', unsafe_allow_html=True)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"GaussianNB (AUC = {metrics['AUC-ROC']:.3f})",
        line=dict(color=C_BLUE, width=2),
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random classifier",
        line=dict(color=C_MUTED, width=1, dash="dash"),
    ))
    fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    plot(fig_roc, 340)
    st.caption(f"Gaussian Naive Bayes · ColumnTransformer pipeline · 70/30 stratified split · random_state={RANDOM_STATE}")

    st.divider()
    st.markdown('<div class="lsa-section">// Metric explanations</div>', unsafe_allow_html=True)
    for name, expl in METRIC_EXPL.items():
        with st.expander(f"{name}  —  {metrics[name]:.3f}"):
            st.write(expl)

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    left, right = st.columns([1, 2])
    with left:
        st.markdown('<div class="lsa-section">// Defect probability</div>', unsafe_allow_html=True)
        gauge_color = C_DANGER if pred_class == 1 else C_TEAL
        fg = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_prob * 100,
            number={"suffix": "%", "font": {"size": 38, "family": "JetBrains Mono", "color": "#fff"}},
            title={"text": "P(Surface Defect)", "font": {"size": 11, "family": "JetBrains Mono", "color": C_MUTED}},
            gauge={
                "axis": {"range": [0, 100],
                         "tickfont": {"size": 9, "family": "JetBrains Mono"},
                         "tickcolor": "#1e2d45"},
                "bar": {"color": gauge_color, "thickness": 0.22},
                "bgcolor": "#0e1420", "bordercolor": "#1e2d45",
                "steps": [
                    {"range": [0,  30], "color": "rgba(45,212,191,0.08)"},
                    {"range": [30, 60], "color": "rgba(251,191,36,0.08)"},
                    {"range": [60, 100], "color": "rgba(248,113,113,0.10)"},
                ],
                "threshold": {"line": {"color": "#fff", "width": 2}, "thickness": 0.75, "value": 50},
            }
        ))
        fg.update_layout(height=290, paper_bgcolor="rgba(0,0,0,0)",
                         margin=dict(t=50, b=10, l=20, r=20))
        st.plotly_chart(fg, use_container_width=True)
        if pred_class == 1:
            st.error("DEFECT RISK — HOLD BEFORE PAINT CABIN")
        else:
            st.success("LOW RISK — PROCEED TO PAINTING")
        st.caption(f"Dataset avg: {defect_rate:.1%}  ·  This part: {pred_prob:.1%}  ·  Δ {pred_prob - defect_rate:+.1%}")

    with right:
        st.markdown('<div class="lsa-section">// Log-likelihood contribution (class 1 vs 0)</div>', unsafe_allow_html=True)
        clf_obj  = model.named_steps["clf"]
        prep_obj = model.named_steps["prep"]
        X_new_t  = prep_obj.transform(pd.DataFrame([{
            "regrind_pct": regrind_pct, "resin_temp_c": resin_temp_c,
            "cooling_time_s": cooling_time_s, "paint_viscosity": paint_viscosity,
            "film_thickness_um": film_thickness_um, "booth_humidity_pct": booth_humidity_pct,
            "pre_paint_storage_hrs": pre_paint_storage_hrs, "num_handlings": num_handlings,
            "container_type": container_type, "part_protection": part_protection
        }]))
        eps  = 1e-9
        ll0  = -0.5*((X_new_t - clf_obj.theta_[0])**2 / (clf_obj.var_[0]+eps)) - 0.5*np.log(2*np.pi*(clf_obj.var_[0]+eps))
        ll1  = -0.5*((X_new_t - clf_obj.theta_[1])**2 / (clf_obj.var_[1]+eps)) - 0.5*np.log(2*np.pi*(clf_obj.var_[1]+eps))
        diff = (ll1 - ll0).flatten()
        ohe_feats   = list(prep_obj.named_transformers_["cat"].get_feature_names_out(CAT_COLS))
        all_feats   = NUM_COLS + ohe_feats
        diff_series = pd.Series(diff, index=all_feats)
        top5        = diff_series.abs().nlargest(5).index
        contrib_df  = diff_series[top5].reset_index()
        contrib_df.columns = ["Feature", "Contribution"]
        contrib_df["Label"] = contrib_df["Feature"].map(
            lambda x: FEAT_LABELS.get(x, x.replace("_", " ").title())
        )
        fc = go.Figure(go.Bar(
            x=contrib_df["Contribution"],
            y=contrib_df["Label"],
            orientation="h",
            marker_color=[C_DANGER if v > 0 else C_TEAL for v in contrib_df["Contribution"]],
            text=[f"{v:+.2f}" for v in contrib_df["Contribution"]],
            textfont=dict(family="JetBrains Mono", size=10, color="#c8d8f0"),
            textposition="outside",
        ))
        fc.update_layout(xaxis_title="Log-likelihood diff (class Defect vs Pass)")
        plot(fc, 270)
        if pred_prob >= 0.60:
            st.error("Priority: HIGH · Hold batch — root cause review before painting")
        elif pred_prob >= 0.30:
            st.warning("Priority: MEDIUM · Sample inspection before releasing to paint cabin")
        else:
            st.success("Priority: LOW · Standard process — proceed to painting")

    st.divider()
    st.markdown('<div class="lsa-section">// Scenario comparison</div>', unsafe_allow_html=True)
    bp = predict_scenario("plastic_box", "protected", 1.0, 2, 25.0, 30.0, 50.0, 5.0, 235.0, 18.0)[0]
    wp = predict_scenario("metal_rack",  "unprotected", 18.0, 6, 21.0, 38.0, 68.0, 28.0, 248.0, 13.0)[0]
    cdf = pd.DataFrame([
        {"Scenario": "Best case — controlled process",  "P(Defect)": f"{bp:.1%}",
         "Status": "Pass" if bp < 0.5 else "Defect",
         "Δ vs current": f"{bp - pred_prob:+.1%}"},
        {"Scenario": "Your current profile",            "P(Defect)": f"{pred_prob:.1%}",
         "Status": "Pass" if pred_class==0 else "Defect",
         "Δ vs current": "—"},
        {"Scenario": "Worst case — high-risk profile",  "P(Defect)": f"{wp:.1%}",
         "Status": "Pass" if wp < 0.5 else "Defect",
         "Δ vs current": f"{wp - pred_prob:+.1%}"},
    ])
    st.dataframe(cdf, use_container_width=True, hide_index=True)

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    cd_scores = {c: cohens_d(df[df[TARGET]==0][c], df[df[TARGET]==1][c]) for c in NUM_COLS}
    cd_series = pd.Series(cd_scores).sort_values(ascending=True)
    df_dum2   = pd.get_dummies(df.drop(TARGET, axis=1), drop_first=True)
    mi        = mutual_info_classif(df_dum2, df[TARGET], random_state=RANDOM_STATE)
    mi_s      = pd.Series(mi, index=df_dum2.columns).sort_values(ascending=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="lsa-section">// Cohen\'s d — numeric class separation</div>', unsafe_allow_html=True)
        fig_cd = go.Figure(go.Bar(
            x=cd_series.values,
            y=[FEAT_LABELS.get(f, f.replace("_"," ").title()) for f in cd_series.index],
            orientation="h",
            marker_color=[C_DANGER if v > 0.2 else C_BLUE for v in cd_series.values],
            text=[f"{v:.3f}" for v in cd_series.values],
            textfont=dict(family="JetBrains Mono", size=9, color="#c8d8f0"),
            textposition="outside",
        ))
        fig_cd.add_vline(x=0.2, line_dash="dash", line_color=C_WARN, line_width=1,
                         annotation_text="small effect",
                         annotation_font=dict(family="JetBrains Mono", size=8, color=C_WARN))
        fig_cd.update_layout(xaxis_title="Cohen's d (pooled effect size)")
        plot(fig_cd, 370)

    with c2:
        st.markdown('<div class="lsa-section">// Mutual information — all features</div>', unsafe_allow_html=True)
        fig_mi = go.Figure(go.Bar(
            x=mi_s.values,
            y=[FEAT_LABELS.get(f, f.replace("_"," ").title()) for f in mi_s.index],
            orientation="h",
            marker_color=[C_DANGER if v > mi_s.median() else C_BLUE for v in mi_s.values],
            text=[f"{v:.4f}" for v in mi_s.values],
            textfont=dict(family="JetBrains Mono", size=9, color="#c8d8f0"),
            textposition="outside",
        ))
        fig_mi.update_layout(xaxis_title="Mutual Information (bits)")
        plot(fig_mi, 370)

    st.divider()
    st.markdown('<div class="lsa-section">// Digital Pareto — combined driver ranking</div>', unsafe_allow_html=True)
    cd_norm = pd.Series(cd_scores) / pd.Series(cd_scores).max() * 100
    mi_full = pd.Series(mi, index=df_dum2.columns)
    mi_norm = mi_full / mi_full.max() * 100 if mi_full.max() > 0 else mi_full
    pareto  = pd.concat([cd_norm, mi_norm])
    pareto  = pareto[~pareto.index.duplicated(keep="first")].sort_values(ascending=False)
    cumul   = np.cumsum(pareto.values) / np.cumsum(pareto.values)[-1] * 100
    labels  = [FEAT_LABELS.get(f, f.replace("_"," ").title()) for f in pareto.index]

    fig_par = go.Figure()
    fig_par.add_trace(go.Bar(
        x=labels, y=pareto.values,
        marker_color=[C_DANGER if v >= 50 else C_BLUE for v in pareto.values],
        name="Importance (%)",
    ))
    fig_par.add_trace(go.Scatter(
        x=labels, y=cumul, mode="lines+markers",
        line=dict(color=C_DANGER, width=2),
        name="Cumulative %", yaxis="y2",
    ))
    fig_par.add_hline(y=80, line_dash="dash", line_color=C_TEAL, line_width=1,
                      annotation_text="80%", yref="y2",
                      annotation_font=dict(family="JetBrains Mono", size=8, color=C_TEAL))
    fig_par.update_layout(
        yaxis=dict(title="Normalized Importance (%)"),
        yaxis2=dict(title="Cumulative (%)", overlaying="y", side="right", range=[0, 110]),
        legend=dict(orientation="h", y=-0.25),
        xaxis_tickangle=-35,
    )
    plot(fig_par, 440)
    st.caption("Cohen's d (numeric features) + Mutual Information (all features). Directional influence, not causation.")

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    if pred_prob >= 0.60:
        pl, hz, ac = "HIGH",   "Before next run",     "Hold batch — do not release to paint cabin until root cause is identified"
    elif pred_prob >= 0.30:
        pl, hz, ac = "MEDIUM", "Pre-paint inspection", "Pull sample for visual pre-inspection — check handling and protection coverage"
    else:
        pl, hz, ac = "LOW",    "Standard",             "Standard process — proceed to paint cabin with normal checklist"

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
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;">Est. Defect Prob</div>
                <div style="font-family:var(--fm);font-size:1.3rem;font-weight:600;color:#fff;">{pred_prob:.1%}</div>
            </div>
            <div>
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;">Horizon</div>
                <div style="font-family:var(--fm);font-size:0.85rem;color:var(--text);">{hz}</div>
            </div>
            <div>
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;">Owner</div>
                <div style="font-family:var(--fm);font-size:0.85rem;color:var(--text);">Quality Inspector / Process Engineer</div>
            </div>
        </div>
        <div style="margin-top:12px;font-family:var(--fm);font-size:0.72rem;color:var(--text);">
            <span style="color:var(--muted);">Action → </span>{ac}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="lsa-section">// Active risk factors</div>', unsafe_allow_html=True)
    risk_feats = []
    if pre_paint_storage_hrs > 8:    risk_feats.append("pre_paint_storage_hrs")
    if num_handlings >= 4:           risk_feats.append("num_handlings")
    if part_protection == "unprotected":        risk_feats.append("part_protection_unprotected")
    if container_type  == "metal_rack":         risk_feats.append("container_type_metal_rack")
    if container_type  == "cardboard_pallet":   risk_feats.append("container_type_cardboard_pallet")
    if booth_humidity_pct > 60:      risk_feats.append("booth_humidity_pct")
    if regrind_pct > 25:             risk_feats.append("regrind_pct")

    if risk_feats:
        for feat in risk_feats[:4]:
            if feat in ACTION_MAP:
                label = FEAT_LABELS.get(feat, feat.replace("_", " ").title())
                with st.expander(f"▲  {label}  —  active risk factor"):
                    st.write(ACTION_MAP[feat])
    else:
        st.success("No elevated risk factors detected in the current configuration.")

    st.divider()
    st.caption("This tool supports quality decisions — it does not replace inspection protocols or process engineering judgment.")

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lsa-footer">
    LozanoLsa · Turning Operations into Predictive Systems · Surface Defect Predictor · Project 02 · v2.0
</div>
""", unsafe_allow_html=True)
