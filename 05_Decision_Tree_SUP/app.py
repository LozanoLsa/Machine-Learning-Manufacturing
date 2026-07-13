"""
app.py — Stamping Press Scrap Risk Dashboard
LozanoLsa · Project 05 · Decision Tree · 2026

Model: Decision Tree (multiclass — Low / Medium / High scrap risk)
Domain: Process Quality — Stamping Press Production
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DT · Scrap Risk Predictor",
    page_icon="🏭",
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
.lsa-rule { background: var(--card); border: 1px solid var(--border); border-left: 3px solid var(--blue); padding: 0.7rem 1rem; border-radius: 2px; margin-bottom: 8px; font-family: var(--fm); font-size: 0.72rem; color: var(--text); line-height: 1.6; }
.lsa-footer { margin-top: 2.5rem; padding-top: 0.8rem; border-top: 1px solid var(--border); font-family: var(--fm); font-size: 0.58rem; color: var(--muted); letter-spacing: 0.1em; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATA_PATH    = "scrap_risk_data.csv"
RANDOM_STATE = 42
NUM_COLS = ["press_speed_spm", "raw_material_hardness_hrb", "operator_experience_yrs",
            "ambient_temp_c", "critical_supplier_lot", "recent_model_change",
            "setup_checklist_complete"]
CAT_COLS    = ["shift"]
TARGET      = "scrap_risk"
RISK_ORDER  = ["Low", "Medium", "High"]
RISK_COLORS = {"Low": "#3b82f6", "Medium": "#fbbf24", "High": "#f87171"}

FEAT_LABELS = {
    "press_speed_spm":           "Press Speed (spm)",
    "raw_material_hardness_hrb": "Material Hardness (HRB)",
    "operator_experience_yrs":   "Operator Experience (yrs)",
    "ambient_temp_c":            "Ambient Temp (°C)",
    "critical_supplier_lot":     "Critical Supplier Lot",
    "recent_model_change":       "Recent Model Change",
    "setup_checklist_complete":  "Setup Checklist Complete",
    "shift_Night":               "Shift: Night",
    "shift_Early_Morning":       "Shift: Early Morning",
}

METRIC_EXPL = {
    "Accuracy":    "Out of every 100 production runs, the model classifies this many correctly.",
    "F1 Macro":    "Average F1 across all three risk classes — treats Low, Medium, and High equally.",
    "F1 Weighted": "F1 averaged by class frequency — closer to overall accuracy.",
    "Train Acc":   "Training set accuracy — compare with test to assess overfitting.",
}

ACTION_MAP = {
    "operator_experience_yrs":   "Assign a senior operator or pair with a mentor before running critical lots",
    "setup_checklist_complete":  "Enforce complete pre-run checklist — gate production release on checklist sign-off",
    "critical_supplier_lot":     "Increase incoming inspection frequency — consider sample testing before full run",
    "press_speed_spm":           "Reduce press speed to below 45 spm until conditions improve",
    "recent_model_change":       "Require post-changeover trial pieces before full production — validate setup output",
    "raw_material_hardness_hrb": "Verify material certificate — hardness outside 72–88 HRB warrants additional checks",
    "ambient_temp_c":            "Check shop floor temperature — extreme conditions affect die and material behavior",
    "shift":                     "Schedule critical runs during Day shift when supervisory coverage is highest",
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
        return pd.read_csv("https://raw.githubusercontent.com/LozanoLsa/Process_Decisions_Optimization/main/scrap_risk_data.csv")

@st.cache_resource
def train_model(df):
    X, y    = df.drop(TARGET, axis=1), df[TARGET]
    le_y    = LabelEncoder()
    y_enc   = le_y.fit_transform(y)
    prep    = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", sparse_output=False), CAT_COLS),
        ("num", "passthrough", NUM_COLS)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.3, random_state=RANDOM_STATE, stratify=y_enc
    )
    pipe = Pipeline([
        ("preprocessor", prep),
        ("model", DecisionTreeClassifier(criterion="gini", max_depth=5,
                                         min_samples_leaf=50, random_state=RANDOM_STATE))
    ])
    pipe.fit(X_train, y_train)
    y_pred_train = pipe.predict(X_train)
    y_pred       = pipe.predict(X_test)
    metrics = {
        "Accuracy":    accuracy_score(y_test, y_pred),
        "F1 Macro":    f1_score(y_test, y_pred, average="macro"),
        "F1 Weighted": f1_score(y_test, y_pred, average="weighted"),
        "Train Acc":   accuracy_score(y_train, y_pred_train),
    }
    ohe       = pipe.named_steps["preprocessor"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CAT_COLS))
    all_names = cat_names + NUM_COLS
    imp_df    = pd.DataFrame({
        "Feature":    all_names,
        "Importance": pipe.named_steps["model"].feature_importances_,
    }).sort_values("Importance", ascending=False)
    return pipe, le_y, X_train, X_test, y_train, y_test, y_pred, metrics, imp_df, all_names

df = load_data()
pipe_dt, le_y, X_train, X_test, y_train, y_test, y_pred, metrics, imp_df, all_names = train_model(df)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="lsa-project-tag">LozanoLsa · Project 05</div>
    <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;
                color:#fff;margin-bottom:6px;">Scrap Risk<br>Predictor</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                color:#4e6a8a;line-height:1.7;margin-bottom:12px;">
        Decision Tree on 2,317 stamping press runs<br>Low / Medium / High scrap risk
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="lsa-section">// Process Parameters</div>', unsafe_allow_html=True)
    press_speed  = st.slider("Press Speed (spm)", 20, 60, 35)
    hardness     = st.slider("Material Hardness (HRB)", 60, 100, 80)
    ambient_temp = st.slider("Ambient Temperature (°C)", 15, 35, 24)

    st.markdown('<div class="lsa-section">// Operator</div>', unsafe_allow_html=True)
    exp_yrs = st.slider("Operator Experience (years)", 0.0, 20.0, 5.0, 0.5)
    shift   = st.selectbox("Shift", ["Day", "Night", "Early_Morning"])

    st.markdown('<div class="lsa-section">// Setup & Supply</div>', unsafe_allow_html=True)
    checklist     = st.selectbox("Setup Checklist", [(1, "Complete"), (0, "Incomplete")],
                                  format_func=lambda x: x[1])[0]
    model_change  = st.selectbox("Recent Model Change", [(0, "No"), (1, "Yes")],
                                  format_func=lambda x: x[1])[0]
    crit_supplier = st.selectbox("Supplier Lot", [(0, "Standard"), (1, "Critical")],
                                  format_func=lambda x: x[1])[0]

    st.divider()
    st.caption("gini · max_depth=5 · min_samples_leaf=50 · 70/30 stratified")
    st.caption("Where f(x) meets Kaizen · 2026")

# ─── PREDICT ──────────────────────────────────────────────────────────────────
def predict_run(speed, hard, exp, temp, chk, mc, sup, sh):
    row  = pd.DataFrame([{
        "press_speed_spm": speed, "raw_material_hardness_hrb": hard,
        "operator_experience_yrs": exp, "ambient_temp_c": temp,
        "critical_supplier_lot": sup, "shift": sh,
        "recent_model_change": mc, "setup_checklist_complete": chk
    }])
    enc  = pipe_dt.predict(row)[0]
    prob = pipe_dt.predict_proba(row)[0]
    cls  = le_y.inverse_transform([enc])[0]
    return cls, dict(zip(le_y.classes_, prob))

pred_class, pred_probs = predict_run(
    press_speed, hardness, exp_yrs, ambient_temp,
    checklist, model_change, crit_supplier, shift
)
high_prob = pred_probs.get("High", 0)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-header">
    <div class="lsa-project-tag">ML Project #05 · Decision Tree · Stamping Press Quality</div>
    <div class="lsa-title">Scrap Is a Decision, Not an Accident</div>
    <div class="lsa-tagline">Every High-risk run was predictable. The checklist, the operator, the lot — the tree already knew.</div>
    <div style="margin-top:10px;">
        <span class="lsa-chip">DECISION TREE</span>
        <span class="lsa-chip">3-CLASS OUTPUT</span>
        <span class="lsa-chip">{metrics['Accuracy']:.1%} ACCURACY</span>
        <span class="lsa-chip">F1 MACRO {metrics['F1 Macro']:.3f}</span>
        <span class="lsa-chip">MAX DEPTH 5</span>
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
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Runs",     f"{len(df):,}")
    k2.metric("High Risk",      f"{(df[TARGET]=='High').sum():,}")
    k3.metric("Medium Risk",    f"{(df[TARGET]=='Medium').sum():,}")
    k4.metric("Low Risk",       f"{(df[TARGET]=='Low').sum():,}")
    k5.metric("High Risk Rate", f"{(df[TARGET]=='High').mean():.1%}")
    st.divider()

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown('<div class="lsa-section">// Risk class distribution</div>', unsafe_allow_html=True)
        counts  = df[TARGET].value_counts().reindex(RISK_ORDER)
        fig_pie = go.Figure(go.Pie(
            labels=RISK_ORDER, values=counts.values,
            marker_colors=[RISK_COLORS[r] for r in RISK_ORDER],
            hole=0.52, textinfo="percent+label",
            textfont=dict(family="JetBrains Mono", size=11),
        ))
        plot(fig_pie, 270)

    with c2:
        st.markdown('<div class="lsa-section">// Risk distribution by category</div>', unsafe_allow_html=True)
        cat_sel = st.selectbox("Category:", ["shift", "critical_supplier_lot",
                                              "setup_checklist_complete", "recent_model_change"])
        grp = df.groupby(cat_sel)[TARGET].value_counts(normalize=True).unstack().reindex(
            columns=RISK_ORDER, fill_value=0).reset_index()
        grp_melt = grp.melt(id_vars=cat_sel, var_name="Risk", value_name="Rate")
        fig_grp  = go.Figure()
        for risk in RISK_ORDER:
            subset = grp_melt[grp_melt["Risk"] == risk]
            fig_grp.add_trace(go.Bar(
                name=risk, x=subset[cat_sel].astype(str), y=subset["Rate"],
                marker_color=RISK_COLORS[risk],
                text=[f"{v:.1%}" for v in subset["Rate"]],
                textfont=dict(family="JetBrains Mono", size=9),
                textposition="outside",
            ))
        fig_grp.update_layout(barmode="group", yaxis_tickformat=".0%")
        plot(fig_grp, 270)

    st.divider()
    st.markdown('<div class="lsa-section">// Feature distributions</div>', unsafe_allow_html=True)
    num_sel = st.selectbox("Numeric feature:",
                           ["press_speed_spm", "raw_material_hardness_hrb",
                            "operator_experience_yrs", "ambient_temp_c"],
                           format_func=lambda x: FEAT_LABELS.get(x, x))
    c3, c4 = st.columns(2)
    with c3:
        fig_hist = go.Figure()
        for risk in RISK_ORDER:
            fig_hist.add_trace(go.Histogram(
                x=df[df[TARGET]==risk][num_sel], name=risk,
                marker_color=RISK_COLORS[risk], opacity=0.65, nbinsx=25,
            ))
        fig_hist.update_layout(barmode="overlay",
                               xaxis_title=FEAT_LABELS.get(num_sel, num_sel),
                               yaxis_title="Count")
        plot(fig_hist, 300)

    with c4:
        st.markdown('<div class="lsa-section">// Press speed vs operator experience</div>', unsafe_allow_html=True)
        fig_sc = px.scatter(
            df.sample(500, random_state=42),
            x="press_speed_spm", y="operator_experience_yrs",
            color=TARGET, color_discrete_map=RISK_COLORS,
            category_orders={TARGET: RISK_ORDER},
            opacity=0.5,
        )
        fig_sc.update_traces(marker=dict(size=5))
        fig_sc.update_layout(
            xaxis_title=FEAT_LABELS["press_speed_spm"],
            yaxis_title=FEAT_LABELS["operator_experience_yrs"],
        )
        plot(fig_sc, 300)

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="lsa-section">// Model metrics</div>', unsafe_allow_html=True)
    m_cols = st.columns(4)
    for i, (name, val) in enumerate(metrics.items()):
        m_cols[i].metric(name, f"{val:.3f}")
    st.divider()

    cm_arr = confusion_matrix(y_test, y_pred)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="lsa-section">// Confusion matrix (3 classes)</div>', unsafe_allow_html=True)
        fig_cm = go.Figure(go.Heatmap(
            z=cm_arr,
            x=[f"Pred: {c}" for c in le_y.classes_],
            y=[f"True: {c}"  for c in le_y.classes_],
            colorscale=[[0, "#0a1525"], [1, C_BLUE]],
            text=cm_arr, texttemplate="%{text}",
            textfont=dict(family="JetBrains Mono", size=16, color="#fff"),
            showscale=False,
        ))
        plot(fig_cm, 360)
        st.caption("Rows = actual class · Columns = predicted class · Diagonal = correct predictions")

    with c2:
        st.markdown('<div class="lsa-section">// Predicted probability — true class vs others</div>', unsafe_allow_html=True)
        y_prob_all = pipe_dt.predict_proba(X_test)
        fig_prob   = go.Figure()
        for i, cls in enumerate(le_y.classes_):
            fig_prob.add_trace(go.Histogram(
                x=y_prob_all[y_test == i, i], name=f"True: {cls}",
                marker_color=RISK_COLORS.get(cls, C_MUTED), opacity=0.65, nbinsx=20,
            ))
        fig_prob.update_layout(barmode="overlay",
                               xaxis_title="Predicted Class Probability",
                               yaxis_title="Count")
        plot(fig_prob, 360)

    st.divider()
    st.markdown('<div class="lsa-section">// Classification report</div>', unsafe_allow_html=True)
    rep_df = pd.DataFrame(
        classification_report(y_test, y_pred, target_names=le_y.classes_, output_dict=True)
    ).T.round(3)
    st.dataframe(rep_df.style.background_gradient(cmap="Blues",
                                                   subset=["precision", "recall", "f1-score"]),
                 use_container_width=True)
    st.caption(f"Decision Tree · gini · max_depth=5 · min_samples_leaf=50 · 70/30 stratified · random_state={RANDOM_STATE}")

    st.divider()
    st.markdown('<div class="lsa-section">// Metric explanations</div>', unsafe_allow_html=True)
    for name, expl in METRIC_EXPL.items():
        with st.expander(f"{name}  —  {metrics[name]:.3f}"):
            st.write(expl)

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    left, right = st.columns([1, 2])
    with left:
        st.markdown('<div class="lsa-section">// P(High scrap risk)</div>', unsafe_allow_html=True)
        pred_color  = RISK_COLORS.get(pred_class, C_MUTED)
        gauge_color = C_DANGER if pred_class == "High" else (C_WARN if pred_class == "Medium" else C_TEAL)
        fg = go.Figure(go.Indicator(
            mode="gauge+number",
            value=high_prob * 100,
            number={"suffix": "%", "font": {"size": 38, "family": "JetBrains Mono", "color": "#fff"}},
            title={"text": "P(High Scrap Risk)", "font": {"size": 11, "family": "JetBrains Mono", "color": C_MUTED}},
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
        if pred_class == "High":
            st.error("HIGH SCRAP RISK — HOLD RUN")
        elif pred_class == "Medium":
            st.warning("MEDIUM RISK — REVIEW CONDITIONS")
        else:
            st.success("LOW SCRAP RISK — PROCEED")
        overall_high = (df[TARGET] == "High").mean()
        st.caption(f"Fleet avg P(High): {overall_high:.1%}  ·  This run: {high_prob:.1%}  ·  Δ {high_prob - overall_high:+.1%}")

    with right:
        st.markdown('<div class="lsa-section">// Risk class probabilities for this run</div>', unsafe_allow_html=True)
        fig_probs = go.Figure(go.Bar(
            x=list(pred_probs.keys()),
            y=list(pred_probs.values()),
            marker_color=[RISK_COLORS.get(k, C_MUTED) for k in pred_probs],
            text=[f"{v:.1%}" for v in pred_probs.values()],
            textfont=dict(family="JetBrains Mono", size=11, color="#c8d8f0"),
            textposition="outside",
        ))
        fig_probs.update_layout(yaxis_title="Probability", yaxis_range=[0, 1.15],
                                yaxis_tickformat=".0%")
        plot(fig_probs, 280)
        if high_prob >= 0.60:
            st.error("Priority: HIGH · Hold this run — review setup and operator assignment")
        elif high_prob >= 0.30:
            st.warning("Priority: MEDIUM · Proceed with caution — check checklist and supplier lot")
        else:
            st.success("Priority: LOW · Standard process — approve run")

    st.divider()
    st.markdown('<div class="lsa-section">// Scenario comparison</div>', unsafe_allow_html=True)
    best_cls,  best_p  = predict_run(30, 80, 10.0, 24, 1, 0, 0, "Day")
    worst_cls, worst_p = predict_run(55, 92, 0.5,  32, 0, 1, 1, "Night")
    cdf = pd.DataFrame([
        {"Scenario": "Best case — standard conditions",       "Predicted Risk": best_cls,
         "P(High)": f"{best_p.get('High',0):.1%}",
         "Δ P(High) vs current": f"{best_p.get('High',0) - high_prob:+.1%}"},
        {"Scenario": "Current run configuration",             "Predicted Risk": pred_class,
         "P(High)": f"{high_prob:.1%}",
         "Δ P(High) vs current": "—"},
        {"Scenario": "Worst case — compounded risk factors",  "Predicted Risk": worst_cls,
         "P(High)": f"{worst_p.get('High',0):.1%}",
         "Δ P(High) vs current": f"{worst_p.get('High',0) - high_prob:+.1%}"},
    ])
    st.dataframe(cdf, use_container_width=True, hide_index=True)

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="lsa-section">// Gini feature importance</div>', unsafe_allow_html=True)
        imp_sorted = imp_df.sort_values("Importance", ascending=True)
        fig_imp = go.Figure(go.Bar(
            x=imp_sorted["Importance"],
            y=[FEAT_LABELS.get(f, f.replace("_", " ").title()) for f in imp_sorted["Feature"]],
            orientation="h",
            marker_color=[C_DANGER if v > 0.1 else C_BLUE for v in imp_sorted["Importance"]],
            text=[f"{v:.3f}" for v in imp_sorted["Importance"]],
            textfont=dict(family="JetBrains Mono", size=9, color="#c8d8f0"),
            textposition="outside",
        ))
        fig_imp.add_vline(x=0.1, line_dash="dash", line_color=C_WARN, line_width=1,
                          annotation_text="10% threshold",
                          annotation_font=dict(family="JetBrains Mono", size=8, color=C_WARN))
        fig_imp.update_layout(xaxis_title="Gini Importance (reduction in impurity)")
        plot(fig_imp, 400)
        st.caption("How much each variable reduces impurity across all tree splits. Higher = more predictive.")

    with c2:
        st.markdown('<div class="lsa-section">// Top decision rules from the tree</div>', unsafe_allow_html=True)
        rules = [
            ("IF setup_checklist_complete = 0 AND critical_supplier_lot = 1",  "High Risk"),
            ("IF operator_experience_yrs ≤ 1.0 AND press_speed_spm > 50",      "High Risk"),
            ("IF setup_checklist_complete = 1 AND operator_experience_yrs > 5", "Low Risk"),
            ("IF recent_model_change = 1 AND shift ≠ Day AND press_speed_spm > 45", "High Risk"),
            ("IF setup_checklist_complete = 1 AND critical_supplier_lot = 0 AND experience > 3", "Low / Medium"),
        ]
        risk_border = {"High Risk": C_DANGER, "Low Risk": C_TEAL, "Low / Medium": C_BLUE}
        for i, (condition, outcome) in enumerate(rules, 1):
            border = risk_border.get(outcome, C_MUTED)
            st.markdown(f"""
            <div style="background:var(--card);border:1px solid var(--border);
                        border-left:3px solid {border};padding:0.7rem 1rem;
                        border-radius:2px;margin-bottom:8px;">
                <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                            letter-spacing:.15em;margin-bottom:4px;">// RULE {i}</div>
                <div style="font-family:var(--fm);font-size:0.72rem;color:var(--text);
                            line-height:1.6;">{condition}</div>
                <div style="font-family:var(--fm);font-size:0.7rem;font-weight:600;
                            color:{border};margin-top:4px;">→ {outcome}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="lsa-section">// Full tree rules (first 60 lines)</div>', unsafe_allow_html=True)
    with st.expander("View operational rules extracted from the tree"):
        ohe_exp   = pipe_dt.named_steps["preprocessor"].named_transformers_["cat"]
        cat_names = list(ohe_exp.get_feature_names_out(CAT_COLS))
        feat_names = cat_names + NUM_COLS
        rules_txt  = export_text(pipe_dt.named_steps["model"],
                                  feature_names=feat_names, decimals=2)
        lines = rules_txt.split("\n")
        st.code("\n".join(lines[:60]))

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    if pred_class == "High":
        pl, hz, ac = "HIGH",   "Before run start", "HOLD production release — review setup, operator, and supplier lot before proceeding"
    elif pred_class == "Medium":
        pl, hz, ac = "MEDIUM", "Pre-run review",   "Verify checklist completion and operator readiness — consider reducing press speed"
    else:
        pl, hz, ac = "LOW",    "Standard",          "Approve run — standard monitoring applies"

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
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;">Predicted Risk · P(High)</div>
                <div style="font-family:var(--fm);font-size:1.1rem;font-weight:600;color:#fff;">{pred_class} · {high_prob:.1%}</div>
            </div>
            <div>
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;">Horizon</div>
                <div style="font-family:var(--fm);font-size:0.85rem;color:var(--text);">{hz}</div>
            </div>
            <div>
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;">Owner</div>
                <div style="font-family:var(--fm);font-size:0.85rem;color:var(--text);">Process Engineer / Shift Supervisor</div>
            </div>
        </div>
        <div style="margin-top:12px;font-family:var(--fm);font-size:0.72rem;color:var(--text);">
            <span style="color:var(--muted);">Action → </span>{ac}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="lsa-section">// Active risk factors</div>', unsafe_allow_html=True)
    risk_feats = []
    if exp_yrs < 2:                                   risk_feats.append("operator_experience_yrs")
    if checklist == 0:                                risk_feats.append("setup_checklist_complete")
    if crit_supplier == 1:                            risk_feats.append("critical_supplier_lot")
    if press_speed > 50:                              risk_feats.append("press_speed_spm")
    if model_change == 1:                             risk_feats.append("recent_model_change")
    if hardness > 90 or hardness < 70:               risk_feats.append("raw_material_hardness_hrb")
    if shift != "Day":                                risk_feats.append("shift")
    if ambient_temp > 30 or ambient_temp < 18:        risk_feats.append("ambient_temp_c")

    if risk_feats:
        for feat in risk_feats[:5]:
            if feat in ACTION_MAP:
                label = FEAT_LABELS.get(feat, feat.replace("_", " ").title())
                with st.expander(f"▲  {label}  —  active risk factor"):
                    st.write(ACTION_MAP[feat])
    else:
        st.success("No elevated risk factors detected in the current configuration — proceed with standard monitoring.")

    st.divider()
    st.caption("This tool supports production decisions — it does not replace process engineering judgment or quality protocols. A High prediction is a flag for review, not an automatic line stop.")

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lsa-footer">
    LozanoLsa · Turning Operations into Predictive Systems · Scrap Risk Predictor · Project 05 · v2.0
</div>
""", unsafe_allow_html=True)
