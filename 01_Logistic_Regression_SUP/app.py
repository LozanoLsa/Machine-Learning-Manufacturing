"""
app.py — Logistics Delay Prediction Dashboard
LozanoLsa · Project 01 · Logistic Regression · 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LR · Logistics Delay Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── FULL CSS INJECTION ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&family=Instrument+Serif:ital@0;1&display=swap');

/* ╔══ VARIABLES ══╗ */
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

/* ╔══ BASE ══╗ */
.stApp {
    background: var(--bg) !important;
    color: var(--text);
    font-family: var(--fh);
}
.block-container {
    padding: 1.8rem 2.4rem 3rem !important;
    max-width: 1400px !important;
}
#MainMenu, footer, header { visibility: hidden; }

/* ╔══ SIDEBAR ══╗ */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1rem !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    font-family: var(--fm) !important;
    font-size: 0.72rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.05em;
}
[data-testid="stSidebar"] label {
    font-family: var(--fm) !important;
    font-size: 0.7rem !important;
    color: var(--text) !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* ╔══ SLIDERS ══╗ */
[data-testid="stSlider"] [role="slider"] {
    background: var(--blue) !important;
    border: 2px solid var(--blue2) !important;
    box-shadow: 0 0 8px rgba(59,130,246,0.5) !important;
}
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
    font-family: var(--fm) !important;
    font-size: 0.65rem !important;
    color: var(--blue2) !important;
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    padding: 1px 5px !important;
    border-radius: 3px !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: var(--blue) !important;
}

/* ╔══ SELECTBOX ══╗ */
[data-testid="stSelectbox"] > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--fm) !important;
    font-size: 0.78rem !important;
    border-radius: 3px !important;
}

/* ╔══ METRIC CARDS ══╗ */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-top: 2px solid var(--blue) !important;
    padding: 1rem 1.1rem 0.9rem !important;
    border-radius: 3px !important;
}
[data-testid="stMetricLabel"] > div {
    font-family: var(--fm) !important;
    font-size: 0.6rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.18em !important;
    color: var(--muted) !important;
    font-weight: 400 !important;
}
[data-testid="stMetricValue"] > div {
    font-family: var(--fm) !important;
    font-size: 1.7rem !important;
    font-weight: 600 !important;
    color: var(--blue2) !important;
    line-height: 1.1 !important;
}
[data-testid="stMetricDelta"] > div {
    font-family: var(--fm) !important;
    font-size: 0.68rem !important;
}

/* ╔══ TABS ══╗ */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: var(--fm) !important;
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: var(--muted) !important;
    padding: 0.5rem 1.2rem !important;
    border: none !important;
    border-radius: 0 !important;
    background: transparent !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
    color: var(--blue2) !important;
    background: rgba(59,130,246,0.06) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--blue) !important;
    border-bottom: 2px solid var(--blue) !important;
    background: transparent !important;
}
[data-testid="stTabsContent"] {
    padding-top: 1.4rem !important;
}

/* ╔══ ALERTS ══╗ */
[data-testid="stAlert"] {
    border-radius: 2px !important;
    font-family: var(--fm) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
}

/* ╔══ EXPANDER ══╗ */
[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    margin-bottom: 6px !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--fm) !important;
    font-size: 0.72rem !important;
    color: var(--text) !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stExpander"] p {
    font-family: var(--fm) !important;
    font-size: 0.72rem !important;
    color: var(--muted) !important;
    line-height: 1.7 !important;
}

/* ╔══ DATAFRAME ══╗ */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
}
[data-testid="stDataFrame"] th {
    font-family: var(--fm) !important;
    font-size: 0.62rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    background: var(--card2) !important;
    color: var(--muted) !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] td {
    font-family: var(--fm) !important;
    font-size: 0.72rem !important;
    color: var(--text) !important;
    background: var(--card) !important;
}

/* ╔══ DIVIDER / CAPTION ══╗ */
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
[data-testid="stCaptionContainer"] p {
    font-family: var(--fm) !important;
    font-size: 0.62rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.08em !important;
}

/* ╔══ HEADINGS ══╗ */
h1, h2, h3 {
    font-family: var(--fh) !important;
    color: var(--text) !important;
    letter-spacing: -0.01em !important;
}
h2 { font-size: 1.1rem !important; font-weight: 700 !important; }
h3 { font-size: 0.9rem !important; font-weight: 600 !important; }
p, li { font-family: var(--fh) !important; font-size: 0.88rem !important; }

/* ╔══ CUSTOM COMPONENTS ══╗ */
.lsa-header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.2rem;
    margin-bottom: 0.2rem;
}
.lsa-project-tag {
    font-family: var(--fm);
    font-size: 0.6rem;
    color: var(--blue);
    text-transform: uppercase;
    letter-spacing: 0.22em;
    margin-bottom: 4px;
}
.lsa-title {
    font-family: var(--fh);
    font-size: 1.85rem;
    font-weight: 800;
    color: #fff;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.lsa-tagline {
    font-family: var(--fs);
    font-style: italic;
    font-size: 0.9rem;
    color: var(--muted);
    margin-top: 4px;
}
.lsa-chip {
    display: inline-block;
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.3);
    color: var(--blue2);
    font-family: var(--fm);
    font-size: 0.58rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 2px;
    margin-right: 5px;
}
.lsa-section {
    font-family: var(--fm);
    font-size: 0.6rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.2em;
    margin-bottom: 10px;
    padding-bottom: 5px;
    border-bottom: 1px solid var(--border);
}
.lsa-footer {
    margin-top: 2.5rem;
    padding-top: 0.8rem;
    border-top: 1px solid var(--border);
    font-family: var(--fm);
    font-size: 0.58rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATA_PATH    = "logistics_shipments_data.csv"
RANDOM_STATE = 42

FEAT_LABELS = {
    "distance_km":             "Distance (km)",
    "weight_kg":               "Weight (kg)",
    "num_stops":               "Num Stops",
    "priority":                "Priority Flag",
    "loading_time_min":        "Loading Time (min)",
    "dock_wait_time_min":      "Dock Wait (min)",
    "operator_experience_yrs": "Operator Experience (yrs)",
    "route_type_mixed":        "Route: Mixed",
    "route_type_urban":        "Route: Urban",
    "departure_shift_night":   "Departure: Night",
}

METRIC_EXPL = {
    "Accuracy":  "Out of every 100 shipments, the model gets this many right overall.",
    "Precision": "When the model flags a delay, how often it's actually late. High = fewer false alarms.",
    "Recall":    "Out of all truly delayed shipments, how many the model catches. High = fewer missed delays.",
    "F1 Score":  "Balances precision and recall in one number. Closer to 1.0 = better at both.",
    "AUC-ROC":   "How well the model separates delayed vs on-time across all thresholds. 1.0 = perfect.",
}

ACTION_MAP = {
    "route_type_urban":        "Evaluate route re-classification or consolidation to reduce urban segments",
    "num_stops":               "Optimize stop sequencing — reduce unnecessary intermediate stops when possible",
    "loading_time_min":        "Audit loading procedures — target under 60 min for standard loads",
    "dock_wait_time_min":      "Improve dock scheduling and yard management to reduce pre-loading wait time",
    "departure_shift_night":   "Review night-shift coordination protocols and pre-departure checklists",
    "route_type_mixed":        "Consider dedicated routes for mixed-type corridors with high delay history",
    "weight_kg":               "Review heavy-load handling procedures — delays often correlate with overweight cargo",
    "distance_km":             "Consider relay drivers or intermediate hubs for high-distance routes",
    "operator_experience_yrs": "Prioritize experienced operators for high-risk urban routes",
    "priority":                "Leverage priority flagging more broadly — it demonstrably reduces delay risk",
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
            "https://raw.githubusercontent.com/LozanoLsa/Delays_Are_Not_Random/main/logistics_shipments_data.csv"
        )

@st.cache_resource
def train_model(df):
    df_enc = pd.get_dummies(df, drop_first=True)
    X = df_enc.drop("delayed", axis=1)
    y = df_enc["delayed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]
    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1 Score":  f1_score(y_test, y_pred),
        "AUC-ROC":   roc_auc_score(y_test, y_prob),
    }
    feature_names = X.columns.tolist()
    return model, scaler, feature_names, X_train, X_test, y_train, y_test, y_pred, y_prob, metrics

df = load_data()
model, scaler, feature_names, X_train, X_test, y_train, y_test, y_pred, y_prob, metrics = train_model(df)
delay_rate = df["delayed"].mean()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="lsa-project-tag">LozanoLsa · Project 01</div>
    <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;
                color:#fff;margin-bottom:6px;">Logistics Delay<br>Predictor</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                color:#4e6a8a;line-height:1.7;margin-bottom:12px;">
        Logistic Regression trained on 1,847<br>shipments · 9 features
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="lsa-section">// Route & Cargo</div>', unsafe_allow_html=True)
    route_type  = st.selectbox("Route Type", ["urban", "mixed", "long_haul"], index=1)
    distance_km = st.slider("Distance (km)", 10, 1000, 300)
    weight_kg   = st.slider("Weight (kg)", 50, 3000, 800)
    num_stops   = st.slider("Number of Stops", 1, 5, 2)
    priority    = st.selectbox("Priority", [("Normal", 0), ("Urgent", 1)],
                               format_func=lambda x: x[0])[1]

    st.markdown('<div class="lsa-section">// Facility & Timing</div>', unsafe_allow_html=True)
    loading_time    = st.slider("Loading Time (min)", 10, 180, 60)
    dock_wait       = st.slider("Dock Wait (min)", 0, 240, 30)
    departure_shift = st.selectbox("Departure Shift", ["day", "night"])

    st.markdown('<div class="lsa-section">// Operator</div>', unsafe_allow_html=True)
    operator_exp = st.slider("Experience (yrs)", 0, 30, 5)

    st.divider()
    st.caption("C=1.0 · max_iter=1000 · lbfgs · scaled features")
    st.caption("Where f(x) meets Kaizen · 2026")

# ─── PREDICT ──────────────────────────────────────────────────────────────────
def predict_scenario(rt, dist, wt, stops, prio, load, dock, exp, shift):
    row = pd.DataFrame([{
        "distance_km": dist, "weight_kg": wt, "num_stops": stops,
        "priority": prio, "loading_time_min": load, "dock_wait_time_min": dock,
        "operator_experience_yrs": exp, "route_type": rt, "departure_shift": shift
    }])
    enc    = pd.get_dummies(row, drop_first=True).reindex(columns=feature_names, fill_value=0)
    scaled = scaler.transform(enc)
    prob   = model.predict_proba(scaled)[0, 1]
    return prob, int(prob >= 0.5)

pred_prob, pred_class = predict_scenario(
    route_type, distance_km, weight_kg, num_stops, priority,
    loading_time, dock_wait, operator_exp, departure_shift
)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-header">
    <div class="lsa-project-tag">ML Project #01 · Logistic Regression · Logistics</div>
    <div class="lsa-title">Delays Are Not Random</div>
    <div class="lsa-tagline">The probability of a late shipment is written in the data before the truck leaves the dock.</div>
    <div style="margin-top:10px;">
        <span class="lsa-chip">LOGISTIC REGRESSION</span>
        <span class="lsa-chip">9 FEATURES</span>
        <span class="lsa-chip">{metrics['Accuracy']:.1%} ACCURACY</span>
        <span class="lsa-chip">AUC {metrics['AUC-ROC']:.3f}</span>
        <span class="lsa-chip">SCALED FEATURES</span>
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
    k1.metric("Total Shipments", f"{len(df):,}")
    k2.metric("Delayed",         f"{df['delayed'].sum():,}")
    k3.metric("On Time",         f"{(df['delayed']==0).sum():,}")
    k4.metric("Delay Rate",      f"{delay_rate:.1%}")
    st.divider()

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown('<div class="lsa-section">// Class distribution</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["On Time", "Delayed"],
            values=[(df["delayed"]==0).sum(), df["delayed"].sum()],
            marker_colors=[C_BLUE, C_DANGER],
            hole=0.52, textinfo="percent+label",
            textfont=dict(family="JetBrains Mono", size=11),
        ))
        plot(fig_pie, 270)

    with c2:
        st.markdown('<div class="lsa-section">// Delay rate by category</div>', unsafe_allow_html=True)
        cat_col = st.selectbox("Category:", ["route_type", "departure_shift"])
        rates   = df.groupby(cat_col)["delayed"].mean().reset_index()
        rates.columns = [cat_col, "delay_rate"]
        rates   = rates.sort_values("delay_rate", ascending=True)
        fig_cat = go.Figure(go.Bar(
            x=rates["delay_rate"],
            y=rates[cat_col],
            orientation="h",
            marker_color=[C_DANGER if v > delay_rate else C_BLUE for v in rates["delay_rate"]],
            text=[f"{v:.1%}" for v in rates["delay_rate"]],
            textfont=dict(family="JetBrains Mono", size=10, color="#c8d8f0"),
            textposition="outside",
        ))
        fig_cat.update_layout(xaxis_tickformat=".0%")
        plot(fig_cat, 270)

    st.divider()
    st.markdown('<div class="lsa-section">// Feature distributions</div>', unsafe_allow_html=True)
    num_feat = st.selectbox("Numeric feature:",
                            ["distance_km", "weight_kg", "loading_time_min",
                             "dock_wait_time_min", "operator_experience_yrs"],
                            format_func=lambda x: FEAT_LABELS.get(x, x))
    c3, c4 = st.columns(2)
    with c3:
        fig_hist = go.Figure()
        for cls, color, name in [(0, C_BLUE, "On Time"), (1, C_DANGER, "Delayed")]:
            fig_hist.add_trace(go.Histogram(
                x=df[df["delayed"]==cls][num_feat], name=name,
                marker_color=color, opacity=0.65, nbinsx=30,
            ))
        fig_hist.update_layout(barmode="overlay",
                               xaxis_title=FEAT_LABELS.get(num_feat, num_feat),
                               yaxis_title="Count")
        plot(fig_hist, 300)
    with c4:
        fig_box = px.box(
            df, x=df["delayed"].map({0: "On Time", 1: "Delayed"}), y=num_feat,
            color=df["delayed"].map({0: "On Time", 1: "Delayed"}),
            color_discrete_map={"On Time": C_BLUE, "Delayed": C_DANGER},
        )
        fig_box.update_layout(showlegend=False,
                              xaxis_title="",
                              yaxis_title=FEAT_LABELS.get(num_feat, num_feat))
        plot(fig_box, 300)

    st.divider()
    st.markdown('<div class="lsa-section">// Feature correlation matrix</div>', unsafe_allow_html=True)
    df_enc_corr = pd.get_dummies(df, drop_first=True)
    corr        = df_enc_corr.corr().round(2)
    fig_heat    = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            text_auto=True, aspect="auto")
    fig_heat.update_layout(coloraxis_colorbar=dict(
        tickfont=dict(family="JetBrains Mono", size=9)
    ))
    plot(fig_heat, 500)

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
            x=["Pred: On Time", "Pred: Delayed"],
            y=["Actual: On Time", "Actual: Delayed"],
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
            x=y_prob[y_test == 0], name="Actual: On Time",
            marker_color=C_BLUE, opacity=0.65, nbinsx=30,
        ))
        fig_prob.add_trace(go.Histogram(
            x=y_prob[y_test == 1], name="Actual: Delayed",
            marker_color=C_DANGER, opacity=0.65, nbinsx=30,
        ))
        fig_prob.add_vline(x=0.5, line_dash="dash", line_color="#fff", line_width=1,
                           annotation_text="threshold 0.5",
                           annotation_font=dict(family="JetBrains Mono", size=9, color=C_MUTED))
        fig_prob.update_layout(barmode="overlay",
                               xaxis_title="Predicted Delay Probability",
                               yaxis_title="Count")
        plot(fig_prob, 300)

    st.divider()
    st.markdown('<div class="lsa-section">// ROC curve</div>', unsafe_allow_html=True)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines", name=f"LR (AUC = {metrics['AUC-ROC']:.3f})",
        line=dict(color=C_BLUE, width=2),
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random classifier",
        line=dict(color=C_MUTED, width=1, dash="dash"),
    ))
    fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    plot(fig_roc, 360)
    st.caption(f"Logistic Regression · C=1.0 · max_iter=1000 · random_state={RANDOM_STATE}")

    st.divider()
    st.markdown('<div class="lsa-section">// Metric explanations</div>', unsafe_allow_html=True)
    for name, expl in METRIC_EXPL.items():
        with st.expander(f"{name}  —  {metrics[name]:.3f}"):
            st.write(expl)

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    left, right = st.columns([1, 2])
    with left:
        st.markdown('<div class="lsa-section">// Delay probability</div>', unsafe_allow_html=True)
        gauge_color = C_DANGER if pred_class == 1 else C_TEAL
        fg = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_prob * 100,
            number={"suffix": "%", "font": {"size": 38, "family": "JetBrains Mono", "color": "#fff"}},
            title={"text": "P(Delay)", "font": {"size": 11, "family": "JetBrains Mono", "color": C_MUTED}},
            gauge={
                "axis": {"range": [0, 100],
                         "tickfont": {"size": 9, "family": "JetBrains Mono"},
                         "tickcolor": "#1e2d45"},
                "bar": {"color": gauge_color, "thickness": 0.22},
                "bgcolor": "#0e1420", "bordercolor": "#1e2d45",
                "steps": [
                    {"range": [0,  40], "color": "rgba(45,212,191,0.08)"},
                    {"range": [40, 65], "color": "rgba(251,191,36,0.08)"},
                    {"range": [65, 100], "color": "rgba(248,113,113,0.10)"},
                ],
                "threshold": {"line": {"color": "#fff", "width": 2}, "thickness": 0.75, "value": 50},
            }
        ))
        fg.update_layout(height=290, paper_bgcolor="rgba(0,0,0,0)",
                         margin=dict(t=50, b=10, l=20, r=20))
        st.plotly_chart(fg, use_container_width=True)
        if pred_class == 1:
            st.error("DELAY EXPECTED — REVIEW BEFORE DISPATCH")
        else:
            st.success("ON TIME EXPECTED — STANDARD CHECKLIST")
        st.caption(f"Fleet avg: {delay_rate:.1%}  ·  This shipment: {pred_prob:.1%}  ·  Δ {pred_prob - delay_rate:+.1%}")

    with right:
        st.markdown('<div class="lsa-section">// Log-odds contribution by feature</div>', unsafe_allow_html=True)
        row_df = pd.DataFrame([{
            "distance_km": distance_km, "weight_kg": weight_kg, "num_stops": num_stops,
            "priority": priority, "loading_time_min": loading_time,
            "dock_wait_time_min": dock_wait, "operator_experience_yrs": operator_exp,
            "route_type": route_type, "departure_shift": departure_shift
        }])
        row_enc    = pd.get_dummies(row_df, drop_first=True).reindex(columns=feature_names, fill_value=0)
        row_scaled = scaler.transform(row_enc)
        contrib    = model.coef_[0] * row_scaled[0]
        contrib_df = pd.DataFrame({
            "Feature":      feature_names,
            "Contribution": contrib,
        }).sort_values("Contribution", key=abs, ascending=False).head(6)
        contrib_df["Label"] = contrib_df["Feature"].map(
            lambda x: FEAT_LABELS.get(x, x.replace("_", " ").title())
        )
        fc = go.Figure(go.Bar(
            x=contrib_df["Contribution"], y=contrib_df["Label"], orientation="h",
            marker_color=[C_DANGER if v > 0 else C_TEAL for v in contrib_df["Contribution"]],
            text=[f"{v:+.3f}" for v in contrib_df["Contribution"]],
            textfont=dict(family="JetBrains Mono", size=10, color="#c8d8f0"),
            textposition="outside",
        ))
        fc.update_layout(xaxis_title="Log-odds contribution (coef × scaled input)")
        plot(fc, 270)
        if pred_prob >= 0.70:
            st.error("Priority: HIGH · Escalate pre-departure review")
        elif pred_prob >= 0.40:
            st.warning("Priority: MEDIUM · Flag for dispatcher review")
        else:
            st.success("Priority: LOW · Standard departure checklist")

    st.divider()
    st.markdown('<div class="lsa-section">// Scenario comparison</div>', unsafe_allow_html=True)
    bp = predict_scenario("long_haul", 120, 400, 1, 1, 35, 5, 12, "day")[0]
    wp = predict_scenario("urban", 800, 2800, 5, 0, 160, 200, 0, "night")[0]
    cdf = pd.DataFrame([
        {"Scenario": "Best case — low-risk profile",  "P(Delay)": f"{bp:.1%}",
         "Status": "On Time" if bp < 0.5 else "Delayed",
         "Δ vs current": f"{bp - pred_prob:+.1%}"},
        {"Scenario": "Your current configuration",   "P(Delay)": f"{pred_prob:.1%}",
         "Status": "On Time" if pred_class == 0 else "Delayed",
         "Δ vs current": "—"},
        {"Scenario": "Worst case — high-risk profile", "P(Delay)": f"{wp:.1%}",
         "Status": "On Time" if wp < 0.5 else "Delayed",
         "Δ vs current": f"{wp - pred_prob:+.1%}"},
    ])
    st.dataframe(cdf, use_container_width=True, hide_index=True)

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="lsa-section">// Logistic regression coefficients (log-odds)</div>', unsafe_allow_html=True)
    coef_df = pd.DataFrame({
        "Feature":     feature_names,
        "Coefficient": model.coef_[0],
        "Odds Ratio":  np.exp(model.coef_[0]),
    }).sort_values("Coefficient", ascending=True)
    coef_df["Label"] = coef_df["Feature"].map(
        lambda x: FEAT_LABELS.get(x, x.replace("_", " ").title())
    )
    fig_coef = go.Figure(go.Bar(
        x=coef_df["Coefficient"],
        y=coef_df["Label"],
        orientation="h",
        marker_color=[C_TEAL if c < 0 else C_DANGER for c in coef_df["Coefficient"]],
        text=[f"{c:+.3f}" for c in coef_df["Coefficient"]],
        textfont=dict(family="JetBrains Mono", size=9, color="#c8d8f0"),
        textposition="outside",
    ))
    fig_coef.add_vline(x=0, line_color="#fff", line_width=1)
    fig_coef.update_layout(xaxis_title="Coefficient — log-odds scale (scaled features)")
    plot(fig_coef, 400)
    st.caption("Red = increases delay probability · Teal = reduces delay probability · Coefficients on scaled features — magnitude is comparable across features.")

    st.divider()
    c1, c2 = st.columns(2)
    risk_df = coef_df[coef_df["Coefficient"] > 0].sort_values("Coefficient", ascending=False)
    prot_df = coef_df[coef_df["Coefficient"] < 0].sort_values("Coefficient")

    with c1:
        st.markdown('<div class="lsa-section">// Top risk factors (pushing delay up)</div>', unsafe_allow_html=True)
        st.dataframe(
            risk_df[["Label", "Coefficient", "Odds Ratio"]].rename(columns={"Label": "Feature"}).round(3),
            hide_index=True, use_container_width=True
        )
    with c2:
        st.markdown('<div class="lsa-section">// Protective factors (reducing delay risk)</div>', unsafe_allow_html=True)
        st.dataframe(
            prot_df[["Label", "Coefficient", "Odds Ratio"]].rename(columns={"Label": "Feature"}).round(3),
            hide_index=True, use_container_width=True
        )

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    if pred_prob >= 0.70:
        pl, hz, ac = "HIGH",   "24 hours",    "Escalate for pre-departure review — dispatcher + operations manager sign-off"
    elif pred_prob >= 0.40:
        pl, hz, ac = "MEDIUM", "Pre-departure", "Flag for dispatcher review — check dock schedule and operator assignment"
    else:
        pl, hz, ac = "LOW",    "Standard",    "Standard departure checklist — no special intervention required"

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
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;">Est. Delay Prob</div>
                <div style="font-family:var(--fm);font-size:1.3rem;font-weight:600;color:#fff;">{pred_prob:.1%}</div>
            </div>
            <div>
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;">Horizon</div>
                <div style="font-family:var(--fm);font-size:0.85rem;color:var(--text);">{hz}</div>
            </div>
            <div>
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;">Owner</div>
                <div style="font-family:var(--fm);font-size:0.85rem;color:var(--text);">Dispatcher / Operations Manager</div>
            </div>
        </div>
        <div style="margin-top:12px;font-family:var(--fm);font-size:0.72rem;color:var(--text);">
            <span style="color:var(--muted);">Action → </span>{ac}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="lsa-section">// Active risk factors</div>', unsafe_allow_html=True)
    row_df2    = pd.DataFrame([{
        "distance_km": distance_km, "weight_kg": weight_kg, "num_stops": num_stops,
        "priority": priority, "loading_time_min": loading_time,
        "dock_wait_time_min": dock_wait, "operator_experience_yrs": operator_exp,
        "route_type": route_type, "departure_shift": departure_shift
    }])
    row_enc2    = pd.get_dummies(row_df2, drop_first=True).reindex(columns=feature_names, fill_value=0)
    row_scaled2 = scaler.transform(row_enc2)
    contrib_all = model.coef_[0] * row_scaled2[0]

    shown = 0
    for feat, val in sorted(zip(feature_names, contrib_all), key=lambda x: abs(x[1]), reverse=True):
        if val > 0 and feat in ACTION_MAP and shown < 4:
            label = FEAT_LABELS.get(feat, feat.replace("_", " ").title())
            with st.expander(f"▲  {label}  —  increasing delay risk"):
                st.write(ACTION_MAP[feat])
            shown += 1

    if shown == 0:
        st.success("No active risk factors detected — configuration is within low-risk parameters.")

    st.divider()
    st.caption("This tool supports dispatch decisions — it does not replace dispatcher judgment or operational procedures.")

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lsa-footer">
    LozanoLsa · Turning Operations into Predictive Systems · Logistics Delay Predictor · Project 01 · v3    .0
</div>
""", unsafe_allow_html=True)
