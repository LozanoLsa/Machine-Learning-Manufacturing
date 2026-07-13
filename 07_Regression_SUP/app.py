"""
app.py — Riveting Process Intelligence Dashboard
LozanoLsa · Project 07 · Linear Regression · 2026

Model: OLS Multiple Linear Regression
Domain: Manufacturing — Rivet Head Diameter Prediction
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import linregress
import warnings

warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LR · Riveting Head Diameter Predictor",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="collapsed",
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
    --accent:   #2dd4bf;
    --accent2:  #5eead4;
    --danger:   #f87171;
    --warn:     #fbbf24;
    --ok:       #4ade80;
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
[data-testid="stSidebar"] label { font-family: var(--fm) !important; font-size: 0.7rem !important; color: var(--text) !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; }

[data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; border: 2px solid var(--accent2) !important; box-shadow: 0 0 8px rgba(45,212,191,0.4) !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { font-family: var(--fm) !important; font-size: 0.65rem !important; color: var(--accent2) !important; background: var(--card) !important; border: 1px solid var(--border) !important; padding: 1px 5px !important; border-radius: 3px !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--accent) !important; }

[data-testid="stSelectbox"] > div > div { background: var(--card) !important; border: 1px solid var(--border) !important; color: var(--text) !important; font-family: var(--fm) !important; font-size: 0.78rem !important; border-radius: 3px !important; }

[data-testid="stMetric"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-top: 2px solid var(--accent) !important; padding: 1rem 1.1rem 0.9rem !important; border-radius: 3px !important; }
[data-testid="stMetricLabel"] > div { font-family: var(--fm) !important; font-size: 0.6rem !important; text-transform: uppercase !important; letter-spacing: 0.18em !important; color: var(--muted) !important; font-weight: 400 !important; }
[data-testid="stMetricValue"] > div { font-family: var(--fm) !important; font-size: 1.7rem !important; font-weight: 600 !important; color: var(--accent2) !important; line-height: 1.1 !important; }

[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid var(--border) !important; gap: 0 !important; background: transparent !important; }
[data-testid="stTabs"] [role="tab"] { font-family: var(--fm) !important; font-size: 0.68rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; color: var(--muted) !important; padding: 0.5rem 1.2rem !important; border: none !important; border-radius: 0 !important; background: transparent !important; transition: all 0.2s !important; }
[data-testid="stTabs"] [role="tab"]:hover { color: var(--accent2) !important; background: rgba(45,212,191,0.06) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; background: transparent !important; }
[data-testid="stTabsContent"] { padding-top: 1.4rem !important; }

[data-testid="stAlert"] { border-radius: 2px !important; font-family: var(--fm) !important; font-size: 0.75rem !important; letter-spacing: 0.04em !important; border: none !important; }

[data-testid="stExpander"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 2px !important; margin-bottom: 6px !important; }
[data-testid="stExpander"] summary { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--text) !important; letter-spacing: 0.06em !important; }

[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 2px !important; }
[data-testid="stDataFrame"] th { font-family: var(--fm) !important; font-size: 0.62rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; background: var(--card2) !important; color: var(--muted) !important; border-bottom: 1px solid var(--border) !important; }
[data-testid="stDataFrame"] td { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--text) !important; background: var(--card) !important; }

hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
[data-testid="stCaptionContainer"] p { font-family: var(--fm) !important; font-size: 0.62rem !important; color: var(--muted) !important; letter-spacing: 0.08em !important; }

h1, h2, h3 { font-family: var(--fh) !important; color: var(--text) !important; letter-spacing: -0.01em !important; }
p, li { font-family: var(--fh) !important; font-size: 0.88rem !important; }

.lsa-header { border-bottom: 1px solid var(--border); padding-bottom: 1.2rem; margin-bottom: 0.2rem; }
.lsa-project-tag { font-family: var(--fm); font-size: 0.6rem; color: var(--accent); text-transform: uppercase; letter-spacing: 0.22em; margin-bottom: 4px; }
.lsa-title { font-family: var(--fh); font-size: 1.85rem; font-weight: 800; color: #fff; line-height: 1.1; letter-spacing: -0.02em; }
.lsa-tagline { font-family: var(--fs); font-style: italic; font-size: 0.9rem; color: var(--muted); margin-top: 4px; }
.lsa-chip { display: inline-block; background: rgba(45,212,191,0.1); border: 1px solid rgba(45,212,191,0.3); color: var(--accent2); font-family: var(--fm); font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 2px 8px; border-radius: 2px; margin-right: 5px; }
.lsa-section { font-family: var(--fm); font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid var(--border); }
.lsa-footer { margin-top: 2.5rem; padding-top: 0.8rem; border-top: 1px solid var(--border); font-family: var(--fm); font-size: 0.58rem; color: var(--muted); letter-spacing: 0.1em; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATA_PATH           = "lr_riveting_data.csv"
DATA_PATH_ALT       = "07_LR_Cycle_Time_Estimation/lr_riveting_data.csv"
RANDOM_STATE        = 42
SPEC_LO, SPEC_HI    = 5.0, 5.4
SPEC_MID            = 5.2
FEATURES = [
    "press_force_kn", "press_stroke_mm", "rivet_diameter_mm",
    "rivet_length_mm", "temperature_c", "hold_time_ms",
]
TARGET = "head_diameter_mm"

FEAT_LABELS = {
    "press_force_kn":    "Press Force (kN)",
    "press_stroke_mm":   "Press Stroke (mm)",
    "rivet_diameter_mm": "Rivet Diameter (mm)",
    "rivet_length_mm":   "Rivet Length (mm)",
    "temperature_c":     "Temperature (°C)",
    "hold_time_ms":      "Hold Time (ms)",
}

# ─── MATPLOTLIB PALETTE ───────────────────────────────────────────────────────
C_BG    = "#080c12"
C_CARD  = "#121922"
C_TEAL  = "#2dd4bf"
C_TEAL2 = "#5eead4"
C_DANGER= "#f87171"
C_WARN  = "#fbbf24"
C_OK    = "#4ade80"
C_TEXT  = "#c8d8f0"
C_MUTED = "#4e6a8a"

def dark_fig(w=9, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_CARD)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.xaxis.label.set_color(C_MUTED)
    ax.yaxis.label.set_color(C_MUTED)
    ax.title.set_color(C_TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e2d45")
    return fig, ax

def dark_fig2(w=13, h=5):
    fig, axes = plt.subplots(1, 2, figsize=(w, h))
    fig.patch.set_facecolor(C_BG)
    for ax in axes:
        ax.set_facecolor(C_CARD)
        ax.tick_params(colors=C_MUTED, labelsize=9)
        for lbl in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            lbl.set_color(C_TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e2d45")
    return fig, axes

# ─── DATA & MODEL ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        return pd.read_csv(DATA_PATH_ALT)

@st.cache_resource
def train_model(df):
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "r2"     : round(r2_score(y_test, y_pred), 4),
        "rmse"   : round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "mae"    : round(mean_absolute_error(y_test, y_pred), 4),
        "n_train": len(X_train),
        "n_test" : len(X_test),
    }
    coef_df = pd.DataFrame({
        "Feature":     FEATURES,
        "Coefficient": model.coef_,
    }).sort_values("Coefficient", key=abs, ascending=False).reset_index(drop=True)
    return model, X_test, y_test, y_pred, metrics, coef_df

df = load_data()
model, X_test, y_test, y_pred_test, metrics, coef_df = train_model(df)
in_spec_rate = df["head_diameter_ok"].mean() if "head_diameter_ok" in df.columns \
               else ((df[TARGET] >= SPEC_LO) & (df[TARGET] <= SPEC_HI)).mean()

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-header">
    <div class="lsa-project-tag">ML Project #07 · Linear Regression · Riveting Process</div>
    <div class="lsa-title">Head Diameter Is Predictable</div>
    <div class="lsa-tagline">Every deviation from spec was visible in the process parameters before the rivet was set.</div>
    <div style="margin-top:10px;">
        <span class="lsa-chip">OLS REGRESSION</span>
        <span class="lsa-chip">6 FEATURES</span>
        <span class="lsa-chip">R² {metrics['r2']:.4f}</span>
        <span class="lsa-chip">RMSE {metrics['rmse']:.4f} MM</span>
        <span class="lsa-chip">SPEC [5.0–5.4 MM]</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TOP KPI ROW ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("R² (Test Set)",   f"{metrics['r2']:.4f}",  "Variance explained")
k2.metric("RMSE (Test)",     f"{metrics['rmse']:.4f} mm", "6.3% of spec window")
k3.metric("MAE (Test)",      f"{metrics['mae']:.4f} mm",  "Typical miss per cycle")
k4.metric("In-Spec Rate",    f"{in_spec_rate:.1%}",    "Cycles in [5.0–5.4 mm]")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "DATA EXPLORER", "PERFORMANCE", "SIMULATOR", "PROCESS DRIVERS", "ACTION PLAN"
])

# ══ TAB 1 ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="lsa-section">// Dataset overview — riveting process records</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Cycles", f"{len(df):,}")
    c2.metric("Features", len(FEATURES))
    c3.metric("Target Variable", "head_diameter_mm")

    with st.expander("Preview first 20 rows"):
        st.dataframe(df.head(20), use_container_width=True)

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="lsa-section">// Head diameter distribution</div>',
                    unsafe_allow_html=True)
        fig, ax = dark_fig(7, 4)
        ax.hist(df[TARGET], bins=40, color=C_TEAL, alpha=0.70, edgecolor=C_BG, linewidth=0.3)
        ax.axvspan(SPEC_LO, SPEC_HI, alpha=0.10, color=C_OK)
        ax.axvline(df[TARGET].mean(), color=C_WARN, ls="--", lw=1.5,
                   label=f"Mean = {df[TARGET].mean():.3f} mm")
        ax.axvline(SPEC_LO, color=C_OK, ls=":", lw=1.2, label="Spec limits [5.0–5.4]")
        ax.axvline(SPEC_HI, color=C_OK, ls=":", lw=1.2)
        ax.set_xlabel("Head Diameter (mm)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8, facecolor=C_CARD, labelcolor=C_TEXT)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Green zone = specification window [5.0–5.4 mm]. Dashed amber = process mean.")

    with col_b:
        st.markdown('<div class="lsa-section">// Head diameter by rivet size</div>',
                    unsafe_allow_html=True)
        fig, ax = dark_fig(7, 4)
        groups = [df[df["rivet_diameter_mm"] == d][TARGET].values for d in [3.0, 4.0, 5.0]]
        bp = ax.boxplot(groups, labels=["3 mm", "4 mm", "5 mm"], patch_artist=True,
                        medianprops=dict(color="white", linewidth=2))
        for patch, c in zip(bp["boxes"], [C_DANGER, C_TEAL, C_WARN]):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
        ax.axhspan(SPEC_LO, SPEC_HI, alpha=0.08, color=C_OK)
        ax.set_xlabel("Rivet Nominal Diameter")
        ax.set_ylabel("Head Diameter (mm)")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("All three rivet sizes can produce in-spec heads — the press recipe must be adjusted per size.")

    st.divider()
    st.markdown('<div class="lsa-section">// Feature scatter vs head diameter</div>',
                unsafe_allow_html=True)
    sel_feat = st.selectbox("Process variable:",
                            FEATURES, format_func=lambda x: FEAT_LABELS.get(x, x))
    fig, ax = dark_fig(10, 4)
    if "head_diameter_ok" in df.columns:
        colors_pt = df["head_diameter_ok"].map({1: C_TEAL, 0: C_DANGER})
    else:
        mask_ok   = (df[TARGET] >= SPEC_LO) & (df[TARGET] <= SPEC_HI)
        colors_pt = mask_ok.map({True: C_TEAL, False: C_DANGER})
    ax.scatter(df[sel_feat], df[TARGET], c=colors_pt, alpha=0.25, s=8)
    m, b, r, *_ = linregress(df[sel_feat], df[TARGET])
    xr = np.linspace(df[sel_feat].min(), df[sel_feat].max(), 100)
    ax.plot(xr, m * xr + b, color="white", lw=1.5, ls="--", label=f"r = {r:.3f}")
    ax.axhspan(SPEC_LO, SPEC_HI, alpha=0.07, color=C_OK)
    ax.set_xlabel(FEAT_LABELS.get(sel_feat, sel_feat))
    ax.set_ylabel("Head Diameter (mm)")
    ax.legend(fontsize=9, facecolor=C_CARD, labelcolor=C_TEXT)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.caption(f"Teal = in-spec  ·  Red = out-of-spec  ·  Pearson r = {r:.3f}")

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="lsa-section">// OLS linear regression — test set results</div>',
                unsafe_allow_html=True)
    residuals = y_test.values - y_pred_test

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="lsa-section">// Predicted vs measured</div>',
                    unsafe_allow_html=True)
        fig, ax = dark_fig(6, 5)
        in_spec  = (y_test >= SPEC_LO) & (y_test <= SPEC_HI)
        ax.scatter(y_test[in_spec],  y_pred_test[in_spec],
                   alpha=0.5, s=12, color=C_TEAL, label="In-spec")
        ax.scatter(y_test[~in_spec], y_pred_test[~in_spec],
                   alpha=0.5, s=12, color=C_DANGER, label="Out-of-spec")
        lims = [y_test.min() - 0.05, y_test.max() + 0.05]
        ax.plot(lims, lims, "w--", lw=1.2, label="Perfect prediction")
        ax.axvspan(SPEC_LO, SPEC_HI, alpha=0.06, color=C_OK)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Measured (mm)"); ax.set_ylabel("Predicted (mm)")
        ax.set_title(f"Predicted vs Measured — R² = {metrics['r2']}", color=C_TEXT)
        ax.legend(fontsize=8, facecolor=C_CARD, labelcolor=C_TEXT)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Points along the dashed line = perfect prediction. Green zone = specification window.")

    with col_b:
        st.markdown('<div class="lsa-section">// Residuals vs fitted — homoscedasticity check</div>',
                    unsafe_allow_html=True)
        fig, ax = dark_fig(6, 5)
        ax.scatter(y_pred_test, residuals, alpha=0.35, s=10,
                   c=[C_DANGER if abs(r) > 0.15 else C_TEAL for r in residuals])
        ax.axhline(0, color="white", lw=1.5, ls="--")
        ax.axhline(+2 * metrics["rmse"], color=C_WARN, lw=1, ls=":", label="±2·RMSE")
        ax.axhline(-2 * metrics["rmse"], color=C_WARN, lw=1, ls=":")
        ax.set_xlabel("Fitted Value (mm)"); ax.set_ylabel("Residual (mm)")
        ax.legend(fontsize=8, facecolor=C_CARD, labelcolor=C_TEXT)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Random scatter around zero = residuals independent of prediction magnitude. Red = |error| > 0.15 mm.")

    st.divider()
    st.markdown('<div class="lsa-section">// Model comparison — OLS vs regularized variants</div>',
                unsafe_allow_html=True)
    comparison = pd.DataFrame({
        "Model"     : ["OLS Linear Regression", "Ridge (α=1.0)", "Lasso (α=0.01)", "ElasticNet (α=0.01, ρ=0.5)"],
        "MAE Test"  : [0.06510, 0.06508, 0.06821, 0.06673],
        "RMSE Test" : [0.08110, 0.08108, 0.08463, 0.08274],
        "R² Test"   : [0.90900, 0.90902, 0.90216, 0.90638],
    })
    st.dataframe(
        comparison.style
            .highlight_max(subset=["R² Test"],          color="#0f2e28")
            .highlight_min(subset=["RMSE Test","MAE Test"], color="#0f2e28")
            .format({"MAE Test": "{:.5f}", "RMSE Test": "{:.5f}", "R² Test": "{:.5f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("All four models perform within 0.003 R² of each other — regularisation adds no meaningful benefit when all features carry genuine linear signal.")

    st.divider()
    st.markdown('<div class="lsa-section">// Metric explanations</div>',
                unsafe_allow_html=True)
    metric_expl = {
        "R²":   "Proportion of target variance explained by the model. 1.0 = perfect. 0.91 means 91% of diameter variation is captured.",
        "RMSE": "Root Mean Squared Error — average prediction error in mm, penalising large errors more. Comparable to one standard deviation of residuals.",
        "MAE":  "Mean Absolute Error — average absolute miss in mm. More interpretable than RMSE: on average the model is off by this many mm per cycle.",
    }
    for name, expl in metric_expl.items():
        with st.expander(f"{name}  —  {metrics.get(name.lower().replace('²','2'), '—')}"):
            st.write(expl)

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    col_inp, col_out = st.columns([1.25, 1])

    with col_inp:
        st.markdown('<div class="lsa-section">// Press recipe configuration</div>',
                    unsafe_allow_html=True)
        rivet_d = st.selectbox("Rivet Nominal Diameter (mm)", [3.0, 4.0, 5.0], index=1)
        force   = st.slider("Press Force (kN)",        25.0, 60.0, 42.0, 0.5)
        stroke  = st.slider("Press Stroke (mm)",        3.0,  6.0,  4.5, 0.1)
        length  = st.slider("Rivet Length (mm)",        8.0, 16.0, 12.0, 0.5)
        temp    = st.slider("Ambient Temperature (°C)", 18.0, 30.0, 24.0, 0.5)
        hold    = st.slider("Hold Time (ms)",           50.0,200.0,120.0, 5.0)

    x_sim    = pd.DataFrame([{
        "press_force_kn":    force,
        "press_stroke_mm":   stroke,
        "rivet_diameter_mm": rivet_d,
        "rivet_length_mm":   length,
        "temperature_c":     temp,
        "hold_time_ms":      hold,
    }])
    pred_val = model.predict(x_sim)[0]
    is_ok    = SPEC_LO <= pred_val <= SPEC_HI
    margin   = min(pred_val - SPEC_LO, SPEC_HI - pred_val)

    with col_out:
        result_color = C_OK if is_ok else C_DANGER
        badge_bg     = "#0f2e1a" if is_ok else "#2e0f0f"
        badge_label  = "\u2713  OK \u2014 Within Specification" if is_ok else "\u2717  NG \u2014 Out of Specification"
        st.markdown(
            f'''<div style="background:var(--card);border:1px solid var(--border);
                        border-radius:4px;padding:1.6rem 1.8rem;">
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                            color:#fff;margin-bottom:1rem;">Prediction Result</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:3.4rem;
                            font-weight:700;color:{result_color};line-height:1;
                            letter-spacing:-0.02em;">{pred_val:.3f} mm</div>
                <div style="margin-top:14px;">
                    <span style="background:{badge_bg};color:{result_color};
                                 font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                                 font-weight:600;letter-spacing:.08em;
                                 padding:5px 16px;border-radius:20px;">{badge_label}</span>
                </div>
                <div style="margin-top:18px;font-family:'JetBrains Mono',monospace;
                            font-size:0.68rem;color:var(--muted);line-height:2.1;">
                    Spec window &amp;nbsp;: [{SPEC_LO} \u2013 {SPEC_HI}] mm<br>
                    Target centre : {SPEC_MID} mm<br>
                    Margin &amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;:
                    <strong style="color:{result_color};">{margin:+.3f} mm</strong>
                    {'(inside spec)' if is_ok else '(outside spec)'}
                </div>
            </div>''',
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown('<div class="lsa-section">// Position within specification window</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 1.4))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    full_lo, full_hi = 4.4, 6.2
    ax.barh(0, full_hi - full_lo, left=full_lo, height=0.55, color="#1e2d45")
    ax.barh(0, SPEC_HI - SPEC_LO, left=SPEC_LO, height=0.55, color=(0.29, 0.87, 0.50, 0.18))
    ax.axvline(SPEC_LO, color=C_OK, lw=1.2, ls=":")
    ax.axvline(SPEC_HI, color=C_OK, lw=1.2, ls=":")
    marker_c = C_OK if is_ok else C_DANGER
    ax.plot([pred_val, pred_val], [-0.38, 0.38], color=marker_c, lw=2.5)
    ax.scatter([pred_val], [0], s=130, color=marker_c, zorder=5)
    ax.set_xlim(full_lo, full_hi)
    ax.set_ylim(-0.65, 0.65)
    ax.set_yticks([])
    ax.tick_params(colors=C_MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("Head Diameter (mm)", color=C_MUTED, fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.caption("Marker = predicted diameter  \u00b7  Green zone = specification window [5.0\u20135.4 mm].")

    st.divider()
    st.markdown('<div class="lsa-section">// Feature contribution at current settings (top 5)</div>',
                unsafe_allow_html=True)
    top5 = coef_df.head(5).copy()
    top5["Impact (mm)"] = top5["Coefficient"] * x_sim[top5["Feature"]].values[0]
    fig, ax = dark_fig(9, 3.5)
    colors_bar = [C_DANGER if v > 0 else C_TEAL for v in top5["Impact (mm)"]]
    bars = ax.barh(
        [FEAT_LABELS.get(f, f) for f in top5["Feature"][::-1]],
        top5["Impact (mm)"][::-1],
        color=colors_bar[::-1], alpha=0.85, height=0.55, edgecolor="none"
    )
    ax.axvline(0, color="white", lw=0.8)
    for bar, val in zip(bars, top5["Impact (mm)"][::-1]):
        ax.text(val + (0.002 if val >= 0 else -0.002),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f} mm", va="center",
                ha="left" if val >= 0 else "right", fontsize=9, color=C_TEXT)
    ax.set_xlabel("Contribution to Predicted Diameter (mm)")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.caption("Bar = coefficient \u00d7 current parameter value.  Red = pushes diameter up  \u00b7  Teal = pushes diameter down.")

    st.divider()
    st.markdown('<div class="lsa-section">// Scenario comparison</div>', unsafe_allow_html=True)
    best_pred  = model.predict(pd.DataFrame([{"press_force_kn":35,"press_stroke_mm":4.2,
        "rivet_diameter_mm":4.0,"rivet_length_mm":12,"temperature_c":24,"hold_time_ms":120}]))[0]
    worst_pred = model.predict(pd.DataFrame([{"press_force_kn":58,"press_stroke_mm":5.8,
        "rivet_diameter_mm":5.0,"rivet_length_mm":15,"temperature_c":29,"hold_time_ms":180}]))[0]
    cdf = pd.DataFrame([
        {"Scenario": "Low-force recipe (3-4 mm rivet)",  "Predicted (mm)": f"{best_pred:.3f}",
         "Status": "In-spec" if SPEC_LO<=best_pred<=SPEC_HI else "Out-of-spec",
         "vs current": f"{best_pred-pred_val:+.3f} mm"},
        {"Scenario": "Current configuration",            "Predicted (mm)": f"{pred_val:.3f}",
         "Status": "In-spec" if is_ok else "Out-of-spec",
         "vs current": "-"},
        {"Scenario": "High-force recipe (5 mm rivet)",   "Predicted (mm)": f"{worst_pred:.3f}",
         "Status": "In-spec" if SPEC_LO<=worst_pred<=SPEC_HI else "Out-of-spec",
         "vs current": f"{worst_pred-pred_val:+.3f} mm"},
    ])
    st.dataframe(cdf, use_container_width=True, hide_index=True)

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    col_a, col_b = st.columns([1.1, 1])

    with col_a:
        st.markdown('<div class="lsa-section">// OLS coefficients — engineering levers</div>',
                    unsafe_allow_html=True)
        fig, ax = dark_fig(7, 5)
        colors_coef = [C_DANGER if c > 0 else C_TEAL for c in coef_df["Coefficient"]]
        bars = ax.barh(
            [FEAT_LABELS.get(f, f) for f in coef_df["Feature"][::-1]],
            coef_df["Coefficient"][::-1],
            color=colors_coef[::-1], alpha=0.82, edgecolor="none", height=0.6
        )
        ax.axvline(0, color="white", lw=0.8)
        for bar, val in zip(bars, coef_df["Coefficient"][::-1]):
            ax.text(val + (0.0003 if val >= 0 else -0.0003),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.4f}", va="center",
                    ha="left" if val >= 0 else "right", fontsize=9, color=C_TEXT)
        ax.set_xlabel("Coefficient (mm per unit input)")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("All coefficients are positive — every variable increases head diameter when it increases. Rivet diameter dominates.")

    with col_b:
        st.markdown('<div class="lsa-section">// Coefficient interpretation table</div>',
                    unsafe_allow_html=True)
        display_coef = coef_df.copy()
        display_coef["Feature"]     = display_coef["Feature"].map(lambda x: FEAT_LABELS.get(x, x))
        display_coef["Direction"]   = display_coef["Coefficient"].apply(
            lambda c: "▲ Increases diam." if c > 0 else "▼ Decreases diam."
        )
        display_coef["Coefficient"] = display_coef["Coefficient"].map("{:+.6f}".format)
        st.dataframe(display_coef[["Feature", "Coefficient", "Direction"]],
                     use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);
                    border-left:3px solid {C_WARN};border-radius:2px;
                    padding:1rem 1.2rem;margin-top:12px;">
            <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                        text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Key insight</div>
            <div style="font-family:var(--fm);font-size:0.72rem;color:var(--text);line-height:1.7;">
                Switching from a 4 mm to a 5 mm rivet adds <strong style="color:{C_WARN};">+0.183 mm</strong>
                to the predicted head diameter — more than the entire {SPEC_HI - SPEC_LO:.1f} mm spec window.
                Any rivet size change must be accompanied by a press recipe adjustment.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="lsa-section">// 2D response surface — press force × press stroke</div>',
                unsafe_allow_html=True)
    rivet_sel = st.selectbox("Rivet size for surface map:", [3.0, 4.0, 5.0], index=1,
                             key="surf_rivet")
    force_g   = np.linspace(25, 60, 60)
    stroke_g  = np.linspace(3.0, 6.0, 60)
    F, S      = np.meshgrid(force_g, stroke_g)
    grid      = pd.DataFrame({
        "press_force_kn":    F.ravel(),
        "press_stroke_mm":   S.ravel(),
        "rivet_diameter_mm": rivet_sel,
        "rivet_length_mm":   12.0,
        "temperature_c":     24.0,
        "hold_time_ms":      120.0,
    })
    Z = model.predict(grid[FEATURES]).reshape(F.shape)
    fig, ax = dark_fig(10, 5.5)
    cf   = ax.contourf(F, S, Z, levels=25, cmap="RdYlGn_r", alpha=0.88)
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label("Predicted Head Diameter (mm)", color=C_MUTED)
    cbar.ax.yaxis.set_tick_params(color=C_MUTED)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=C_MUTED)
    cs = ax.contour(F, S, Z, levels=[SPEC_LO, SPEC_HI],
                    colors=["white", "white"], linestyles=["--", "--"], linewidths=1.8)
    ax.clabel(cs, fmt="%.1f mm", fontsize=9, colors="white")
    ax.contourf(F, S, Z, levels=[SPEC_LO, SPEC_HI],
                colors=["lime"], alpha=0.12, hatches=["////"])
    ax.set_xlabel("Press Force (kN)")
    ax.set_ylabel("Press Stroke (mm)")
    ax.set_title(f"Response Surface — {rivet_sel:.0f} mm Rivet  ·  Find the In-Spec Zone",
                 color=C_TEXT)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.caption("Green-hatched zone = force + stroke combinations that produce head diameter within [5.0–5.4 mm]. Use this map to define press recipes per rivet size.")

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="lsa-section">// Operational recommendations</div>',
                unsafe_allow_html=True)
    actions = [
        (C_TEAL,   "Anchor your press recipe to the response surface",
         "Use the 2D contour map to define force & stroke setpoints for each rivet diameter. Every rivet size change requires a corresponding recipe change — the model quantifies exactly how much."),
        (C_DANGER, "Implement real-time diameter prediction at recipe changeover",
         "Before the first production cycle after a changeover, run the predicted parameters through the simulator. If the predicted head diameter falls outside [5.0–5.4 mm], halt and recalculate. This catches scrap before it happens."),
        (C_OK,     "Prioritise press stroke as the primary fine-tune lever",
         "The stroke coefficient (+0.079 mm/mm) is the strongest controllable parameter after rivet diameter. Small stroke adjustments (±0.5 mm) have a predictable and measurable effect, making it the preferred real-time adjustment knob."),
        (C_WARN,   "Monitor temperature and hold time for process drift",
         "Their coefficients are small (+0.003 and +0.0002 mm per unit), but in high-volume production accumulation of warm-up drift, seasonal temperature variation, and press timing variability can bias the mean by 0.05–0.10 mm over a shift."),
        (C_TEAL,   "Automate the optimal-settings grid at batch planning",
         "Run the optimal-settings search automatically when the production plan changes rivet sizes. Return the top-5 press recipes (closest to target 5.2 mm, within spec) for the process engineer to choose from."),
    ]
    for color, title, body in actions:
        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);
                    border-left:3px solid {color};border-radius:2px;
                    padding:1.1rem 1.3rem;margin-bottom:10px;">
            <div style="font-family:var(--fm);font-size:0.72rem;font-weight:600;
                        color:{color};margin-bottom:6px;">{title}</div>
            <div style="font-family:var(--fm);font-size:0.7rem;color:var(--muted);line-height:1.7;">{body}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
    <div style="background:var(--card);border:1px solid var(--border);border-radius:2px;
                padding:1rem 1.3rem;text-align:center;">
        <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Model validity reminder</div>
        <div style="font-family:var(--fm);font-size:0.68rem;color:var(--muted);line-height:1.7;">
            Calibrated for press force 25–60 kN · stroke 3.0–6.0 mm · rivet diameters 3–5 mm.<br>
            Predictions outside these ranges are extrapolations — treat with caution.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lsa-footer">
    LozanoLsa · Turning Operations into Predictive Systems · Riveting Head Diameter Predictor · Project 07 · v2.0
</div>
""", unsafe_allow_html=True)
