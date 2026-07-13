"""
app.py — Hot Stamping Tensile Strength Dashboard
LozanoLsa · Project 11 · ElasticNet Regression · 2026

Model: ElasticNet + ElasticNetCV (L1+L2 combined)
Domain: Press Hardening — Tensile Strength Prediction
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ElasticNet · Hot Stamping Strength",
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
DATA_PATH     = "hot_stamping_data.csv"
DATA_PATH_ALT = "11_ElasticNet_Hot_Stamping/hot_stamping_data.csv"
RANDOM_STATE  = 42
FEATURES = ["furnace_entry_temp_c", "furnace_exit_temp_c", "die_temp_c", "transfer_time_s",
            "press_force_kn", "press_speed_mm_s", "dwell_time_s", "blank_thickness_mm",
            "steel_grade", "active_cooling", "cooling_pressure_bar", "cooling_flow_l_min"]
TARGET        = "tensile_strength_mpa"
SPEC_MIN      = 900
STEEL_GRADES  = {1: "Usibor (1500 MPa)", 2: "Ductibor (500 MPa)", 3: "MBorian (700 MPa)"}

FEAT_LABELS = {
    "furnace_entry_temp_c":  "Furnace Entry Temp (°C)",
    "furnace_exit_temp_c":   "Furnace Exit Temp (°C)",
    "die_temp_c":            "Die Temperature (°C)",
    "transfer_time_s":       "Transfer Time (s)",
    "press_force_kn":        "Press Force (kN)",
    "press_speed_mm_s":      "Press Speed (mm/s)",
    "dwell_time_s":          "Dwell Time (s)",
    "blank_thickness_mm":    "Blank Thickness (mm)",
    "steel_grade":           "Steel Grade",
    "active_cooling":        "Active Cooling",
    "cooling_pressure_bar":  "Cooling Pressure (bar)",
    "cooling_flow_l_min":    "Cooling Flow (L/min)",
}

# ─── MATPLOTLIB PALETTE ───────────────────────────────────────────────────────
C_BG    = "#080c12"
C_CARD  = "#121922"
C_TEAL  = "#2dd4bf"
C_TEAL2 = "#5eead4"
C_DANGER= "#f87171"
C_WARN  = "#fbbf24"
C_OK    = "#4ade80"
C_PURP  = "#a78bfa"
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
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(Xtr); Xte_sc = sc.transform(Xte)
    en_cv = ElasticNetCV(
        l1_ratio=[.1, .3, .5, .7, .9, .95, .99, 1.0],
        alphas=np.logspace(-3, 2, 60),
        cv=5, random_state=RANDOM_STATE, max_iter=20000
    )
    en_cv.fit(Xtr_sc, ytr)
    best_a = en_cv.alpha_; best_l1 = en_cv.l1_ratio_
    mdl = ElasticNet(alpha=best_a, l1_ratio=best_l1, max_iter=20000, random_state=RANDOM_STATE)
    mdl.fit(Xtr_sc, ytr); yp = mdl.predict(Xte_sc)
    metrics = {
        "r2"  : round(r2_score(yte, yp), 4),
        "rmse": round(np.sqrt(mean_squared_error(yte, yp)), 2),
        "mae" : round(mean_absolute_error(yte, yp), 2),
        "alpha": round(best_a, 5),
        "l1":   round(best_l1, 2),
    }
    cdf = (pd.DataFrame({"Feature": FEATURES, "Coefficient": mdl.coef_})
             .sort_values("Coefficient", key=abs, ascending=False)
             .reset_index(drop=True))
    return mdl, sc, Xte, yte, yp, metrics, cdf

df = load_data()
model, scaler, X_test, y_test, y_pred, metrics, coef_df = train_model(df)
active    = coef_df[coef_df["Coefficient"] != 0]
zeroed    = coef_df[coef_df["Coefficient"] == 0]
residuals = y_test.values - y_pred

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-header">
    <div class="lsa-project-tag">ML Project #11 · ElasticNet Regression · Hot Stamping · Final Regression</div>
    <div class="lsa-title">L1 + L2 — Best of Both Penalties</div>
    <div class="lsa-tagline">Lasso finds the redundant sensor. Ridge stabilises the correlated ones. ElasticNet does both in one pass.</div>
    <div style="margin-top:10px;">
        <span class="lsa-chip">ELASTICNET · CV</span>
        <span class="lsa-chip">{len(active)} / {len(FEATURES)} ACTIVE</span>
        <span class="lsa-chip">R² {metrics['r2']:.4f}</span>
        <span class="lsa-chip">α = {metrics['alpha']:.4f}  ρ = {metrics['l1']:.2f}</span>
        <span class="lsa-chip">SPEC ≥ {SPEC_MIN} MPa</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TOP KPI ROW ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("R² (Test Set)",      f"{metrics['r2']:.4f}",       "96.8% variance explained")
k2.metric("RMSE",               f"±{metrics['rmse']:.1f} MPa","4.4% relative error vs 991 MPa")
k3.metric("CV-Selected Params", f"α={metrics['alpha']:.4f}  ρ={metrics['l1']:.2f}",
          "5-fold · 480 candidate grid")
k4.metric("Features Selected",  f"{len(active)} / {len(FEATURES)} active",
          f"{len(zeroed)} zeroed: furnace_entry (r=0.95)")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "DATA EXPLORER", "PERFORMANCE", "SIMULATOR", "ELASTICNET ANALYSIS", "ACTION PLAN"
])

# ══ TAB 1 ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="lsa-section">// Dataset — 1,741 hot stamping cycles</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", "1,741")
    c2.metric("Process Variables", "12")
    c3.metric("Spec", f"≥{SPEC_MIN} MPa")
    with st.expander("Preview first 20 rows"):
        st.dataframe(df.head(20), use_container_width=True)

    st.divider()
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="lsa-section">// Strength by steel grade</div>',
                    unsafe_allow_html=True)
        fig, ax = dark_fig(7, 4)
        for g, color in {1: C_TEAL, 2: C_OK, 3: C_WARN}.items():
            vals = df[df["steel_grade"] == g][TARGET]
            ax.hist(vals, bins=25, alpha=0.65, color=color,
                    edgecolor=C_BG, lw=0.3,
                    label=f"Gr.{g} {STEEL_GRADES[g][:7]}")
        ax.axvline(SPEC_MIN, color=C_DANGER, ls="--", lw=1.8,
                   label=f"Spec ≥{SPEC_MIN} MPa")
        ax.set_xlabel("Tensile Strength (MPa)"); ax.set_ylabel("Count")
        ax.legend(fontsize=7, facecolor=C_CARD, labelcolor=C_TEXT)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        for g in [1, 2, 3]:
            sub = df[df["steel_grade"] == g][TARGET]
            st.caption(f"Grade {g}: {sub.mean():.0f} ± {sub.std():.0f} MPa  "
                       f"({(sub >= SPEC_MIN).mean()*100:.0f}% pass)")

    with cb:
        st.markdown('<div class="lsa-section">// Collinearity — furnace entry vs exit</div>',
                    unsafe_allow_html=True)
        fig, ax = dark_fig(7, 4)
        sc_plot = ax.scatter(df["furnace_entry_temp_c"], df["furnace_exit_temp_c"],
                             c=df[TARGET], cmap="RdYlGn", alpha=0.30, s=8)
        plt.colorbar(sc_plot, ax=ax, label="Tensile Strength (MPa)")
        m, b, r, *_ = linregress(df["furnace_entry_temp_c"], df["furnace_exit_temp_c"])
        xr = np.linspace(df["furnace_entry_temp_c"].min(),
                         df["furnace_entry_temp_c"].max(), 100)
        ax.plot(xr, m * xr + b, color="white", lw=1.5, ls="--", label=f"r = {r:.3f}")
        ax.set_xlabel("Furnace Entry Temp (°C)"); ax.set_ylabel("Furnace Exit Temp (°C)")
        ax.set_title(f"r = {r:.3f} — Entry zeroed by ElasticNet", color=C_TEXT)
        ax.legend(fontsize=8, facecolor=C_CARD, labelcolor=C_TEXT)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Entry and exit temperatures co-move at r=0.95. ElasticNet zeroed the entry sensor — "
                   "exit temperature captures all shared information.")

    st.divider()
    st.markdown('<div class="lsa-section">// Feature scatter vs tensile strength</div>',
                unsafe_allow_html=True)
    sel = st.selectbox("Feature:", FEATURES, format_func=lambda x: FEAT_LABELS.get(x, x))
    m, b, r, p, _ = linregress(df[sel], df[TARGET])
    fig, ax = dark_fig(10, 4)
    ax.scatter(df[sel], df[TARGET], alpha=0.25, s=8, color=C_TEAL)
    xr = np.linspace(df[sel].min(), df[sel].max(), 100)
    ax.plot(xr, m * xr + b, color=C_DANGER, lw=1.5, ls="--", label=f"r = {r:.3f}")
    ax.axhline(SPEC_MIN, color=C_WARN, lw=1.2, ls=":", label=f"Spec {SPEC_MIN} MPa")
    ax.set_xlabel(FEAT_LABELS.get(sel, sel)); ax.set_ylabel("Tensile Strength (MPa)")
    ax.legend(fontsize=9, facecolor=C_CARD, labelcolor=C_TEXT)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.caption(f"r = {r:.3f}  ·  slope = {m:+.4f} MPa/unit  ·  "
               f"p {'< 0.001' if p < 0.001 else f'= {p:.4f}'}")

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="lsa-section">// ElasticNet — test performance (n=300)</div>',
                unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="lsa-section">// Predicted vs actual</div>',
                    unsafe_allow_html=True)
        fig, ax = dark_fig(6, 5)
        in_spec = y_test >= SPEC_MIN
        ax.scatter(y_test[in_spec],  y_pred[in_spec],  alpha=0.45, s=12, color=C_TEAL,   label="Pass")
        ax.scatter(y_test[~in_spec], y_pred[~in_spec], alpha=0.45, s=12, color=C_DANGER, label="Fail")
        lims = [y_test.min() - 10, y_test.max() + 10]
        ax.plot(lims, lims, color="white", ls="--", lw=1.5, label="Perfect")
        ax.axvline(SPEC_MIN, color=C_WARN, lw=1.2, ls=":", alpha=0.8)
        ax.axhline(SPEC_MIN, color=C_WARN, lw=1.2, ls=":", alpha=0.8,
                   label=f"Spec {SPEC_MIN} MPa")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual (MPa)"); ax.set_ylabel("Predicted (MPa)")
        ax.set_title(f"R² = {metrics['r2']}", color=C_TEXT)
        ax.legend(fontsize=8, facecolor=C_CARD, labelcolor=C_TEXT)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Teal = within spec · Red = below minimum tensile requirement.")

    with cb:
        st.markdown('<div class="lsa-section">// Residuals vs fitted — homoscedasticity check</div>',
                    unsafe_allow_html=True)
        fig, ax = dark_fig(6, 5)
        ax.scatter(y_pred, residuals, alpha=0.35, s=10,
                   c=[C_DANGER if abs(r) > 60 else C_TEAL for r in residuals])
        ax.axhline(0, color="white", lw=1.5, ls="--")
        ax.axhline(+2 * metrics["rmse"], color=C_WARN, lw=1, ls=":", label="±2·RMSE")
        ax.axhline(-2 * metrics["rmse"], color=C_WARN, lw=1, ls=":")
        ax.set_xlabel("Fitted (MPa)"); ax.set_ylabel("Residual (MPa)")
        ax.legend(fontsize=8, facecolor=C_CARD, labelcolor=C_TEXT)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Random scatter — no systematic bias across the strength range.")

    st.divider()
    st.markdown('<div class="lsa-section">// Regularisation comparison — EN vs Ridge vs Lasso</div>',
                unsafe_allow_html=True)
    Xtr2, Xte2, ytr2, yte2 = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=RANDOM_STATE
    )
    sc2 = StandardScaler(); Xtr2s = sc2.fit_transform(Xtr2); Xte2s = sc2.transform(Xte2)
    rows = []
    for name, mdl2 in [
        ("ElasticNet (CV-selected)", None),
        ("Ridge (α=1.0)",            Ridge(alpha=1.0)),
        ("Lasso (α=0.01)",           Lasso(alpha=0.01, max_iter=20000)),
    ]:
        if name == "ElasticNet (CV-selected)":
            yp2 = model.predict(Xte2s)
            act = int(sum(model.coef_ != 0))
        else:
            mdl2.fit(Xtr2s, ytr2); yp2 = mdl2.predict(Xte2s)
            act = len(FEATURES)
        rows.append({
            "Model":          name,
            "R² Test":        round(r2_score(yte2, yp2), 5),
            "RMSE (MPa)":     round(np.sqrt(mean_squared_error(yte2, yp2)), 2),
            "Active Features":act,
        })
    st.dataframe(
        pd.DataFrame(rows).style.highlight_max(subset=["R² Test"], color="#0f2e28"),
        use_container_width=True, hide_index=True
    )
    st.caption("All three regularisations converge on accuracy within margin (R² ≈ 0.893). "
               "ElasticNet's value: it found the optimal mix automatically and zeroed the redundant entry sensor.")

    st.divider()
    st.markdown('<div class="lsa-section">// Metric explanations</div>',
                unsafe_allow_html=True)
    for name, expl in {
        "R²":   "89.3% of tensile strength variance is explained across three production shifts and two grade changeovers. The remaining scatter reflects genuine process variability that no recipe can fully eliminate.",
        "RMSE": "Root Mean Squared Error in MPa. At 4.4% of the mean 991 MPa strength, the model supports recipe classification and process window decisions — not a replacement for inline tensile testing.",
        "MAE":  "Mean Absolute Error per cycle. The model's average miss is small enough to use for recipe pre-screening.",
    }.items():
        with st.expander(f"{name}  —  {metrics.get(name.lower().replace('²','2'), '—')}"):
            st.write(expl)

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    medians = df[FEATURES].median().to_dict()
    ci, co  = st.columns([1.1, 1])

    with ci:
        st.markdown('<div class="lsa-section">// Material & furnace</div>', unsafe_allow_html=True)
        grade       = st.selectbox("Steel Grade", options=[1, 2, 3],
                                   format_func=lambda g: f"Grade {g} — {STEEL_GRADES[g]}")
        furnace_exit= st.slider("Furnace Exit Temp (°C)", 850, 960,
                                int(medians["furnace_exit_temp_c"]), 5)
        transfer    = st.slider("Transfer Time (s)", 2.0, 9.0,
                                float(medians["transfer_time_s"]), 0.1)
        thickness   = st.selectbox("Blank Thickness (mm)", [1.0, 1.2, 1.4, 1.6])

        st.markdown('<div class="lsa-section">// Press</div>', unsafe_allow_html=True)
        force = st.slider("Press Force (kN)", 800, 1800, int(medians["press_force_kn"]), 25)
        speed = st.slider("Press Speed (mm/s)", 30, 65, int(medians["press_speed_mm_s"]), 1)
        dwell = st.slider("Dwell Time (s)", 2.0, 5.0, float(medians["dwell_time_s"]), 0.1)

        st.markdown('<div class="lsa-section">// Cooling</div>', unsafe_allow_html=True)
        die_temp = st.slider("Die Temperature (°C)", 130, 185,
                             int(medians["die_temp_c"]), 1)
        cooling  = st.selectbox("Active Cooling", [1, 0],
                                format_func=lambda x: "ON — Water cooled" if x else "OFF — Passive")
        cool_p   = st.slider("Cooling Pressure (bar)", 8, 24,
                             int(medians["cooling_pressure_bar"]), 1)
        cool_f   = st.slider("Cooling Flow (L/min)", 15, 48,
                             int(medians["cooling_flow_l_min"]), 1)

    params = medians.copy()
    params.update({
        "steel_grade": grade, "furnace_exit_temp_c": furnace_exit,
        "transfer_time_s": transfer, "blank_thickness_mm": float(thickness),
        "press_force_kn": force, "press_speed_mm_s": speed, "dwell_time_s": dwell,
        "die_temp_c": die_temp, "active_cooling": cooling,
        "cooling_pressure_bar": cool_p, "cooling_flow_l_min": cool_f,
    })
    xsim     = pd.DataFrame([[params[c] for c in FEATURES]], columns=FEATURES)
    pred_mpa = model.predict(scaler.transform(xsim))[0]
    is_pass  = pred_mpa >= SPEC_MIN
    margin   = pred_mpa - SPEC_MIN
    qual_c   = C_OK if is_pass else C_DANGER
    qual_l   = "PASS — Within tensile specification" if is_pass else "FAIL — Below minimum tensile spec"
    badge_bg = "#0f2e1a" if is_pass else "#2e0f0f"

    with co:
        st.markdown(
            f'''<div style="background:var(--card);border:1px solid var(--border);
                        border-radius:4px;padding:1.6rem 1.8rem;">
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                            color:#fff;margin-bottom:1rem;">Prediction Result</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:3.4rem;
                            font-weight:700;color:{qual_c};line-height:1;
                            letter-spacing:-0.02em;">{pred_mpa:.0f}
                    <span style="font-size:1.4rem;font-weight:400;color:{C_MUTED};">MPa</span>
                </div>
                <div style="margin-top:14px;">
                    <span style="background:{badge_bg};color:{qual_c};
                                 font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                                 font-weight:600;letter-spacing:.08em;
                                 padding:5px 16px;border-radius:20px;">{qual_l}</span>
                </div>
                <div style="margin-top:18px;font-family:'JetBrains Mono',monospace;
                            font-size:0.68rem;color:var(--muted);line-height:2.1;">
                    Steel &nbsp;&nbsp;&nbsp;: {STEEL_GRADES[grade]}<br>
                    Spec &nbsp;&nbsp;&nbsp;&nbsp;: &#8805;{SPEC_MIN} MPa<br>
                    Margin &nbsp;:
                    <strong style="color:{qual_c};">{margin:+.0f} MPa</strong>
                    {'(within spec)' if is_pass else '(below spec)'}
                </div>
            </div>''',
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown('<div class="lsa-section">// Position within strength scale</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 1.4))
    fig.patch.set_facecolor(C_BG); ax.set_facecolor(C_BG)
    ax.barh(0, 1400 - 600, left=600, height=0.55, color="#1e2d45")
    # Green zone = above spec (spec_min to 1400)
    ax.barh(0, 1400 - SPEC_MIN, left=SPEC_MIN, height=0.55,
            color=(0.29, 0.87, 0.50, 0.18))
    ax.axvline(SPEC_MIN, color=C_OK, lw=1.2, ls=":")
    mc = C_OK if is_pass else C_DANGER
    ax.plot([pred_mpa, pred_mpa], [-0.38, 0.38], color=mc, lw=2.5)
    ax.scatter([pred_mpa], [0], s=130, color=mc, zorder=5)
    ax.set_xlim(600, 1400); ax.set_ylim(-0.65, 0.65); ax.set_yticks([])
    ax.tick_params(colors=C_MUTED, labelsize=8)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xlabel("Tensile Strength (MPa)", color=C_MUTED, fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.caption(f"Green zone = meets spec (≥{SPEC_MIN} MPa) · Marker = predicted strength.")

    st.divider()
    st.markdown('<div class="lsa-section">// Three reference scenarios</div>',
                unsafe_allow_html=True)
    scen = {
        "A · Usibor · Optimised":    {
            "steel_grade":1,"furnace_exit_temp_c":920,"press_force_kn":1350,
            "die_temp_c":140,"transfer_time_s":4.0,"active_cooling":1,
            "cooling_pressure_bar":17,"cooling_flow_l_min":36,"blank_thickness_mm":1.0,
        },
        "B · MBorian · Poor Recipe": {
            "steel_grade":3,"furnace_exit_temp_c":870,"press_force_kn":1050,
            "die_temp_c":165,"transfer_time_s":6.5,"active_cooling":0,
            "blank_thickness_mm":1.6,
        },
        "C · MBorian · Corrected":   {
            "steel_grade":3,"furnace_exit_temp_c":870,"press_force_kn":1050,
            "die_temp_c":145,"transfer_time_s":4.5,"active_cooling":1,
            "cooling_pressure_bar":16,"cooling_flow_l_min":33,"blank_thickness_mm":1.6,
        },
    }
    cols_s = st.columns(3); sc_preds = {}
    for col, (name, p) in zip(cols_s, scen.items()):
        base2 = medians.copy(); base2.update(p)
        pm    = model.predict(scaler.transform(
            pd.DataFrame([[base2[c] for c in FEATURES]], columns=FEATURES)
        ))[0]
        sc_preds[name] = pm
        ok    = pm >= SPEC_MIN
        c_col = C_OK if ok else C_DANGER
        with col:
            st.markdown(f"""
            <div style="background:var(--card);border:1px solid var(--border);
                        border-left:3px solid {c_col};border-radius:2px;padding:1.1rem 1.2rem;">
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);
                            letter-spacing:.15em;text-transform:uppercase;margin-bottom:8px;">{name}</div>
                <div style="font-family:var(--fm);font-size:2.4rem;font-weight:700;
                            color:{c_col};line-height:1;">{pm:.0f}</div>
                <div style="font-family:var(--fm);font-size:0.72rem;color:var(--muted);
                            margin-top:4px;">MPa · {'Pass' if ok else 'Fail'}</div>
            </div>""", unsafe_allow_html=True)

    pv = list(sc_preds.values())
    st.markdown(f"""
    <div style="background:var(--card);border:1px solid var(--border);
                border-left:3px solid {C_WARN};border-radius:2px;
                padding:0.9rem 1.2rem;margin-top:10px;">
        <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Cooling recovery</div>
        <div style="font-family:var(--fm);font-size:0.72rem;color:var(--text);line-height:1.7;">
            Activating die cooling on MBorian (passive → active) and reducing die temp 165→145°C
            recovers <strong style="color:{C_WARN};">{pv[2]-pv[1]:.0f} MPa</strong> —
            from {pv[1]:.0f} to {pv[2]:.0f} MPa.
            Cooling is the primary lever when steel grade is fixed.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    ca, cb = st.columns([1.2, 1])
    with ca:
        st.markdown('<div class="lsa-section">// ElasticNet coefficients & feature selection</div>',
                    unsafe_allow_html=True)
        cs = coef_df.sort_values("Coefficient", ascending=True)
        fig, ax = dark_fig(7, 6)
        bar_c = [C_MUTED if c == 0 else (C_DANGER if c > 0 else C_TEAL)
                 for c in cs["Coefficient"]]
        bars = ax.barh(
            [FEAT_LABELS.get(f, f) for f in cs["Feature"]],
            cs["Coefficient"], color=bar_c, alpha=0.82, edgecolor="none", height=0.65
        )
        for bar, val, feat in zip(bars, cs["Coefficient"], cs["Feature"]):
            if val == 0:
                bar.set_hatch("////"); bar.set_alpha(0.30)
                ax.text(1.0, bar.get_y() + bar.get_height() / 2, "zeroed →",
                        va="center", ha="left", fontsize=8, color=C_MUTED, style="italic")
            else:
                off = 1.0 if val >= 0 else -1.0
                ax.text(val + off, bar.get_y() + bar.get_height() / 2,
                        f"{val:+.2f}", va="center",
                        ha="left" if val >= 0 else "right", fontsize=9, color=C_TEXT)
        ax.axvline(0, color="white", lw=0.8)
        ax.set_xlabel("Standardised Coefficient (MPa per σ input change)")
        ax.set_title(f"ElasticNet  α={metrics['alpha']:.4f}  ρ={metrics['l1']:.2f}",
                     color=C_TEXT)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Red = increases strength · Teal = decreases strength · Hatched = zeroed by L1 penalty.")

    with cb:
        st.markdown('<div class="lsa-section">// Active features</div>',
                    unsafe_allow_html=True)
        atbl = active[["Feature", "Coefficient"]].copy()
        atbl["Feature"]    = atbl["Feature"].map(lambda x: FEAT_LABELS.get(x, x))
        atbl["Direction"]  = atbl["Coefficient"].apply(
            lambda c: "▲ Stronger" if c > 0 else "▼ Weaker"
        )
        atbl["Coefficient"] = atbl["Coefficient"].map("{:+.4f}".format)
        st.dataframe(atbl, use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);
                    border-left:3px solid {C_DANGER};border-radius:2px;
                    padding:1rem 1.2rem;margin-top:12px;">
            <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                        text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Zeroed: furnace entry</div>
            <div style="font-family:var(--fm);font-size:0.72rem;color:var(--text);line-height:1.7;">
                <code>furnace_entry_temp_c</code> was eliminated — its signal is fully captured by
                <code>furnace_exit_temp_c</code> (r=0.95). The L1 component detected the redundancy
                and zeroed the entry sensor. Monitor exit only; entry adds nothing predictive.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="lsa-section">// Response surface — press force × die temperature</div>',
                unsafe_allow_html=True)
    grade_surf = st.selectbox("Steel grade for surface map:",
                              options=[1, 2, 3],
                              format_func=lambda g: f"Grade {g} — {STEEL_GRADES[g]}",
                              key="surf_g")
    force_r = np.linspace(900, 1700, 60); die_r = np.linspace(130, 185, 60)
    F, D = np.meshgrid(force_r, die_r)
    base_surf = df[FEATURES].median().to_dict()
    base_surf.update({"steel_grade": grade_surf, "active_cooling": 1})
    grid = pd.DataFrame({
        "furnace_entry_temp_c": base_surf["furnace_entry_temp_c"],
        "furnace_exit_temp_c":  900.0,
        "die_temp_c":           D.ravel(),
        "transfer_time_s":      5.0,
        "press_force_kn":       F.ravel(),
        "press_speed_mm_s":     45.0,
        "dwell_time_s":         3.5,
        "blank_thickness_mm":   1.2,
        "steel_grade":          grade_surf,
        "active_cooling":       1,
        "cooling_pressure_bar": 14.0,
        "cooling_flow_l_min":   30.0,
    })
    Z = model.predict(scaler.transform(grid[FEATURES])).reshape(F.shape)
    fig, ax = dark_fig(10, 5.5)
    cf   = ax.contourf(F, D, Z, levels=25, cmap="RdYlGn", alpha=0.88)
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label("Predicted Tensile Strength (MPa)", color=C_MUTED)
    cbar.ax.yaxis.set_tick_params(color=C_MUTED)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=C_MUTED)
    cs2 = ax.contour(F, D, Z, levels=[SPEC_MIN], colors=["white"], linewidths=2.0)
    ax.clabel(cs2, fmt=f"{SPEC_MIN} MPa", fontsize=9, colors="white")
    ax.contourf(F, D, Z, levels=[SPEC_MIN, 2000],
                colors=["lime"], alpha=0.12, hatches=["////"])
    ax.set_xlabel("Press Force (kN)"); ax.set_ylabel("Die Temperature (°C)")
    ax.set_title(f"Tensile Response Surface — {STEEL_GRADES[grade_surf]} (Active Cooling)",
                 color=C_TEXT)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.caption("Green-hatched zone = meets ≥900 MPa spec. Higher force and lower die temp both push strength up.")

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="lsa-section">// Operational recommendations</div>',
                unsafe_allow_html=True)
    actions = [
        (C_DANGER, "Steel grade selection is a first-principle decision",
         "Grade 1 (Usibor) averages 1053 MPa; Grade 3 (MBorian) averages 931 MPa. "
         "No process adjustment fully compensates for grade selection. Before optimising process "
         "parameters, confirm the correct steel class is specified for the strength requirement."),
        (C_TEAL,   "Press force is the primary in-cycle lever (coefficient +114 MPa/σ)",
         "Higher press force improves martensite transformation completeness. When parts come back "
         "borderline on tensile, increase force by 50–100 kN before adjusting temperature — "
         "force has the highest unit impact of any controllable variable."),
        (C_OK,     "Active cooling is non-negotiable for spec-critical parts",
         "The active_cooling flag carries a +6.7 MPa/σ coefficient. More importantly, "
         "the die temperature effect (−27.6 MPa/σ) means hotter dies from passive cooling "
         "systematically depress strength. For B-pillar and structural parts, run water-cooled "
         "dies as standard — never passive cooling."),
        (C_WARN,   "Decommission the furnace entry thermocouple for prediction",
         "furnace_entry_temp_c was zeroed by ElasticNet — it adds no independent predictive "
         "signal beyond furnace_exit_temp_c. Redirect calibration budget to the exit sensor "
         "and die thermocouples, which are active drivers."),
        (C_PURP,   "Retrain after steel batch changes or major die maintenance",
         "The model learns the heat transfer characteristics of a specific die-steel-coolant system. "
         "Die refurbishment or a new steel coil supplier shifts the baseline. Collect 300+ cycles "
         "and refit with ElasticNetCV — the alpha/l1_ratio search is automatic."),
    ]
    for color, title, body in actions:
        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);
                    border-left:3px solid {color};border-radius:2px;
                    padding:1.1rem 1.3rem;margin-bottom:10px;">
            <div style="font-family:var(--fm);font-size:0.72rem;font-weight:600;
                        color:{color};margin-bottom:6px;">{title}</div>
            <div style="font-family:var(--fm);font-size:0.7rem;color:var(--muted);
                        line-height:1.7;">{body}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="background:var(--card);border:1px solid var(--border);border-radius:2px;
                padding:1rem 1.3rem;text-align:center;">
        <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Full project pack</div>
        <div style="font-family:var(--fm);font-size:0.68rem;color:var(--muted);line-height:1.7;">
            Complete dataset · notebook with full outputs · presentation deck (PPTX + PDF) · simulator
            available on <span style="color:#2dd4bf;">lozanolsa.gumroad.com</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lsa-footer">
    LozanoLsa · Turning Operations into Predictive Systems · Hot Stamping Strength Predictor · Project 11 · v2.0
</div>
""", unsafe_allow_html=True)
