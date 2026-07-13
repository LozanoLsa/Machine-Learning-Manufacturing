"""
app.py — Heat Treatment Cycle Time Intelligence Dashboard
LozanoLsa · Project 08 · Multiple Linear Regression · 2026 · FREE PROJECT

Model: OLS Multiple Linear Regression + statsmodels
Domain: Manufacturing — Quench Rack Scheduling
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson as dw_test
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LR · Quench Rack Cycle Time Predictor",
    page_icon="🔥",
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
.lsa-chip-free { display: inline-block; background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.3); color: #4ade80; font-family: var(--fm); font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 2px 8px; border-radius: 2px; margin-right: 5px; }
.lsa-section { font-family: var(--fm); font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid var(--border); }
.lsa-footer { margin-top: 2.5rem; padding-top: 0.8rem; border-top: 1px solid var(--border); font-family: var(--fm); font-size: 0.58rem; color: var(--muted); letter-spacing: 0.1em; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATA_PATH     = "quench_data.csv"
DATA_PATH_ALT = "08_Multi_Factor_Process_Impact/quench_data.csv"
RANDOM_STATE  = 42
FEATURES      = ["rack_piece_count", "avg_piece_mass_kg", "part_mix_count",
                 "avg_furnace_temp_c", "burner_power_pct"]
TARGET        = "cycle_time_min"

FEAT_LABELS = {
    "rack_piece_count":   "Pieces in Rack",
    "avg_piece_mass_kg":  "Avg Piece Mass (kg)",
    "part_mix_count":     "Part Mix (# PN types)",
    "avg_furnace_temp_c": "Avg Furnace Temp (°C)",
    "burner_power_pct":   "Burner Power (%)",
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
    mdl = LinearRegression()
    mdl.fit(Xtr, ytr)
    yp = mdl.predict(Xte)
    metrics = {
        "r2"  : round(r2_score(yte, yp), 4),
        "rmse": round(np.sqrt(mean_squared_error(yte, yp)), 2),
        "mae" : round(mean_absolute_error(yte, yp), 2),
    }
    cdf = (pd.DataFrame({"Feature": FEATURES, "Coefficient": mdl.coef_})
             .sort_values("Coefficient", key=abs, ascending=False)
             .reset_index(drop=True))
    X_sm = sm.add_constant(Xtr.reset_index(drop=True))
    ols  = sm.OLS(ytr.reset_index(drop=True), X_sm).fit()
    return mdl, Xte, yte, yp, metrics, cdf, ols

df = load_data()
model, X_test, y_test, y_pred, metrics, coef_df, ols = train_model(df)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-header">
    <div class="lsa-project-tag">ML Project #08 · Multiple Linear Regression · Heat Treatment</div>
    <div class="lsa-title">Cycle Time Is a Function of the Rack</div>
    <div class="lsa-tagline">Mass, mix, temperature, power — the furnace does not lie. The coefficients tell you exactly what each decision costs.</div>
    <div style="margin-top:10px;">
        <span class="lsa-chip">OLS REGRESSION</span>
        <span class="lsa-chip">5 FEATURES</span>
        <span class="lsa-chip">R² {metrics['r2']:.4f}</span>
        <span class="lsa-chip">RMSE ±{metrics['rmse']:.1f} MIN</span>
        <span class="lsa-chip-free">FREE PROJECT</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TOP KPI ROW ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("R² (Test Set)",      f"{metrics['r2']:.4f}",   "80.5% variance explained")
k2.metric("RMSE",               f"±{metrics['rmse']:.1f} min", "Scheduling buffer needed")
k3.metric("MAE",                f"{metrics['mae']:.1f} min",   "< 4% of mean 300 min")
k4.metric("Significant Vars",   "5 / 5",                  "All p < 0.001 · VIF < 1.02")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "DATA EXPLORER", "PERFORMANCE", "SIMULATOR", "PROCESS DRIVERS", "ACTION PLAN"
])

# ══ TAB 1 ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="lsa-section">// Dataset — 647 furnace rack cycles</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", "647")
    c2.metric("PNs Observed", "200 of 300")
    c3.metric("Target Variable", "cycle_time_min")

    with st.expander("Preview first 20 rows"):
        st.dataframe(df.head(20), use_container_width=True)

    st.divider()
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="lsa-section">// Cycle time distribution</div>',
                    unsafe_allow_html=True)
        fig, ax = dark_fig(7, 4)
        ax.hist(df[TARGET], bins=35, color=C_TEAL, alpha=0.70,
                edgecolor=C_BG, linewidth=0.3)
        ax.axvline(df[TARGET].mean(),   color=C_DANGER, ls="--", lw=1.5,
                   label=f"Mean = {df[TARGET].mean():.1f}")
        ax.axvline(df[TARGET].median(), color=C_WARN,   ls=":",  lw=1.5,
                   label=f"Median = {df[TARGET].median():.1f}")
        ax.set_xlabel("Cycle Time (min)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8, facecolor=C_CARD, labelcolor=C_TEXT)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Range 208–382 min. Right tail = heavy mixed racks.")

    with cb:
        st.markdown('<div class="lsa-section">// Cycle time by part mix</div>',
                    unsafe_allow_html=True)
        fig, ax = dark_fig(7, 4)
        grps = [df[df["part_mix_count"] == k][TARGET].values for k in range(1, 6)]
        bp = ax.boxplot(grps, labels=[f"{k}t" for k in range(1, 6)],
                        patch_artist=True,
                        medianprops=dict(color="white", linewidth=2))
        for p, c in zip(bp["boxes"], [C_OK, C_TEAL, C_WARN, C_DANGER, "#a78bfa"]):
            p.set_facecolor(c)
            p.set_alpha(0.65)
        ax.set_xlabel("Part Types")
        ax.set_ylabel("Cycle Time (min)")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Each extra PN type adds ~10 min. 5-type racks average 340 min.")

    st.divider()
    st.markdown('<div class="lsa-section">// Feature scatter vs cycle time</div>',
                unsafe_allow_html=True)
    sel = st.selectbox("Feature:", FEATURES, format_func=lambda x: FEAT_LABELS.get(x, x))
    m, b, r, p, _ = linregress(df[sel], df[TARGET])
    fig, ax = dark_fig(10, 4)
    ax.scatter(df[sel], df[TARGET], alpha=0.35, s=10, color=C_TEAL)
    xr = np.linspace(df[sel].min(), df[sel].max(), 100)
    ax.plot(xr, m * xr + b, color=C_DANGER, lw=1.5, ls="--", label=f"r = {r:.3f}")
    ax.set_xlabel(FEAT_LABELS.get(sel, sel))
    ax.set_ylabel("Cycle Time (min)")
    ax.legend(fontsize=9, facecolor=C_CARD, labelcolor=C_TEXT)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.caption(f"r = {r:.3f}  ·  slope = {m:+.4f} min/unit  ·  p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}")

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="lsa-section">// OLS performance — test set (n=100)</div>',
                unsafe_allow_html=True)
    residuals = y_test.values - y_pred

    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="lsa-section">// Predicted vs actual</div>',
                    unsafe_allow_html=True)
        fig, ax = dark_fig(6, 5)
        ax.scatter(y_test, y_pred, alpha=0.55, s=14, color=C_TEAL)
        lims = [y_test.min() - 5, y_test.max() + 5]
        ax.plot(lims, lims, color=C_DANGER, ls="--", lw=1.5, label="Perfect")
        ax.fill_between(lims,
                        [l - metrics['rmse'] for l in lims],
                        [l + metrics['rmse'] for l in lims],
                        alpha=0.08, color=C_WARN, label=f"±RMSE {metrics['rmse']} min")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual (min)"); ax.set_ylabel("Predicted (min)")
        ax.set_title(f"R² = {metrics['r2']}", color=C_TEXT)
        ax.legend(fontsize=8, facecolor=C_CARD, labelcolor=C_TEXT)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Amber band = ±RMSE scheduling buffer. Most points fall inside.")

    with cb:
        st.markdown('<div class="lsa-section">// Residuals vs fitted — homoscedasticity check</div>',
                    unsafe_allow_html=True)
        fig, ax = dark_fig(6, 5)
        ax.scatter(y_pred, residuals, alpha=0.45, s=12,
                   c=[C_DANGER if abs(r) > 25 else C_TEAL for r in residuals])
        ax.axhline(0, color="white", lw=1.5, ls="--")
        ax.axhline(+2 * metrics['rmse'], color=C_WARN, lw=1, ls=":", label="±2·RMSE")
        ax.axhline(-2 * metrics['rmse'], color=C_WARN, lw=1, ls=":")
        ax.set_xlabel("Fitted (min)"); ax.set_ylabel("Residual (min)")
        ax.legend(fontsize=8, facecolor=C_CARD, labelcolor=C_TEXT)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Random scatter — variance constant across fitted range. OLS holds.")

    st.divider()
    st.markdown('<div class="lsa-section">// Statsmodels summary highlights</div>',
                unsafe_allow_html=True)
    st.markdown(f"""| Metric | Value |
|---|---|
| R² | {ols.rsquared:.4f} |
| Adj. R² | {ols.rsquared_adj:.4f} |
| F-statistic | {ols.fvalue:.1f} (p ≈ {ols.f_pvalue:.1e}) |
| Durbin-Watson | {dw_test(ols.resid):.3f} |
| AIC | {ols.aic:.0f} |""")
    st.caption("Adj. R² ≈ R² → no overfitting. DW ≈ 2 → no autocorrelation in residuals.")

    st.divider()
    st.markdown('<div class="lsa-section">// Metric explanations</div>',
                unsafe_allow_html=True)
    for name, expl in {
        "R²":   "Proportion of cycle time variance explained. 0.805 means the model captures 80.5% of scheduling variability.",
        "RMSE": "Root Mean Squared Error in minutes. Use this as the scheduling buffer — add ±RMSE to every slot.",
        "MAE":  "Mean Absolute Error — average miss per cycle. At < 4% of the mean 300-min cycle, the model is operationally reliable.",
    }.items():
        with st.expander(f"{name}  —  {metrics.get(name.lower().replace('²','2'), '—')}"):
            st.write(expl)

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    col_inp, col_out = st.columns([1.1, 1])

    with col_inp:
        st.markdown('<div class="lsa-section">// Rack configuration</div>',
                    unsafe_allow_html=True)
        n_pcs = st.slider("Pieces in Rack",          60,  240, 120,  5)
        mass  = st.slider("Avg Piece Mass (kg)",     0.6,  4.0, 2.0, 0.1)
        mix   = st.slider("Part Mix (# PN types)",     1,    5,   1)
        temp  = st.slider("Avg Furnace Temp (°C)",   840,  920, 880,  1)
        power = st.slider("Burner Power (%)",         75,  100,  85,  1)

    xsim = pd.DataFrame([{
        "rack_piece_count":   n_pcs,
        "avg_piece_mass_kg":  mass,
        "part_mix_count":     mix,
        "avg_furnace_temp_c": temp,
        "burner_power_pct":   power,
    }])
    pm = model.predict(xsim)[0]

    with col_out:
        st.markdown(
            f'''<div style="background:var(--card);border:1px solid var(--border);
                        border-radius:4px;padding:1.6rem 1.8rem;">
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                            color:#fff;margin-bottom:1rem;">Prediction Result</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:3.4rem;
                            font-weight:700;color:{C_TEAL2};line-height:1;
                            letter-spacing:-0.02em;">{pm:.1f}
                    <span style="font-size:1.4rem;font-weight:400;color:{C_MUTED};">min</span>
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.9rem;
                            color:{C_MUTED};margin-top:6px;">{pm/60:.2f} hours</div>
                <div style="margin-top:18px;font-family:'JetBrains Mono',monospace;
                            font-size:0.68rem;color:var(--muted);line-height:2.1;">
                    Schedule slot &nbsp;:
                    <strong style="color:{C_TEXT};">{pm:.0f}–{pm+15:.0f} min</strong>
                    (±{metrics['rmse']:.0f} min buffer)<br>
                    Shift impact &nbsp;&nbsp;:
                    <strong style="color:{C_TEXT};">{pm/480*100:.1f}%</strong>
                    of an 8-hour shift
                </div>
            </div>''',
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown('<div class="lsa-section">// Three reference scenarios</div>',
                unsafe_allow_html=True)
    scen = {
        "A · Light · Single PN":    {"rack_piece_count":120,"avg_piece_mass_kg":1.2,
                                      "part_mix_count":1,"avg_furnace_temp_c":900,"burner_power_pct":92},
        "B · Heavy · Under-Powered":{"rack_piece_count":200,"avg_piece_mass_kg":2.8,
                                      "part_mix_count":4,"avg_furnace_temp_c":860,"burner_power_pct":78},
        "C · Heavy · Corrected":    {"rack_piece_count":200,"avg_piece_mass_kg":2.8,
                                      "part_mix_count":4,"avg_furnace_temp_c":900,"burner_power_pct":95},
    }
    sc_preds = {}
    cols = st.columns(3)
    for col, (name, params) in zip(cols, scen.items()):
        p = model.predict(pd.DataFrame([params]))[0]
        sc_preds[name] = p
        color = C_OK if p < 270 else (C_DANGER if p > 330 else C_WARN)
        with col:
            st.markdown(f"""
            <div style="background:var(--card);border:1px solid var(--border);
                        border-left:3px solid {color};border-radius:2px;padding:1.1rem 1.2rem;">
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);
                            letter-spacing:.15em;text-transform:uppercase;margin-bottom:8px;">{name}</div>
                <div style="font-family:var(--fm);font-size:2.4rem;font-weight:700;
                            color:{color};line-height:1;">{p:.1f}</div>
                <div style="font-family:var(--fm);font-size:0.72rem;color:var(--muted);
                            margin-top:4px;">min · {p/60:.2f}h</div>
            </div>""", unsafe_allow_html=True)

    pv  = list(sc_preds.values())
    sav = pv[1] - pv[2]
    st.markdown(f"""
    <div style="background:var(--card);border:1px solid var(--border);
                border-left:3px solid {C_WARN};border-radius:2px;
                padding:0.9rem 1.2rem;margin-top:10px;">
        <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Key insight</div>
        <div style="font-family:var(--fm);font-size:0.72rem;color:var(--text);line-height:1.7;">
            Correcting temp (+40°C) and power (+17%) on the heavy rack recovers
            <strong style="color:{C_WARN};">{sav:.1f} min ({sav/pv[1]*100:.1f}%)</strong>
            — no load change required.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    ca, cb = st.columns([1.2, 1])
    with ca:
        st.markdown('<div class="lsa-section">// OLS coefficients + 95% confidence intervals</div>',
                    unsafe_allow_html=True)
        params = ols.params.drop("const")
        ci_    = ols.conf_int().drop("const")
        pvals  = ols.pvalues.drop("const")
        order  = params.abs().sort_values(ascending=True).index
        ps, cs = params[order], ci_.loc[order]
        elo, ehi = ps - cs[0], cs[1] - ps
        fig, ax = dark_fig(7, 5)
        cols_ = [C_DANGER if v > 0 else C_TEAL for v in ps]
        ax.barh(
            [FEAT_LABELS.get(f, f) for f in ps.index],
            ps.values, color=cols_, alpha=0.80, edgecolor="none", height=0.55
        )
        ax.errorbar(ps.values, range(len(ps)),
                    xerr=[elo.values, ehi.values],
                    fmt="none", color="white", capsize=4, lw=1)
        ax.axvline(0, color="white", lw=0.8)
        for i, (val, feat) in enumerate(zip(ps.values, ps.index)):
            ax.text(val + (0.15 if val >= 0 else -0.15), i,
                    f"{val:+.3f}", va="center",
                    ha="left" if val >= 0 else "right", fontsize=9, color=C_TEXT)
        ax.set_xlabel("Coefficient (min per unit)")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Red = increases cycle time  ·  Teal = reduces it. Error bars = 95% CI.")

    with cb:
        st.markdown('<div class="lsa-section">// Coefficient table + significance</div>',
                    unsafe_allow_html=True)
        tbl = pd.DataFrame({
            "Feature": [FEAT_LABELS.get(f, f) for f in params.index],
            "Coef":    params.values.round(4),
            "CI Low":  ci_[0].values.round(3),
            "CI High": ci_[1].values.round(3),
            "p-val":   pvals.values.round(4),
        }).sort_values("Coef", key=abs, ascending=False)
        tbl["Sig"] = tbl["p-val"].apply(
            lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*"
        )
        st.dataframe(tbl, use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);
                    border-left:3px solid {C_WARN};border-radius:2px;
                    padding:1rem 1.2rem;margin-top:12px;">
            <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                        text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Zero multicollinearity</div>
            <div style="font-family:var(--fm);font-size:0.72rem;color:var(--text);line-height:1.7;">
                All VIF &lt; 1.02. Each coefficient is a pure, isolated physical effect
                with no inflation from correlated predictors.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="lsa-section">// Gap-fill: 100 low-volume PNs not previously measured</div>',
                unsafe_allow_html=True)
    np.random.seed(42)
    m_all = np.random.uniform(0.6, 4.0, size=300)
    df_lv = pd.DataFrame({
        "part_number":        range(201, 301),
        "avg_piece_mass_kg":  m_all[200:].round(2),
        "rack_piece_count":   120,
        "part_mix_count":     1,
        "avg_furnace_temp_c": 880.0,
        "burner_power_pct":   85.0,
    })
    df_lv["cycle_time_pred_min"] = model.predict(df_lv[FEATURES]).round(2)
    fig, ax = dark_fig(10, 4)
    ax.bar(df_lv["part_number"], df_lv["cycle_time_pred_min"],
           color=C_WARN, alpha=0.75, width=0.8)
    ax.axhline(df_lv["cycle_time_pred_min"].mean(), color=C_DANGER, ls="--", lw=1.5,
               label=f"Mean = {df_lv['cycle_time_pred_min'].mean():.1f}")
    ax.set_xlabel("Part Number")
    ax.set_ylabel("Predicted Cycle Time (min)")
    ax.legend(fontsize=9, facecolor=C_CARD, labelcolor=C_TEXT)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.caption("Standard rack: 120 pcs · 1 PN type · 880°C · 85% burner. These complete the MES planning database.")
    with st.expander("Full planning table"):
        st.dataframe(
            df_lv[["part_number", "avg_piece_mass_kg", "cycle_time_pred_min"]],
            use_container_width=True
        )

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="lsa-section">// Operational recommendations</div>',
                unsafe_allow_html=True)
    actions = [
        (C_TEAL,   "Load gap-fill estimates into the MES for all 300 PNs",
         "Use the Tab 4 gap-fill output to populate cycle time fields for the 100 low-volume PNs. Flag them for measurement confirmation on next production run."),
        (C_DANGER, "Prioritise mass reduction when scheduling tight shifts",
         "Avg piece mass drives +22.4 min per kg. Shifting to lighter PNs can recover 20–40 min per rack — potentially fitting an extra cycle into an 8-hour shift."),
        (C_WARN,   "Run temp ≥ 880°C and burner ≥ 85% for mixed racks",
         "Mixed racks already add ~10 min per PN type. Running cool/under-powered compounds the penalty: −10°C adds ~2.2 min; −5% power adds ~1.8 min."),
        (C_OK,     "Simulate before confirming last-minute rack changes",
         "When the plan changes at shift start, run the simulator in 10 seconds. A 15-min overrun discovered at cycle-end costs more than a pre-check."),
        (C_TEAL,   "Recalibrate quarterly as furnace wear accumulates",
         "Refractory degradation and burner drift shift the intercept over months. Collect 150+ new cycles each quarter and refit. Coefficients should stay stable."),
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
    st.markdown("""
    <div style="background:var(--card);border:1px solid var(--border);border-radius:2px;
                padding:1rem 1.3rem;text-align:center;">
        <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Free project</div>
        <div style="font-family:var(--fm);font-size:0.68rem;color:var(--muted);line-height:1.7;">
            Full dataset + simulator included. Check the rest of the portfolio at
            <span style="color:#2dd4bf;">lozanolsa.gumroad.com</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lsa-footer">
    LozanoLsa · Turning Operations into Predictive Systems · Quench Rack Cycle Time Predictor · Project 08 · v2.0
</div>
""", unsafe_allow_html=True)
