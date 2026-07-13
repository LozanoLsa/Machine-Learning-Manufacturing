"""
app.py — Industrial Motor Health Monitoring Dashboard
LozanoLsa · Project 17 · PCA Predictive Maintenance · 2026

Algorithm: PCA (3 components) + SPE + Hotelling T² + Reliability Index
Domain: Rotating Equipment — Motor Condition Monitoring
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCA · Motor Predictive Maintenance",
    page_icon="⚙️",
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
    --accent:   #f97316;
    --accent2:  #fb923c;
    --danger:   #f87171;
    --warn:     #fbbf24;
    --ok:       #4ade80;
    --blue:     #60a5fa;
    --text:     #c8d8f0;
    --muted:    #4e6a8a;
    --fh: 'Syne', sans-serif;
    --fm: 'JetBrains Mono', monospace;
    --fs: 'Instrument Serif', Georgia, serif;
}

.stApp { background: var(--bg) !important; color: var(--text); font-family: var(--fh); }
.block-container { padding: 1.8rem 2.4rem 3rem !important; max-width: 1400px !important; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; border: 2px solid var(--accent2) !important; box-shadow: 0 0 8px rgba(249,115,22,0.5) !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { font-family: var(--fm) !important; font-size: 0.65rem !important; color: var(--accent2) !important; background: var(--card) !important; border: 1px solid var(--border) !important; padding: 1px 5px !important; border-radius: 3px !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--accent) !important; }

[data-testid="stSelectbox"] > div > div { background: var(--card) !important; border: 1px solid var(--border) !important; color: var(--text) !important; font-family: var(--fm) !important; font-size: 0.78rem !important; border-radius: 3px !important; }

[data-testid="stMetric"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-top: 2px solid var(--accent) !important; padding: 1rem 1.1rem 0.9rem !important; border-radius: 3px !important; }
[data-testid="stMetricLabel"] > div { font-family: var(--fm) !important; font-size: 0.6rem !important; text-transform: uppercase !important; letter-spacing: 0.18em !important; color: var(--muted) !important; font-weight: 400 !important; }
[data-testid="stMetricValue"] > div { font-family: var(--fm) !important; font-size: 1.7rem !important; font-weight: 600 !important; color: var(--accent2) !important; line-height: 1.1 !important; }

[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid var(--border) !important; gap: 0 !important; background: transparent !important; }
[data-testid="stTabs"] [role="tab"] { font-family: var(--fm) !important; font-size: 0.68rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; color: var(--muted) !important; padding: 0.5rem 1.2rem !important; border: none !important; border-radius: 0 !important; background: transparent !important; transition: all 0.2s !important; }
[data-testid="stTabs"] [role="tab"]:hover { color: var(--accent2) !important; background: rgba(249,115,22,0.06) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; background: transparent !important; }
[data-testid="stTabsContent"] { padding-top: 1.4rem !important; }

[data-testid="stAlert"] { border-radius: 2px !important; font-family: var(--fm) !important; font-size: 0.75rem !important; border: none !important; }
[data-testid="stExpander"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 2px !important; }
[data-testid="stExpander"] summary { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--text) !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 2px !important; }
[data-testid="stDataFrame"] th { font-family: var(--fm) !important; font-size: 0.62rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; background: var(--card2) !important; color: var(--muted) !important; }
[data-testid="stDataFrame"] td { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--text) !important; background: var(--card) !important; }

hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
[data-testid="stCaptionContainer"] p { font-family: var(--fm) !important; font-size: 0.62rem !important; color: var(--muted) !important; letter-spacing: 0.08em !important; }
h1, h2, h3 { font-family: var(--fh) !important; color: var(--text) !important; }
p, li { font-family: var(--fh) !important; font-size: 0.88rem !important; }

.lsa-header { border-bottom: 1px solid var(--border); padding-bottom: 1.2rem; margin-bottom: 0.2rem; }
.lsa-project-tag { font-family: var(--fm); font-size: 0.6rem; color: var(--accent); text-transform: uppercase; letter-spacing: 0.22em; margin-bottom: 4px; }
.lsa-title { font-family: var(--fh); font-size: 1.85rem; font-weight: 800; color: #fff; line-height: 1.1; letter-spacing: -0.02em; }
.lsa-tagline { font-family: var(--fs); font-style: italic; font-size: 0.9rem; color: var(--muted); margin-top: 4px; }
.lsa-chip { display: inline-block; background: rgba(249,115,22,0.1); border: 1px solid rgba(249,115,22,0.3); color: var(--accent2); font-family: var(--fm); font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 2px 8px; border-radius: 2px; margin-right: 5px; }
.lsa-section { font-family: var(--fm); font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid var(--border); }
.lsa-footer { margin-top: 2.5rem; padding-top: 0.8rem; border-top: 1px solid var(--border); font-family: var(--fm); font-size: 0.58rem; color: var(--muted); letter-spacing: 0.1em; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATA_PATH     = "mtto_data.csv"
DATA_PATH_ALT = "17_PCA_Predictive_Maintenance/mtto_data.csv"
RANDOM_STATE  = 42
N_COMPONENTS  = 3

FEATURES = [
    "vibration_rms_mm_s", "vibration_hf_g", "bearing_temp_c", "motor_current_a",
    "voltage_v", "power_factor_pct", "speed_rpm", "lube_flow_l_min",
    "lube_pressure_bar", "internal_humidity_pct",
]
FEAT_LABELS = {
    "vibration_rms_mm_s":   "Vibration RMS (mm/s)",
    "vibration_hf_g":       "Vibration HF (g)",
    "bearing_temp_c":       "Bearing Temp (°C)",
    "motor_current_a":      "Motor Current (A)",
    "voltage_v":            "Voltage (V)",
    "power_factor_pct":     "Power Factor (%)",
    "speed_rpm":            "Speed (rpm)",
    "lube_flow_l_min":      "Lube Flow (L/min)",
    "lube_pressure_bar":    "Lube Pressure (bar)",
    "internal_humidity_pct":"Internal Humidity (%)",
}

# ─── MATPLOTLIB PALETTE ───────────────────────────────────────────────────────
C_BG     = "#080c12"
C_CARD   = "#121922"
C_ORANGE = "#f97316"
C_ORANGE2= "#fb923c"
C_DANGER = "#f87171"
C_OK     = "#4ade80"
C_WARN   = "#fbbf24"
C_AMBER  = "#fb923c"
C_BLUE   = "#60a5fa"
C_TEXT   = "#c8d8f0"
C_MUTED  = "#4e6a8a"

STATUS_COLORS = {"Normal": C_OK, "Alert": C_WARN, "Severe": C_AMBER, "Critical": C_DANGER}

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
    for path in [DATA_PATH, DATA_PATH_ALT]:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            continue
    st.error("mtto_data.csv not found. Place the file in the same folder as app.py.")
    st.stop()

@st.cache_resource
def fit_model(df):
    X        = df[FEATURES].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca      = PCA(n_components=N_COMPONENTS)
    X_pca    = pca.fit_transform(X_scaled)
    X_recon  = pca.inverse_transform(X_pca)
    SPE      = np.sum((X_scaled - X_recon) ** 2, axis=1)
    var_comps= np.var(X_pca, axis=0)
    T2       = np.sum((X_pca ** 2) / var_comps, axis=1)
    SPE_norm = (SPE - SPE.min()) / (SPE.max() - SPE.min())
    T2_norm  = (T2  - T2.min())  / (T2.max()  - T2.min())
    IR       = 1 / (1 + SPE_norm + T2_norm)
    pca_full = PCA(n_components=10)
    X_pca_full = pca_full.fit_transform(X_scaled)
    return scaler, pca, pca_full, X_scaled, X_pca, X_pca_full, SPE, T2, IR, var_comps

df = load_data()
scaler, pca, pca_full, X_scaled, X_pca, X_pca_full, SPE, T2, IR, var_comps = fit_model(df)
df_vis        = df.copy()
df_vis["IR"]  = IR
var_3         = pca.explained_variance_ratio_
var_full      = pca_full.explained_variance_ratio_
SPE_UCL       = np.percentile(SPE, 95)
T2_UCL        = np.percentile(T2,  95)

def ir_classify(ir):
    if ir >= 0.86: return "Normal",   C_OK,     "Continue operation. Log reading."
    if ir >= 0.74: return "Alert",    C_WARN,   "Monitor closely. Inspect at next stop."
    if ir >= 0.62: return "Severe",   C_AMBER,  "Schedule inspection within 48 hours."
    return              "Critical", C_DANGER, "Halt for immediate maintenance."

def score_motor(vals):
    x    = scaler.transform([vals])
    xp   = pca.transform(x)
    xr   = pca.inverse_transform(xp)
    spe  = float(np.sum((x - xr) ** 2))
    t2   = float(np.sum((xp ** 2) / var_comps))
    sn   = (spe - SPE.min()) / (SPE.max() - SPE.min())
    tn   = (t2  - T2.min())  / (T2.max()  - T2.min())
    ir   = 1 / (1 + sn + tn)
    return xp[0, 0], xp[0, 1], xp[0, 2], spe, t2, ir

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-header">
    <div class="lsa-project-tag">ML Project #17 · PCA · Predictive Maintenance · Rotating Equipment</div>
    <div class="lsa-title">10 Sensors. 3 Components. One Reliability Index.</div>
    <div class="lsa-tagline">PCA compresses correlated sensor signals into a health axis. SPE and T² tell you how far a motor has drifted from normality.</div>
    <div style="margin-top:10px;">
        <span class="lsa-chip">PCA  k=3</span>
        <span class="lsa-chip">10 SENSORS</span>
        <span class="lsa-chip">{var_3.sum()*100:.1f}% VARIANCE</span>
        <span class="lsa-chip">SPE · T² · IR</span>
        <span class="lsa-chip">UCL 95%</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TOP KPI ROW ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Snapshots",  f"{len(df):,}",              "Motor condition readings")
k2.metric("PC1 Variance",     f"{var_3[0]*100:.1f}%",     "Dominant degradation axis")
k3.metric("3-PC Coverage",    f"{var_3.sum()*100:.1f}%",  "Sufficient for monitoring")
k4.metric("IR Normal mean",
          f"{df_vis[df_vis['op_status']=='Normal']['IR'].mean():.4f}",
          "Well within safe zone")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "DATA EXPLORER", "SCREE & PCA SPACE",
    "MOTOR CLASSIFIER", "LOADINGS & HEALTH", "ACTION PLAN"
])

# ══ TAB 1 ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="lsa-section">// Motor sensor dataset — 1,687 condition snapshots</div>',
                unsafe_allow_html=True)
    cl, cr = st.columns([3, 2])
    with cl:
        st.markdown('<div class="lsa-section">// Dataset sample</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True, height=310)
    with cr:
        st.markdown('<div class="lsa-section">// Status distribution</div>', unsafe_allow_html=True)
        vc = df["op_status"].value_counts()
        fig0, ax0 = dark_fig(4.5, 3)
        ax0.bar(vc.index, vc.values,
                color=[STATUS_COLORS.get(s, C_ORANGE) for s in vc.index],
                alpha=0.85, edgecolor="#1e2d45")
        for i, (k, v) in enumerate(vc.items()):
            ax0.text(i, v + 5, str(v), ha="center", color=C_TEXT, fontsize=10)
        ax0.set_ylabel("Count")
        ax0.tick_params(colors=C_MUTED)
        fig0.tight_layout()
        st.pyplot(fig0, use_container_width=True); plt.close()

    st.divider()
    st.markdown('<div class="lsa-section">// Sensor correlation heatmap</div>',
                unsafe_allow_html=True)
    corr = df[FEATURES].corr()
    fig_c, ax_c = plt.subplots(figsize=(9, 7))
    fig_c.patch.set_facecolor(C_BG); ax_c.set_facecolor(C_BG)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, linewidths=0.3, linecolor="#1e2d45",
                annot_kws={"size": 8, "color": C_TEXT}, ax=ax_c,
                xticklabels=[FEAT_LABELS.get(f, f) for f in FEATURES],
                yticklabels=[FEAT_LABELS.get(f, f) for f in FEATURES])
    ax_c.set_xticklabels(ax_c.get_xticklabels(), color=C_TEXT, fontsize=8, rotation=35, ha="right")
    ax_c.set_yticklabels(ax_c.get_yticklabels(), color=C_TEXT, fontsize=8, rotation=0)
    plt.tight_layout()
    st.pyplot(fig_c, use_container_width=True); plt.close()
    st.caption("Strong multi-collinearity: vibration ↔ temperature ↔ current ↔ lube. PCA compresses this into 3 uncorrelated axes.")

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="lsa-section">// PCA variance decomposition — 10-sensor dataset</div>',
                unsafe_allow_html=True)
    comps   = list(range(1, 11))
    var_cum = np.cumsum(var_full)

    c_scree, c_cum = st.columns(2)
    with c_scree:
        st.markdown('<div class="lsa-section">// Scree plot — individual variance per PC</div>',
                    unsafe_allow_html=True)
        fig_s, ax_s = dark_fig(6, 4)
        ax_s.bar(comps, var_full * 100,
                 color=[C_DANGER if c <= 3 else C_ORANGE for c in comps],
                 alpha=0.85, edgecolor="#1e2d45")
        ax_s.plot(comps, var_full * 100, "o-", color="white", lw=1.8, ms=6)
        ax_s.axvline(x=3.5, color=C_WARN, ls="--", lw=1.5, label="Retained (k=3)")
        for i, v in enumerate(var_full * 100):
            ax_s.text(i + 1, v + 0.3, f"{v:.1f}%", ha="center", fontsize=7.5, color=C_TEXT)
        ax_s.set_xlabel("PC"); ax_s.set_ylabel("Variance (%)")
        ax_s.set_xticks(comps)
        ax_s.legend(facecolor=C_CARD, labelcolor=C_TEXT, fontsize=8)
        ax_s.grid(axis="y", alpha=0.15, color="#1e2d45")
        fig_s.tight_layout()
        st.pyplot(fig_s, use_container_width=True); plt.close()

    with c_cum:
        st.markdown('<div class="lsa-section">// Cumulative variance — how many PCs to keep?</div>',
                    unsafe_allow_html=True)
        fig_cv, ax_cv = dark_fig(6, 4)
        ax_cv.plot(comps, var_cum * 100, "o-", color=C_ORANGE, lw=2.5, ms=8)
        for thresh, color, lbl in [(75, C_WARN, "75%"), (90, C_OK, "90%"), (95, C_DANGER, "95%")]:
            ax_cv.axhline(y=thresh, color=color, ls="--", lw=1.3, alpha=0.8, label=lbl)
        ax_cv.axvline(x=3, color=C_WARN, ls="--", lw=1.5)
        for i, v in enumerate(var_cum * 100):
            ax_cv.text(i + 1, v + 0.8, f"{v:.0f}%", ha="center", fontsize=7.5, color=C_TEXT)
        ax_cv.set_xlabel("Components"); ax_cv.set_ylabel("Cumulative Variance (%)")
        ax_cv.set_ylim(50, 107); ax_cv.set_xticks(comps)
        ax_cv.legend(facecolor=C_CARD, labelcolor=C_TEXT, fontsize=8, loc="lower right")
        ax_cv.grid(True, alpha=0.1, color="#1e2d45")
        fig_cv.tight_layout()
        st.pyplot(fig_cv, use_container_width=True); plt.close()

    st.divider()
    st.markdown('<div class="lsa-section">// PC1–PC2 motor health map — colored by reliability index</div>',
                unsafe_allow_html=True)
    fig_map, ax_map = dark_fig(10, 5.5)
    sc_map = ax_map.scatter(X_pca[:, 0], X_pca[:, 1], c=IR, cmap="RdYlGn",
                            vmin=0.5, vmax=1.0, alpha=0.6, s=14, edgecolors="none")
    cbar = plt.colorbar(sc_map, ax=ax_map)
    cbar.set_label("Reliability Index (IR)", color=C_MUTED)
    cbar.ax.yaxis.set_tick_params(color=C_MUTED)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=C_MUTED)
    ax_map.axhline(0, color="#1e2d45", lw=0.8, ls="--")
    ax_map.axvline(0, color="#1e2d45", lw=0.8, ls="--")
    ax_map.set_xlabel(f"PC1 — Mechanical Degradation ({var_3[0]*100:.1f}%)")
    ax_map.set_ylabel(f"PC2 — Electrical Supply ({var_3[1]*100:.1f}%)")
    ax_map.set_title("IR Heat Map on PC1–PC2 Health Space  (Green = healthy · Red = degraded)",
                     color=C_TEXT, fontsize=11, fontweight="bold")
    ax_map.grid(True, alpha=0.1, color="#1e2d45")
    fig_map.tight_layout()
    st.pyplot(fig_map, use_container_width=True); plt.close()
    st.caption("PC1 = primary health axis. Motors drift rightward as they degrade. "
               "PC2 (voltage) is orthogonal to mechanical health.")

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PC1 Variance",     f"{var_3[0]*100:.1f}%",  "Single dominant axis")
    m2.metric("3-PC Coverage",    f"{var_3.sum()*100:.1f}%","Sufficient for monitoring")
    m3.metric("IR Normal mean",
              f"{df_vis[df_vis['op_status']=='Normal']['IR'].mean():.4f}",   "Well within safe zone")
    m4.metric("IR Critical mean",
              f"{df_vis[df_vis['op_status']=='Critical']['IR'].mean():.4f}", "Correctly flagged")

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="lsa-section">// Real-time motor health classifier</div>',
                unsafe_allow_html=True)
    st.caption("Enter current sensor readings to compute the Reliability Index (IR) and recommended action.")

    # Result LEFT · Controls RIGHT
    col_out, col_inp = st.columns([3, 2])

    with col_inp:
        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);
                    border-left:3px solid var(--accent);border-radius:2px;
                    padding:1rem 1.2rem;margin-bottom:14px;">
            <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                        text-transform:uppercase;letter-spacing:.18em;">// Sensor inputs</div>
        </div>
        """, unsafe_allow_html=True)
        vib_rms = st.slider("Vibration RMS (mm/s)",   0.1, 12.0,  2.0, 0.1)
        vib_hf  = st.slider("Vibration HF (g)",       0.01,  4.0,  0.5, 0.05)
        temp    = st.slider("Bearing Temp (°C)",      30.0, 130.0, 65.0, 0.5)
        current = st.slider("Motor Current (A)",       0.5,  45.0, 18.0, 0.5)
        voltage = st.slider("Voltage (V)",           430.0, 490.0,460.0, 0.5)
        pf      = st.slider("Power Factor (%)",       70.0, 100.0, 95.0, 0.5)
        rpm     = st.slider("Speed (rpm)",           500.0,1600.0,1480.0, 5.0)
        lf      = st.slider("Lube Flow (L/min)",       5.0,  50.0, 35.0, 0.5)
        lp      = st.slider("Lube Pressure (bar)",     0.1,   5.5,  3.5, 0.1)
        hum     = st.slider("Internal Humidity (%)",   5.0,  95.0, 25.0, 1.0)

    vals = [vib_rms, vib_hf, temp, current, voltage, pf, rpm, lf, lp, hum]
    pc1, pc2, pc3, spe, t2, ir = score_motor(vals)
    status, s_color, action    = ir_classify(ir)
    badge_bg = {"Normal": "#0f2e1a", "Alert": "#2e2a0a",
                "Severe": "#2e1f0a", "Critical": "#2e0f0f"}.get(status, "#121922")

    with col_out:
        ir_pct = int(ir * 100)
        st.markdown(
            f'''<div style="background:var(--card);border:1px solid var(--border);
                        border-radius:4px;padding:1.6rem 1.8rem;margin-bottom:14px;">
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                            color:#fff;margin-bottom:0.8rem;">Motor Health Assessment</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:3rem;
                            font-weight:700;color:{s_color};line-height:1;">IR = {ir:.4f}</div>
                <div style="margin-top:12px;">
                    <span style="background:{badge_bg};color:{s_color};
                                 font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                                 font-weight:600;letter-spacing:.08em;
                                 padding:5px 16px;border-radius:20px;">
                        MOTOR STATUS: {status.upper()}
                    </span>
                </div>
                <div style="margin-top:14px;font-family:var(--fm);font-size:0.7rem;
                            color:var(--muted);line-height:1.8;">{action}</div>
            </div>''',
            unsafe_allow_html=True
        )

        # IR gauge bar
        st.markdown('<div class="lsa-section">// Reliability index gauge</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style="margin-bottom:14px;">
            <div style="display:flex;justify-content:space-between;
                        font-family:var(--fm);font-size:0.68rem;color:var(--text);margin-bottom:4px;">
                <span>Reliability Index</span>
                <span style="color:{s_color};font-weight:600;">{ir:.4f}</span>
            </div>
            <div style="background:#1e2d45;border-radius:4px;height:14px;">
                <div style="background:{s_color};width:{ir_pct}%;height:14px;border-radius:4px;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;
                        font-family:var(--fm);font-size:0.58rem;color:var(--muted);margin-top:3px;">
                <span>0 — Critical</span><span>0.62</span><span>0.74</span>
                <span>0.86</span><span>1.0 — Optimal</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="lsa-section">// PCA health coordinates</div>',
                    unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        for col, name, val in zip([m1, m2, m3], ["PC1", "PC2", "PC3"], [pc1, pc2, pc3]):
            col.metric(name, f"{val:+.3f}")

        st.markdown('<div class="lsa-section">// Statistical process control</div>',
                    unsafe_allow_html=True)
        for name, val, ucl in [("SPE", spe, SPE_UCL), ("T²", t2, T2_UCL)]:
            alarm     = val > ucl
            bar_color = C_DANGER if alarm else C_ORANGE
            bar_w     = min(int(val / (ucl * 2) * 100), 100)
            flag      = "⚠️ Above UCL" if alarm else "✅ Within UCL"
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;
                            font-family:var(--fm);font-size:0.7rem;color:var(--text);margin-bottom:3px;">
                    <span>{name}: {val:.4f}</span>
                    <span style="color:{bar_color};">{flag} (UCL={ucl:.3f})</span>
                </div>
                <div style="background:#1e2d45;border-radius:3px;height:7px;">
                    <div style="background:{bar_color};width:{bar_w}%;height:7px;border-radius:3px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="lsa-section">// PCA loadings — what each sensor contributes</div>',
                unsafe_allow_html=True)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1", "PC2", "PC3"],
        index=[FEAT_LABELS.get(f, f) for f in FEATURES]
    ).round(3)
    st.dataframe(loadings, use_container_width=True)
    st.caption("PC1 = Mechanical Degradation (vibration + temp + current − lube). "
               "PC2 = Voltage supply. PC3 = Speed axis.")

    st.divider()
    st.markdown('<div class="lsa-section">// Loading heatmap</div>', unsafe_allow_html=True)
    fig_load, ax_load = plt.subplots(figsize=(7, 5.5))
    fig_load.patch.set_facecolor(C_BG); ax_load.set_facecolor(C_BG)
    sns.heatmap(loadings, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, linewidths=0.4, linecolor="#1e2d45",
                annot_kws={"size": 9, "color": C_TEXT},
                cbar_kws={"label": "Loading"}, ax=ax_load)
    ax_load.set_xticklabels(ax_load.get_xticklabels(), color=C_TEXT, fontsize=10)
    ax_load.set_yticklabels(ax_load.get_yticklabels(), color=C_TEXT, fontsize=9, rotation=0)
    ax_load.set_title("Sensor Contributions to PC1–PC3", color=C_TEXT, fontsize=12, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_load, use_container_width=True); plt.close()

    st.divider()
    st.markdown('<div class="lsa-section">// SPE & T² control charts</div>',
                unsafe_allow_html=True)
    idx          = np.arange(len(df))
    colors_plot  = [STATUS_COLORS.get(s, C_ORANGE) for s in df["op_status"]]

    fig_ctrl, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.5))
    fig_ctrl.patch.set_facecolor(C_BG)
    for ax in [ax1, ax2]:
        ax.set_facecolor(C_CARD)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e2d45")
        ax.tick_params(colors=C_MUTED)

    ax1.scatter(idx, SPE, c=colors_plot, alpha=0.4, s=6, edgecolors="none")
    ax1.axhline(SPE_UCL, color=C_DANGER, lw=1.8, ls="--",
                label=f"UCL 95% = {SPE_UCL:.2f}")
    ax1.set_ylabel("SPE", color=C_MUTED)
    ax1.set_title("Squared Prediction Error Chart", color=C_TEXT, fontsize=11, fontweight="bold")
    ax1.legend(facecolor=C_CARD, labelcolor=C_TEXT, fontsize=9)
    ax1.grid(True, alpha=0.1, color="#1e2d45")
    ax1.xaxis.label.set_color(C_MUTED); ax1.yaxis.label.set_color(C_MUTED)

    ax2.scatter(idx, T2, c=colors_plot, alpha=0.4, s=6, edgecolors="none")
    ax2.axhline(T2_UCL, color=C_DANGER, lw=1.8, ls="--",
                label=f"UCL 95% = {T2_UCL:.2f}")
    ax2.set_xlabel("Observation", color=C_MUTED)
    ax2.set_ylabel("T²", color=C_MUTED)
    ax2.set_title("Hotelling T² Control Chart", color=C_TEXT, fontsize=11, fontweight="bold")
    ax2.legend(facecolor=C_CARD, labelcolor=C_TEXT, fontsize=9)
    ax2.grid(True, alpha=0.1, color="#1e2d45")
    ax2.xaxis.label.set_color(C_MUTED); ax2.yaxis.label.set_color(C_MUTED)

    patches = [mpatches.Patch(color=c, label=s) for s, c in STATUS_COLORS.items()]
    fig_ctrl.legend(handles=patches, loc="lower right",
                    facecolor=C_CARD, labelcolor=C_TEXT, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_ctrl, use_container_width=True); plt.close()

    st.divider()
    st.markdown('<div class="lsa-section">// IR distribution by operating status</div>',
                unsafe_allow_html=True)
    fig_ir, ax_ir = dark_fig(10, 4)
    for status, color in STATUS_COLORS.items():
        sub = df_vis[df_vis["op_status"] == status]["IR"]
        if len(sub) > 0:
            ax_ir.hist(sub, bins=35, alpha=0.65, color=color,
                       label=status, density=True, edgecolor="none")
    for thresh, color in [(0.86, C_OK), (0.74, C_WARN), (0.62, C_AMBER)]:
        ax_ir.axvline(x=thresh, color=color, ls="--", lw=1.8, alpha=0.9)
    ax_ir.set_xlabel("Reliability Index (IR)")
    ax_ir.set_ylabel("Density")
    ax_ir.legend(facecolor=C_CARD, labelcolor=C_TEXT, fontsize=9)
    ax_ir.grid(axis="y", alpha=0.15, color="#1e2d45")
    fig_ir.tight_layout()
    st.pyplot(fig_ir, use_container_width=True); plt.close()
    st.caption("Vertical lines = IR thresholds. Critical motors cluster at low IR (≤0.62). "
               "Normal motors cluster near IR=0.84–0.97.")

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="lsa-section">// Predictive maintenance response protocol</div>',
                unsafe_allow_html=True)
    st.caption("The Reliability Index (IR) translates PCA statistics into actionable maintenance decisions.")

    protocols = [
        {"title": "IR ≥ 0.86 — Normal Operation", "color": C_OK,
         "freq": "~59% of snapshots",
         "sig":  "Vibration ~2 mm/s · Temp ~65°C · Lube flow ~35 L/min",
         "actions": ["Continue operation. No intervention required.",
                     "Log IR trend in CMMS for baseline tracking.",
                     "Next inspection per standard PM schedule."]},
        {"title": "IR 0.74–0.86 — Alert", "color": C_WARN,
         "freq": "~30% of snapshots",
         "sig":  "Vibration 3–5 mm/s · Temp 75–85°C · Lube flow 28–32 L/min",
         "actions": ["Monitor at increased frequency (every 4 hours).",
                     "Verify lubrication oil level and quality.",
                     "Check bearing alignment and coupling condition.",
                     "Schedule detailed inspection at next planned stoppage."]},
        {"title": "IR 0.62–0.74 — Severe", "color": C_AMBER,
         "freq": "~7% of snapshots",
         "sig":  "Vibration 5–7 mm/s · Temp 85–95°C · Lube pressure dropping",
         "actions": ["Schedule inspection within 24–48 hours.",
                     "Prepare spare bearings and seal kits.",
                     "Verify current draw — check for phase imbalance.",
                     "Review oil sample analysis (contamination / viscosity).",
                     "Alert maintenance supervisor."]},
        {"title": "IR < 0.62 — Critical", "color": C_DANGER,
         "freq": "~4% of snapshots",
         "sig":  "Vibration > 7 mm/s · Temp > 95°C · Lubrication severely degraded",
         "actions": ["HALT MOTOR — immediate maintenance required.",
                     "Do not restart until root cause is identified.",
                     "Replace bearings and inspect shaft for scoring.",
                     "Check motor winding resistance — risk of coil failure.",
                     "Verify cooling system integrity.",
                     "Document event in PSM system and update FMEA."]},
    ]

    for p in protocols:
        with st.expander(p["title"], expanded=True):
            l, r = st.columns([3, 1])
            with l:
                st.markdown(f"""
                <div style="font-family:var(--fm);font-size:0.7rem;color:var(--muted);
                            margin-bottom:10px;">
                    <strong style="color:var(--text);">Typical signature:</strong> {p['sig']}
                </div>""", unsafe_allow_html=True)
                for i, a in enumerate(p["actions"], 1):
                    prefix = "🚨 " if "Critical" in p["title"] and i == 1 else f"{i}. "
                    st.markdown(f"""
                    <div style="font-family:var(--fm);font-size:0.7rem;color:var(--muted);
                                padding:3px 0;border-bottom:1px solid #1e2d4530;line-height:1.6;">
                        <span style="color:{p['color']};font-weight:600;">{prefix}</span>{a}
                    </div>""", unsafe_allow_html=True)
            with r:
                st.markdown(f"""
                <div style="background:var(--card);border:1px solid var(--border);
                            border-top:3px solid {p['color']};border-radius:2px;
                            padding:1rem;text-align:center;margin-top:8px;">
                    <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                                text-transform:uppercase;letter-spacing:.1em;">Frequency</div>
                    <div style="font-family:var(--fm);font-size:1.1rem;font-weight:700;
                                color:{p['color']};margin-top:4px;">{p['freq']}</div>
                </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="background:var(--card);border:1px solid var(--border);border-radius:2px;
                padding:1rem 1.3rem;text-align:center;">
        <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Full project pack</div>
        <div style="font-family:var(--fm);font-size:0.68rem;color:var(--muted);line-height:1.7;">
            Complete dataset · notebook · presentation deck · simulator
            available on <span style="color:#f97316;">lozanolsa.gumroad.com</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lsa-footer">
    LozanoLsa · Turning Operations into Predictive Systems · Motor Predictive Maintenance · Project 17 · v2.0
</div>
""", unsafe_allow_html=True)
