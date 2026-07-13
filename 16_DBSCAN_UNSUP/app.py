"""
app.py — CNC Machining Anomaly Detection Dashboard
LozanoLsa · Project 16 · DBSCAN Anomaly Detection · 2026

Algorithm: DBSCAN (eps=0.45, min_samples=5) + PCA + K-distance elbow
Domain: CNC Machining — Cycle-Level Anomaly Detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DBSCAN · CNC Anomaly Detection",
    page_icon="🔬",
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
DATA_PATH     = "CNC_data.csv"
DATA_PATH_ALT = "16_DBSCAN_Anomaly_CNC_Detection/CNC_data.csv"
RANDOM_STATE  = 42
EPS           = 0.45
MIN_SAMPLES   = 5

FEATURES = ["vibration_mm_s", "spindle_temp_c", "dim_deviation_um"]
FEAT_LABELS = {
    "vibration_mm_s":    "Vibration (mm/s RMS)",
    "spindle_temp_c":    "Spindle Temperature (°C)",
    "dim_deviation_um":  "Dimensional Deviation (µm)",
}

# ─── MATPLOTLIB PALETTE ───────────────────────────────────────────────────────
C_BG     = "#080c12"
C_CARD   = "#121922"
C_ORANGE = "#f97316"
C_ORANGE2= "#fb923c"
C_DANGER = "#f87171"
C_OK     = "#4ade80"
C_WARN   = "#fbbf24"
C_BLUE   = "#60a5fa"
C_TEXT   = "#c8d8f0"
C_MUTED  = "#4e6a8a"

CLUSTER_COLORS = {-1: C_DANGER, 0: C_OK, 1: C_WARN}
CLUSTER_NAMES  = {-1: "Anomaly (Noise)", 0: "Normal Operation", 1: "Borderline Cases"}

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
    st.error("CNC_data.csv not found. Place the file in the same folder as app.py.")
    st.stop()

@st.cache_resource
def fit_model(df):
    X        = df[FEATURES].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    db       = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
    labels   = db.fit_predict(X_scaled)
    pca      = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca    = pca.fit_transform(X_scaled)
    return scaler, db, labels, X_scaled, X_pca, pca

df = load_data()
scaler, db_model, labels, X_scaled, X_pca, pca = fit_model(df)
df_vis           = df.copy()
df_vis["cluster"]= labels
df_vis["status"] = [CLUSTER_NAMES.get(l, f"C{l}") for l in labels]
normal_df        = df_vis[df_vis["cluster"] == 0][FEATURES]
anomaly_df       = df_vis[df_vis["cluster"] == -1][FEATURES]
CORE_SAMPLES     = X_scaled[db_model.core_sample_indices_]
n_out            = int((labels == -1).sum())

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-header">
    <div class="lsa-project-tag">ML Project #16 · DBSCAN · Unsupervised · CNC Machining</div>
    <div class="lsa-title">No Threshold Needed — Density Decides</div>
    <div class="lsa-tagline">DBSCAN doesn't ask where the boundary is. It reads the density of the data and lets isolated points self-identify as anomalies.</div>
    <div style="margin-top:10px;">
        <span class="lsa-chip">DBSCAN</span>
        <span class="lsa-chip">eps={EPS} · min_samples={MIN_SAMPLES}</span>
        <span class="lsa-chip">3 SIGNALS</span>
        <span class="lsa-chip">{n_out} ANOMALIES · {n_out/len(df)*100:.1f}%</span>
        <span class="lsa-chip">K-DISTANCE ELBOW</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TOP KPI ROW ──────────────────────────────────────────────────────────────
mask_v = labels != -1
sil    = silhouette_score(X_scaled[mask_v], labels[mask_v]) if mask_v.sum() > 1 else 0
dbs    = davies_bouldin_score(X_scaled[mask_v], labels[mask_v]) if mask_v.sum() > 1 else 0
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Cycles",    f"{len(df):,}",               "CNC machining records")
k2.metric("Anomalies Found", f"{n_out}",                   f"{n_out/len(df)*100:.1f}% of cycles")
k3.metric("Core Points",     f"{len(db_model.core_sample_indices_)}", "Dense region members")
k4.metric("Silhouette",      f"{sil:.4f}",                 "Normal cluster separation")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "DATA EXPLORER", "EPS & SENSITIVITY", "ANOMALY CLASSIFIER", "ANOMALY PROFILE", "ACTION PLAN"
])

# ══ TAB 1 ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="lsa-section">// CNC machining cycle dataset</div>',
                unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown('<div class="lsa-section">// Dataset sample</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True, height=300)
    with col_right:
        st.markdown('<div class="lsa-section">// Descriptive statistics</div>',
                    unsafe_allow_html=True)
        st.dataframe(df[FEATURES].describe().round(2), use_container_width=True, height=300)

    st.divider()
    st.markdown('<div class="lsa-section">// KDE density map — normal operating envelope</div>',
                unsafe_allow_html=True)
    palette = {"Normal Operation": C_OK, "Anomaly (Noise)": C_DANGER, "Borderline Cases": C_WARN}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(C_BG)
    for ax in [ax1, ax2]:
        ax.set_facecolor(C_CARD)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e2d45")
        ax.tick_params(colors=C_MUTED)

    sns.kdeplot(x=df["vibration_mm_s"], y=df["spindle_temp_c"],
                fill=True, cmap="Blues", levels=10, ax=ax1)
    sns.scatterplot(data=df_vis, x="vibration_mm_s", y="spindle_temp_c",
                    hue="status", palette=palette, alpha=0.5, s=12, ax=ax1, legend=False)
    ax1.set_xlabel("Vibration (mm/s)", color=C_MUTED)
    ax1.set_ylabel("Spindle Temp (°C)", color=C_MUTED)
    ax1.set_title("Vibration vs Temperature", color=C_TEXT, fontweight="bold")

    sns.kdeplot(x=df["vibration_mm_s"], y=df["dim_deviation_um"],
                fill=True, cmap="Blues", levels=10, ax=ax2)
    sns.scatterplot(data=df_vis, x="vibration_mm_s", y="dim_deviation_um",
                    hue="status", palette=palette, alpha=0.5, s=12, ax=ax2)
    ax2.set_xlabel("Vibration (mm/s)", color=C_MUTED)
    ax2.set_ylabel("Deviation (µm)", color=C_MUTED)
    ax2.set_title("Vibration vs Deviation", color=C_TEXT, fontweight="bold")
    ax2.legend(facecolor=C_CARD, labelcolor=C_TEXT, fontsize=8)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()
    st.caption("Dense blue = normal operating envelope · Red × = anomalies · Amber = borderline.")

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="lsa-section">// Hyperparameter selection — eps=0.45 · min_samples=5</div>',
                unsafe_allow_html=True)
    st.caption("K-distance plot reveals the natural density boundary. Points above the elbow become anomalies.")

    c_kdist, c_sens = st.columns(2)
    with c_kdist:
        st.markdown('<div class="lsa-section">// K-distance plot (k=4)</div>',
                    unsafe_allow_html=True)
        nn = NearestNeighbors(n_neighbors=4)
        nn.fit(X_scaled)
        dists_nn, _ = nn.kneighbors(X_scaled)
        k_dists     = np.sort(dists_nn[:, -1])

        fig, ax = dark_fig(6.5, 4.5)
        ax.plot(k_dists, color=C_ORANGE, lw=2, label="4-distance (sorted)")
        ax.axhline(y=EPS, color=C_DANGER, ls="--", lw=1.8, label=f"eps = {EPS}")
        ax.axhspan(0, EPS, alpha=0.06, color=C_OK,     label="Dense core")
        ax.axhspan(EPS, k_dists.max() * 1.05, alpha=0.06, color=C_DANGER, label="Noise candidates")
        ax.set_xlabel("Sorted point index"); ax.set_ylabel("4th-neighbor distance")
        ax.set_title(f"Elbow at eps ≈ {EPS}", color=C_TEXT, fontweight="bold")
        ax.legend(facecolor=C_CARD, labelcolor=C_TEXT, fontsize=8)
        ax.grid(True, alpha=0.15, color="#1e2d45")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with c_sens:
        st.markdown('<div class="lsa-section">// Sensitivity — anomaly rate vs eps</div>',
                    unsafe_allow_html=True)
        eps_vals = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        out_pcts = []
        for eps in eps_vals:
            lt = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit_predict(X_scaled)
            out_pcts.append((lt == -1).sum() / len(lt) * 100)

        fig2, ax2 = dark_fig(6.5, 4.5)
        bar_colors = [C_DANGER if e == EPS else C_ORANGE for e in eps_vals]
        bars = ax2.bar([str(e) for e in eps_vals], out_pcts,
                       color=bar_colors, alpha=0.85, edgecolor="#1e2d45", lw=0.5)
        for bar, p in zip(bars, out_pcts):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f"{p:.1f}%", ha="center", va="bottom", fontsize=9, color=C_TEXT)
        ax2.set_xlabel("eps value"); ax2.set_ylabel("Anomaly Rate (%)")
        ax2.set_title(f"Sensitivity (min_samples={MIN_SAMPLES})", color=C_TEXT, fontweight="bold")
        ax2.grid(axis="y", alpha=0.15, color="#1e2d45")
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True); plt.close()

    st.divider()
    st.markdown('<div class="lsa-section">// DBSCAN cluster map — raw space + PCA</div>',
                unsafe_allow_html=True)
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
    fig3.patch.set_facecolor(C_BG)
    for ax in [ax3, ax4]:
        ax.set_facecolor(C_CARD)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e2d45")
        ax.tick_params(colors=C_MUTED)

    for lbl in sorted(set(labels)):
        mask  = labels == lbl
        color = CLUSTER_COLORS.get(lbl, C_WARN)
        name  = CLUSTER_NAMES.get(lbl, f"C{lbl}")
        alpha = 0.85 if lbl == -1 else 0.4
        size  = 35   if lbl == -1 else 12
        mkr   = "X"  if lbl == -1 else "o"
        ax3.scatter(df_vis.loc[mask, "vibration_mm_s"], df_vis.loc[mask, "spindle_temp_c"],
                    c=color, label=name, alpha=alpha, s=size, marker=mkr, edgecolors="none")
        ax4.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=color, label=name, alpha=alpha, s=size, marker=mkr, edgecolors="none")

    var_exp = pca.explained_variance_ratio_ * 100
    ax3.set_xlabel("Vibration (mm/s)", color=C_MUTED)
    ax3.set_ylabel("Spindle Temp (°C)", color=C_MUTED)
    ax3.set_title("DBSCAN — Raw Feature Space", color=C_TEXT, fontweight="bold")
    ax3.legend(facecolor=C_CARD, labelcolor=C_TEXT, fontsize=8)
    ax3.grid(True, alpha=0.1, color="#1e2d45")

    ax4.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)", color=C_MUTED)
    ax4.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)", color=C_MUTED)
    ax4.set_title("DBSCAN — PCA 2D Projection", color=C_TEXT, fontweight="bold")
    ax4.legend(facecolor=C_CARD, labelcolor=C_TEXT, fontsize=8)
    ax4.grid(True, alpha=0.1, color="#1e2d45")
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True); plt.close()

    st.divider()
    st.markdown('<div class="lsa-section">// Validation metrics</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Silhouette",      f"{sil:.4f}",  "> 0.5 = good")
    m2.metric("Davies-Bouldin",  f"{dbs:.4f}",  "< 0.5 = excellent")
    m3.metric("Anomaly Rate",    f"{n_out/len(labels)*100:.1f}%", "Within 5–10% range")
    m4.metric("Core Points",     f"{len(db_model.core_sample_indices_)}", "Dense region")

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="lsa-section">// Real-time cycle classifier</div>',
                unsafe_allow_html=True)
    st.caption("Enter sensor readings from a machining cycle. DBSCAN assesses whether it falls in the normal envelope.")

    # Result LEFT · Controls RIGHT
    col_out, col_inp = st.columns([3, 2])

    with col_inp:
        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);
                    border-left:3px solid var(--accent);border-radius:2px;
                    padding:1rem 1.2rem;margin-bottom:14px;">
            <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                        text-transform:uppercase;letter-spacing:.18em;">// Cycle parameters</div>
        </div>
        """, unsafe_allow_html=True)
        vib  = st.slider("Vibration (mm/s RMS)",         0.5,  8.0,  2.5, 0.1)
        temp = st.slider("Spindle Temperature (°C)",     35.0, 75.0, 45.0, 0.5)
        dev  = st.slider("Dimensional Deviation (µm)",    2.0, 20.0,  8.0, 0.1)

    # Classify
    X_new      = scaler.transform([[vib, temp, dev]])
    dists_new  = np.linalg.norm(CORE_SAMPLES - X_new, axis=1)
    n_nbrs     = int((dists_new <= EPS).sum())
    is_anomaly = n_nbrs < MIN_SAMPLES

    status_color = C_DANGER if is_anomaly else C_OK
    status_label = "ANOMALY DETECTED" if is_anomaly else "NORMAL OPERATION"
    badge_bg     = "#2e0f0f" if is_anomaly else "#0f2e1a"
    status_icon  = "🚨" if is_anomaly else "✅"

    z_vib  = (vib  - normal_df["vibration_mm_s"].mean()) / normal_df["vibration_mm_s"].std()
    z_temp = (temp - normal_df["spindle_temp_c"].mean())  / normal_df["spindle_temp_c"].std()
    z_dev  = (dev  - normal_df["dim_deviation_um"].mean())/ normal_df["dim_deviation_um"].std()

    with col_out:
        st.markdown(
            f'''<div style="background:var(--card);border:1px solid var(--border);
                        border-radius:4px;padding:1.6rem 1.8rem;margin-bottom:14px;">
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                            color:#fff;margin-bottom:0.8rem;">Cycle Assessment</div>
                <div style="font-size:2rem;margin-bottom:6px;">{status_icon}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.6rem;
                            font-weight:700;color:{status_color};line-height:1.2;">{status_label}</div>
                <div style="margin-top:12px;">
                    <span style="background:{badge_bg};color:{status_color};
                                 font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                                 font-weight:600;letter-spacing:.08em;
                                 padding:4px 14px;border-radius:20px;">
                        {n_nbrs} neighbors within eps={EPS} · threshold={MIN_SAMPLES}
                    </span>
                </div>
            </div>''',
            unsafe_allow_html=True
        )

        st.markdown('<div class="lsa-section">// Signal Z-scores vs normal mean</div>',
                    unsafe_allow_html=True)
        for feat_name, z in [("Vibration", z_vib), ("Temperature", z_temp), ("Deviation", z_dev)]:
            bar_color = C_DANGER if abs(z) > 2.0 else C_ORANGE
            bar_w     = min(int(abs(z) / 4 * 100), 100)
            sign      = "+" if z >= 0 else ""
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;
                            font-family:var(--fm);font-size:0.7rem;color:var(--text);margin-bottom:3px;">
                    <span>{feat_name}</span>
                    <span style="color:{bar_color};font-weight:600;">{sign}{z:.2f}σ</span>
                </div>
                <div style="background:#1e2d45;border-radius:3px;height:7px;">
                    <div style="background:{bar_color};width:{bar_w}%;height:7px;border-radius:3px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        if is_anomaly:
            zs       = [("Vibration", abs(z_vib)), ("Temperature", abs(z_temp)), ("Deviation", abs(z_dev))]
            dominant = max(zs, key=lambda x: x[1])[0]
            st.markdown(f"""
            <div style="background:var(--card);border:1px solid var(--border);
                        border-left:3px solid {C_DANGER};border-radius:2px;
                        padding:0.9rem 1.2rem;margin-top:10px;">
                <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                            text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Dominant driver</div>
                <div style="font-family:var(--fm);font-size:0.72rem;color:var(--text);line-height:1.7;">
                    <strong style="color:{C_DANGER};">{dominant}</strong> is the primary anomaly signal.
                    Halt cycle · Inspect tool and spindle · Measure last 5 parts.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.success("Cycle within normal operating envelope — continue production.")

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="lsa-section">// Anomaly characterization — tool-wear signature</div>',
                unsafe_allow_html=True)
    st.caption("Anomalous cycles show elevated vibration, spindle temperature, and dimensional deviation — simultaneously.")

    st.markdown('<div class="lsa-section">// Normal vs anomaly profile</div>',
                unsafe_allow_html=True)
    profile_df = pd.DataFrame({
        "Normal (Cluster 0)":  normal_df.mean().round(2),
        "Anomaly (Cluster -1)":anomaly_df.mean().round(2),
        "Delta (%)":           ((anomaly_df.mean() - normal_df.mean()) / normal_df.mean() * 100).round(1),
    })
    profile_df.index = [FEAT_LABELS.get(f, f) for f in profile_df.index]
    st.dataframe(profile_df, use_container_width=True)

    st.divider()
    st.markdown('<div class="lsa-section">// Distribution comparison — normal vs anomaly</div>',
                unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor(C_BG)
    for ax, feat in zip(axes, FEATURES):
        ax.set_facecolor(C_CARD)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e2d45")
        ax.hist(normal_df[feat],  bins=30, alpha=0.65, color=C_OK,     label="Normal",  density=True)
        ax.hist(anomaly_df[feat], bins=20, alpha=0.65, color=C_DANGER, label="Anomaly", density=True)
        ax.set_xlabel(FEAT_LABELS.get(feat, feat), color=C_MUTED, fontsize=9)
        ax.set_ylabel("Density", color=C_MUTED)
        ax.set_title(FEAT_LABELS.get(feat, feat), color=C_TEXT, fontsize=10, fontweight="bold")
        ax.legend(facecolor=C_CARD, labelcolor=C_TEXT, fontsize=8)
        ax.tick_params(colors=C_MUTED)
        ax.grid(axis="y", alpha=0.15, color="#1e2d45")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()
    st.caption("Anomaly distributions shift right — higher vibration, temp, and deviation. "
               "No single threshold separates them cleanly — multivariate density is required.")

    st.divider()
    st.markdown('<div class="lsa-section">// Anomaly Z-score heatmap (sample of 40 cycles)</div>',
                unsafe_allow_html=True)
    anomaly_z = (anomaly_df - normal_df.mean()) / normal_df.std()
    sample_z  = anomaly_z.sample(min(40, len(anomaly_z)), random_state=RANDOM_STATE).reset_index(drop=True)
    sample_z.columns = [FEAT_LABELS.get(f, f) for f in sample_z.columns]

    fig2, ax2 = plt.subplots(figsize=(12, 3))
    fig2.patch.set_facecolor(C_BG); ax2.set_facecolor(C_BG)
    sns.heatmap(sample_z.T, cmap="RdYlGn_r", center=0, vmin=-2, vmax=5,
                linewidths=0.2, linecolor="#1e2d45",
                cbar_kws={"label": "Z-score vs normal mean"},
                annot=False, ax=ax2)
    ax2.set_title("Z-Score Displacement per Anomalous Cycle (Red = high deviation)",
                  color=C_TEXT, fontsize=10, fontweight="bold")
    ax2.set_xticklabels([], fontsize=0)
    ax2.set_yticklabels(ax2.get_yticklabels(), color=C_TEXT, rotation=0, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True); plt.close()

    st.divider()
    st.markdown('<div class="lsa-section">// Mean Z-score displacement per signal</div>',
                unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    for col, feat, z_val in zip(
        [a1, a2, a3], FEATURES,
        [anomaly_z["vibration_mm_s"].mean(),
         anomaly_z["spindle_temp_c"].mean(),
         anomaly_z["dim_deviation_um"].mean()],
    ):
        col.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);
                    border-top:3px solid {C_DANGER};border-radius:3px;
                    padding:1rem;text-align:center;">
            <div style="font-family:var(--fm);font-size:1.4rem;font-weight:700;
                        color:{C_DANGER};">+{z_val:.2f}σ</div>
            <div style="font-family:var(--fm);font-size:0.62rem;color:var(--muted);
                        text-transform:uppercase;letter-spacing:.1em;margin-top:4px;">Mean displacement</div>
            <div style="font-family:var(--fm);font-size:0.65rem;color:var(--text);
                        margin-top:4px;">{FEAT_LABELS.get(feat, feat)}</div>
        </div>""", unsafe_allow_html=True)

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="lsa-section">// Operational response protocol</div>',
                unsafe_allow_html=True)
    st.caption("DBSCAN flags anomalies — the playbook converts flags into decisions by dominant driver.")

    protocols = [
        {"title": "NORMAL OPERATION", "color": C_OK,
         "rate": f"{(labels==0).sum()/len(labels)*100:.1f}% of cycles",
         "condition": "Vibration ~2.5 mm/s · Temp ~45°C · Deviation ~8 µm",
         "actions": ["Continue production — all signals within normal envelope.",
                     "Log cycle data to SPC system for trending.",
                     "Monitor for gradual drift — 3 consecutive cycles trending up = early warning."]},
        {"title": "ANOMALY — Dominant driver: VIBRATION", "color": C_DANGER,
         "rate": "~60% of anomalies",
         "condition": "Vibration > 4.0 mm/s | Temp and deviation also elevated",
         "actions": ["Halt cycle immediately — do not continue cutting.",
                     "Inspect tool for wear, chipping, or runout.",
                     "Check tool holder collet for loosening.",
                     "Verify spindle bearing play — replace if excessive.",
                     "Measure last 5 parts before clearing the alarm."]},
        {"title": "ANOMALY — Dominant driver: TEMPERATURE", "color": C_WARN,
         "rate": "~25% of anomalies",
         "condition": "Spindle temp > 55°C | Vibration moderate",
         "actions": ["Check coolant flow rate and inlet temperature.",
                     "Inspect coolant nozzle alignment and blockage.",
                     "Verify spindle warm-up cycle was completed.",
                     "Reduce cutting feed rate by 15% and monitor trend.",
                     "Schedule bearing inspection at next planned stoppage."]},
        {"title": "ANOMALY — Dominant driver: DEVIATION", "color": C_ORANGE,
         "rate": "~15% of anomalies",
         "condition": "Dimensional deviation > 12 µm | Vibration and temp moderate",
         "actions": ["Measure last 10 produced parts — flag for quality review.",
                     "Check workpiece clamping fixture for play or contamination.",
                     "Verify tool offset compensation is current.",
                     "Inspect guide rails for wear or contamination.",
                     "Run test cut on scrap material before resuming production."]},
    ]

    for prot in protocols:
        with st.expander(prot["title"], expanded=True):
            l, r = st.columns([3, 1])
            with l:
                st.markdown(f"""
                <div style="font-family:var(--fm);font-size:0.7rem;color:var(--muted);
                            margin-bottom:10px;line-height:1.6;">
                    <strong style="color:var(--text);">Typical pattern:</strong>
                    {prot['condition']}
                </div>""", unsafe_allow_html=True)
                for i, act in enumerate(prot["actions"], 1):
                    prefix = "🚨 " if "VIBRATION" in prot["title"] and i == 1 else f"{i}. "
                    st.markdown(f"""
                    <div style="font-family:var(--fm);font-size:0.7rem;color:var(--muted);
                                padding:3px 0;border-bottom:1px solid #1e2d4530;line-height:1.6;">
                        <span style="color:{prot['color']};font-weight:600;">{prefix}</span>{act}
                    </div>""", unsafe_allow_html=True)
            with r:
                st.markdown(f"""
                <div style="background:var(--card);border:1px solid var(--border);
                            border-top:3px solid {prot['color']};border-radius:2px;
                            padding:1rem;text-align:center;margin-top:8px;">
                    <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                                text-transform:uppercase;letter-spacing:.1em;">Frequency</div>
                    <div style="font-family:var(--fm);font-size:1.1rem;font-weight:700;
                                color:{prot['color']};margin-top:4px;">{prot['rate']}</div>
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
    LozanoLsa · Turning Operations into Predictive Systems · CNC Anomaly Detection · Project 16 · v2.0
</div>
""", unsafe_allow_html=True)
