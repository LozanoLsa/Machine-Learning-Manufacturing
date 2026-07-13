"""
app.py — Press Anomaly Intelligence Dashboard
LozanoLsa · Project 21 · Isolation Forest · 2026 · FREE PROJECT

Algorithm: Isolation Forest (n=300, contamination=0.03, bootstrap=True)
Domain: Hot Stamping Line — Multi-Type Press Anomaly Detection
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Isolation Forest · Press Anomaly Intelligence",
    page_icon="🏭",
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
    --purp:     #a78bfa;
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
.lsa-chip-free { display: inline-block; background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.3); color: #4ade80; font-family: var(--fm); font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 2px 8px; border-radius: 2px; margin-right: 5px; }
.lsa-section { font-family: var(--fm); font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid var(--border); }
.lsa-footer { margin-top: 2.5rem; padding-top: 0.8rem; border-top: 1px solid var(--border); font-family: var(--fm); font-size: 0.58rem; color: var(--muted); letter-spacing: 0.1em; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ─── MATPLOTLIB PALETTE ───────────────────────────────────────────────────────
C_BG     = "#080c12"
C_CARD   = "#121922"
C_ORANGE = "#f97316"
C_ORANGE2= "#fb923c"
C_DANGER = "#f87171"
C_OK     = "#4ade80"
C_WARN   = "#fbbf24"
C_BLUE   = "#60a5fa"
C_PURP   = "#a78bfa"
C_TEXT   = "#c8d8f0"
C_MUTED  = "#4e6a8a"

plt.rcParams.update({
    "figure.facecolor": C_BG, "axes.facecolor": C_CARD,
    "axes.edgecolor": "#1e2d45", "axes.labelcolor": C_MUTED,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED,
    "text.color": C_TEXT, "grid.color": "#1e2d45",
    "grid.linestyle": "--", "grid.alpha": 0.4,
    "legend.facecolor": C_CARD, "legend.edgecolor": "#1e2d45",
})

SEV_COLORS = {"Normal": C_OK, "Mild": C_BLUE, "Moderate": C_WARN, "Critical": C_DANGER}

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

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATA_PATH     = "Data.csv"
DATA_PATH_ALT = "21_IsoForest_Press_Anomaly/Data.csv"

FEATURES = ["tool_temp_c", "part_temp_c", "vibration_x_mm_s", "vibration_y_mm_s",
            "press_force_ton", "contact_force_kn", "cycle_time_s", "energy_kwh",
            "lubricant_flow_lmin"]
LABELS   = ["Tool Temp (°C)", "Part Temp (°C)", "Vibration X", "Vibration Y",
            "Press Force (ton)", "Contact Force (kN)", "Cycle Time (s)",
            "Energy (kWh)", "Lube Flow (L/min)"]

# ─── DATA & MODEL ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    for path in [DATA_PATH, DATA_PATH_ALT]:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            continue
    st.error("Data.csv not found. Place the file in the same folder as app.py.")
    st.stop()

@st.cache_resource
def train_model(df):
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])
    iso      = IsolationForest(n_estimators=300, contamination=0.03,
                                bootstrap=True, random_state=42)
    iso.fit(X_scaled)
    scores = iso.score_samples(X_scaled)
    pred   = (iso.predict(X_scaled) == -1).astype(int)
    return scaler, iso, scores, pred

df                  = load_data()
scaler, iso, scores, pred_anom = train_model(df)
df["score_if"]      = scores
df["pred_anomaly"]  = pred_anom

decision_thr = float(np.quantile(scores, 0.03))
q05          = float(np.quantile(scores, 0.05))
q10          = float(np.quantile(scores, 0.10))

def severity(score, flag):
    if flag == 0:         return "Normal"
    if score <= q05:      return "Critical"
    if score <= q10:      return "Moderate"
    return                       "Mild"

df["severity"] = df.apply(lambda r: severity(r["score_if"], r["pred_anomaly"]), axis=1)
y_true = df["is_anomaly"]
y_pred = df["pred_anomaly"]
cm     = confusion_matrix(y_true, y_pred)
rep    = classification_report(y_true, y_pred, output_dict=True)
tn, fp, fn, tp = cm.ravel()

means_all = df[FEATURES].mean()
stds_all  = df[FEATURES].std()

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-header">
    <div class="lsa-project-tag">ML Project #21 · Isolation Forest · Hot Stamping Line · Anomaly Detection</div>
    <div class="lsa-title">The Press Knows When Something Is Wrong</div>
    <div class="lsa-tagline">Isolation Forest isolates anomalies by path length — the rarer the pattern, the shorter the tree. No labels needed.</div>
    <div style="margin-top:10px;">
        <span class="lsa-chip">ISOLATION FOREST</span>
        <span class="lsa-chip">9 SENSORS</span>
        <span class="lsa-chip">{pred_anom.sum():,} FLAGGED · {pred_anom.mean()*100:.1f}%</span>
        <span class="lsa-chip">CONTAMINATION 3%</span>
        <span class="lsa-chip">CRITICAL · MODERATE · MILD</span>
        <span class="lsa-chip-free">FREE PROJECT</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TOP KPI ROW ──────────────────────────────────────────────────────────────
prec = rep["1"]["precision"]; rec = rep["1"]["recall"]; f1 = rep["1"]["f1-score"]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Cycles",   f"{len(df):,}",               "Stamping records")
k2.metric("IF Detections",  f"{pred_anom.sum():,}",       f"{pred_anom.mean()*100:.1f}% · contamination=0.03")
k3.metric("Recall (GT)",    f"{rec:.4f}",                 "Anomalies caught vs ground truth")
k4.metric("Severity Tiers", "Critical · Moderate · Mild", "By IF score quantiles")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "DATA EXPLORER", "MODEL PERFORMANCE", "CYCLE CLASSIFIER", "RISK DRIVERS", "ACTION PLAN"
])

# ══ TAB 1 ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="lsa-section">// Hot stamping press — 10,312 cycle dataset</div>',
                unsafe_allow_html=True)
    cl, cr = st.columns([1.3, 1])
    with cl:
        st.markdown('<div class="lsa-section">// Sensor distributions — normal vs anomaly (GT)</div>',
                    unsafe_allow_html=True)
        fig, axes = plt.subplots(3, 3, figsize=(10, 7))
        fig.patch.set_facecolor(C_BG)
        for ax, feat, lbl in zip(axes.flat, FEATURES, LABELS):
            ax.set_facecolor(C_CARD)
            for sp in ax.spines.values(): sp.set_edgecolor("#1e2d45")
            ax.hist(df.loc[df["is_anomaly"] == 0, feat], bins=30,
                    color=C_BLUE,   alpha=0.65, density=True, label="Normal")
            ax.hist(df.loc[df["is_anomaly"] == 1, feat], bins=15,
                    color=C_DANGER, alpha=0.85, density=True, label="Anomaly")
            ax.set_title(lbl, fontsize=8, fontweight="bold")
            ax.legend(fontsize=6)
            ax.tick_params(colors=C_MUTED, labelsize=7)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()
        st.caption("Blue = normal · Red = anomaly. Overlapping distributions require multivariate detection — single threshold fails.")

    with cr:
        st.markdown('<div class="lsa-section">// Sensor statistics</div>', unsafe_allow_html=True)
        desc       = df[FEATURES].describe().T.round(2)
        desc.index = LABELS
        st.dataframe(desc[["mean", "std", "min", "max"]], use_container_width=True)

        st.markdown('<div class="lsa-section" style="margin-top:14px;">// Severity distribution</div>',
                    unsafe_allow_html=True)
        sev_counts = df["severity"].value_counts()
        fig2, ax2  = dark_fig(5, 3)
        colors2    = [SEV_COLORS.get(s, C_ORANGE) for s in sev_counts.index]
        bars = ax2.barh(sev_counts.index, sev_counts.values,
                        color=colors2, edgecolor=C_BG, height=0.5, alpha=0.85)
        for bar, v in zip(bars, sev_counts.values):
            ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                     f"{v:,}", va="center", fontsize=9, color=C_TEXT)
        ax2.set_xlabel("Cycles")
        ax2.grid(True, axis="x", alpha=0.3)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True); plt.close()

    st.divider()
    st.markdown('<div class="lsa-section">// IF score time-series — full production run</div>',
                unsafe_allow_html=True)
    fig3, ax3 = dark_fig(13, 3)
    idx = np.arange(0, len(df), 5)
    ax3.plot(idx, df["score_if"].iloc[idx], lw=0.6, color=C_BLUE, alpha=0.7)
    ax3.axhline(decision_thr, color=C_DANGER, lw=1.8, ls="--",
                label=f"Decision thr = {decision_thr:.4f}")
    ax3.axhline(q05, color=C_WARN, lw=1.2, ls=":", label=f"Critical q05 = {q05:.4f}")
    ax3.fill_between(idx,
                     df["score_if"].iloc[idx].clip(upper=decision_thr),
                     decision_thr,
                     alpha=0.25, color=C_DANGER, label="Anomaly zone")
    ax3.set_xlabel("Cycle Index"); ax3.set_ylabel("IF Score")
    ax3.set_title("Isolation Forest Score — lower = more anomalous", color=C_TEXT, fontweight="bold")
    ax3.legend(facecolor=C_CARD, labelcolor=C_TEXT, fontsize=8)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True); plt.close()
    st.caption("Score below the red dashed line = anomaly. The lower the score, the more isolated the point in the forest.")

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="lsa-section">// Isolation Forest performance vs ground truth labels</div>',
                unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  f"{rep['accuracy']:.4f}")
    m2.metric("Precision", f"{prec:.4f}")
    m3.metric("Recall",    f"{rec:.4f}")
    m4.metric("F1",        f"{f1:.4f}")
    m5.metric("Contamination", "0.03")

    st.divider()
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="lsa-section">// Confusion matrix</div>', unsafe_allow_html=True)
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5))
        fig_cm.patch.set_facecolor(C_BG); ax_cm.set_facecolor(C_BG)
        sns.heatmap(cm, annot=True, fmt="d",
                    cmap=sns.light_palette(C_ORANGE, as_cmap=True),
                    xticklabels=["Pred Normal", "Pred Anomaly"],
                    yticklabels=["True Normal", "True Anomaly"],
                    linewidths=0.5, linecolor="#1e2d45",
                    annot_kws={"size": 14, "weight": "bold", "color": C_TEXT},
                    cbar=False, ax=ax_cm)
        ax_cm.tick_params(colors=C_MUTED, labelsize=9)
        fig_cm.tight_layout()
        st.pyplot(fig_cm, use_container_width=True); plt.close()
        st.caption(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    with cb:
        st.markdown('<div class="lsa-section">// IF score distribution with severity zones</div>',
                    unsafe_allow_html=True)
        fig_sc, ax_sc = dark_fig(6, 4.5)
        ax_sc.hist(scores, bins=60, color=C_ORANGE, alpha=0.80, edgecolor="none", density=True)
        ax_sc.axvline(decision_thr, color=C_DANGER, lw=2, ls="--",
                      label=f"Decision q03 = {decision_thr:.4f}")
        ax_sc.axvline(q05, color=C_WARN, lw=1.5, ls=":",
                      label=f"Critical q05 = {q05:.4f}")
        ax_sc.axvline(q10, color=C_BLUE, lw=1.5, ls=":",
                      label=f"Moderate q10 = {q10:.4f}")
        ax_sc.axvspan(scores.min(), decision_thr, alpha=0.10, color=C_DANGER, label="Anomaly zone")
        ax_sc.set_xlabel("IF Score"); ax_sc.set_ylabel("Density")
        ax_sc.set_title("Score Distribution — left tail = anomalies", color=C_TEXT, fontweight="bold")
        ax_sc.legend(facecolor=C_CARD, labelcolor=C_TEXT, fontsize=8)
        ax_sc.grid(True, axis="y", alpha=0.3)
        fig_sc.tight_layout()
        st.pyplot(fig_sc, use_container_width=True); plt.close()
        st.caption("The left tail is where anomalies live. Severity is determined by how far left the score falls.")

    st.divider()
    st.markdown('<div class="lsa-section">// Contamination sensitivity — how the flag rate changes</div>',
                unsafe_allow_html=True)
    cont_vals = [0.01, 0.02, 0.03, 0.04, 0.05]
    cont_rows = []
    X_sc2 = StandardScaler().fit_transform(df[FEATURES])
    for c in cont_vals:
        iso_c = IsolationForest(n_estimators=100, contamination=c, random_state=42)
        prd_c = (iso_c.fit_predict(X_sc2) == -1).astype(int)
        cm_c  = confusion_matrix(y_true, prd_c)
        tn_c, fp_c, fn_c, tp_c = cm_c.ravel()
        rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
        cont_rows.append({"Contamination": c, "Flagged": int(prd_c.sum()),
                           "Recall": f"{rec_c:.4f}", "FP": fp_c})
    st.dataframe(pd.DataFrame(cont_rows), use_container_width=True, hide_index=True)
    st.caption("contamination=0.03 matches the known 3% anomaly rate and maximises recall without excessive false positives.")

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="lsa-section">// Real-time cycle classifier</div>',
                unsafe_allow_html=True)
    st.caption("Enter sensor readings from a stamping cycle. Isolation Forest scores it and assigns severity.")

    col_out, col_inp = st.columns([3, 2])

    with col_inp:
        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);
                    border-left:3px solid var(--accent);border-radius:2px;
                    padding:1rem 1.2rem;margin-bottom:14px;">
            <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                        text-transform:uppercase;letter-spacing:.18em;">// Cycle sensor readings</div>
        </div>
        """, unsafe_allow_html=True)
        tool_t  = st.slider("Tool Temp (°C)",         300.0, 520.0,
                            float(means_all["tool_temp_c"]),          1.0)
        part_t  = st.slider("Part Temp (°C)",          600.0, 950.0,
                            float(means_all["part_temp_c"]),          1.0)
        vib_x   = st.slider("Vibration X (mm/s)",       0.5,   8.0,
                            float(means_all["vibration_x_mm_s"]),    0.1)
        vib_y   = st.slider("Vibration Y (mm/s)",       0.5,   8.0,
                            float(means_all["vibration_y_mm_s"]),    0.1)
        press   = st.slider("Press Force (ton)",        150.0, 450.0,
                            float(means_all["press_force_ton"]),      1.0)
        contact = st.slider("Contact Force (kN)",       100.0, 350.0,
                            float(means_all["contact_force_kn"]),     1.0)
        cycle_t = st.slider("Cycle Time (s)",             8.0,  22.0,
                            float(means_all["cycle_time_s"]),        0.1)
        energy  = st.slider("Energy (kWh)",               5.0,  35.0,
                            float(means_all["energy_kwh"]),          0.1)
        lube    = st.slider("Lube Flow (L/min)",          0.5,   5.5,
                            float(means_all["lubricant_flow_lmin"]), 0.05)

    vals    = [tool_t, part_t, vib_x, vib_y, press, contact, cycle_t, energy, lube]
    X_in    = scaler.transform([vals])
    score_in= float(iso.score_samples(X_in)[0])
    flag_in = int(iso.predict(X_in)[0] == -1)
    sev_in  = severity(score_in, flag_in)
    sev_col = SEV_COLORS.get(sev_in, C_ORANGE)
    badge_bg= {"Normal": "#0f2e1a", "Mild": "#0f1e2e",
               "Moderate": "#2e2a0a", "Critical": "#2e0f0f"}.get(sev_in, "#121922")

    with col_out:
        st.markdown(
            f'''<div style="background:var(--card);border:1px solid var(--border);
                        border-radius:4px;padding:1.6rem 1.8rem;margin-bottom:14px;">
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                            color:#fff;margin-bottom:0.8rem;">Cycle Assessment</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:2.4rem;
                            font-weight:700;color:{sev_col};line-height:1;">
                    Score = {score_in:.4f}
                </div>
                <div style="margin-top:12px;">
                    <span style="background:{badge_bg};color:{sev_col};
                                 font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                                 font-weight:600;letter-spacing:.08em;
                                 padding:5px 16px;border-radius:20px;">
                        SEVERITY: {sev_in.upper()}
                    </span>
                </div>
                <div style="margin-top:14px;font-family:var(--fm);font-size:0.68rem;
                            color:var(--muted);line-height:1.9;">
                    Decision thr : {decision_thr:.4f}<br>
                    Critical q05 : {q05:.4f}<br>
                    Moderate q10 : {q10:.4f}
                </div>
            </div>''',
            unsafe_allow_html=True
        )

        # Gauge bar
        st.markdown('<div class="lsa-section">// Score position</div>', unsafe_allow_html=True)
        score_min  = float(scores.min())
        score_max  = 0.0
        gauge_pct  = max(0, (score_in - score_min) / (score_max - score_min))
        fig_g, ax_g = plt.subplots(figsize=(5.5, 1.1))
        fig_g.patch.set_facecolor(C_CARD); ax_g.set_facecolor(C_CARD)
        ax_g.barh([0], [1.0],       color="#1e2d45", height=0.5)
        ax_g.barh([0], [gauge_pct], color=sev_col,  height=0.5, alpha=0.9)
        dec_pct = (decision_thr - score_min) / (score_max - score_min)
        ax_g.axvline(dec_pct, color=C_DANGER, lw=2, ls="--", label="Decision thr")
        ax_g.set_xlim(0, 1); ax_g.set_yticks([])
        ax_g.set_xticks([0, 0.5, 1.0])
        ax_g.set_xticklabels(["Anomalous", "Mid", "Normal"], fontsize=8, color=C_MUTED)
        for sp in ax_g.spines.values(): sp.set_edgecolor("#1e2d45")
        plt.tight_layout()
        st.pyplot(fig_g, use_container_width=True); plt.close()

        # Top 5 sensor deviations
        st.markdown('<div class="lsa-section">// Top 5 sensor deviations</div>',
                    unsafe_allow_html=True)
        deltas = {lbl: abs((v - means_all[f]) / stds_all[f])
                  for f, v, lbl in zip(FEATURES, vals, LABELS)}
        top5   = sorted(deltas.items(), key=lambda x: x[1], reverse=True)[:5]
        fig_d, ax_d = plt.subplots(figsize=(5.5, 3))
        fig_d.patch.set_facecolor(C_CARD); ax_d.set_facecolor(C_CARD)
        lbls5  = [t[0] for t in top5]; vals5 = [t[1] for t in top5]
        cols5  = [C_DANGER if v > 2 else C_WARN if v > 1 else C_BLUE for v in vals5]
        ax_d.barh(lbls5, vals5, color=cols5, edgecolor=C_BG, height=0.5, alpha=0.85)
        ax_d.axvline(2, color=C_DANGER, lw=1.5, ls="--", alpha=0.7, label="|Z|=2")
        ax_d.axvline(3, color=C_WARN,   lw=1.5, ls=":",  alpha=0.7, label="|Z|=3")
        ax_d.set_xlabel("|Z| deviation from mean", color=C_MUTED)
        ax_d.grid(True, axis="x", alpha=0.3)
        ax_d.tick_params(colors=C_MUTED)
        for sp in ax_d.spines.values(): sp.set_edgecolor("#1e2d45")
        ax_d.legend(facecolor=C_CARD, labelcolor=C_TEXT, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_d, use_container_width=True); plt.close()

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="lsa-section">// Anomaly risk driver analysis</div>',
                unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="lsa-section">// Sensor ↔ IF score correlation</div>',
                    unsafe_allow_html=True)
        correlations = {lbl: float(np.corrcoef(df[f], df["score_if"])[0, 1])
                        for f, lbl in zip(FEATURES, LABELS)}
        corr_df = pd.Series(correlations).sort_values()
        fig, ax = dark_fig(6.5, 5)
        colors_c = [C_DANGER if v < 0 else C_BLUE for v in corr_df.values]
        bars = ax.barh(corr_df.index, corr_df.values, color=colors_c,
                       edgecolor=C_BG, height=0.55, alpha=0.85)
        for bar, val in zip(bars, corr_df.values):
            off = 0.003 if val >= 0 else -0.003
            ax.text(val + off, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center",
                    ha="left" if val >= 0 else "right", fontsize=8, color=C_TEXT)
        ax.axvline(0, color="white", lw=0.8)
        ax.set_xlabel("Pearson r with IF Score")
        ax.set_title("Feature → Score Correlation\n(Negative r = anomaly driver)",
                     color=C_TEXT, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()
        st.caption("Vibration X/Y are the strongest anomaly drivers. "
                   "Press force is inversely correlated — low force = more isolated = lower score.")

    with col_r:
        st.markdown('<div class="lsa-section">// Normal vs anomaly mean profile</div>',
                    unsafe_allow_html=True)
        profile          = df.groupby("pred_anomaly")[FEATURES].mean().T.round(2)
        profile.columns  = ["Normal", "Predicted Anomaly"]
        profile["Δ%"]    = ((profile["Predicted Anomaly"] - profile["Normal"]) /
                             profile["Normal"] * 100).round(1)
        profile.index    = LABELS
        st.dataframe(profile, use_container_width=True)

        st.markdown('<div class="lsa-section" style="margin-top:14px;">// Z-Score by severity class</div>',
                    unsafe_allow_html=True)
        fig2, axes2 = plt.subplots(2, 2, figsize=(6, 5))
        fig2.patch.set_facecolor(C_BG)
        for ax2, feat, lbl in zip(axes2.flat, FEATURES[:4], LABELS[:4]):
            ax2.set_facecolor(C_CARD)
            for sp in ax2.spines.values(): sp.set_edgecolor("#1e2d45")
            z_col  = (df[feat] - means_all[feat]) / stds_all[feat]
            for sev, col in SEV_COLORS.items():
                mask = df["severity"] == sev
                if mask.sum() > 0:
                    ax2.hist(z_col[mask].abs(), bins=20, color=col, alpha=0.65,
                             label=sev, density=True)
            ax2.axvline(3.0, color=C_DANGER, lw=1.5, ls="--", alpha=0.8)
            ax2.set_title(lbl, fontsize=8, fontweight="bold")
            ax2.tick_params(colors=C_MUTED, labelsize=7)
            ax2.legend(fontsize=6)
            ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True); plt.close()

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="lsa-section">// Anomaly response protocol — hot stamping line</div>',
                unsafe_allow_html=True)
    st.caption("3 anomaly types × 4 severity tiers = structured intervention matrix.")

    anomaly_types = [
        {"name": "Type A — Mechanical Anomaly", "color": C_DANGER, "severity": "Critical",
         "trigger": "vibration_x_mm_s > 5.5  or  energy_kwh > 28",
         "root_cause": "Bearing wear · Axis looseness · Spindle imbalance · Drive coupling failure",
         "immediate": "Stop press immediately → Inspect X/Y axis bearings → Check drive coupling torque",
         "preventive": "Vibration trending via CMMS · Bearing replacement at 80% wear indicator"},
        {"name": "Type B — Thermal Anomaly", "color": C_WARN, "severity": "Moderate",
         "trigger": "tool_temp_c > 445°C  or  part_temp_c > 875°C",
         "root_cause": "Cooling circuit failure · Furnace temperature drift · Die coating degradation",
         "immediate": "Reduce press cycle rate → Verify coolant circuit pressure → Check furnace setpoints",
         "preventive": "Pyrometer calibration weekly · Die coating inspection every 5,000 cycles"},
        {"name": "Type C — Low-Pressure Anomaly", "color": C_PURP, "severity": "Moderate–Critical",
         "trigger": "press_force_ton < 200  or  contact_force_kn < 160",
         "root_cause": "Hydraulic pressure loss · Die misalignment · Tonnage setpoint drift",
         "immediate": "Inspect hydraulic system pressure → Verify die clamping alignment",
         "preventive": "Tonnage monitoring per run · Hydraulic oil analysis monthly"},
    ]

    for atype in anomaly_types:
        with st.expander(f"{atype['name']}  ·  Severity: {atype['severity']}", expanded=True):
            cl, cr = st.columns([3, 1])
            with cl:
                for lbl, text, color_lbl in [
                    ("Trigger Condition", atype["trigger"],    C_ORANGE),
                    ("Root Cause",        atype["root_cause"], C_MUTED),
                    ("Immediate Action",  atype["immediate"],  C_DANGER),
                    ("Preventive",        atype["preventive"], C_OK),
                ]:
                    st.markdown(f"""
                    <div style="background:var(--card);border:1px solid var(--border);
                                border-left:3px solid {color_lbl};border-radius:2px;
                                padding:8px 12px;margin-bottom:7px;">
                        <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                                    text-transform:uppercase;letter-spacing:.15em;margin-bottom:4px;">{lbl}</div>
                        <div style="font-family:var(--fm);font-size:0.72rem;color:var(--text);
                                    line-height:1.6;">{text}</div>
                    </div>""", unsafe_allow_html=True)
            with cr:
                st.markdown(f"""
                <div style="background:var(--card);border:1px solid var(--border);
                            border-top:3px solid {atype['color']};border-radius:2px;
                            padding:1rem;text-align:center;margin-top:8px;">
                    <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                                text-transform:uppercase;letter-spacing:.1em;">Severity</div>
                    <div style="font-family:var(--fm);font-size:1rem;font-weight:700;
                                color:{atype['color']};margin-top:4px;">{atype['severity']}</div>
                </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
    <div style="background:var(--card);border:1px solid var(--border);border-radius:4px;
                padding:1.2rem 1.6rem;">
        <div style="font-family:var(--fh);font-size:1rem;font-weight:700;
                    color:#fff;margin-bottom:12px;">IF Score → Severity → Response Matrix</div>
        <table style="width:100%;font-family:var(--fm);font-size:0.72rem;border-collapse:collapse;">
            <tr style="border-bottom:1px solid var(--border);background:var(--card2);">
                <th style="padding:8px;text-align:left;color:var(--muted);">Severity</th>
                <th style="padding:8px;color:var(--muted);">IF Score Range</th>
                <th style="padding:8px;color:var(--muted);">Response</th>
            </tr>
            <tr style="border-bottom:1px solid var(--border);">
                <td style="padding:8px;color:{C_OK};font-weight:600;">Normal</td>
                <td style="padding:8px;text-align:center;color:{C_TEXT};">score &gt; {decision_thr:.4f}</td>
                <td style="padding:8px;color:{C_MUTED};">Continue — log for trend monitoring</td>
            </tr>
            <tr style="border-bottom:1px solid var(--border);">
                <td style="padding:8px;color:{C_BLUE};font-weight:600;">Mild</td>
                <td style="padding:8px;text-align:center;color:{C_TEXT};">{q10:.4f} to {decision_thr:.4f}</td>
                <td style="padding:8px;color:{C_MUTED};">Monitor — plan inspection at shift end</td>
            </tr>
            <tr style="border-bottom:1px solid var(--border);">
                <td style="padding:8px;color:{C_WARN};font-weight:600;">Moderate</td>
                <td style="padding:8px;text-align:center;color:{C_TEXT};">{q05:.4f} to {q10:.4f}</td>
                <td style="padding:8px;color:{C_MUTED};">Reduce rate — inspect within 2 hours</td>
            </tr>
            <tr>
                <td style="padding:8px;color:{C_DANGER};font-weight:600;">Critical</td>
                <td style="padding:8px;text-align:center;color:{C_TEXT};">score &#8804; {q05:.4f}</td>
                <td style="padding:8px;color:{C_MUTED};">STOP CYCLE — immediate inspection</td>
            </tr>
        </table>
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
            <span style="color:#f97316;">lozanolsa.gumroad.com</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lsa-footer">
    LozanoLsa · Turning Operations into Predictive Systems · Press Anomaly Intelligence · Project 21 · v2.0
</div>
""", unsafe_allow_html=True)
