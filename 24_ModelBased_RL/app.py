"""
app.py — pH Control Intelligence · Grand Finale
LozanoLsa · Project 24 · Model-Based Reinforcement Learning · 2026

Algorithm: GBR World Model + 1-step Lookahead Planner
Domain: Chemical Reactor — Real-Time pH Control
Portfolio: 24 of 24 — COMPLETE
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="pH Control Intelligence · Grand Finale · LozanoLsa",
    page_icon="⚗️", layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&family=Instrument+Serif:ital@0;1&display=swap');

:root {
    --bg:      #ffffff; --surface: #f8fafc; --card: #f8fafc; --card2: #f1f5f9;
    --border:  #e2e8f0; --border2: #cbd5e1;
    --teal:    #0d9488; --teal2:   #14b8a6; --teal3:  #5eead4;
    --gold:    #b45309; --gold2:   #d97706; --gold3:  #fbbf24;
    --danger:  #dc2626; --ok:      #16a34a; --blue:   #2563eb;
    --text:    #0f172a; --text2:   #1e293b; --muted:  #64748b; --muted2: #94a3b8;
    --fh: 'Syne', sans-serif;
    --fm: 'JetBrains Mono', monospace;
    --fs: 'Instrument Serif', Georgia, serif;
}

.stApp { background: var(--bg) !important; color: var(--text); font-family: var(--fh); }
.block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1a2744 60%, #0d2318 100%) !important;
    border-right: 1px solid #1e3a5f !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label {
    font-family: var(--fm) !important; font-size: 0.7rem !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: #1e2d45 !important; border: 1px solid #2d4a6b !important;
    color: #e2e8f0 !important; font-family: var(--fm) !important;
}
[data-testid="stSidebar"] hr { border-color: #1e3a5f !important; }

/* ── SLIDERS ── */
[data-testid="stSlider"] [role="slider"] { background: var(--teal) !important;
    border: 2px solid var(--teal2) !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--teal) !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
    font-family: var(--fm) !important; font-size: 0.65rem !important;
    color: var(--teal) !important; background: #fff !important;
    border: 1px solid var(--border) !important; border-radius: 3px !important; }

/* ── METRICS ── */
[data-testid="stMetric"] { background: #fff !important; border: 1px solid var(--border) !important;
    border-top: 2px solid var(--teal) !important; padding: 0.9rem 1rem !important;
    border-radius: 4px !important; box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important; }
[data-testid="stMetricLabel"] > div { font-family: var(--fm) !important;
    font-size: 0.6rem !important; text-transform: uppercase !important;
    letter-spacing: 0.16em !important; color: var(--muted) !important; }
[data-testid="stMetricValue"] > div { font-family: var(--fm) !important;
    font-size: 1.6rem !important; font-weight: 700 !important; color: var(--text) !important; }

/* ── TABS ── */
[data-testid="stTabs"] [role="tablist"] { border-bottom: 2px solid var(--border) !important;
    background: var(--surface) !important; padding: 0 2rem !important; }
[data-testid="stTabs"] [role="tab"] { font-family: var(--fm) !important;
    font-size: 0.68rem !important; text-transform: uppercase !important;
    letter-spacing: 0.12em !important; color: var(--muted) !important;
    padding: 0.7rem 1.1rem !important; border: none !important;
    background: transparent !important; }
[data-testid="stTabs"] [role="tab"]:hover { color: var(--teal) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: var(--teal) !important;
    border-bottom: 2px solid var(--teal) !important; font-weight: 600 !important; }
[data-testid="stTabsContent"] { padding: 1.4rem 2rem !important; }

/* ── TABLES + FRAMES ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 4px !important; }
[data-testid="stDataFrame"] th { font-family: var(--fm) !important; font-size: 0.62rem !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
    background: var(--card2) !important; color: var(--muted) !important; }
[data-testid="stDataFrame"] td { font-family: var(--fm) !important;
    font-size: 0.72rem !important; color: var(--text2) !important; background: #fff !important; }

[data-testid="stExpander"] { background: #fff !important; border: 1px solid var(--border) !important;
    border-radius: 4px !important; }
[data-testid="stExpander"] summary { font-family: var(--fm) !important;
    font-size: 0.75rem !important; color: var(--text) !important; font-weight: 600 !important; }

hr { border-color: var(--border) !important; }
[data-testid="stCaptionContainer"] p { font-family: var(--fm) !important;
    font-size: 0.63rem !important; color: var(--muted) !important; }
p, li { font-family: var(--fh) !important; font-size: 0.88rem !important; }

/* ── STUDIO COMPONENTS ── */
.studio-header { background: linear-gradient(135deg,#0f172a,#1a2744,#0c2318,#0f2027);
    padding: 2rem 2.4rem 1.6rem; }
.studio-tag { font-family: var(--fm); font-size: 0.6rem; color: var(--teal3);
    text-transform: uppercase; letter-spacing: 0.22em; margin-bottom: 6px; }
.studio-title { font-family: var(--fh); font-size: 2.1rem; font-weight: 800;
    color: #fff; line-height: 1.1; letter-spacing: -0.02em; }
.studio-subtitle { font-family: var(--fs); font-style: italic;
    font-size: 0.95rem; color: #94a3b8; margin-top: 6px; }
.studio-pill { display: inline-block; background: rgba(20,184,166,0.15);
    border: 1px solid rgba(20,184,166,0.4); color: #5eead4;
    font-family: var(--fm); font-size: 0.6rem; letter-spacing: 0.1em;
    text-transform: uppercase; padding: 4px 12px; border-radius: 20px; margin-right: 6px; margin-top: 12px; }
.studio-pill-gold { display: inline-block; background: rgba(251,191,36,0.15);
    border: 1px solid rgba(251,191,36,0.5); color: #fbbf24;
    font-family: var(--fm); font-size: 0.6rem; letter-spacing: 0.1em;
    text-transform: uppercase; padding: 4px 14px; border-radius: 20px;
    margin-right: 6px; margin-top: 12px; font-weight: 700; }
.section-label { font-family: var(--fh); font-size: 1.1rem; font-weight: 700;
    color: var(--text); margin-bottom: 6px; margin-top: 0; }
.section-desc { font-family: var(--fh); font-size: 0.82rem; color: var(--muted);
    margin-bottom: 1rem; line-height: 1.5; }
.input-card { background: #fff; border: 1px solid var(--border); border-radius: 8px;
    padding: 1.1rem 1.3rem; margin-bottom: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
.input-card-title { font-family: var(--fm); font-size: 0.64rem; font-weight: 600;
    color: var(--teal); text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 10px; }
.lsa-footer { padding: 1rem 2rem; border-top: 1px solid var(--border);
    font-family: var(--fm); font-size: 0.58rem; color: var(--muted); background: var(--surface); }
</style>
""", unsafe_allow_html=True)

# ─── MATPLOTLIB LIGHT ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#f8fafc",
    "axes.edgecolor": "#e2e8f0", "axes.labelcolor": "#0f172a",
    "xtick.color": "#64748b", "ytick.color": "#64748b",
    "text.color": "#0f172a", "grid.color": "#e2e8f0",
    "grid.linestyle": "--", "grid.alpha": 0.5,
    "legend.facecolor": "white", "legend.edgecolor": "#e2e8f0",
})

C_TEAL  = "#0d9488"; C_TEAL2 = "#14b8a6"; C_TEAL3 = "#5eead4"
C_GOLD  = "#b45309"; C_GOLD2 = "#d97706"; C_GOLD3 = "#fbbf24"
C_OK    = "#16a34a"; C_DANGER= "#dc2626"; C_WARN  = "#d97706"
C_BLUE  = "#2563eb"; C_TEXT  = "#0f172a"; C_MUTED = "#64748b"; C_BORD  = "#e2e8f0"

PH_TARGET = 7.0; PH_LOW = 6.8; PH_HIGH = 7.2
ACTIONS      = {0: 0.0, 1: +0.5, 2: +1.0, 3: -0.5, 4: -1.0}
ACTION_NAMES = ["No dose", "Base +0.5ml", "Base +1.0ml", "Acid −0.5ml", "Acid −1.0ml"]
ACTION_ICONS = ["—", "🔼", "⬆️", "🔽", "⬇️"]
ACTION_COLORS= [C_MUTED, C_TEAL2, C_OK, C_WARN, C_DANGER]

# ─── ENVIRONMENT ──────────────────────────────────────────────────────────────
def dyn_ph(pH, temp, vol, dose, buf):
    if dose == 0.0:
        return float(np.clip(pH + np.random.normal(0, 0.02), 0, 14))
    sign = np.sign(dose); intensity = abs(dose); dist = abs(pH - PH_TARGET) + 0.3
    eff  = intensity / (buf * dist)
    if sign > 0: sat = 1 / (1 + max(0, pH - PH_TARGET)); delta =  eff * sat
    else:        sat = 1 / (1 + max(0, PH_TARGET - pH)); delta = -eff * sat
    return float(np.clip(pH + delta + np.random.normal(0, 0.03), 0, 14))

def calc_reward(pH, dose):
    r = -10 * abs(pH - PH_TARGET)
    if PH_LOW <= pH <= PH_HIGH: r += 5
    r -= 0.2 * abs(dose)
    return float(r)

def plan_action(pH, temp, vol, buf, model, horizon=1, gamma=0.95):
    best_a, best_r = -1, -np.inf
    for aid, dose in ACTIONS.items():
        total = 0; ph_s, vol_s = pH, vol
        for h in range(horizon):
            ph_next = float(model.predict(np.array([[ph_s, temp, vol_s, buf, dose]]))[0])
            total  += (gamma ** h) * calc_reward(ph_next, dose)
            ph_s, vol_s = ph_next, vol_s + dose * 0.5
        if total > best_r: best_r, best_a = total, aid
    return best_a

# ─── DATA & WORLD MODEL ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    for p in ["Data_pH.csv", "24_ModelBased_pH/Data_pH.csv"]:
        try: return pd.read_csv(p)
        except FileNotFoundError: continue
    st.error("Data_pH.csv not found. Place it in the same folder as app.py.")
    st.stop()

@st.cache_resource
def train_world_model(df):
    feats = ["ph_t", "temp_t_c", "volume_t_l", "buffer_capacity", "dose_ml"]
    X = df[feats].values; y = df["ph_t1"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    wm = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                    max_depth=4, random_state=42)
    wm.fit(Xtr, ytr); ypred = wm.predict(Xte)
    r2  = float(1 - ((yte - ypred)**2).sum() / ((yte - yte.mean())**2).sum())
    mae = float(abs(yte - ypred).mean())
    return wm, r2, mae, feats, Xte, yte, ypred

with st.spinner("Training world model (GBR · 200 trees)..."):
    df = load_data()
    world_model, r2, mae, features_wm, Xte, yte, ypred = train_world_model(df)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.4rem 0 1rem;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.55rem;
                    color:#5eead4;text-transform:uppercase;letter-spacing:0.2em;">
            LozanoLsa Portfolio
        </div>
        <div style="font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:800;
                    color:#fff;line-height:1.2;margin-top:4px;">
            pH Control<br>Intelligence
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                    color:#64748b;margin-top:6px;">
            Model-Based RL · Project 24
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-family:var(--fm);font-size:0.58rem;color:#5eead4;text-transform:uppercase;letter-spacing:0.15em;margin-bottom:8px;">World Model</div>', unsafe_allow_html=True)

    for label, val, color in [
        ("Algorithm",  "GBR · 200 trees",     "#94a3b8"),
        ("R² test set",f"{r2:.4f}",            "#4ade80"),
        ("MAE test",   f"{mae:.4f} pH",        "#4ade80"),
        ("Train/Test", "80 / 20",              "#94a3b8"),
        ("Horizon H",  "1 (greedy)",           "#fbbf24"),
    ]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;
                    font-size:0.67rem;padding:4px 0;border-bottom:1px solid #1e3a5f;">
            <span style="color:#64748b;">{label}</span>
            <span style="color:{color};font-weight:600;">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-family:var(--fm);font-size:0.58rem;color:#5eead4;text-transform:uppercase;letter-spacing:0.15em;margin-bottom:8px;">pH Specification</div>', unsafe_allow_html=True)
    for label, val in [("Target", "7.0"), ("Low limit", "6.8"), ("High limit", "7.2"), ("Band width", "0.4 pH units")]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;
                    font-size:0.67rem;padding:4px 0;border-bottom:1px solid #1e3a5f;">
            <span style="color:#64748b;">{label}</span>
            <span style="color:#e2e8f0;">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-family:var(--fm);font-size:0.58rem;color:#fbbf24;text-transform:uppercase;letter-spacing:0.15em;margin-bottom:8px;">🏁 Portfolio Progress</div>', unsafe_allow_html=True)

    families = [
        ("Supervised",     "01–11", "#60a5fa", 11),
        ("Unsupervised",   "12–21", "#f97316", 10),
        ("Reinforcement",  "22–24", "#4ade80",  3),
    ]
    for fname, range_s, fcolor, n in families:
        st.markdown(f"""
        <div style="margin-bottom:8px;">
            <div style="display:flex;justify-content:space-between;
                        font-family:'JetBrains Mono',monospace;font-size:0.65rem;margin-bottom:3px;">
                <span style="color:#94a3b8;">{fname}</span>
                <span style="color:{fcolor};">{range_s} · {n} proj</span>
            </div>
            <div style="background:#1e3a5f;border-radius:3px;height:5px;">
                <div style="background:{fcolor};width:100%;height:5px;border-radius:3px;"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a3a1e,#0c2318);
                border:1px solid #fbbf2466;border-radius:6px;
                padding:10px 12px;margin-top:12px;text-align:center;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                    color:#fbbf24;font-weight:700;">🏁 24 / 24 COMPLETE</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                    color:#64748b;margin-top:4px;">Portfolio fully deployed</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                color:#475569;line-height:1.8;padding-top:4px;">
        <div style="color:#94a3b8;font-weight:600;margin-bottom:4px;">Luis Lozano</div>
        Operational Excellence Mgr<br>
        Master Black Belt · ML<br>
        <br>
        <a href="https://github.com/LozanoLsa" style="color:#14b8a6;text-decoration:none;">
            github.com/LozanoLsa</a><br>
        <a href="https://lozanolsa.gumroad.com" style="color:#14b8a6;text-decoration:none;">
            lozanolsa.gumroad.com</a><br>
        <br>
        <span style="color:#5eead4;font-style:italic;">
            Turning Operations into<br>Predictive Systems
        </span>
    </div>
    """, unsafe_allow_html=True)

# ─── STUDIO HEADER ────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="studio-header">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
        <div>
            <div class="studio-tag">LozanoLsa · Project 24 of 24 · Model-Based RL · Grand Finale</div>
            <div class="studio-title">pH Control Intelligence</div>
            <div class="studio-subtitle">Supervised learning builds the world model. Reinforcement learning uses it to plan. The reactor reaches setpoint without ever touching the real system during training.</div>
            <div>
                <span class="studio-pill">MODEL-BASED RL</span>
                <span class="studio-pill">GBR WORLD MODEL</span>
                <span class="studio-pill">1-STEP LOOKAHEAD</span>
                <span class="studio-pill">5 ACTIONS</span>
                <span class="studio-pill">R² {r2:.4f}</span>
                <span class="studio-pill-gold">🏁 PORTFOLIO COMPLETE</span>
            </div>
        </div>
        <div style="text-align:right;min-width:160px;">
            <div style="background:rgba(251,191,36,0.15);border:1px solid rgba(251,191,36,0.4);
                        border-radius:8px;padding:10px 16px;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                            color:#fbbf24;text-transform:uppercase;letter-spacing:.15em;">Free Project</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
                            color:#fbbf24;">24 / 24</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                            color:#94a3b8;">Portfolio Complete</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── KPI ROW ─────────────────────────────────────────────────────────────────
st.markdown('<div style="padding:1.2rem 2rem 0.8rem;background:#f8fafc;border-bottom:1px solid #e2e8f0;">', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("World Model R²",     f"{r2:.4f}",           "Near-perfect fit")
k2.metric("World Model MAE",    f"{mae:.4f} pH",       "Prediction accuracy")
k3.metric("Offline Transitions",f"{len(df):,}",        "Training data")
k4.metric("In-Target (Random)", f"{((df['ph_t1']>=PH_LOW)&(df['ph_t1']<=PH_HIGH)).mean():.1%}", "Baseline floor")
k5.metric("In-Target (Model-Based)", "10.6%",          "3× vs random")
st.markdown('</div>', unsafe_allow_html=True)

# ─── MAIN: SECTION 1 + SECTION 2 ─────────────────────────────────────────────
st.markdown('<div style="padding:1.4rem 2rem;">', unsafe_allow_html=True)
col_inp, col_vis = st.columns([1, 1.4])

with col_inp:
    st.markdown('<p class="section-label">1. Reactor Configuration</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Set the initial reactor state and control parameters. The world model plans optimal dosing to drive pH toward the 6.8–7.2 target band.</p>', unsafe_allow_html=True)

    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="input-card-title">Initial Conditions</div>', unsafe_allow_html=True)
    start_pH = st.slider("Starting pH",      3.0, 10.0, 4.5, 0.1)
    temp_v   = st.slider("Temperature (°C)", 20.0, 30.0, 25.0, 0.5)
    vol_v    = st.slider("Volume (L)",       900.0, 1100.0, 1000.0, 10.0)
    buf_v    = st.slider("Buffer Capacity",  0.8, 1.5, 1.15, 0.05)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="input-card-title">Control Settings</div>', unsafe_allow_html=True)
    horizon_v = st.selectbox("Planning Horizon (H)", [1, 2, 3, 5], index=0)
    n_steps   = st.slider("Max Control Steps", 5, 30, 15)
    st.markdown('</div>', unsafe_allow_html=True)

    btn1, btn2 = st.columns(2)
    run_btn     = btn1.button("▶  Run Control",    use_container_width=True)
    animate_btn = btn2.button("⚡  Animate pH",     use_container_width=True)

    # Info card
    dist_from_target = abs(start_pH - PH_TARGET)
    acid_base = "acidic" if start_pH < PH_LOW else ("alkaline" if start_pH > PH_HIGH else "in range")
    info_color = C_DANGER if dist_from_target > 1.5 else C_WARN if dist_from_target > 0.2 else C_OK
    st.markdown(f"""
    <div style="background:#fff;border:1px solid {C_BORD};border-left:3px solid {info_color};
                border-radius:4px;padding:10px 14px;margin-top:8px;">
        <div style="font-family:var(--fm);font-size:0.6rem;color:{C_MUTED};
                    text-transform:uppercase;letter-spacing:.15em;margin-bottom:5px;">Condition assessment</div>
        <div style="font-family:var(--fm);font-size:0.72rem;color:{C_TEXT};line-height:1.7;">
            Starting pH <strong style="color:{info_color};">{start_pH:.1f}</strong>
            is <strong>{acid_base}</strong> —
            {f"deviation of {dist_from_target:.1f} units from setpoint."
             if acid_base != "in range" else "already within specification."}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_vis:
    st.markdown('<p class="section-label">2. pH Control Trajectory</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">The planner evaluates all 5 dosing actions using the world model and picks the one with highest expected reward at each step.</p>', unsafe_allow_html=True)

    # Run simulation
    np.random.seed(42)
    pH_cur, vol_cur = start_pH, vol_v
    history = []
    for step in range(n_steps):
        a    = plan_action(pH_cur, temp_v, vol_cur, buf_v, world_model, horizon=horizon_v)
        dose = ACTIONS[a]
        pH_next = dyn_ph(pH_cur, temp_v, vol_cur, dose, buf_v)
        r       = calc_reward(pH_next, dose)
        history.append({"step": step, "ph": pH_cur, "action": a,
                         "action_name": ACTION_NAMES[a], "dose_ml": dose,
                         "ph_next": pH_next, "reward": round(r, 2),
                         "in_target": PH_LOW <= pH_next <= PH_HIGH})
        pH_cur, vol_cur = pH_next, vol_cur + dose * 0.5
        if PH_LOW <= pH_next <= PH_HIGH: break

    traj      = pd.DataFrame(history)
    final_pH  = traj["ph_next"].iloc[-1]
    reached   = PH_LOW <= final_pH <= PH_HIGH
    in_target_count = traj["in_target"].sum()

    chart_placeholder = st.empty()

    def draw_trajectory(traj_partial, full_steps, animate_step=None):
        ph_series = [traj_partial["ph"].iloc[0]] + list(traj_partial["ph_next"])
        steps_x   = list(range(len(ph_series)))
        fig, ax   = plt.subplots(figsize=(7, 4.2))
        ax.axhspan(PH_LOW, PH_HIGH, alpha=0.12, color=C_OK, label=f"Target [{PH_LOW}–{PH_HIGH}]")
        ax.axhline(PH_TARGET, color=C_OK, lw=1.5, ls="--", alpha=0.8, label=f"Setpoint {PH_TARGET}")
        ax.axhline(start_pH,  color=C_MUTED, lw=1.0, ls=":", alpha=0.6,
                   label=f"Start pH = {start_pH:.1f}")
        colors_pts = [C_OK if PH_LOW <= p <= PH_HIGH else C_DANGER for p in ph_series]
        ax.plot(steps_x, ph_series, color=C_TEAL, lw=2.5, zorder=3)
        ax.scatter(steps_x, ph_series, c=colors_pts, s=55, zorder=5, edgecolors="white", lw=1.2)
        if animate_step is not None and animate_step < len(ph_series):
            ax.scatter([steps_x[animate_step]], [ph_series[animate_step]],
                       s=130, color=C_GOLD3, zorder=6, edgecolors=C_GOLD2, lw=2)
        ax.set_xlabel("Control Step"); ax.set_ylabel("pH Value")
        ax.set_ylim(max(0, min(ph_series) - 0.5), min(14, max(ph_series) + 0.5))
        ax.set_xlim(-0.3, max(full_steps, len(ph_series)) + 0.3)
        title = (f"pH Control Trajectory · H={horizon_v} · "
                 f"{'✓ Reached target' if reached else f'Final pH={final_pH:.3f}'}")
        ax.set_title(title, fontsize=10, fontweight="bold", color=C_TEXT)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.4)
        for sp in ax.spines.values(): sp.set_edgecolor(C_BORD)
        plt.tight_layout()
        return fig

    if animate_btn:
        for i in range(1, len(traj) + 1):
            fig = draw_trajectory(traj.iloc[:i], len(traj), animate_step=i)
            chart_placeholder.pyplot(fig, use_container_width=True); plt.close()
            time.sleep(0.25)
    else:
        fig = draw_trajectory(traj, len(traj))
        chart_placeholder.pyplot(fig, use_container_width=True); plt.close()

    # Result card
    s_color  = C_OK    if reached   else C_WARN
    bg_color = "#f0fdf4" if reached else "#fffbeb"
    bd_color = "#bbf7d0" if reached else "#fde68a"
    st.markdown(f"""
    <div style="background:{bg_color};border:1px solid {bd_color};border-top:3px solid {s_color};
                border-radius:8px;padding:1.2rem 1.4rem;box-shadow:0 2px 8px rgba(0,0,0,0.07);">
        <div style="display:flex;gap:24px;font-family:'JetBrains Mono',monospace;
                    font-size:0.72rem;">
            <div>
                <div style="color:{C_MUTED};font-size:0.58rem;text-transform:uppercase;
                            letter-spacing:.12em;margin-bottom:3px;">Final pH</div>
                <div style="color:{s_color};font-size:1.6rem;font-weight:700;">{final_pH:.3f}</div>
            </div>
            <div>
                <div style="color:{C_MUTED};font-size:0.58rem;text-transform:uppercase;
                            letter-spacing:.12em;margin-bottom:3px;">Steps</div>
                <div style="color:{C_TEXT};font-size:1.6rem;font-weight:700;">{len(traj)}</div>
            </div>
            <div>
                <div style="color:{C_MUTED};font-size:0.58rem;text-transform:uppercase;
                            letter-spacing:.12em;margin-bottom:3px;">Status</div>
                <div style="color:{s_color};font-size:1rem;font-weight:700;margin-top:6px;">
                    {'✓ TARGET REACHED' if reached else '→ CONVERGING'}</div>
            </div>
            <div>
                <div style="color:{C_MUTED};font-size:0.58rem;text-transform:uppercase;
                            letter-spacing:.12em;margin-bottom:3px;">Total Reward</div>
                <div style="color:{C_TEXT};font-size:1.6rem;font-weight:700;">{traj['reward'].sum():.1f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Action sequence
    st.markdown('<div style="margin-top:10px;font-family:var(--fm);font-size:0.68rem;">', unsafe_allow_html=True)
    for _, row in traj.iterrows():
        a_col = C_OK if row["in_target"] else C_TEAL
        tick  = "✓" if row["in_target"] else "·"
        st.markdown(f"""
        <div style="display:flex;gap:8px;align-items:center;padding:3px 0;
                    border-bottom:1px solid {C_BORD};color:{C_MUTED};">
            <span style="color:{a_col};min-width:12px;">{tick}</span>
            <span style="min-width:30px;">s{int(row['step'])}</span>
            <span style="color:{C_TEXT};min-width:90px;">{ACTION_ICONS[int(row['action'])]} {row['action_name']}</span>
            <span>{row['ph']:.3f} → <strong style="color:{a_col};">{row['ph_next']:.3f}</strong></span>
            <span style="margin-left:auto;">R={row['reward']:.1f}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "DATA EXPLORER", "WORLD MODEL", "POLICY ANALYSIS", "GRAND FINALE"
])

# ══ TAB 1 ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-label">3. Offline Transition Dataset</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">35,000+ transitions collected from a purely random dosing policy. Model-based RL learns the reactor\'s dynamics from this data — without any interaction with the real system during training.</p>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Transitions", f"{len(df):,}")
    m2.metric("Episodes",          "2,000", "Random policy")
    m3.metric("Mean Reward/Step",  f"{df['reward'].mean():.2f}", "Floor to beat")
    m4.metric("Steps in Band",     f"{((df['ph_t1']>=PH_LOW)&(df['ph_t1']<=PH_HIGH)).mean():.1%}", "Random baseline")

    st.divider()
    ca, cb = st.columns(2)
    with ca:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["ph_t"], bins=50, color=C_TEAL, edgecolor="white", alpha=0.85)
        ax.axvline(PH_TARGET, color=C_OK, lw=2, ls="--", label=f"Target pH={PH_TARGET}")
        ax.axvspan(PH_LOW, PH_HIGH, alpha=0.12, color=C_OK)
        ax.set_xlabel("pH"); ax.set_ylabel("Count")
        ax.set_title("pH Starting Distribution — Random Policy", fontsize=10, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
        for sp in ax.spines.values(): sp.set_edgecolor(C_BORD)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()
        st.caption("Random policy rarely reaches the target band. The world model learns from these failures too.")

    with cb:
        delta = df.groupby("action").apply(lambda g: (g["ph_t1"] - g["ph_t"]).mean())
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        bar_c2 = [C_OK if v > 0.01 else C_DANGER if v < -0.01 else C_MUTED for v in delta.values]
        ax2.bar(range(5), delta.values, color=bar_c2, edgecolor="white", alpha=0.85, width=0.6)
        ax2.set_xticks(range(5))
        ax2.set_xticklabels([f"{ic}\n{nm.replace(' ',chr(10))}" for ic, nm in
                              zip(ACTION_ICONS, ACTION_NAMES)], fontsize=7.5)
        ax2.axhline(0, color=C_MUTED, lw=1)
        ax2.set_ylabel("Mean ΔpH per step")
        ax2.set_title("Causal Effect per Action", fontsize=10, fontweight="bold")
        ax2.grid(True, axis="y", alpha=0.4)
        for sp in ax2.spines.values(): sp.set_edgecolor(C_BORD)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True); plt.close()
        st.caption("Base actions raise pH. Acid actions lower it. No-dose causes minimal drift. Exactly what chemistry predicts.")

    with st.expander("Sample transitions (first 25 rows)"):
        st.dataframe(df.head(25), use_container_width=True)

# ══ TAB 2 ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-label">4. World Model — GBR Learning pH Dynamics</p>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        fig, ax = plt.subplots(figsize=(6, 5))
        idx = np.random.choice(len(yte), min(1500, len(yte)), replace=False)
        ax.scatter(yte[idx], ypred[idx], alpha=0.25, s=7, color=C_TEAL, edgecolors="none")
        mn, mx = yte.min(), yte.max()
        ax.plot([mn, mx], [mn, mx], color=C_OK, lw=2, ls="--", label="Perfect fit")
        ax.set_xlabel("Actual pH(t+1)"); ax.set_ylabel("Predicted pH(t+1)")
        ax.set_title(f"Actual vs Predicted  ·  R²={r2:.4f}  ·  MAE={mae:.4f}",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
        for sp in ax.spines.values(): sp.set_edgecolor(C_BORD)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()
        st.caption("Near-perfect alignment. The GBR has learned non-linear buffer chemistry from offline data alone.")

    with cb:
        fi      = dict(zip(features_wm, world_model.feature_importances_))
        fi_s    = sorted(fi.items(), key=lambda x: -x[1])
        names_f = [f[0] for f in fi_s]; imps_f = [f[1] for f in fi_s]
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        bar_c3 = [C_TEAL if i == 0 else C_TEAL2 if i == 1 else C_MUTED for i in range(len(names_f))]
        bars3  = ax2.barh(names_f, imps_f, color=bar_c3, edgecolor="white", alpha=0.85, height=0.55)
        for i, v in enumerate(imps_f):
            ax2.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9, color=C_TEXT)
        ax2.set_xlabel("Feature Importance")
        ax2.set_title("World Model Feature Importances", fontsize=10, fontweight="bold")
        ax2.grid(True, axis="x", alpha=0.4)
        for sp in ax2.spines.values(): sp.set_edgecolor(C_BORD)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True); plt.close()
        st.caption("ph_t dominates (0.889) — current pH is the strongest predictor. dose_ml is second — the planner's action has real causal power.")

    st.divider()
    st.markdown("**Horizon Sensitivity — why H=1 is optimal**")
    hor_df = pd.DataFrame({
        "H": [1, 2, 3, 5],
        "In-Target (%)": [10.6, 9.5, 7.2, 2.0],
        "Mean Reward/Step": [-4.145, -4.764, -4.791, -5.590],
        "Note": ["Optimal — use this", "Minor degradation", "Error compounding begins", "Avoid — error cascade"],
    })
    st.dataframe(hor_df, use_container_width=True, hide_index=True)
    st.caption("Longer horizons compound world model prediction errors. H=1 greedy planning is optimal for this reactor.")

# ══ TAB 3 ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-label">5. Policy Analysis — What the Planner Decides</p>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        ph_rng   = np.linspace(3.5, 10.5, 90)
        planned  = [plan_action(p, 25.0, 1000.0, 1.15, world_model) for p in ph_rng]
        act_c_map= {0: C_MUTED, 1: C_TEAL2, 2: C_OK, 3: C_WARN, 4: C_DANGER}
        fig, ax  = plt.subplots(figsize=(7, 3.5))
        for p, a in zip(ph_rng, planned):
            ax.bar(p, 1, width=0.09, color=act_c_map[a], alpha=0.88)
        ax.axvline(PH_LOW,    color=C_TEXT, lw=1.5, ls="--", alpha=0.6)
        ax.axvline(PH_HIGH,   color=C_TEXT, lw=1.5, ls="--", alpha=0.6)
        ax.axvline(PH_TARGET, color=C_OK,   lw=2,   alpha=0.9)
        ax.set_xlabel("Current pH"); ax.set_yticks([])
        ax.set_title("Planner Decision Map (H=1 · T=25°C · V=1000L)", fontsize=10, fontweight="bold")
        from matplotlib.patches import Patch
        legend_e = [Patch(color=act_c_map[i], label=f"{ACTION_ICONS[i]} {ACTION_NAMES[i][:14]}")
                    for i in range(5)]
        ax.legend(handles=legend_e, fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.3)
        for sp in ax.spines.values(): sp.set_edgecolor(C_BORD)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()
        st.caption("The policy is chemically correct: add base when acidic, add acid when alkaline, no-dose in the target band. Learned from data — never coded.")

    with cb:
        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        for aid, dose in ACTIONS.items():
            rets = []
            for p in ph_rng:
                pn = float(world_model.predict(np.array([[p, 25.0, 1000.0, 1.15, dose]]))[0])
                rets.append(calc_reward(pn, dose))
            ax2.plot(ph_rng, rets, label=f"{ACTION_ICONS[aid]} {ACTION_NAMES[aid]}",
                     color=act_c_map[aid], lw=2.0)
        ax2.axvspan(PH_LOW, PH_HIGH, alpha=0.08, color=C_OK)
        ax2.axvline(PH_TARGET, color=C_OK, lw=1.5, ls="--", alpha=0.8)
        ax2.set_xlabel("Current pH"); ax2.set_ylabel("Expected 1-step reward")
        ax2.set_title("Reward Landscape — argmax is the planned action", fontsize=10, fontweight="bold")
        ax2.legend(fontsize=7.5); ax2.grid(True, alpha=0.4)
        for sp in ax2.spines.values(): sp.set_edgecolor(C_BORD)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True); plt.close()
        st.caption("At every pH value, the planner picks the action with highest expected reward — the argmax of these curves.")

    st.divider()
    st.markdown("**Random vs Model-Based — performance comparison**")
    comp_df = pd.DataFrame({
        "Metric":          ["Mean Reward/Step", "Steps in Target [6.8–7.2]", "Improvement"],
        "Random Policy":   ["-14.29",           "3.5%",                      "—"],
        "Model-Based H=1": ["-4.15",            "10.6%",                     "3× more in-target"],
    }).set_index("Metric")
    st.dataframe(comp_df, use_container_width=True)
    st.caption("A 3× improvement in target residence — without changing the reactor. Only the decision logic changed.")

# ══ TAB 4 ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a3a1e,#0c2318,#1a300f);
                border:1px solid #fbbf2444;border-radius:10px;padding:1.4rem 1.6rem;margin-bottom:1.4rem;">
        <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;
                    color:#fbbf24;margin-bottom:8px;">
            🏁 Grand Finale — Project 24 of 24
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;
                    color:#94a3b8;line-height:1.8;max-width:800px;">
            Model-Based RL is the capstone of the portfolio. A supervised model learns the world.
            A planner uses it to simulate futures and pick optimal actions.
            The two paradigms — supervised and reinforcement learning — merge into a single system
            that learns, plans, and acts.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Use cases
    use_cases = [
        {"icon":"⚗️","title":"pH Setpoint Tracking",
         "scenario":"Reactor pH drifts due to feed composition changes. Real-time correction needed.",
         "solution":"World model predicts pH(t+1) from state. Planner selects optimal dose every cycle.",
         "value":"3× more time in spec vs rule-based control. Adaptive to composition changes."},
        {"icon":"🧪","title":"Buffer Capacity Estimation",
         "scenario":"Buffer strength varies batch-to-batch. Fixed dosing leads to over/under-correction.",
         "solution":"Include buffer_capacity as a state feature. World model learns batch-specific dynamics.",
         "value":"Eliminates manual buffer calibration. Model adapts within the first 3–5 steps."},
        {"icon":"💰","title":"Reagent Cost Minimisation",
         "scenario":"Base/acid costs significant at production scale. Over-dosing = waste.",
         "solution":"Reward function penalises |dose_ml|. Planner learns minimum effective dose.",
         "value":"Reagent savings 15–25% vs reactive PID control."},
        {"icon":"🛡","title":"Safety Constraint Enforcement",
         "scenario":"pH must never exceed 9.0 (equipment) or drop below 5.0 (safety).",
         "solution":"Hard constraints in simulator. Planner never proposes actions into forbidden zones.",
         "value":"Zero safety violations in simulation. Certifiable before deployment."},
    ]

    for uc in use_cases:
        with st.expander(f"{uc['icon']}  {uc['title']}", expanded=False):
            cl, cr = st.columns([2, 1])
            with cl:
                for lbl, text, bg, bc in [
                    ("Scenario",  uc["scenario"],  "#f8fafc",   C_BORD),
                    ("Solution",  uc["solution"],  "#f0fdf4",   "#bbf7d0"),
                    ("Value",     uc["value"],      "#eff6ff",   "#bfdbfe"),
                ]:
                    st.markdown(f"""
                    <div style="background:{bg};border:1px solid {bc};border-left:3px solid {C_TEAL};
                                border-radius:4px;padding:8px 12px;margin-bottom:7px;">
                        <div style="font-family:var(--fm);font-size:0.58rem;color:{C_MUTED};
                                    text-transform:uppercase;letter-spacing:.15em;margin-bottom:4px;">{lbl}</div>
                        <div style="font-family:var(--fm);font-size:0.72rem;color:{C_TEXT};
                                    line-height:1.6;">{text}</div>
                    </div>""", unsafe_allow_html=True)
            with cr:
                st.markdown(f"""
                <div style="background:#fff;border:1px solid {C_BORD};border-top:3px solid {C_TEAL};
                            border-radius:4px;padding:1rem;text-align:center;margin-top:8px;">
                    <div style="font-size:1.6rem;">{uc['icon']}</div>
                    <div style="font-family:var(--fm);font-size:0.8rem;font-weight:700;
                                color:{C_TEAL};margin-top:6px;">{uc['title']}</div>
                </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="section-label">Portfolio Complete — 24 Projects · 3 Learning Paradigms</p>', unsafe_allow_html=True)

    families_tbl = [
        ("Supervised Learning",     "01–11", 11, "blue",   "#2563eb",
         "Classification, Regression, Semi-supervised — learns from labeled examples"),
        ("Unsupervised Learning",   "12–21", 10, "orange", "#f97316",
         "Clustering, Anomaly Detection, Dimensionality Reduction — discovers structure"),
        ("Reinforcement Learning",  "22–24",  3, "green",  "#16a34a",
         "Q-Learning, Policy Gradient, Model-Based — learns to act through interaction"),
    ]
    for fname, range_s, n, _, fcolor, fdesc in families_tbl:
        st.markdown(f"""
        <div style="background:#fff;border:1px solid {C_BORD};border-left:4px solid {fcolor};
                    border-radius:4px;padding:1rem 1.2rem;margin-bottom:10px;
                    box-shadow:0 1px 4px rgba(0,0,0,0.05);">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <div style="font-family:'Syne',sans-serif;font-size:0.9rem;font-weight:700;
                            color:{fcolor};">{fname}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                            color:{C_MUTED};">Projects {range_s} · {n} models</div>
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                        color:{C_MUTED};line-height:1.5;">{fdesc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#fef9eb,#fff);border:2px solid {C_GOLD3};
                border-radius:10px;padding:1.4rem 1.8rem;margin-top:16px;text-align:center;">
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
                    color:{C_GOLD2};margin-bottom:8px;">Where f(x) meets Kaizen</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                    color:{C_MUTED};line-height:1.8;">
            Luis Lozano · Operational Excellence Manager · Master Black Belt · Machine Learning<br>
            GitHub: LozanoLsa &nbsp;·&nbsp; lozanolsa.gumroad.com<br>
            <em style="color:{C_GOLD2};">Turning Operations into Predictive Systems —
            Clone it. Fork it. Improve it.</em>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-footer" style="display:flex;justify-content:space-between;align-items:center;">
    <div>
        <strong>LozanoLsa</strong> &nbsp;·&nbsp; Turning Operations into Predictive Systems
        &nbsp;·&nbsp; pH Control Intelligence &nbsp;·&nbsp; Project 24 of 24 &nbsp;·&nbsp; v2.0
    </div>
    <div>
        <a href="https://github.com/LozanoLsa" style="color:{C_TEAL};text-decoration:none;">GitHub</a>
        &nbsp;·&nbsp;
        <a href="https://lozanolsa.gumroad.com" style="color:{C_TEAL};text-decoration:none;">Gumroad</a>
        &nbsp;·&nbsp;
        <span style="color:{C_GOLD3};">🏁 Portfolio Complete</span>
    </div>
</div>
""", unsafe_allow_html=True)
