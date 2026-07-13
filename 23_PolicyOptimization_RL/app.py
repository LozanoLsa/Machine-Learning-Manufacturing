"""
app.py — Assembly Line Intelligence Dashboard
LozanoLsa · Project 23 · REINFORCE Policy Gradient · 2026 · FREE PROJECT

Algorithm: REINFORCE (Monte Carlo Policy Gradient) · Softmax Policy
Domain: Assembly Line — Production Decision Optimization
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="REINFORCE · Assembly Line Intelligence · LozanoLsa",
    layout="wide", page_icon="🏭",
    initial_sidebar_state="collapsed",
)

# ─── CSS — DISTILLATION STUDIO STYLE ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&family=Instrument+Serif:ital@0;1&display=swap');

:root {
    --bg:       #ffffff;
    --surface:  #f8fafc;
    --card:     #f8fafc;
    --card2:    #f1f5f9;
    --border:   #e2e8f0;
    --border2:  #cbd5e1;
    --green:    #16a34a;
    --green2:   #4ade80;
    --green3:   #86efac;
    --danger:   #dc2626;
    --warn:     #d97706;
    --blue:     #2563eb;
    --purp:     #7c3aed;
    --text:     #0f172a;
    --text2:    #1e293b;
    --muted:    #64748b;
    --muted2:   #94a3b8;
    --fh: 'Syne', sans-serif;
    --fm: 'JetBrains Mono', monospace;
    --fs: 'Instrument Serif', Georgia, serif;
}

.stApp { background: var(--bg) !important; color: var(--text); font-family: var(--fh); }
.block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSlider"] [role="slider"] { background: var(--green) !important;
    border: 2px solid var(--green2) !important; box-shadow: 0 0 6px rgba(22,163,74,0.3) !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { font-family: var(--fm) !important;
    font-size: 0.65rem !important; color: var(--green) !important; background: #fff !important;
    border: 1px solid var(--border) !important; padding: 1px 5px !important; border-radius: 3px !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--green) !important; }

[data-testid="stNumberInput"] input { font-family: var(--fm) !important;
    font-size: 1rem !important; font-weight: 600 !important; color: var(--text) !important;
    background: #fff !important; border: 1px solid var(--border2) !important; border-radius: 4px !important; }
[data-testid="stNumberInput"] button { background: var(--card2) !important;
    border: 1px solid var(--border) !important; color: var(--text) !important; }

[data-testid="stMetric"] { background: #fff !important;
    border: 1px solid var(--border) !important;
    border-top: 2px solid var(--green) !important;
    padding: 0.9rem 1rem !important; border-radius: 4px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important; }
[data-testid="stMetricLabel"] > div { font-family: var(--fm) !important; font-size: 0.6rem !important;
    text-transform: uppercase !important; letter-spacing: 0.16em !important; color: var(--muted) !important; }
[data-testid="stMetricValue"] > div { font-family: var(--fm) !important;
    font-size: 1.6rem !important; font-weight: 700 !important; color: var(--text) !important; }

[data-testid="stTabs"] [role="tablist"] { border-bottom: 2px solid var(--border) !important;
    background: var(--surface) !important; padding: 0 2.4rem !important; }
[data-testid="stTabs"] [role="tab"] { font-family: var(--fm) !important;
    font-size: 0.68rem !important; text-transform: uppercase !important;
    letter-spacing: 0.12em !important; color: var(--muted) !important;
    padding: 0.7rem 1.2rem !important; border: none !important;
    background: transparent !important; transition: all 0.2s !important; }
[data-testid="stTabs"] [role="tab"]:hover { color: var(--green) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: var(--green) !important;
    border-bottom: 2px solid var(--green) !important; font-weight: 600 !important; }
[data-testid="stTabsContent"] { padding: 1.6rem 2.4rem !important; }

[data-testid="stExpander"] { background: #fff !important;
    border: 1px solid var(--border) !important; border-radius: 4px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important; }
[data-testid="stExpander"] summary { font-family: var(--fm) !important;
    font-size: 0.75rem !important; color: var(--text) !important; font-weight: 600 !important; }

[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 4px !important; }
[data-testid="stDataFrame"] th { font-family: var(--fm) !important; font-size: 0.62rem !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
    background: var(--card2) !important; color: var(--muted) !important; }
[data-testid="stDataFrame"] td { font-family: var(--fm) !important;
    font-size: 0.72rem !important; color: var(--text2) !important; background: #fff !important; }

[data-testid="stAlert"] { border-radius: 4px !important;
    font-family: var(--fm) !important; font-size: 0.75rem !important; }

hr { border-color: var(--border) !important; }
[data-testid="stCaptionContainer"] p { font-family: var(--fm) !important;
    font-size: 0.63rem !important; color: var(--muted) !important; }
p, li { font-family: var(--fh) !important; font-size: 0.88rem !important; }

/* ── STUDIO COMPONENTS ── */
.studio-header { background: linear-gradient(135deg,#0f2027,#203a43,#1a3a2a);
    padding:2rem 2.4rem 1.6rem; margin-bottom:0; }
.studio-tag { font-family:var(--fm); font-size:0.6rem; color:var(--green2);
    text-transform:uppercase; letter-spacing:0.2em; margin-bottom:6px; }
.studio-title { font-family:var(--fh); font-size:2rem; font-weight:800;
    color:#fff; line-height:1.1; letter-spacing:-0.02em; }
.studio-subtitle { font-family:var(--fs); font-style:italic;
    font-size:0.95rem; color:#94a3b8; margin-top:6px; }
.studio-pill { display:inline-block; background:rgba(74,222,128,0.15);
    border:1px solid rgba(74,222,128,0.4); color:#86efac;
    font-family:var(--fm); font-size:0.6rem; letter-spacing:0.1em;
    text-transform:uppercase; padding:4px 12px; border-radius:20px;
    margin-right:6px; margin-top:12px; }
.studio-pill-free { display:inline-block; background:rgba(74,222,128,0.2);
    border:1px solid rgba(74,222,128,0.5); color:#4ade80;
    font-family:var(--fm); font-size:0.6rem; letter-spacing:0.1em;
    text-transform:uppercase; padding:4px 12px; border-radius:20px;
    margin-right:6px; margin-top:12px; font-weight:700; }

.section-label { font-family:var(--fh); font-size:1.1rem; font-weight:700;
    color:var(--text); margin-bottom:6px; margin-top:0; }
.section-desc { font-family:var(--fh); font-size:0.82rem; color:var(--muted);
    margin-bottom:1.2rem; line-height:1.5; }
.input-card { background:#fff; border:1px solid var(--border); border-radius:8px;
    padding:1.2rem 1.4rem; margin-bottom:12px; box-shadow:0 1px 4px rgba(0,0,0,0.05); }
.input-card-title { font-family:var(--fm); font-size:0.65rem; font-weight:600;
    color:var(--green); text-transform:uppercase; letter-spacing:0.15em; margin-bottom:10px; }
.lsa-footer { padding:1rem 2.4rem; border-top:1px solid var(--border);
    font-family:var(--fm); font-size:0.58rem; color:var(--muted);
    letter-spacing:0.1em; background:var(--surface); }
</style>
""", unsafe_allow_html=True)

# ─── MATPLOTLIB LIGHT ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":"white", "axes.facecolor":"#f8fafc",
    "axes.edgecolor":"#e2e8f0", "axes.labelcolor":"#0f172a",
    "xtick.color":"#64748b", "ytick.color":"#64748b",
    "text.color":"#0f172a", "grid.color":"#e2e8f0",
    "grid.linestyle":"--", "grid.alpha":0.5,
    "legend.facecolor":"white", "legend.edgecolor":"#e2e8f0",
})

C_GREEN  = "#16a34a"; C_GREEN2 = "#4ade80"; C_GREEN3 = "#86efac"
C_DANGER = "#dc2626"; C_WARN   = "#d97706"; C_BLUE   = "#2563eb"
C_PURP   = "#7c3aed"; C_TEXT   = "#0f172a"; C_MUTED  = "#64748b"
C_BORD   = "#e2e8f0"

ACTION_NAMES = ["Reassign Operator","Increase Speed","Decrease Speed",
                "Quick Maintenance","Redirect Flow","No Action"]
ACTION_ICONS = ["👷","⚡","🐢","🔧","↪️","—"]
ACTION_COLORS= [C_GREEN, C_WARN, C_MUTED, C_BLUE, C_GREEN, C_MUTED]

# ─── ENVIRONMENT & POLICY ─────────────────────────────────────────────────────
class AssemblyLineEnv:
    def __init__(self, n=6):
        self.n=n; self.action_space=6
        self.ct_min,self.ct_max=20,45
        self.wip_max,self.vel_max,self.mic_max=40,1.2,15
    def _norm(self,x,lo,hi): return (x-lo)/(hi-lo)
    def reset(self):
        self.ct=np.random.uniform(28,35,self.n); self.wip=np.random.randint(6,15)
        self.vel=np.random.uniform(0.80,1.00); self.p_f=np.random.uniform(0.05,0.12)
        self.micro=np.random.randint(0,5); self.efic=np.random.uniform(0.75,0.95)
        return self._state()
    def _state(self):
        return np.array([self._norm(c,self.ct_min,self.ct_max) for c in self.ct]+
                        [self.wip/self.wip_max,self.vel/self.vel_max,
                         self.p_f,self.micro/self.mic_max,self.efic],dtype=np.float32)
    def step(self,a):
        if a==0: self.efic+=np.random.uniform(0,0.05); self.micro=max(0,self.micro-1)
        elif a==1: self.vel+=0.03; self.ct-=np.random.uniform(0.3,0.8,self.n); self.p_f+=0.02
        elif a==2: self.vel-=0.03; self.ct+=np.random.uniform(0.5,1.2,self.n); self.p_f-=0.015
        elif a==3: self.micro=max(0,self.micro-3); self.p_f-=0.03; self.ct+=np.random.uniform(0.2,0.6,self.n)
        elif a==4: self.wip-=np.random.randint(1,4); self.vel+=0.01
        self.ct+=np.random.uniform(-0.5,0.5,self.n); self.wip+=np.random.randint(-1,2)
        self.micro+=np.random.randint(0,2); self.p_f+=np.random.uniform(-0.01,0.01)
        self.ct=np.clip(self.ct,20,45); self.wip=np.clip(self.wip,0,40)
        self.vel=np.clip(self.vel,0.5,1.2); self.p_f=np.clip(self.p_f,0,0.25)
        self.micro=np.clip(self.micro,0,15); self.efic=np.clip(self.efic,0.60,1.00)
        tp=(3600/np.mean(self.ct))*self.vel*self.efic
        r=1.2*tp-4.0*self.micro-1.5*self.wip-10.0*self.p_f
        done=self.wip>=40 or self.p_f>0.20
        if done: r-=200
        return self._state(),r,done,{"throughput":tp}

class SoftmaxPolicy:
    def __init__(self,state_dim,n_actions,lr=0.002):
        self.W=np.zeros((n_actions,state_dim)); self.b=np.zeros(n_actions); self.lr=lr
    def probs(self,s):
        lg=self.W@s+self.b; lg-=lg.max(); e=np.exp(lg); return e/e.sum()
    def select_action(self,s): return int(np.random.choice(len(self.b),p=self.probs(s)))
    def update(self,states,actions,returns):
        for s,a,G in zip(states,actions,returns):
            p=self.probs(s); I=np.zeros(len(self.b)); I[a]=1.0
            self.W+=self.lr*G*np.outer(I-p,s); self.b+=self.lr*G*(I-p)

def reinforce_train(env,policy,episodes=400,gamma=0.99,max_steps=100):
    rewards=[]; np.random.seed(42)
    for _ in range(episodes):
        s=env.reset(); sts,acts,rws=[],[],[]
        for _ in range(max_steps):
            a=policy.select_action(s); ns,r,done,_=env.step(a)
            sts.append(s); acts.append(a); rws.append(r); s=ns
            if done: break
        G=0; rets=[]
        for r in reversed(rws): G=r+gamma*G; rets.insert(0,G)
        R=np.array(rets)
        if R.std()>1e-6: R=(R-R.mean())/R.std()
        policy.update(sts,acts,R.tolist()); rewards.append(sum(rws))
    return rewards

@st.cache_resource
def get_model():
    env=AssemblyLineEnv(); p=SoftmaxPolicy(11,6,0.002)
    with st.spinner("Training REINFORCE agent — 400 episodes..."):
        rewards=reinforce_train(env,p)
    return env,p,rewards

@st.cache_data
def load_data():
    for path in ["Data_people.csv","23_REINFORCE_Assembly/Data_people.csv"]:
        try: return pd.read_csv(path)
        except FileNotFoundError: continue
    st.error("Data_people.csv not found. Place the file in the same folder as app.py.")
    st.stop()

env, policy, episode_rewards = get_model()
df = load_data()

# ─── STUDIO HEADER ────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="studio-header">
    <div class="studio-tag">LozanoLsa · Project 23 · Reinforcement Learning · Final RL</div>
    <div class="studio-title">Assembly Line Intelligence</div>
    <div class="studio-subtitle">REINFORCE learns which intervention maximises throughput — without being told the rules of the factory.</div>
    <div>
        <span class="studio-pill">REINFORCE</span>
        <span class="studio-pill">SOFTMAX POLICY</span>
        <span class="studio-pill">6 ACTIONS</span>
        <span class="studio-pill">400 EPISODES</span>
        <span class="studio-pill">W MATRIX</span>
        <span class="studio-pill-free">FREE PROJECT</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── KPI ROW ─────────────────────────────────────────────────────────────────
st.markdown('<div style="padding:1.2rem 2.4rem 0.8rem;background:#f8fafc;border-bottom:1px solid #e2e8f0;">', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Training Episodes",   "400",                            "γ=0.99 · lr=0.002")
k2.metric("Line Stations",       "6",                              "Independent cycle times")
k3.metric("Action Space",        "6 interventions",                "Reassign→No Action")
k4.metric("Final Avg Reward",    f"{np.mean(episode_rewards[-50:]):.0f}", "Last 50 episodes")
st.markdown('</div>', unsafe_allow_html=True)

# ─── MAIN: SECTION 1 + SECTION 2 ─────────────────────────────────────────────
st.markdown('<div style="padding:1.6rem 2.4rem;">', unsafe_allow_html=True)
col_inp, col_vis = st.columns([1, 1.3])

with col_inp:
    st.markdown('<p class="section-label">1. Assembly Line State</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Set the current production conditions. The trained REINFORCE policy will recommend the optimal intervention in real time.</p>', unsafe_allow_html=True)

    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="input-card-title">Production Parameters</div>', unsafe_allow_html=True)
    wip_n   = st.slider("WIP Level (normalized)",        0.0, 1.0, 0.30, 0.05)
    speed_n = st.slider("Line Speed (normalized)",       0.4, 1.0, 0.80, 0.02)
    fail_p  = st.slider("Failure Probability",           0.0, 0.25,0.07, 0.01)
    micro_n = st.slider("Micro-Stops (normalized)",      0.0, 1.0, 0.20, 0.05)
    op_eff  = st.slider("Operator Efficiency",           0.6, 1.0, 0.85, 0.01)
    ct_val  = st.slider("Avg Cycle Time (normalized)",   0.1, 0.9, 0.40, 0.05)
    st.markdown('</div>', unsafe_allow_html=True)

    # Quick scenarios
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="input-card-title">Quick Scenarios</div>', unsafe_allow_html=True)
    scenario = st.selectbox("Load preset:", [
        "— custom —",
        "Optimal: low WIP, high speed, no failures",
        "Crisis: high WIP, high failure risk",
        "Bottleneck: slow line, high cycle time",
        "Fatigue: low operator efficiency",
    ])
    st.markdown('</div>', unsafe_allow_html=True)

    presets = {
        "Optimal: low WIP, high speed, no failures":
            dict(wip_n=0.10, speed_n=0.95, fail_p=0.02, micro_n=0.05, op_eff=0.95, ct_val=0.20),
        "Crisis: high WIP, high failure risk":
            dict(wip_n=0.80, speed_n=0.60, fail_p=0.18, micro_n=0.70, op_eff=0.72, ct_val=0.70),
        "Bottleneck: slow line, high cycle time":
            dict(wip_n=0.50, speed_n=0.50, fail_p=0.06, micro_n=0.30, op_eff=0.80, ct_val=0.85),
        "Fatigue: low operator efficiency":
            dict(wip_n=0.35, speed_n=0.75, fail_p=0.10, micro_n=0.55, op_eff=0.65, ct_val=0.45),
    }
    if scenario in presets:
        pr       = presets[scenario]
        wip_n    = pr["wip_n"]; speed_n = pr["speed_n"]; fail_p  = pr["fail_p"]
        micro_n  = pr["micro_n"]; op_eff = pr["op_eff"]; ct_val  = pr["ct_val"]

with col_vis:
    st.markdown('<p class="section-label">2. Policy Recommendation</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">REINFORCE evaluates all 6 possible interventions and recommends the one that maximises expected cumulative reward.</p>', unsafe_allow_html=True)

    state_in = np.array([ct_val]*6 + [wip_n, speed_n, fail_p, micro_n, op_eff], dtype=np.float32)
    probs_in = policy.probs(state_in)
    top_a    = int(np.argmax(probs_in))
    top3     = np.argsort(probs_in)[::-1][:3]
    rec_col  = ACTION_COLORS[top_a]
    badge_bg = {"Reassign Operator":"#f0fdf4","Increase Speed":"#fffbeb","Decrease Speed":"#f8fafc",
                "Quick Maintenance":"#eff6ff","Redirect Flow":"#f0fdf4","No Action":"#f8fafc"
                }.get(ACTION_NAMES[top_a],"#f8fafc")

    # Recommendation card
    st.markdown(f"""
    <div style="background:#fff;border:1px solid {C_BORD};border-top:3px solid {rec_col};
                border-radius:8px;padding:1.4rem 1.6rem;margin-bottom:14px;
                box-shadow:0 2px 8px rgba(0,0,0,0.07);text-align:center;">
        <div style="font-size:2.2rem;margin-bottom:6px;">{ACTION_ICONS[top_a]}</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;
                    color:{rec_col};">{ACTION_NAMES[top_a].upper()}</div>
        <div style="margin-top:10px;">
            <span style="background:{badge_bg};color:{rec_col};
                         font-family:var(--fm);font-size:0.7rem;font-weight:700;
                         padding:5px 16px;border-radius:20px;
                         border:1px solid {rec_col}44;">
                Policy confidence: {probs_in[top_a]:.1%}
            </span>
        </div>
        <div style="margin-top:14px;display:flex;justify-content:center;gap:16px;
                    font-family:var(--fm);font-size:0.72rem;color:{C_MUTED};">
            <span>🥇 {ACTION_NAMES[top3[0]]} — {probs_in[top3[0]]:.1%}</span>
            <span>·</span>
            <span>🥈 {ACTION_NAMES[top3[1]]} — {probs_in[top3[1]]:.1%}</span>
            <span>·</span>
            <span>🥉 {ACTION_NAMES[top3[2]]} — {probs_in[top3[2]]:.1%}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Action probability bars
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bar_c = [ACTION_COLORS[i] if i == top_a else "#e2e8f0" for i in range(6)]
    bars  = ax.barh([f"{ic} {nm}" for ic, nm in zip(ACTION_ICONS, ACTION_NAMES)],
                    probs_in, color=bar_c, edgecolor="white", height=0.55)
    for i, bar in enumerate(bars):
        bar.set_alpha(1.0 if i == top_a else 0.75)
    ax.axvline(1/6, color=C_DANGER, lw=1.5, ls="--", alpha=0.7, label="Random baseline (1/6)")
    for bar, p in zip(bars, probs_in):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f"{p:.3f}", va="center", fontsize=9, color=C_TEXT)
    ax.set_xlabel("Action Probability")
    ax.set_title("Policy Distribution — REINFORCE", fontsize=10, fontweight="bold", color=C_TEXT)
    ax.legend(fontsize=8); ax.grid(True, axis="x", alpha=0.5)
    ax.set_xlim(0, max(probs_in)*1.25)
    for sp in ax.spines.values(): sp.set_edgecolor(C_BORD)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()
    st.caption("Green bar = recommended action. Red dashed = random policy baseline (1/6 ≈ 16.7%).")

st.markdown('</div>', unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "DATA EXPLORER", "LEARNING CURVE", "POLICY WEIGHTS", "ACTION PLAN"
])

# ══ TAB 1 ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-label">3. Random Baseline Dataset — Performance Floor</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Collected using a random policy (uniform action selection). This is the performance floor the REINFORCE agent must exceed.</p>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Records",          f"{len(df):,}")
    m2.metric("Episodes",         "300",                       "Random policy baseline")
    m3.metric("Mean Reward/Step", f"{df['reward'].mean():.1f}", "REINFORCE exceeds this")
    m4.metric("Mean Throughput",  f"{df['throughput_uph'].mean():.1f} UPH", "Units per hour")

    st.divider()
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Reward distribution — random policy**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["reward"], bins=50, color=C_BLUE, alpha=0.80, edgecolor="white")
        ax.axvline(df["reward"].mean(), color=C_DANGER, lw=2, ls="--",
                   label=f"Mean = {df['reward'].mean():.1f}")
        ax.set_xlabel("Step Reward"); ax.set_ylabel("Count")
        ax.set_title("Random Policy Reward Distribution", fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)
        for sp in ax.spines.values(): sp.set_edgecolor(C_BORD)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with cb:
        st.markdown("**Action frequency — random policy**")
        act_map = {i: n for i, n in enumerate(ACTION_NAMES)}
        action_counts = df["action"].value_counts().sort_index()
        action_counts.index = [f"{ACTION_ICONS[i]} {ACTION_NAMES[i]}" for i in action_counts.index]
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        bars2 = ax2.barh(action_counts.index, action_counts.values,
                         color=C_BLUE, alpha=0.80, edgecolor="white", height=0.55)
        for bar, v in zip(bars2, action_counts.values):
            ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                     f"{v:,}", va="center", fontsize=9, color=C_TEXT)
        ax2.set_xlabel("Times selected"); ax2.set_title("Action Frequency (Random)", fontsize=10, fontweight="bold")
        ax2.grid(True, axis="x", alpha=0.4)
        for sp in ax2.spines.values(): sp.set_edgecolor(C_BORD)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True); plt.close()

    st.divider()
    st.markdown("**Action preference: Random vs. REINFORCE**")
    rand_probs   = [1/6]*6
    sampled      = [policy.probs(env.reset()) for _ in range(300)]
    trained_probs= np.mean(sampled, axis=0)
    comp_df = pd.DataFrame({
        "Action":    [f"{ic} {nm}" for ic, nm in zip(ACTION_ICONS, ACTION_NAMES)],
        "Random":    [f"{p:.3f}" for p in rand_probs],
        "REINFORCE": [f"{p:.3f}" for p in trained_probs],
        "Δ vs random": [f"{(t - 1/6):+.3f}" for t in trained_probs],
        "Signal":    ["▲ Over-weights" if t > 1/6*1.1 else "▼ Suppresses" if t < 1/6*0.9 else "≈ Neutral"
                      for t in trained_probs],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.caption("REINFORCE over-weights Redirect Flow and Reassign Operator. "
               "Decrease Speed is suppressed — the policy learned it hurts throughput without enough benefit.")

# ══ TAB 2 ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-label">4. REINFORCE Learning Curve</p>', unsafe_allow_html=True)
    ca, cb = st.columns([1.4, 1])
    with ca:
        window = 30
        smooth = pd.Series(episode_rewards).rolling(window).mean()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(episode_rewards, alpha=0.20, color=C_GREEN, lw=0.8, label="Raw")
        ax.plot(smooth, color=C_GREEN, lw=2.5, label=f"{window}-ep moving avg")
        ax.axhline(np.mean(episode_rewards[-50:]), color=C_DANGER, ls="--", lw=1.8,
                   label=f"Final avg = {np.mean(episode_rewards[-50:]):.0f}")
        ax.axhline(df["reward"].mean() * 100, color=C_BLUE, ls=":", lw=1.5, alpha=0.8,
                   label=f"Random baseline ≈ {df['reward'].mean()*100:.0f}")
        ax.set_xlabel("Episode"); ax.set_ylabel("Total Episode Reward")
        ax.set_title("REINFORCE Training Progress — 400 Episodes", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)
        for sp in ax.spines.values(): sp.set_edgecolor(C_BORD)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()
        st.caption("Reward increases as the policy learns to avoid failure states and redirect flow earlier. "
                   "High variance is expected — stochastic environment + Monte Carlo updates.")

    with cb:
        st.markdown("**Training hyperparameters**")
        params = pd.DataFrame({
            "Parameter": ["Algorithm","Episodes","γ (discount)","α (learning rate)",
                          "Max steps/ep","Policy","State dim","Action dim"],
            "Value":     ["REINFORCE","400","0.99","0.002","100",
                          "Softmax (linear)","11","6"],
        })
        st.dataframe(params, use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-left:3px solid {C_GREEN};
                    border-radius:4px;padding:10px 14px;margin-top:10px;">
            <div style="font-family:var(--fm);font-size:0.6rem;color:{C_GREEN};
                        text-transform:uppercase;letter-spacing:.15em;margin-bottom:5px;">Why γ=0.99?</div>
            <div style="font-family:var(--fm);font-size:0.72rem;color:{C_TEXT};line-height:1.6;">
                Assembly line rewards are delayed — a good action now (e.g. flow redirect)
                prevents WIP buildup 10 steps later. High γ ensures the policy credits these
                long-range cause-effect relationships correctly.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══ TAB 3 ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-label">5. Policy Weight Matrix — What REINFORCE Learned</p>', unsafe_allow_html=True)
    feature_names = ["CT-S1","CT-S2","CT-S3","CT-S4","CT-S5","CT-S6",
                     "WIP","Speed","FailP","MicroSt","OperEff"]
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**W matrix heatmap — feature influence per action**")
        fig, ax = plt.subplots(figsize=(7.5, 5))
        im = ax.imshow(policy.W, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, label="Weight", shrink=0.85)
        ax.set_xticks(range(11))
        ax.set_xticklabels(feature_names, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(6))
        ax.set_yticklabels([f"{ic} {nm[:14]}" for ic, nm in zip(ACTION_ICONS, ACTION_NAMES)], fontsize=8)
        ax.set_title("W Matrix — Feature Influence per Action", fontsize=11, fontweight="bold")
        for sp in ax.spines.values(): sp.set_edgecolor(C_BORD)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()
        st.caption("Green = feature strongly drives this action. Red = feature suppresses it. "
                   "Each row is the policy's 'opinion' about when to use that action.")

    with cb:
        st.markdown("**State → Reward correlation (baseline data)**")
        num_cols = [c for c in df.columns if c not in ["action","episode","step","reward"]]
        corr     = df[num_cols + ["reward"]].corr()["reward"].drop("reward").sort_values()
        fig2, ax2= plt.subplots(figsize=(6, 4.5))
        bar_c2   = [C_DANGER if v < 0 else C_GREEN for v in corr.values]
        bars2    = ax2.barh(corr.index, corr.values, color=bar_c2,
                            edgecolor="white", height=0.55, alpha=0.85)
        for bar, val in zip(bars2, corr.values):
            off = 0.004 if val >= 0 else -0.004
            ax2.text(val + off, bar.get_y() + bar.get_height()/2,
                     f"{val:+.3f}", va="center",
                     ha="left" if val >= 0 else "right", fontsize=8, color=C_TEXT)
        ax2.axvline(0, color=C_MUTED, lw=1)
        ax2.set_xlabel("Pearson r with Step Reward")
        ax2.set_title("State → Reward Correlation", fontsize=10, fontweight="bold")
        ax2.grid(True, axis="x", alpha=0.4)
        for sp in ax2.spines.values(): sp.set_edgecolor(C_BORD)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True); plt.close()
        st.caption("operator_efficiency and speed_norm drive reward up. "
                   "failure_prob and wip_norm drag it down.")

        st.markdown("**W matrix — numerical table (top weights)**")
        w_df = pd.DataFrame(policy.W, index=ACTION_NAMES, columns=feature_names).round(4)
        st.dataframe(w_df.style.background_gradient(cmap="RdYlGn", axis=None, vmin=-1, vmax=1),
                     use_container_width=True)

# ══ TAB 4 ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-label">6. Operational Action Guide — 6 Interventions Decoded</p>', unsafe_allow_html=True)

    actions_detail = [
        {"name":"[0] Reassign Operator","icon":"👷","color":C_GREEN,"severity":"Standard",
         "trigger":"operator_efficiency < 0.80","effect":"↑ efficiency +0–5%  ·  ↓ micro-stops −1",
         "when":"Low operator efficiency is limiting throughput without other active failures",
         "avoid":"When WIP is critically high — flow redirection takes priority"},
        {"name":"[1] Increase Speed","icon":"⚡","color":C_WARN,"severity":"Use with Caution",
         "trigger":"speed_norm < 0.75 AND failure_prob < 0.08","effect":"↑ velocity +0.03  ·  ↑ failure_prob +0.02  ·  ↓ cycle times",
         "when":"Line is running slow and failure risk is low — safe to push throughput",
         "avoid":"When failure_prob > 0.10 — speed amplifies failure risk exponentially"},
        {"name":"[2] Decrease Speed","icon":"🐢","color":C_MUTED,"severity":"Protective",
         "trigger":"failure_prob > 0.15","effect":"↓ velocity −0.03  ·  ↓ failure_prob −0.015  ·  ↑ cycle times",
         "when":"Failure probability is elevated — sacrifice throughput to protect the line",
         "avoid":"As first response — REINFORCE deprioritizes this action in most states"},
        {"name":"[3] Quick Maintenance","icon":"🔧","color":C_BLUE,"severity":"Reactive",
         "trigger":"micro_stops > 0.50 OR failure_prob > 0.12","effect":"↓ micro-stops −3  ·  ↓ failure_prob −0.03  ·  ↑ cycle time +0.2–0.6",
         "when":"Micro-stop accumulation or failure risk requires immediate intervention",
         "avoid":"When WIP is the primary constraint — redirect flow first"},
        {"name":"[4] Redirect Flow","icon":"↪️","color":C_GREEN,"severity":"Preferred",
         "trigger":"wip_norm > 0.40","effect":"↓ WIP −1 to −3 units  ·  ↑ speed +0.01",
         "when":"WIP accumulation is building congestion risk",
         "avoid":"When operator efficiency is the binding constraint"},
        {"name":"[5] No Action","icon":"—","color":C_MUTED,"severity":"Passive",
         "trigger":"All KPIs within normal range","effect":"No change to any variable",
         "when":"Line is running stably — interventions may introduce unnecessary variance",
         "avoid":"When any KPI is trending toward its limit"},
    ]

    for act in actions_detail:
        trained_p = float(policy.probs(env.reset())[actions_detail.index(act)])
        delta_vs_rand = (trained_p - 1/6) * 100
        with st.expander(f"{act['icon']}  {act['name']}  ·  {act['severity']}", expanded=False):
            cl, cr = st.columns([2.5, 1])
            with cl:
                for lbl, text, bg, border in [
                    ("Trigger Condition", act["trigger"],  "#f8fafc",   C_BORD),
                    ("Effect",           act["effect"],   "#f0fdf4",   "#bbf7d0"),
                    ("Use When",         act["when"],     "#eff6ff",   "#bfdbfe"),
                    ("Avoid When",       act["avoid"],    "#fffbeb",   "#fde68a"),
                ]:
                    st.markdown(f"""
                    <div style="background:{bg};border:1px solid {border};
                                border-left:3px solid {act['color']};border-radius:4px;
                                padding:8px 12px;margin-bottom:7px;">
                        <div style="font-family:var(--fm);font-size:0.58rem;color:{C_MUTED};
                                    text-transform:uppercase;letter-spacing:.15em;margin-bottom:4px;">{lbl}</div>
                        <div style="font-family:var(--fm);font-size:0.72rem;color:{C_TEXT};
                                    line-height:1.6;">{text}</div>
                    </div>""", unsafe_allow_html=True)
            with cr:
                delta_c = C_GREEN if delta_vs_rand > 0 else C_DANGER
                st.markdown(f"""
                <div style="background:#fff;border:1px solid {C_BORD};
                            border-top:3px solid {act['color']};border-radius:4px;
                            padding:1rem;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                    <div style="font-family:var(--fm);font-size:0.58rem;color:{C_MUTED};
                                text-transform:uppercase;letter-spacing:.1em;">Trained Policy Pref</div>
                    <div style="font-family:var(--fm);font-size:1.4rem;font-weight:700;
                                color:{act['color']};margin:6px 0;">{trained_p:.1%}</div>
                    <div style="font-family:var(--fm);font-size:0.72rem;
                                color:{delta_c};font-weight:600;">{delta_vs_rand:+.1f}% vs random</div>
                </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
    <div style="background:#fff;border:1px solid {C_BORD};border-radius:8px;
                padding:1.2rem 1.6rem;box-shadow:0 2px 6px rgba(0,0,0,0.06);">
        <div style="font-family:var(--fh);font-size:1rem;font-weight:700;
                    color:{C_TEXT};margin-bottom:12px;">REINFORCE vs. Rule-Based Control</div>
        <table style="width:100%;font-family:var(--fm);font-size:0.72rem;border-collapse:collapse;">
            <tr style="border-bottom:1px solid {C_BORD};background:#f8fafc;">
                <th style="padding:8px;text-align:left;color:{C_MUTED};">Criterion</th>
                <th style="padding:8px;color:{C_MUTED};">Rule-Based (IF-THEN)</th>
                <th style="padding:8px;color:{C_MUTED};">REINFORCE (Learned)</th>
            </tr>
            <tr style="border-bottom:1px solid {C_BORD};">
                <td style="padding:8px;color:{C_TEXT};">Multi-variable interactions</td>
                <td style="padding:8px;text-align:center;color:{C_DANGER};">Hard-coded combinations</td>
                <td style="padding:8px;text-align:center;color:{C_GREEN};">Learned automatically</td>
            </tr>
            <tr style="border-bottom:1px solid {C_BORD};">
                <td style="padding:8px;color:{C_TEXT};">New environment layout</td>
                <td style="padding:8px;text-align:center;color:{C_DANGER};">Rewrite all rules</td>
                <td style="padding:8px;text-align:center;color:{C_GREEN};">Retrain policy</td>
            </tr>
            <tr style="border-bottom:1px solid {C_BORD};">
                <td style="padding:8px;color:{C_TEXT};">Delay-credit assignment</td>
                <td style="padding:8px;text-align:center;color:{C_DANGER};">Not possible</td>
                <td style="padding:8px;text-align:center;color:{C_GREEN};">Returns + γ discounting</td>
            </tr>
            <tr>
                <td style="padding:8px;color:{C_TEXT};">Interpretability</td>
                <td style="padding:8px;text-align:center;color:{C_GREEN};">Fully transparent</td>
                <td style="padding:8px;text-align:center;color:{C_WARN};">W matrix (partial)</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-footer">
    LozanoLsa · Turning Operations into Predictive Systems · Assembly Line Intelligence · Project 23 · v2.0
    &nbsp;·&nbsp;
    <a href="https://github.com/LozanoLsa" style="color:{C_GREEN};text-decoration:none;">GitHub</a>
    &nbsp;·&nbsp;
    <a href="https://lozanolsa.gumroad.com" style="color:{C_GREEN};text-decoration:none;">Gumroad</a>
</div>
""", unsafe_allow_html=True)
