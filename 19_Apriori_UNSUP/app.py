"""
app.py — HR People Analytics Dashboard
LozanoLsa · Project 19 · Apriori Association Rules · 2026

Algorithm: Apriori + Association Rules (mlxtend)
Domain: People Analytics — HR Risk Detection via Rule Mining
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Apriori · HR People Analytics",
    page_icon="👥",
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
.lsa-chip-free { display: inline-block; background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.3); color: #4ade80; font-family: var(--fm); font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 2px 8px; border-radius: 2px; margin-right: 5px; }
.lsa-section { font-family: var(--fm); font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid var(--border); }
.lsa-footer { margin-top: 2.5rem; padding-top: 0.8rem; border-top: 1px solid var(--border); font-family: var(--fm); font-size: 0.58rem; color: var(--muted); letter-spacing: 0.1em; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATA_PATH     = "people_data_analysis.csv"
DATA_PATH_ALT = "19_Apriori_HR_People/people_data_analysis.csv"

HR_FEATURES = ["department", "shift", "seniority", "contract_type",
               "remote_work", "training_status", "overtime", "manager_rating"]
OUTCOMES    = ["absenteeism", "performance", "attrition"]
CRITICAL_OUTCOMES = ["attrition=Yes", "attrition=No", "performance=High",
                     "performance=Low", "absenteeism=High", "absenteeism=Low"]

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

OUTCOME_COLORS = {
    "attrition=Yes":    C_DANGER,
    "attrition=No":     C_OK,
    "performance=High": C_BLUE,
    "performance=Low":  C_WARN,
    "absenteeism=High": C_PURP,
    "absenteeism=Low":  C_OK,
}
OUTCOME_ICONS = {
    "attrition=Yes":    "🔴",
    "attrition=No":     "🟢",
    "performance=High": "⭐",
    "performance=Low":  "⚠️",
    "absenteeism=High": "📉",
    "absenteeism=Low":  "✅",
}

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
    st.error("people_data_analysis.csv not found. Place the file in the same folder as app.py.")
    st.stop()

@st.cache_resource
def build_rules(df):
    def encode(df):
        return [[f"{c}={v}" for c, v in zip(df.columns, row)] for _, row in df.iterrows()]
    te     = TransactionEncoder()
    df_enc = pd.DataFrame(te.fit_transform(encode(df)), columns=te.columns_)
    fi     = apriori(df_enc, min_support=0.04, use_colnames=True, max_len=4)
    rules  = association_rules(fi, metric="lift", min_threshold=1.0)
    rules_s  = rules[rules["consequents"].apply(lambda x: len(x) == 1)].copy()
    hr_rules = rules_s[rules_s["consequents"].apply(
        lambda x: any(c in CRITICAL_OUTCOMES for c in list(x)))].copy()
    hr_rules = hr_rules.sort_values("lift", ascending=False).reset_index(drop=True)
    return df_enc, fi, rules_s, hr_rules, te

df = load_data()
df_enc, fi, rules_s, hr_rules, te = build_rules(df)

baselines = {}
for col in OUTCOMES:
    for val in df[col].unique():
        baselines[f"{col}={val}"] = (df[col] == val).mean()

def fmt_items(itemset):
    return " + ".join(sorted([i.replace("_", " ") for i in itemset]))

def analyze_profile(profile_dict):
    items   = frozenset(f"{k}={v}" for k, v in profile_dict.items())
    matched = [r for _, r in hr_rules.iterrows() if r["antecedents"].issubset(items)]
    if not matched:
        return pd.DataFrame()
    return pd.DataFrame(matched).nlargest(8, "lift")

att_rate = (df["attrition"]   == "Yes").mean()
abs_rate = (df["absenteeism"] == "High").mean()

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-header">
    <div class="lsa-project-tag">ML Project #19 · Apriori · Association Rules · HR People Analytics</div>
    <div class="lsa-title">Rules the Manager Never Wrote Down</div>
    <div class="lsa-tagline">Night shift + no remote + no training = attrition. Apriori reads the pattern before HR feels the loss.</div>
    <div style="margin-top:10px;">
        <span class="lsa-chip">APRIORI</span>
        <span class="lsa-chip">{len(hr_rules):,} HR RULES</span>
        <span class="lsa-chip">8 FEATURES · 3 OUTCOMES</span>
        <span class="lsa-chip">LIFT · CONFIDENCE · SUPPORT</span>
        <span class="lsa-chip-free">FREE PROJECT</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TOP KPI ROW ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Employees",  f"{len(df):,}",        "HR records analysed")
k2.metric("HR Rules Found",   f"{len(hr_rules):,}",  "Pointing to outcomes")
k3.metric("Attrition Rate",   f"{att_rate:.1%}",     "Baseline to beat")
k4.metric("High Absenteeism", f"{abs_rate:.1%}",     "Preventable with conditions")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "DATA EXPLORER", "SUPPORT & RULES", "PEOPLE RISK ANALYZER", "RULE LIBRARY", "ACTION PLAN"
])

# ══ TAB 1 ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="lsa-section">// Employee dataset — 1,748 people analytics records</div>',
                unsafe_allow_html=True)
    cl, cr = st.columns([3, 2])
    with cl:
        st.markdown('<div class="lsa-section">// Dataset sample</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True, height=310)
    with cr:
        st.markdown('<div class="lsa-section">// Outcome distributions</div>',
                    unsafe_allow_html=True)
        fig0, axes0 = plt.subplots(1, 3, figsize=(7, 3.5))
        fig0.patch.set_facecolor(C_BG)
        for ax, col, colors in zip(
            axes0,
            ["attrition", "absenteeism", "performance"],
            [[C_OK, C_DANGER], [C_OK, C_PURP], [C_BLUE, C_WARN]],
        ):
            ax.set_facecolor(C_CARD)
            for sp in ax.spines.values(): sp.set_edgecolor("#1e2d45")
            vc = df[col].value_counts()
            ax.bar(vc.index, vc.values, color=colors[:len(vc)], alpha=0.85, edgecolor="#1e2d45")
            for i, (k, v) in enumerate(vc.items()):
                ax.text(i, v + 5, str(v), ha="center", fontsize=9, color=C_TEXT)
            ax.set_title(col.capitalize(), color=C_TEXT, fontsize=9, fontweight="bold")
            ax.tick_params(colors=C_MUTED, labelsize=8)
        plt.tight_layout()
        st.pyplot(fig0, use_container_width=True); plt.close()

    st.divider()
    st.markdown('<div class="lsa-section">// Attrition rate by key categorical features</div>',
                unsafe_allow_html=True)
    feats_plot = ["shift", "seniority", "remote_work", "training_status", "overtime"]
    fig1, axes1 = plt.subplots(1, 5, figsize=(15, 4))
    fig1.patch.set_facecolor(C_BG)
    for ax, feat in zip(axes1, feats_plot):
        ax.set_facecolor(C_CARD)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e2d45")
        rates = df.groupby(feat)["attrition"].apply(lambda x: (x == "Yes").mean() * 100)
        bar_c = [C_DANGER if r > att_rate * 100 * 1.1 else C_ORANGE for r in rates.values]
        ax.bar(rates.index, rates.values, color=bar_c, alpha=0.85, edgecolor="#1e2d45")
        ax.axhline(att_rate * 100, color="white", ls="--", lw=1.2, alpha=0.7)
        ax.set_title(feat.replace("_", " ").title(), color=C_TEXT, fontsize=8.5, fontweight="bold")
        ax.set_ylabel("Attrition %" if feat == feats_plot[0] else "")
        ax.tick_params(colors=C_MUTED, labelsize=7)
        for label in ax.get_xticklabels():
            label.set_rotation(25); label.set_ha("right")
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True); plt.close()
    st.caption("Red bars = above-average attrition rate. White dashed = overall baseline.")

    st.divider()
    st.markdown('<div class="lsa-section">// Frequent itemset support distribution</div>',
                unsafe_allow_html=True)
    fig2, ax2 = dark_fig(10, 3.5)
    ax2.hist(fi["support"], bins=40, color=C_ORANGE, alpha=0.80, edgecolor="#1e2d45")
    ax2.axvline(0.04, color=C_DANGER, ls="--", lw=1.8, label="min_support = 0.04")
    ax2.set_xlabel("Support"); ax2.set_ylabel("Frequency")
    ax2.set_title(f"Support distribution — {len(fi):,} frequent itemsets found", color=C_TEXT, fontweight="bold")
    ax2.legend(facecolor=C_CARD, labelcolor=C_TEXT, fontsize=9)
    ax2.grid(axis="y", alpha=0.15, color="#1e2d45")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True); plt.close()

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="lsa-section">// Rule quality — support, confidence, lift</div>',
                unsafe_allow_html=True)
    st.caption("Good rules have high lift (≫1.0), meaningful support (≥4%), and high confidence.")

    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="lsa-section">// Lift vs confidence scatter</div>',
                    unsafe_allow_html=True)
        fig_sc, ax_sc = dark_fig(6.5, 5)
        for outcome in CRITICAL_OUTCOMES:
            mask = hr_rules["consequents"].apply(lambda x: outcome in x)
            sub  = hr_rules[mask]
            if len(sub):
                ax_sc.scatter(sub["confidence"], sub["lift"],
                              color=OUTCOME_COLORS.get(outcome, C_ORANGE),
                              alpha=0.5, s=20, label=outcome, edgecolors="none")
        ax_sc.axhline(1.0, color="white", ls="--", lw=1.0, alpha=0.5)
        ax_sc.set_xlabel("Confidence"); ax_sc.set_ylabel("Lift")
        ax_sc.set_title("Rule Quality Map", color=C_TEXT, fontweight="bold")
        ax_sc.legend(facecolor=C_CARD, labelcolor=C_TEXT, fontsize=7)
        ax_sc.grid(True, alpha=0.1, color="#1e2d45")
        fig_sc.tight_layout()
        st.pyplot(fig_sc, use_container_width=True); plt.close()
        st.caption("Points above lift=1.0 = non-random associations. Higher = stronger rule.")

    with cb:
        st.markdown('<div class="lsa-section">// Top 12 rules by lift — all outcomes</div>',
                    unsafe_allow_html=True)
        top12 = hr_rules.head(12).copy()
        top12["ant_str"] = top12["antecedents"].apply(fmt_items)
        top12["cons"]    = top12["consequents"].apply(lambda x: list(x)[0])
        fig_top, ax_top = dark_fig(6.5, 5)
        bar_c = [OUTCOME_COLORS.get(c, C_ORANGE) for c in top12["cons"]]
        bars  = ax_top.barh(range(len(top12)), top12["lift"].values,
                            color=bar_c, alpha=0.82, edgecolor="none", height=0.65)
        ax_top.set_yticks(range(len(top12)))
        ax_top.set_yticklabels(
            [f"{r['cons'].split('=')[0][:8]}={r['cons'].split('=')[1][:3]}"
             for _, r in top12.iterrows()],
            color=C_TEXT, fontsize=8
        )
        for bar, lift in zip(bars, top12["lift"]):
            ax_top.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{lift:.3f}", va="center", fontsize=8, color=C_TEXT)
        ax_top.axvline(1.0, color="white", ls="--", lw=1.0, alpha=0.5)
        ax_top.set_xlabel("Lift")
        ax_top.set_title("Top 12 Rules by Lift", color=C_TEXT, fontweight="bold")
        ax_top.grid(axis="x", alpha=0.1, color="#1e2d45")
        fig_top.tight_layout()
        st.pyplot(fig_top, use_container_width=True); plt.close()

    st.divider()
    st.markdown('<div class="lsa-section">// Rule metric explanations</div>',
                unsafe_allow_html=True)
    for name, expl in {
        "Support":    "Fraction of employees matching the full rule (antecedents + consequent). Low support = rare pattern.",
        "Confidence": "Given the antecedents, how often the consequent occurs. P(outcome | conditions).",
        "Lift":       "How much more likely the consequent is given the antecedents vs by chance. Lift > 1.0 = positive association.",
    }.items():
        with st.expander(f"{name}"):
            st.write(expl)

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="lsa-section">// People risk analyzer</div>',
                unsafe_allow_html=True)
    st.caption("Select an employee profile — the engine scans all Apriori rules that match and ranks risks by lift.")

    # Result LEFT · Controls RIGHT
    col_out, col_inp = st.columns([3, 2])

    with col_inp:
        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);
                    border-left:3px solid var(--accent);border-radius:2px;
                    padding:1rem 1.2rem;margin-bottom:14px;">
            <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                        text-transform:uppercase;letter-spacing:.18em;">// Employee profile</div>
        </div>
        """, unsafe_allow_html=True)
        p_dept  = st.selectbox("Department",      ["Engineering", "HR", "Operations", "Sales"])
        p_shift = st.selectbox("Shift",           ["Day", "Night"])
        p_sen   = st.selectbox("Seniority",       ["Junior", "Mid", "Senior"])
        p_ct    = st.selectbox("Contract Type",   ["Permanent", "Temporary"])
        p_rm    = st.selectbox("Remote Work",     ["No", "Yes"])
        p_tr    = st.selectbox("Training Status", ["Completed", "Partial", "None"])
        p_ot    = st.selectbox("Overtime",        ["No", "Yes"])
        p_mg    = st.selectbox("Manager Rating",  ["High", "Medium", "Low"])

    profile = {
        "department": p_dept, "shift": p_shift, "seniority": p_sen,
        "contract_type": p_ct, "remote_work": p_rm, "training_status": p_tr,
        "overtime": p_ot, "manager_rating": p_mg,
    }
    matched = analyze_profile(profile)

    with col_out:
        if len(matched) == 0:
            st.markdown(f"""
            <div style="background:var(--card);border:1px solid var(--border);
                        border-radius:4px;padding:1.6rem 1.8rem;">
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                            color:#fff;margin-bottom:0.8rem;">Risk Assessment</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;
                            color:{C_OK};">No rules matched this profile.</div>
                <div style="font-family:var(--fm);font-size:0.7rem;color:var(--muted);
                            margin-top:8px;">This combination has no significant patterns in the training data.</div>
            </div>""", unsafe_allow_html=True)
        else:
            risks    = matched[matched["consequents"].apply(
                lambda x: any(c in ["attrition=Yes", "absenteeism=High", "performance=Low"]
                              for c in list(x)))]
            positive = matched[matched["consequents"].apply(
                lambda x: any(c in ["attrition=No", "performance=High", "absenteeism=Low"]
                              for c in list(x)))]

            top_risk_color = C_DANGER if len(risks) > 0 else C_OK
            badge_bg       = "#2e0f0f" if len(risks) > 0 else "#0f2e1a"
            badge_label    = f"{len(risks)} RISK RULES · {len(positive)} POSITIVE RULES"

            st.markdown(
                f'''<div style="background:var(--card);border:1px solid var(--border);
                            border-radius:4px;padding:1.6rem 1.8rem;margin-bottom:14px;">
                    <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                                color:#fff;margin-bottom:0.8rem;">Risk Assessment</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:2.6rem;
                                font-weight:700;color:{top_risk_color};line-height:1;">
                        {len(matched)} rules matched
                    </div>
                    <div style="margin-top:12px;">
                        <span style="background:{badge_bg};color:{top_risk_color};
                                     font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                                     font-weight:600;letter-spacing:.08em;
                                     padding:5px 14px;border-radius:20px;">{badge_label}</span>
                    </div>
                </div>''',
                unsafe_allow_html=True
            )

            st.markdown('<div class="lsa-section">// Matched rules (ranked by lift)</div>',
                        unsafe_allow_html=True)
            for _, r in matched.iterrows():
                cons   = list(r["consequents"])[0]
                icon   = OUTCOME_ICONS.get(cons, "")
                color  = OUTCOME_COLORS.get(cons, C_ORANGE)
                ant    = fmt_items(r["antecedents"])
                bl     = baselines.get(cons, 0)
                lift_c = C_DANGER if r["lift"] > 1.3 else C_WARN if r["lift"] > 1.1 else C_OK
                st.markdown(f"""
                <div style="background:var(--card);border:1px solid var(--border);
                            border-left:3px solid {color};border-radius:2px;
                            padding:10px 14px;margin-bottom:8px;">
                    <div style="font-family:var(--fm);font-size:0.72rem;font-weight:700;
                                color:{color};margin-bottom:4px;">{icon} {cons}</div>
                    <div style="font-family:var(--fm);font-size:0.65rem;color:var(--muted);
                                margin-bottom:6px;">{ant}</div>
                    <div style="display:flex;gap:16px;font-family:var(--fm);font-size:0.68rem;">
                        <span>Lift: <strong style="color:{lift_c};">{r['lift']:.3f}</strong></span>
                        <span>Conf: <strong style="color:var(--text);">{r['confidence']:.3f}</strong></span>
                        <span>Supp: <strong style="color:var(--text);">{r['support']:.3f}</strong></span>
                        <span>Base: <strong style="color:var(--muted);">{bl:.1%}</strong></span>
                    </div>
                </div>""", unsafe_allow_html=True)

            # Top intervention
            if len(risks) > 0:
                top_cons = list(risks.iloc[0]["consequents"])[0]
                INTERVENTIONS = {
                    "attrition=Yes":    ["Conduct stay interview within 30 days.",
                                         "Review career path alignment.",
                                         "Consider schedule or remote work options."],
                    "absenteeism=High": ["Offer flexible scheduling or partial remote.",
                                         "Review manager relationship.",
                                         "Check overtime hours for burnout signs."],
                    "performance=Low":  ["Enroll in next training cycle immediately.",
                                         "Assign weekly coaching 1:1.",
                                         "Review workload and support resources."],
                }
                acts = INTERVENTIONS.get(top_cons, [])
                if acts:
                    st.markdown(f"""
                    <div style="background:var(--card);border:1px solid var(--border);
                                border-left:3px solid {C_DANGER};border-radius:2px;
                                padding:0.9rem 1.2rem;margin-top:10px;">
                        <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">
                            // Top intervention: {top_cons}</div>
                        {"".join(f'<div style="font-family:var(--fm);font-size:0.7rem;color:var(--muted);padding:2px 0;line-height:1.7;">{i+1}. {a}</div>' for i,a in enumerate(acts))}
                    </div>""", unsafe_allow_html=True)

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="lsa-section">// HR rule library — top rules by outcome category</div>',
                unsafe_allow_html=True)

    outcome_sel = st.selectbox(
        "Select outcome category:", CRITICAL_OUTCOMES,
        format_func=lambda x: f"{OUTCOME_ICONS.get(x,'')}  {x}"
    )

    sub_rules = hr_rules[hr_rules["consequents"].apply(lambda x: outcome_sel in x)]
    top_rules = sub_rules.nlargest(15, "lift").copy()
    top_rules["antecedents_str"] = top_rules["antecedents"].apply(fmt_items)
    top_rules["consequents_str"] = top_rules["consequents"].apply(lambda x: list(x)[0])

    if len(top_rules):
        color    = OUTCOME_COLORS.get(outcome_sel, C_ORANGE)
        baseline = baselines.get(outcome_sel, 0)
        st.caption(f"{len(sub_rules)} rules found · showing top 15 by lift · Base rate: {baseline:.1%}")

        top_plot = top_rules.nlargest(12, "lift")
        labels   = [
            "\n+ ".join([i.split("=")[1] for i in sorted(list(r["antecedents"]))[:3]])
            for _, r in top_plot.iterrows()
        ]
        fig_lib, ax_lib = dark_fig(11, 6.5)
        bars = ax_lib.barh(range(len(top_plot)), top_plot["lift"].values,
                           color=color, alpha=0.85, edgecolor="#1e2d45", height=0.65)
        ax_lib.set_yticks(range(len(top_plot)))
        ax_lib.set_yticklabels(labels, color=C_TEXT, fontsize=8)
        ax_lib.axvline(1.0, color="white", lw=1.0, ls="--", alpha=0.5)
        for bar, lift, conf in zip(bars, top_plot["lift"], top_plot["confidence"]):
            ax_lib.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"lift={lift:.3f}  conf={conf:.3f}", va="center", fontsize=7.5, color=C_TEXT)
        ax_lib.set_xlabel("Lift")
        ax_lib.set_title(f"Top Rules → {outcome_sel}", color=C_TEXT, fontsize=11, fontweight="bold")
        ax_lib.grid(axis="x", alpha=0.1, color="#1e2d45")
        fig_lib.tight_layout()
        st.pyplot(fig_lib, use_container_width=True); plt.close()

        st.divider()
        st.markdown('<div class="lsa-section">// Full rule table</div>', unsafe_allow_html=True)
        disp = top_rules[["antecedents_str", "consequents_str",
                           "support", "confidence", "lift"]].copy()
        disp.columns = ["Antecedents", "→ Consequent", "Support", "Confidence", "Lift"]
        st.dataframe(disp.reset_index(drop=True).style.format(
            {"Support": "{:.4f}", "Confidence": "{:.4f}", "Lift": "{:.4f}"}
        ), use_container_width=True, height=350)
    else:
        st.info("No rules found for this outcome. Try a different category.")

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="lsa-section">// HR intervention playbook — rule-based action protocols</div>',
                unsafe_allow_html=True)
    st.caption("Each rule cluster maps to a specific HR intervention pathway.")

    playbook = [
        {"title": "Attrition Risk — training_status=None", "color": C_DANGER,
         "trigger": "Rule: training_status=None → attrition=Yes (lift=1.46)",
         "freq": "~6.6% of employees",
         "actions": ["Immediate enrollment in mandatory onboarding training program.",
                     "Assign a peer mentor before end of first month.",
                     "Block temporary + no-training combination in scheduling.",
                     "HR check-in at 30, 60, 90 days."]},
        {"title": "Attrition Risk — Night Shift + No Remote", "color": C_DANGER,
         "trigger": "Rule: shift=Night + remote_work=No → attrition=Yes (lift=1.37)",
         "freq": "~7.0% of employees",
         "actions": ["Introduce Night Shift Equity Program — flexible hours/rotations.",
                     "Evaluate remote work eligibility for night-shift roles.",
                     "Monthly check-in meeting with night-shift cohort.",
                     "Night-shift premium or compensatory time policy review."]},
        {"title": "Absenteeism Risk — Low Manager + No Remote", "color": C_PURP,
         "trigger": "Rule: manager_rating=Low + remote_work=No → absenteeism=High (lift=1.70)",
         "freq": "~6.5% of employees",
         "actions": ["Priority: address manager effectiveness — 360° feedback review.",
                     "Pilot partial remote for this cohort — measurable attendance impact.",
                     "Manager coaching program for low-rated managers.",
                     "EAP referral if absenteeism already elevated."]},
        {"title": "Performance Driver — Remote + No Overtime + Completed Training", "color": C_BLUE,
         "trigger": "Rule: remote_work=Yes + overtime=No + training_status=Completed → performance=High (lift=1.52)",
         "freq": "~6.8% of employees",
         "actions": ["This is the retention and performance formula — protect it.",
                     "Replicate conditions for similar profiles (Junior/Mid prioritized).",
                     "Avoid mandatory overtime for top performers — productivity drops.",
                     "Ensure training completion rates are tracked as KPI."]},
        {"title": "Low Absenteeism — Remote + Day Shift + Completed Training", "color": C_OK,
         "trigger": "Rule: remote_work=Yes + shift=Day + training_status=Completed → absenteeism=Low (lift=1.63)",
         "freq": "~10.9% of employees",
         "actions": ["This profile is the attendance benchmark — document and share.",
                     "Use as success case in HR communications and culture programs.",
                     "Prioritize completed training before granting remote eligibility.",
                     "Track this combination in quarterly HR scorecard."]},
    ]

    for p in playbook:
        with st.expander(p["title"], expanded=True):
            l, r = st.columns([3, 1])
            with l:
                st.markdown(f"""
                <div style="font-family:var(--fm);font-size:0.7rem;color:var(--muted);
                            margin-bottom:10px;">
                    <strong style="color:var(--text);">Trigger rule:</strong> {p['trigger']}
                </div>""", unsafe_allow_html=True)
                for i, a in enumerate(p["actions"], 1):
                    prefix = "🚨 " if "Attrition" in p["title"] and i == 1 else f"{i}. "
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
                                text-transform:uppercase;letter-spacing:.1em;">Coverage</div>
                    <div style="font-family:var(--fm);font-size:1rem;font-weight:700;
                                color:{p['color']};margin-top:4px;">{p['freq']}</div>
                </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
    <div style="background:var(--card);border:1px solid var(--border);
                border-left:3px solid {C_ORANGE};border-radius:2px;padding:1rem 1.3rem;">
        <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Ethical use reminder</div>
        <div style="font-family:var(--fm);font-size:0.72rem;color:var(--text);line-height:1.7;">
            Association rules are population-level patterns — not predictions about individual employees.
            Never use a rule to penalise, discriminate, or pre-judge any person.
            The rules surface conditions to address, not people to flag.
        </div>
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
    LozanoLsa · Turning Operations into Predictive Systems · HR People Analytics · Project 19 · v2.0
</div>
""", unsafe_allow_html=True)
