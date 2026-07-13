# Project 24 · pH Control Intelligence — Grand Finale
### Model-Based Reinforcement Learning for Chemical Reactor Control

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/ModelBased_pH/blob/main/24_ModelBased_pH.ipynb)

> *"For twenty-three projects, the models observed and decided. This one plans."*

---

Twenty-three projects.

From a logistic regression trained to classify weld defects from labeled examples — to an agent that builds a mental model of a chemical reactor from historical data, simulates possible futures through that model, and selects the action predicted to produce the best outcome.

Not memorizing values in a table. Not following the gradient of past rewards. **Planning.**

Model-Based RL is the synthesis of everything that came before it in this portfolio. It requires a world model trained like a supervised learner (Project 09), a planner that reasons about sequences like a reinforcement agent (Projects 22–23), and an offline data paradigm that respects the constraints of real industrial deployment. It is not the most complex algorithm in the portfolio. It is the most complete.

This project applies it to pH control in a 1,000 L chemical reactor — a problem where random dosing keeps the process in the acceptable zone **3.5% of the time**, and where the model-based planner reaches **11.3%** without touching the reactor during training.

---

## Question 1 — What Happens Next?

*The world model learns the reactor's physics from offline data.*

### The Offline Dataset

35,672 state-action-reward transitions collected from **2,000 episodes** of random dosing. The agent never interacted with the reactor during training. This is the offline RL paradigm: learn a model of the world without additional exploration, then plan within that model.

| Variable | Unit | Role in the World Model |
|---|---|---|
| `ph_t` | pH | Current acidity — dominant predictor of next pH |
| `temp_t_c` | °C | Reaction temperature (20–30°C) |
| `volume_t_l` | L | Reactor volume (900–1,100 L) |
| `buffer_capacity` | — | Buffer strength — resistance to pH change (0.8–1.5) |
| `dose_ml` | ml | Reagent applied — the control variable |
| `ph_t1` | pH | **Next-step pH — what the model predicts** |

**Random policy baseline** (from dataset): mean reward −13.94 per step · 3.0% of steps in target zone [6.8, 7.2].

**Reward function:**

$$r_t = -10 \cdot |pH_t - 7.0| + \underbrace{5.0}_{\text{if } pH \in [6.8, 7.2]} - 0.2 \cdot |\text{dose}|$$

The −10 deviation penalty is the dominant term. The +5 bonus for hitting the target zone is the reward the planner hunts for. The −0.2 dosing cost prevents the agent from over-dosing when the reactor is already close.

### The World Model

$$\hat{pH}_{t+1} = \hat{f}_\theta(pH_t,\; T_t,\; V_t,\; \text{buffer},\; \text{dose})$$

**Architecture:** Gradient Boosting Regressor — 200 trees, learning rate 0.1, max depth 4, `random_state=42`.

**Why GBR for the world model?**
The pH–buffer–dose interaction is non-linear. Near pH extremes, the same dose produces a smaller change than near neutral — the saturation effect. GBR captures this without explicit physics assumptions, without overfitting, and with interpretable feature importances.

**Validation results (test set: 7,135 held-out transitions):**

| Metric | Value | Operational meaning |
|---|---|---|
| **R²** | **0.9987** | The model explains 99.87% of next-step pH variance |
| **MAE** | **0.0281 pH units** | Average prediction error below 0.03 pH — sub-threshold accuracy |
| **Bias** | ~0.000 | Symmetric residuals — the model is unbiased |

**Feature importance — what drives next-step pH:**

| Feature | Importance | Interpretation |
|---|---|---|
| `ph_t` | **0.889** | The current pH is overwhelmingly the best predictor of the next pH |
| `dose_ml` | **0.107** | The dosing action has meaningful, secondary influence |
| `buffer_capacity` | 0.003 | Minor effect — buffer modulates dose impact at the margin |
| `volume_t_l` | 0.000 | Negligible in this operating range |
| `temp_t_c` | 0.000 | Negligible in this operating range |

**World model accuracy by pH zone:**

| pH Zone | MAE (pH units) | Count | Note |
|---|---|---|---|
| < 5.0 | 0.024 | 5,907 | Accurate — many training examples |
| 5.0 – 6.0 | 0.025 | 18,742 | Most accurate region — bulk of random data |
| 6.0 – 6.5 | 0.030 | 4,922 | Good |
| 6.5 – 6.8 | 0.042 | 1,401 | Slightly elevated — approaching target zone |
| **6.8 – 7.2 (target)** | **0.186** | **41** | Highest error — only 41 training samples here |
| 7.2 – 7.5 | 0.050 | 653 | Acceptable |
| 7.5 – 8.0 | 0.033 | 1,205 | Good |
| > 8.0 | 0.025 | 2,801 | Accurate |

The target zone has the highest MAE (0.186) because random dosing almost never reaches pH 6.8–7.2 — only 41 of 35,672 transitions land there. This is a fundamental limitation of offline learning from random exploration data: the world model is least accurate exactly where it matters most. The planner must be aware of this.

---

## Question 2 — Which Future Is Best?

*The planner simulates H steps ahead for each action and selects the one with the highest expected return.*

### The Planning Algorithm

For each of the 5 possible actions, the planner queries the world model once (H=1) and computes the expected reward of that transition:

$$a^* = \argmax_{a \in \mathcal{A}} \;\; \hat{G}(a) = \sum_{h=0}^{H-1} \gamma^h \cdot \hat{r}\!\left(\hat{f}^h_\theta(s_t, a), a\right)$$

The world model is deterministic (no sampling noise in simulation). The real reactor is stochastic. This gap between simulated and real dynamics is what makes horizon selection critical.

**The planner's decision logic across the pH spectrum** (T=25°C, V=1,000 L, buffer=1.15):

| Current pH | Planner chooses | Reasoning |
|---|---|---|
| pH < 5.5 | Base +1.0 ml | Large deviation → aggressive correction |
| 5.5 ≤ pH < 6.5 | Base +1.0 or +0.5 ml | Moderate acid → proportional base addition |
| 6.5 ≤ pH < 6.8 | Base +0.5 ml | Near target → gentle approach |
| **6.8 ≤ pH ≤ 7.2** | **No dose** | **In target — minimize reagent cost** |
| 7.2 < pH ≤ 7.8 | Acid −0.5 ml | Slightly alkaline → gentle acid correction |
| pH > 7.8 | Acid −1.0 ml | Large alkaline deviation → aggressive acid |

The planner never selects Decrease Speed or No-Action when the pH is outside the target — it always applies a corrective dose proportional to the deviation. This behavior is emergent from the reward function, not hard-coded.

### The Horizon Problem — Why Longer Is Worse

**Horizon sensitivity (100 evaluation episodes each, seed 99):**

| Horizon H | Mean Reward/Step | In-Target Rate | vs H=1 |
|---|---|---|---|
| **H=1** | **−4.19** | **11.3%** | **← optimal** |
| H=2 | −4.15 | 7.0% | −4.3 pp |
| H=3 | −4.43 | 6.5% | −4.8 pp |
| H=5 | −5.41 | 0.9% | −10.4 pp |

H=5 achieves 0.9% in-target — **worse than the random baseline (3.5%)**. The world model's per-step MAE of 0.028 pH compounds across 5 steps. By step 5, the simulated pH has accumulated enough error that the planner is optimizing a fictional trajectory, not the real reactor.

This is a universal trade-off in Model-Based RL: **longer horizon ≠ better planning**. The optimal horizon is determined by the ratio of world model accuracy to environment stochasticity — not by intuition about planning depth.

---

## Question 3 — What Should I Do?

*The controller executes the plan and measures the outcome.*

### Performance Against the Baseline

**Evaluation: 100 fresh episodes, seed 99:**

| Metric | Random Policy | Model-Based (H=1) | Improvement |
|---|---|---|---|
| Mean reward / step | −14.29 | **−4.19** | **+10.10 per step** |
| Steps in target [6.8–7.2] | 3.5% | **11.3%** | **+7.9 pp · 3.26×** |

The model-based planner achieves **3.26× more time in the target zone** without ever interacting with the reactor during training. Every planning decision uses the world model as a proxy for the reactor — and at MAE = 0.028 pH units, that proxy is accurate enough to produce real improvement.

### Three Reactor Scenarios

**Scenario A — Strongly Acidic Start** `pH₀ = 4.2`

The planner applies Large Base (+1.0 ml) at every step until the reactor approaches the target zone, then switches to a gentler intervention as the pH climbs past 6.1.

| Step | pH before | pH after | Action | Reward |
|---|---|---|---|---|
| 0 | 4.200 | 4.495 | Base +1.0 ml | −25.25 |
| 1 | 4.495 | 4.801 | Base +1.0 ml | −22.19 |
| 2 | 4.801 | 5.169 | Base +1.0 ml | −18.51 |
| 3 | 5.169 | 5.622 | Base +1.0 ml | −13.98 |
| 4 | 5.622 | 6.134 | Base +1.0 ml | −8.86 |
| **5** | **6.134** | **6.872** | **Base +1.0 ml** | **+3.52 ✓** |

**Target reached at step 5** — 6 steps from pH 4.2 to pH 6.87. The planner drives the reactor across 2.67 pH units using sustained large-dose correction, stopping as soon as the reactor enters [6.8, 7.2].

**Scenario B — Near-Neutral Start** `pH₀ = 6.5`

Single-step correction. The planner selects Small Base (+0.5 ml) — gentle enough to push the pH from 6.5 to 7.09 without overshooting.

| Step | pH before | pH after | Action | Reward |
|---|---|---|---|---|
| **0** | **6.500** | **7.091** | **Base +0.5 ml** | **+3.99 ✓** |

**Target reached at step 0** — 1 step. This is the planner at its best: a single precise intervention from near the boundary.

**Scenario C — Alkaline Start** `pH₀ = 8.5`

The planner drops Acid −1.0 ml for the first 2 steps, switching to Acid −0.5 ml as the pH approaches 7.4, then holding No Dose while the reactor drifts naturally toward the target under process noise. Target reached at step 23.

The 24-step trajectory reveals a key behavior: once the pH enters the range [7.22–7.29], the planner recognizes that additional acid risks undershooting below 6.8, and waits — applying No Dose for 13 consecutive steps — until natural drift carries the reactor across the lower boundary at step 23.

---

## 📊 Statistical Validation

**World model residuals:** symmetric around zero, mean residual ≈ 0.0000 — no systematic bias. The model does not consistently over- or under-predict in any direction.

**Compounding error analysis:** at H=1, the model commits a 0.028 pH error once. At H=5, errors compound: the simulated state at step 5 can be 0.028×5 = 0.14 pH units away from reality in the worst case — enough to recommend the wrong corrective direction. The 0.9% in-target rate at H=5 (vs 11.3% at H=1) is the empirical confirmation.

**Key architectural insight:** the target zone [6.8, 7.2] has only 41 training examples — a data coverage problem from random baseline collection. A production deployment would require targeted data collection near the target zone (active learning), or a data-augmentation strategy, to improve world model accuracy exactly where the controller operates most.

---

## 🗂️ Repository Structure

```
ModelBased_pH/
├── 24_ModelBased_pH.ipynb   # Educational notebook (no outputs)
├── Data_pH.csv              # 250-row sample of offline transition dataset
├── requirements.txt
└── README.md
```

**Note on `Data_pH.csv`:** offline transition data collected under a random dosing policy. The world model is trained from this file. The planner is then deployed on a fresh environment using only the world model — no additional reactor interaction during planning.

> 📦 **Full Project Pack** — complete 35,672-row dataset, notebook with full outputs,
> presentation deck (PPTX + PDF), and `app.py` pH control simulator available on
> [Gumroad](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

**Option 1 — Colab:**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/ModelBased_pH/blob/main/24_ModelBased_pH.ipynb)

**Option 2 — Local:**

```bash
git clone https://github.com/LozanoLsa/ModelBased_pH.git
cd ModelBased_pH
pip install -r requirements.txt
jupyter notebook 24_ModelBased_pH.ipynb
```

**Requirements:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`

---

## 💡 Five Lessons From the Finale

**1 — The world model is the architecture.** In model-free RL, the algorithm is central. In model-based RL, the world model is the system — everything else is downstream of its accuracy. R² = 0.9987 and MAE = 0.028 pH are not just metrics; they are the engineering budget that determines how far the planner can look ahead before reality diverges from simulation.

**2 — Longer horizon is not better planning — it is compounded error.** The H=5 planner achieves 0.9% in-target, worse than random (3.5%). This is not a bug. It is the correct and expected result when world model error is not negligible. The optimal horizon is a function of accuracy divided by stochasticity — not of planning depth for its own sake.

**3 — Offline learning creates a coverage problem.** The target zone [6.8, 7.2] has 41 training examples out of 35,672. The world model is least accurate (MAE = 0.186) exactly where the controller operates most. In production, this requires active data collection near the target zone — a problem that supervised learning, model-free RL, and model-based RL all face, but only model-based RL can partially solve through planning without additional real-world interaction.

**4 — The planner's behavior is operationally legible.** At pH 4.2 → apply Large Base. At pH 6.5 → apply Small Base. At pH 7.3 → wait. These decisions are not programmed. They emerge from maximizing the reward function through the world model. An engineer can verify each decision by running `plan_action()` at any state — the function returns the best action and its expected return in one call.

**5 — Model-based RL completes the portfolio's arc.** Supervised learning taught the models to see patterns in historical data. Unsupervised learning taught them to discover structure without labels. Reinforcement learning taught them to act. Model-based RL synthesizes all three: it learns from data like a supervised model (world model training), discovers the optimal policy like RL (planning), and does so without wasting real-world interaction. It is not the end of the field. It is the beginning of the question: *what else can the model simulate?*

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
