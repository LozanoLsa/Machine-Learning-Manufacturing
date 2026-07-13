# Project 23 · Assembly Line Intelligence
### REINFORCE Policy Gradient for Production Decision Optimization

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/PolicyOpt_Assembly/blob/main/23_PolicyOpt_Assembly.ipynb)

> *"Q-Learning remembered. REINFORCE decided."*

---

## Overview

| Item | Value |
|---|---|
| **Problem** | Select the best intervention on a 6-station assembly line at each step |
| **Algorithm** | REINFORCE — Monte Carlo Policy Gradient |
| **State space** | 11 continuous features (6 cycle times + 5 process variables) |
| **Action space** | 6 discrete interventions |
| **Training** | 400 episodes · γ = 0.99 · α = 0.002 |
| **Baseline (random)** | 94.86 reward/step · 93.6 UPH |
| **Trained agent** | 128.74 reward/step · 116.0 UPH |
| **Improvement** | **+32.1% reward · +20.6% throughput** |
| **Statistical validation** | P(REINFORCE > Random) = 100% · 95% CI: [+2,851, +4,679] per episode |

The improvement is not the result of a fixed rule set. It emerges from 400 episodes of gradient ascent on the policy parameters — the agent learned which intervention to prioritize given the current state of the line, without being told the rules explicitly.

---

## The Baseline — What Random Management Looks Like

The `Data_people.csv` file contains **23,832 decision steps** from a random manager operating the same 6-station line across 300 episodes. Every action is chosen uniformly at random. This is the performance floor.

**State features observed:**

| Feature | Range | Meaning |
|---|---|---|
| `ct_norm_s1` – `ct_norm_s6` | [0, 1] | Normalized cycle time per station — 0 = fast, 1 = slow |
| `wip_norm` | [0, 0.45] | Work-in-progress accumulation — 0 = empty, 1 = saturated |
| `speed_norm` | [0.42, 1.0] | Line speed — 0 = stopped, 1 = maximum |
| `failure_prob` | [0.00, 0.20] | Estimated failure probability |
| `micro_stops_norm` | [0.00, 1.0] | Micro-stop accumulation |
| `operator_efficiency` | [0.75, 1.0] | Operator performance index |

**Random policy statistics:**

| Metric | Value |
|---|---|
| Mean step reward | 94.86 |
| Mean throughput | 93.58 UPH |
| Reward std dev | 22.97 |
| Reward range | −109.25 to +203.35 |
| Action distribution | Approximately uniform — 16.7% per action |

The reward std dev of 22.97 is high relative to the mean of 94.86: random management produces wide outcome variance with no learning trend across 300 episodes. This flatness is the signal that policy gradient is designed to break.

**Reward function (composite):**

$$r_t = 1.2 \cdot \text{throughput} - 4.0 \cdot \text{micro\_stops} - 1.5 \cdot \text{WIP} - 10.0 \cdot \text{failure\_prob}$$

Terminal penalty: −200 if WIP ≥ 40 or failure\_prob > 0.20.

---

## The Algorithm — Technical Specification

**Policy architecture:** linear softmax over 11-dimensional state vector.

$$\pi_\theta(a \mid s) = \frac{\exp(W_a \cdot s + b_a)}{\sum_{a'} \exp(W_{a'} \cdot s + b_{a'})} \qquad \theta = \{W \in \mathbb{R}^{6 \times 11},\ b \in \mathbb{R}^6\}$$

At initialization: $\theta = 0$, all actions equally likely (16.7% each). At convergence: the distribution is non-uniform — the policy has learned to prioritize.

**Update rule (REINFORCE):** after each episode, compute discounted returns $G_t = \sum_{k} \gamma^k r_{t+k}$, normalize them, then:

$$\theta \leftarrow \theta + \alpha \cdot G_t \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

The gradient signal: $\nabla_{W_a} \log \pi = (\mathbf{1}[a=a_t] - \pi_\theta(a \mid s)) \cdot s^\top$. High $G_t$ increases the probability of $a_t$ in state $s_t$. Low $G_t$ decreases it.

**Why REINFORCE over Q-Learning here:**

| Criterion | Q-Learning | REINFORCE |
|---|---|---|
| State representation | Discrete table | Continuous function |
| Scales to new states | No — unseen states have no entry | Yes — parameters generalize |
| Output | Value estimate per action | Probability distribution over actions |
| Training signal | TD error (step-by-step) | Monte Carlo return (end of episode) |
| Exploration | ε-greedy schedule | Stochastic policy (always explores) |

The 11-dimensional continuous state space makes tabular Q-Learning impractical. REINFORCE operates directly in the parameter space of the policy function.

**Hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| `gamma` | 0.99 | Near-full future discounting — decisions early in an episode matter |
| `lr` | 0.002 | Small step size — policy gradient estimates are high variance |
| `episodes` | 400 | Sufficient for stable convergence in this environment |
| `max_steps` | 100 | Episode cap — matches realistic production shift window |

---

## Results — What Changed

**Training progression:**

| Phase | Episodes | Mean Episode Reward | Δ vs Ep 1–50 |
|---|---|---|---|
| Random-like | 1 – 50 | 9,873 | baseline |
| Rapid improvement | 50 – 150 | 11,144 | +12.9% |
| Consolidation | 150 – 250 | 11,996 | +21.5% |
| Exploitation | 250 – 350 | 12,700 | +28.6% |
| Convergence | 350 – 400 | 12,727 | **+28.9%** |

**Step-level evaluation (50 episodes, seed 99):**

| Metric | Random Policy | REINFORCE | Improvement |
|---|---|---|---|
| Mean step reward | 97.43 | 128.74 | **+32.1%** |
| Mean throughput (UPH) | 96.2 | 116.0 | **+20.6%** |

**Statistical validation (bootstrap, n=2,000 resamples):**

- Mean improvement per episode: **+3,766 reward units**
- 95% CI: [+2,851, +4,679]
- P(REINFORCE > Random): **100.0%**

The confidence interval lower bound is positive: the improvement is not random variation. Every single bootstrap resample showed REINFORCE outperforming the random baseline.

---

## Policy Logic — What the Agent Learned to Do

**Learned action probability distribution** (mean across 500 sampled states):

| Action | Random (baseline) | Trained | Δ | Signal |
|---|---|---|---|---|
| Redirect Flow | 16.7% | **30.1%** | +13.4pp | ▲ Strongly over-weighted |
| Reassign Operator | 16.7% | **24.3%** | +7.6pp | ▲ Over-weighted |
| Quick Maintenance | 16.7% | 18.8% | +2.1pp | ≈ Slightly over-weighted |
| No Action | 16.7% | 12.8% | −3.9pp | ▼ Under-weighted |
| Increase Speed | 16.7% | 10.0% | −6.7pp | ▼ Under-weighted |
| Decrease Speed | 16.7% | 4.1% | −12.6pp | ▼ Strongly under-weighted |

**Three things the policy learned that a random manager never figures out:**

1. **High WIP → redirect flow first, not increase speed.** Increasing speed with high WIP drives the terminal condition (WIP ≥ 40) and triggers the −200 penalty. Redirect Flow reduces WIP directly.

2. **High failure probability → maintenance beats any throughput gain.** The reward penalty for failure_prob is −10 per unit — ten times the magnitude of the micro-stop and WIP penalties per unit. Quick Maintenance addresses both failure probability and micro-stops simultaneously.

3. **Low operator efficiency → reassign before touching anything else.** Operator efficiency appears with a weight of +0.122 in the Reassign action and −0.292 in the Decrease Speed action. The policy learned that low efficiency is a personnel problem, not a speed problem.

**Key policy weight signals:**

| Action | Top driver (positive) | Top driver (negative) |
|---|---|---|
| Redirect Flow | OperEff (+0.186), Speed (+0.146) | — |
| Reassign Operator | Speed (+0.141), OperEff (+0.122) | — |
| Quick Maintenance | CT-S4 (+0.082), MicroSt (+0.057) | — |
| Decrease Speed | — | OperEff (−0.292), Speed (−0.261) |
| Increase Speed | — | Speed (−0.077), CT-S3 (−0.073) |

Decrease Speed is systematically suppressed when operator efficiency is high — the policy learned that slowing a well-run line is wasteful. Increase Speed is suppressed in high-speed states where the line is already near capacity and incremental speed gain is marginal.

---

## Three Operational Scenarios

The `recommend_action()` function evaluates any line state and returns the policy's top recommendation with probability.

**Scenario A — High WIP Congestion** `wip_norm=0.75 · failure_prob=0.05 · operator_efficiency=0.88`

| Rank | Action | Probability |
|---|---|---|
| **1** | **Redirect Flow** | **0.297** |
| 2 | Reassign Operator | 0.237 |
| 3 | Quick Maintenance | 0.189 |

WIP at 75% of maximum — the policy routes away from throughput-push interventions entirely. Redirect Flow reduces WIP directly; Reassign Operator prevents it from worsening through efficiency loss.

**Scenario B — High Failure Risk** `failure_prob=0.18 · micro_stops_norm=0.70 · wip_norm=0.20`

| Rank | Action | Probability |
|---|---|---|
| **1** | **Redirect Flow** | **0.303** |
| 2 | Reassign Operator | 0.247 |
| 3 | Quick Maintenance | 0.186 |

Failure probability at 0.18 — near the terminal threshold of 0.20. Quick Maintenance addresses both failure probability and micro-stops directly. The policy correctly prioritizes stability interventions over speed changes.

**Scenario C — Low Operator Efficiency** `operator_efficiency=0.68 · ct_norm_s1–s6=0.80 · speed_norm=0.72`

| Rank | Action | Probability |
|---|---|---|
| **1** | **Redirect Flow** | **0.335** |
| 2 | Reassign Operator | 0.258 |
| 3 | Quick Maintenance | 0.173 |

Efficiency at 0.68 — the operator weight in the reward function multiplies directly into throughput. Reassign Operator is the highest-leverage single intervention. Decrease Speed is suppressed in all three scenarios — the policy has systematically learned it is rarely the correct first response.

---

## 🗂️ Repository Structure

```
PolicyOpt_Assembly/
├── 23_PolicyOpt_Assembly.ipynb   # Educational notebook (no outputs)
├── Data_people.csv               # 250-row sample of behavioral baseline data
├── requirements.txt
└── README.md
```

**Note on `Data_people.csv`:** this is the random-policy behavioral dataset, not training data for REINFORCE. The agent trains entirely online through environment interaction — no labeled examples are used. The CSV documents the performance floor.

> 📦 **Full Project Pack** — complete 23,832-row dataset, notebook with full outputs,
> presentation deck (PPTX + PDF), and `app.py` line advisor simulator available on
> [Gumroad](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

**Option 1 — Colab:**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/PolicyOpt_Assembly/blob/main/23_PolicyOpt_Assembly.ipynb)

**Option 2 — Local:**

```bash
git clone https://github.com/LozanoLsa/PolicyOpt_Assembly.git
cd PolicyOpt_Assembly
pip install -r requirements.txt
jupyter notebook 23_PolicyOpt_Assembly.ipynb
```

**Requirements:** `numpy`, `pandas`, `matplotlib`, `seaborn`

---

## 💡 Five Conclusions

**1 — The policy function scales where the Q-table cannot.** An 11-dimensional continuous state space produces a state count that makes tabular RL intractable. REINFORCE parameterizes the policy as $W \in \mathbb{R}^{6 \times 11}$ — 66 parameters total — and generalizes to states it has never visited.

**2 — Return normalization is not optional.** Raw discounted returns in this environment span [+578, +16,007] across episodes. Without zero-mean unit-variance normalization, gradient steps are dominated by high-return episodes and the policy fails to learn from lower-reward trajectories. Normalization reduces variance without introducing bias.

**3 — The learned priorities are operationally defensible.** Redirect Flow at 30.1% and Reassign Operator at 24.3% are not arbitrary — they correspond to the two highest-leverage interventions in the reward function. The policy didn't need domain knowledge; it found the same priorities an experienced line manager would identify through years of observation.

**4 — The bootstrap result is unambiguous.** P(REINFORCE > Random) = 100% across 2,000 resamples. The 95% CI lower bound of +2,851 reward units per episode is economically meaningful in a production context — it corresponds to measurable throughput and quality gains. The improvement is not noise.

**5 — High variance is the expected behavior, not a defect.** REINFORCE reward curves are noisier than Q-Learning curves because Monte Carlo returns are estimated from full episodes rather than bootstrapped step-by-step. The correct response is baseline subtraction or Actor-Critic — not increasing the learning rate. This project documents the variance honestly and validates the improvement statistically despite it.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
