# pH Compliance Is Not Random — Random Forest

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/PH_Adjustment_Syrup/blob/main/06_RF_PH_Adjustment.ipynb)

> *"pH in a pharmaceutical syrup is not a single measurement — it is the outcome of a system. Citric acid pushes it down. The citrate buffer resists the change. Water quality sets the baseline. Mixing conditions determine whether the equilibrium is actually reached. A model that ignores any of these dimensions will miss the story."*

---

## 🎯 Business Problem

In oral liquid pharmaceutical manufacturing, pH is a release criterion — not a preference. A syrup lot with final pH outside the 4.5–5.5 specification cannot be shipped. It goes to reprocess if the batch is salvageable, or to disposal if it isn't. Either outcome triggers a deviation investigation, regulatory documentation, and a delay in the batch release schedule that ripples through production planning.

The challenge is that pH deviation is rarely caused by a single variable. It is the result of a specific combination of formulation choices and process conditions converging unfavorably. Traditional process control monitors each variable individually — citric acid within range, buffer within range, agitation at target — but misses the interaction effects that determine the final pH outcome.

This project builds a **pre-batch risk classifier**: given the planned recipe and process conditions, what is the probability of pH compliance? The model is consulted before synthesis begins — enabling formulation corrections before any material is consumed.

---

## 📊 Dataset

- **1,847 pharmaceutical batch records** from an oral liquid manufacturing line across four formulation campaigns
- **Target:** `lot_approved` (binary) — 1 = final pH within 4.5–5.5 specification
- **Class balance:** 41.4% approved (moderately imbalanced — 58.6% failed pH spec)
- **Final pH range:** 1.89–6.73 (specification window: 4.5–5.5)

| Layer | Feature | Description |
|-------|---------|-------------|
| Formulation | `citric_acid_pct` | Primary acidulant — strongest pH driver |
| Formulation | `citrate_buffer_pct` | Counter-lever — resists pH change |
| Formulation | `api_pct`, `sweetener_pct`, `preservative_pct` | Excipient contributions |
| Water | `water_ph` | Purified water quality at batch start |
| Carryover | `prev_lot_ph` | Previous lot pH — line conditioning effect |
| Process | `mixing_temp_c`, `mixing_time_min`, `agitation_rpm` | Physical mixing conditions |
| Sequence | `addition_order` | 1=Buffer→Acid→API / 2=Acid→Buffer / 3=API→Buffer |

**Key EDA findings:**
- Addition order 2 (Acid→Buffer→API) achieves only 34.6% approval vs 45.0% for order 1 — sequence matters chemically, not just operationally
- Previous lot pH at the lower boundary (≤4.6) measurably increases next-batch deviation risk
- Final pH distribution shows a cluster of lots near 4.0 — slightly below spec — indicating systematic under-buffering

---

## 🤖 Model

**Algorithm:** Random Forest (300 trees) — `sklearn.ensemble.RandomForestClassifier`

pH deviation is not a linear problem. High citric acid combined with high buffer produces a different outcome than high acid with low buffer. These interaction effects are the mechanism — and they are exactly what tree-based ensembles are designed to capture.

A single Decision Tree trained on the same data achieves AUC 0.746. The Random Forest (300 trees, each trained on a random data and feature subset) reaches AUC 0.918. The 17-point gap is not luck — it is the variance reduction effect of ensemble averaging. Individual trees overfit the formulation complexity; the ensemble generalizes.

**Why no scaling:** Random Forest uses threshold splits, not distances. A concentration in % and a temperature in °C compete on equal footing — no StandardScaler needed.

**Dual importance measures:** Gini (built-in, fast) and Permutation (test-set, rigorous) both computed. When they agree — as they do here — the signal is robust.

---

## 📈 Key Results

| Metric | Random Forest | Single Decision Tree |
|--------|--------------|---------------------|
| Test Accuracy | 84.1% | 78.3% |
| ROC-AUC | **0.918** | 0.746 |
| F1 (Approved) | 80.8% | — |
| Recall (Approved) | 80.4% | — |

**Confusion matrix (555 test batches):**

| | Pred: Fail | Pred: Pass |
|---|---|---|
| **Actual: Fail** | 282 (TN) ✅ | 43 (FP) |
| **Actual: Pass** | 45 (FN) | 185 (TP) ✅ |

45 false negatives — batches predicted to pass that would actually fail. In a regulatory context, each FN is a potential deviation event that the model didn't flag. Lowering the classification threshold from 0.5 reduces FN at the cost of more FP (unnecessary interventions).

---

## 🔍 Feature Importance

Both Gini and Permutation importance tell the same story:

| Feature | Gini | Permutation |
|---------|------|-------------|
| `citric_acid_pct` | **23.7%** | **Δ −24.2%** |
| `citrate_buffer_pct` | **21.6%** | **Δ −17.7%** |
| `prev_lot_ph` | 9.2% | Δ −3.8% |
| `water_ph` | 8.4% | Δ −2.9% |
| `mixing_time_min` | 7.5% | Δ −3.1% |
| `agitation_rpm` / `temp` / rest | combined 29% | Δ <1.7% each |

**The formulation insight:** citric acid and citrate buffer together account for 45% of Gini importance and dominate permutation importance by a wide margin. The ratio between these two — not their individual values — determines the pH outcome. This is exactly what physical chemistry predicts.

**The process insight:** agitation and mixing time matter, but far less than the recipe. You cannot fix a bad formulation with better process conditions (see Scenario C).

---

## 🗂️ Repository Structure

```
PH_Adjustment_Syrup/
├── 06_RF_PH_Adjustment.ipynb  # Notebook (no outputs)
├── rf_raw_data.csv            # Complete dataset (1,847 batches)
├── README.md
└── requirements.txt
```

> 📦 **Full Project Pack** — complete dataset (1,847 batches), notebook with full outputs including Gini vs Permutation comparison charts, presentation deck (PPTX + PDF), and `app.py` pre-batch risk simulator available on [Gumroad](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

**Option 1 — Google Colab:** Click the badge above.

**Option 2 — Local:**
```bash
pip install -r requirements.txt
jupyter notebook 06_RF_PH_Adjustment.ipynb
```

---

## 💡 Key Learnings

1. **Ensemble beats single tree by design, not by chance** — the jump from AUC 0.746 (DT) to 0.918 (RF) demonstrates the mechanics of bagging: many imperfect trees, each seeing different data, produce a collective vote that generalizes where any individual tree would overfit.
2. **Two importance measures that agree give high confidence** — Gini and Permutation use different computational approaches and answer slightly different questions. When both name the same top features, the signal is real, not an artifact of either method's bias.
3. **Process corrections cannot fix a formulation root cause** — Scenario C shows this clearly: increasing buffer and agitation on a 1.7% citric acid formula lowers approval probability from 11.7% to 1.0%. The acid concentration must be addressed at the recipe level.
4. **The previous lot effect is underappreciated** — `prev_lot_ph` carries 7.5% Gini importance. Line conditioning effects are real in liquid manufacturing, and they're rarely captured in traditional process control. This model surfaces them.
5. **Addition order is chemistry, not just procedure** — the 10.4 percentage-point approval rate difference between order 1 and order 2 isn't an operational preference — it reflects real differences in acid-buffer equilibration kinetics. The model learned what the chemistry textbook says.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning  
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
