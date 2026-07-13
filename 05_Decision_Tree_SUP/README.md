# Process Decisions Optimization — Decision Tree

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/Process_Decisions_Optimization/blob/main/05_DT_Process_Decisions_Optimization.ipynb)

> *"A new operator running a high-speed setup on a difficult material batch without completing the pre-run checklist. These factors interact in non-linear ways: velocity alone does not cause scrap, but that combination almost always does."*

---

## 🎯 Business Problem

Scrap in a stamping press line doesn't announce itself. By the time it's counted in the bin at end of shift, the material is already lost — along with the machine time, tooling wear, and downstream scheduling impact that came with it. The standard response is reactive: log the defect, investigate the cause, issue a corrective action. Repeat next week.

This project reframes the question: instead of investigating scrap after it happens, can a model score the risk of a bad run **before the press starts**? The answer is yes — because the conditions that produce scrap (operator experience, checklist completion, supplier lot quality, press speed) are all known at setup time. They exist in the data. They just haven't been connected to an actionable risk score.

A Decision Tree doesn't find hidden patterns in this problem. It **formalizes what experienced engineers already know** — and makes those rules auditable, transferable, and documentable.

---

## 📊 Dataset

- **2,317 stamping press production records** from a manufacturing environment spanning two production shifts and four machine lines
- **Target:** `scrap_risk` — three classes: Low / Medium / High
- **Class distribution:** Low 19.9% · Medium 50.4% · High 29.7%
- **Source:** Simulated operational data reflecting real stamping process factor interactions

| Layer | Feature | Description |
|-------|---------|-------------|
| Machine | `press_speed_spm` | Press speed in strokes per minute |
| Machine | `raw_material_hardness_hrb` | Material hardness in HRB scale |
| Operator | `operator_experience_yrs` | Years of operator experience |
| Operator | `shift` | Day / Night / Early_Morning |
| Material | `critical_supplier_lot` | Flag: 1 = lot from critical supplier |
| Environment | `ambient_temp_c` | Shop floor temperature at run start |
| Process | `recent_model_change` | Flag: 1 = model change in last 48h |
| Process | `setup_checklist_complete` | Flag: 1 = pre-run checklist completed |

**Key EDA findings:**
- Critical supplier lots: **53.5% High Risk** vs 19.4% for standard lots — the largest single structural gap
- Incomplete checklist: **42.2% High Risk** vs 16.8% when complete — a process control lever, not a luck factor
- Night shift: 37.6% High Risk vs 22.9% Day — partially explained by operator experience distribution

---

## 🤖 Model

**Algorithm:** Decision Tree (Gini, max_depth=5) — `sklearn.tree.DecisionTreeClassifier`

Decision Trees are the right model here for a reason that goes beyond performance: **the output is a set of if-then-else rules that can be printed and posted at the press**. The model doesn't just classify — it generates process documentation. A LinearSVC coefficient communicates direction; a Decision Tree rule communicates the exact threshold and the path to the decision.

This is a multiclass problem (three risk levels), so **macro-averaged F1** is the primary metric — it penalizes poor performance on any class equally, regardless of frequency.

**Why max_depth=5, min_samples_leaf=50:** Deliberately constrained. Each leaf must represent at least 50 production runs — not a single outlier. The tree is slightly less accurate than an unconstrained version, and significantly more generalizable. In manufacturing, that trade is always worth it.

**Preprocessing:** OneHotEncoder on `shift` (three levels), passthrough on everything else. No scaling — trees split on thresholds, not distances.

---

## 📈 Key Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 72.0% |
| Train Accuracy | 71.6% (near-zero gap — no overfitting) |
| F1 Macro | 71.8% |
| F1 Weighted | 72.0% |
| CV Accuracy (5-fold) | 69.2% ± 0.8% |

**Per-class performance:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| High | 0.71 | 0.74 | 0.73 |
| Low | 0.69 | 0.72 | 0.70 |
| Medium | 0.74 | 0.71 | 0.72 |

**Honest note:** The most common classification error is Medium runs scored as High (59 cases) and Low runs scored as Medium (36 cases). Both are directionally conservative errors — they over-flag risk rather than miss it. High-risk recall is 74%: three in four genuinely dangerous runs are correctly flagged before the press starts.

**Confusion matrix (696 test runs):**

| | Pred: High | Pred: Low | Pred: Medium |
|---|---|---|---|
| **Actual: High** | 154 ✅ | 0 | 53 |
| **Actual: Low** | 3 | 99 ✅ | 36 |
| **Actual: Medium** | 59 | 44 | 248 ✅ |

---

## 🔍 Feature Importance (Gini)

| Feature | Importance | What it means |
|---------|-----------|---------------|
| `setup_checklist_complete` | **23.7%** | Process control lever — the most actionable single intervention |
| `operator_experience_yrs` | **23.2%** | Experience compensates for difficult conditions |
| `critical_supplier_lot` | **19.2%** | Material quality — triggers a mandatory change in run conditions |
| `press_speed_spm` | **17.3%** | Speed interacts with experience — not dangerous alone |
| `recent_model_change` | **16.6%** | Setup instability signal — model changes elevate transition risk |
| `shift`, `hardness`, `temp` | **0.0%** | Zero Gini importance in this tree configuration |

The top five features each contribute meaningfully (16–24%), with no single dominant driver. Checklist completion — the most controllable variable — leads. The tree's early splits route on setup completion and operator experience: two variables that are known before the first stroke fires.

---

## 🗂️ Repository Structure

```
Process_Decisions_Optimization/
├── 05_DT_Process_Decisions_Optimization.ipynb  # Notebook (no outputs)
├── scrap_risk_data.csv                         # Complete dataset (2,317 records)
├── README.md
└── requirements.txt
```

> 📦 **Full Project Pack** — complete dataset (2,317 rows), notebook with full outputs including tree visualization and text rules, presentation deck (PPTX + PDF), and `app.py` pre-run risk simulator available on [Gumroad](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

**Option 1 — Google Colab:** Click the badge above.

**Option 2 — Local:**
```bash
pip install -r requirements.txt
jupyter notebook 05_DT_Process_Decisions_Optimization.ipynb
```

---

## 💡 Key Learnings

1. **The model output is the SOP** — `export_text()` produces if-then-else rules that can be transcribed directly into process control documents. No translation needed between model and practice.
2. **Multiclass F1 Macro is not optional** — with three classes and class imbalance, accuracy is misleading. A model that ignores Low entirely can still score 80% accuracy. Macro F1 prevents that illusion.
3. **Controlling complexity is a design decision** — max_depth=5 and min_samples_leaf=50 are not limitations imposed by the data. They're choices made to produce a model that generalizes to next week's production, not just last week's.
4. **Zero-importance features tell the story too** — shift, hardness, and ambient temperature carry no Gini importance. This doesn't mean they're irrelevant to scrap — it means their effect is already captured by the features that matter (experience, checklist, supplier lot).
5. **The interaction structure matters more than individual variables** — press speed at 55 spm with an experienced operator is manageable. At 55 spm with a 6-month operator and an incomplete checklist, it isn't. Trees capture this logic without feature engineering.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning  
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
