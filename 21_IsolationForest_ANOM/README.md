# Project 21 · Press Anomaly Intelligence — Isolation Forest Detection

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/IsoForest_Anomaly_Detection/blob/main/21_IsoForest_Anomaly_Detection.ipynb)

> *"The forest doesn't know what normal looks like — and that's exactly its advantage."*

---

## 🎯 Business Problem

Every anomaly detection system eventually faces the same question: **what does normal look like?**

Z-Score answers it with a distribution. Control charts answer it with a mean and a standard deviation. Both require you to define the center first — and then measure distance from it. That works well when your process generates one type of failure. A hot stamping line doesn't.

A boron steel blank heated to 950°C, pressed at 220+ tons, cooled in the die under active lubrication — this process can fail in at least three distinct ways simultaneously. A bearing wears and vibration spikes. A furnace trips and the blank arrives underheated. A tonnage valve drifts and forming force drops below spec. None of these failure signatures look alike. A single statistical boundary cannot hold all three.

Isolation Forest approaches the problem differently. It never asks what normal looks like. Instead, it asks: **how many random cuts does it take to isolate this point from all others?** A normal observation sits surrounded by its neighbors — many cuts are needed. An anomaly sits alone — one or two cuts isolate it immediately. The average path length across 300 trees becomes a continuous anomaly score, a severity gradient rather than a binary alarm.

This design has two consequences that matter for industrial monitoring. First, there is no assumption about the shape of the normal distribution — the algorithm works across Gaussian, skewed, and multimodal sensors simultaneously. Second, the continuous score enables tiered maintenance response: Normal → Mild → Moderate → Critical. The response can be proportional to the threat, not just binary.

This notebook demonstrates that architecture on 10,312 stamping cycles across three simultaneous failure modes — without the model ever seeing a single label during training.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Records | 10,312 stamping cycles |
| Features | 9 continuous sensor signals |
| Target | `is_anomaly` — engineer-validated ground truth (used for evaluation only) |
| Anomaly rate | 3.0% (309 flagged cycles) |
| Anomaly types | Type A (Mechanical) · Type B (Thermal) · Type C (Low-Pressure) |
| Source | Simulated hot stamping press line — boron steel automotive components |

### Feature Table

| Feature | Type | Unit | Normal Range | Failure Mode |
|---|---|---|---|---|
| `tool_temp_c` | float | °C | 420 – 440 | Tool overheating / cooling failure |
| `part_temp_c` | float | °C | 820 – 880 | Insufficient blank heating / furnace fault |
| `vibration_x_mm_s` | float | mm/s | 1.5 – 5.0 | Axis bearing wear / mechanical looseness |
| `vibration_y_mm_s` | float | mm/s | 1.5 – 5.0 | Cross-axis resonance / imbalance |
| `press_force_ton` | float | ton | 210 – 250 | Low tonnage → incomplete forming |
| `contact_force_kn` | float | kN | 165 – 200 | Clamping / die alignment issue |
| `cycle_time_s` | float | s | 15 – 22 | Process slowdown / robot delay |
| `energy_kwh` | float | kWh | 20 – 30 | Energy spike / high friction |
| `lubricant_flow_lmin` | float | L/min | 3.5 – 6.5 | Lubrication failure |

### Key EDA Findings

- **Vibration sensors show the clearest bimodal separation.** `vibration_x_mm_s` and `vibration_y_mm_s` exhibit distinct clusters between normal and Type A (mechanical) anomalies. Temperature and pressure shifts are subtler — visible to Isolation Forest in 9-dimensional space but not obviously separable in any single 2D projection.
- **Type C (low-pressure) anomalies are embedded within the normal cloud** in most dimensions. `press_force_ton` and `contact_force_kn` drop only 5% below normal mean — enough to flag with IF scoring but insufficient for a fixed threshold alarm. This is the failure mode that a Z-Score-only approach would miss.
- **Press force and contact force are highly correlated by design.** Isolation Forest exploits this via random partitioning across all 9 sensors simultaneously — a die alignment failure tends to express itself in both channels together.

---

## 🤖 Model

### Why Isolation Forest for This Problem

The case for Isolation Forest here is built on what Z-Score cannot do.

A hot stamping press generates three mechanistically distinct failure modes. Defining a single Gaussian boundary for a 9-sensor space that contains thermal anomalies, mechanical anomalies, and low-pressure anomalies simultaneously is not straightforward — the anomaly region is not convex, not centered, and not the same shape in every sensor dimension.

Isolation Forest partitions the feature space using random trees. Each tree randomly selects a feature and a split threshold, recursively isolating subsets of the data. The number of splits required to isolate a point — the path length — becomes its anomaly score. Short path → easy to isolate → unusual. This mechanism is distribution-agnostic and naturally handles multi-type anomaly structures without prior knowledge of failure modes.

The continuous score also provides something operationally valuable: a severity gradient. A cycle scoring at −0.67 is not just anomalous — it is in a different operational category than a cycle scoring at −0.54. That difference maps to a response: stop vs. monitor.

### Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 300 | More trees → more stable average path lengths across the score range |
| `contamination` | 0.03 | Matches known anomaly rate — confirmed optimal via F1 sensitivity analysis |
| `bootstrap` | True | Subsampling with replacement per tree — reduces correlation between trees |
| `random_state` | 42 | Reproducibility (train/test split only) |

**Preprocessing:** `StandardScaler` is applied before training. Without it, sensors with high absolute ranges (tool_temp_c: 393–497°C) would dominate random split selection over narrow-range sensors (energy_kwh: 17–35 kWh). Scaling ensures every sensor contributes equally to the partitioning process.

**Score interpretation:** `score_samples()` returns values in [−1, 0]. Score near −1 → very short path → highly anomalous. Score near 0 → long path → highly normal. Decision threshold is set at the 3rd percentile of the score distribution (−0.5349), matching the 3% contamination rate.

---

## 📈 Key Results

| Metric | Value | Operational Meaning |
|---|---|---|
| **Precision** | 0.8516 | 85.2% of alarms are real anomalies — 46 false positives per 10,312 cycles |
| **Recall** | 0.8544 | 85.4% of real anomalies are caught — 45 missed events per 10,312 cycles |
| **F1-Score** | 0.8530 | Strong balanced performance across all three anomaly types |
| **Accuracy** | 0.9912 | 10,221 of 10,312 cycles correctly classified |
| **IF Score range** | [−0.708, −0.358] | Full score distribution across 10,312 cycles |
| **Decision threshold** | −0.5326 | 3rd percentile — calibrated to 3% contamination prior |

**Confusion Matrix (N = 10,312 cycles):**

|  | Predicted Normal | Predicted Anomaly |
|---|---|---|
| **Actually Normal** | 9,957 ✅ | 46 ⚠ |
| **Actually Anomaly** | 45 ⚠ | 264 ✅ |

**The 45 false negatives are Type C (low-pressure) anomalies.** Their press force drops only ~5% below normal mean — within the overlap zone of the normal distribution for that sensor. These can be addressed by adding a domain-rule post-filter (`press_force_ton < 200`) without retraining the model.

**Contamination sensitivity confirms the 3% prior is optimal.** F1 degrades at contamination = 0.02 (misses too many) and 0.05 (too many false alarms). The 0.03 setting maximizes balanced detection performance.

---

## 🔍 Feature Impact & Anomaly Profile

Isolation Forest does not output feature importances natively. Feature-score correlation reveals which sensors drive isolation:

| Feature | Correlation with IF Score | Direction | Anomaly Δ |
|---|---|---|---|
| `vibration_x_mm_s` | −0.2406 | High value → more anomalous | +50.0% |
| `vibration_y_mm_s` | −0.2177 | High value → more anomalous | +52.7% |
| `energy_kwh` | −0.1784 | High value → more anomalous | +11.2% |
| `cycle_time_s` | −0.1518 | High value → more anomalous | +13.5% |
| `tool_temp_c` | −0.1084 | High value → more anomalous | +2.2% |
| `press_force_ton` | +0.1220 | Low value → more anomalous | −5.0% |
| `contact_force_kn` | +0.0675 | Low value → more anomalous | −4.1% |

Vibration is the dominant isolation signal — anomalous cycles show 50%+ elevation in both axes. `vibration_x_mm_s` is the single strongest individual isolation driver. Force channels are inverse drivers: their anomaly signature is a drop, not a spike, which is why the positive correlation indicates lower force → shorter path → more anomalous.

**Anomaly type breakdown across detected events:**

| Type | Signature | Primary Sensors |
|---|---|---|
| **Type A — Mechanical** | High vibration + elevated energy | `vibration_x`, `vibration_y`, `energy_kwh` |
| **Type B — Thermal** | High temperatures + extended cycle | `tool_temp_c`, `part_temp_c`, `cycle_time_s` |
| **Type C — Press** | Low tonnage + low contact force | `press_force_ton`, `contact_force_kn` |

---

## 🔭 Simulator — Three Reference Scenarios

The `evaluate_cycle()` function scores a new stamping cycle in real time and returns a severity classification with recommended maintenance action.

| Scenario | Key Conditions | IF Score | Severity | Action |
|---|---|---|---|---|
| **A — Stable** | All sensors nominal | −0.3625 | ✅ Normal | Continue production |
| **B — Mechanical** | Vibration X=10.5, Y=9.8, Energy=32 kWh | −0.6735 | 🚨 Critical | Stop cycle immediately |
| **C — Low Press** | Force=178 ton, Contact=128 kN | −0.5701 | ⚠ Anomaly | Inspect within 2 hours |

Severity tiers are defined by score percentiles: Critical (< 5th pct), Moderate (5th–10th), Mild (above threshold), Normal (below threshold). This converts the binary alarm into a tiered response that maintenance teams can act on proportionally.

---

## 🗂️ Repository Structure

```
IsoForest-Anomaly-Detection/
├── 21_IsoForest_Anomaly_Detection.ipynb   # Educational notebook (no outputs)
├── Data.csv                               
├── requirements.txt
└── README.md
```

> 📦 **Full Project Pack** — complete 10,312-cycle dataset, notebook with full outputs,
> presentation deck (PPTX + PDF), and `app.py` real-time simulator available on
> [Gumroad](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

**Option 1 — Colab (recommended, no setup):**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/IsoForest_Anomaly_Detection/blob/main/21_IsoForest_Anomaly_Detection.ipynb)

**Option 2 — Local:**

```bash
git clone https://github.com/LozanoLsa/IsoForest_Anomaly_Detection.git
cd IsoForest_Anomaly_Detection
pip install -r requirements.txt
jupyter notebook 21_IsoForest_Anomaly_Detection.ipynb
```

**Requirements:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

---

## 💡 Key Learnings

1. **Inversion of perspective is the algorithm's real innovation.** Isolation Forest never defines what normal looks like — it measures how easy a point is to isolate. This shift removes the need for a stable baseline distribution and makes the method robust to multi-type, multi-dimensional anomaly structures that no single statistical boundary can capture.

2. **StandardScaler is not optional here — it changes the result.** Without scaling, sensors with wide absolute ranges dominate random split selection. The scaler is fitted on the training set and must be applied identically to all new cycles in production. Fitting a new scaler on incoming data resets the baseline and invalidates the model.

3. **The 85.4% recall on Type C anomalies is an honest result, not a failure.** Low-pressure failures express themselves as a 5% force drop — within the overlap zone of the normal distribution. No unsupervised method can reliably separate signals that small without domain knowledge. The notebook shows the correct response: add a domain rule post-filter for `press_force_ton < 200` rather than forcing the algorithm to solve a physics problem it cannot see.

4. **Continuous scoring is worth more than the binary flag.** The difference between a score of −0.67 and −0.54 maps directly to a difference in maintenance urgency. A stop-cycle alarm and a monitor-and-inspect recommendation are not the same response. The severity tier system built in Section 10 demonstrates how to operationalize this gradient without additional modeling.

5. **Contamination calibration belongs in the model design loop.** Setting `contamination` arbitrarily at 0.05 or 0.10 because those are "common defaults" changes which cycles get flagged and which get missed. This notebook demonstrates the sensitivity analysis that should accompany every contamination choice — and shows that the correct value is determined by domain knowledge, not convention.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
