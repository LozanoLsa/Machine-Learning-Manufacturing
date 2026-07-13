# Delays Are Not Random — Logistic Regression

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/Delays_Are_Not_Random/blob/main/01_Logistic_Regression_Logistics.ipynb)

> *"A delay is not just a late shipment — it's a break in the operational flow. And breaks in operational flow always leave traces in the data."*

---

## 🎯 Business Problem

Every logistics operation deals with delays. The typical response is reactive: track the KPI, escalate when it's red, firefight until it's green again. This project asks a different question — what if you could score each shipment's probability of being late **before it leaves the dock**?

With 68.0% of shipments arriving delayed in this dataset, the problem isn't rare. It's structural. And structural problems have structural signals — which is exactly what Logistic Regression is built to find.

---

## 📊 Dataset

- **1,847 shipment records** from a road freight logistics operation across three route categories and two departure shifts
- **Target:** `delayed` (binary) — 1 = shipment arrived late, 0 = on time
- **Class balance:** 68.0% delayed (structurally imbalanced — delays are the norm, not the exception)
- **Source:** ERP/TMS operational data covering distance, load, facility timing, and operator assignment

| Feature | Type | Description |
|---------|------|-------------|
| `distance_km` | Numerical | Route distance (km) |
| `weight_kg` | Numerical | Cargo weight (kg) |
| `num_stops` | Integer | Number of intermediate delivery stops |
| `priority` | Binary | 1 = urgent shipment, 0 = standard |
| `loading_time_min` | Numerical | Time spent loading at origin dock (min) |
| `dock_wait_time_min` | Numerical | Pre-loading wait at dock (min) |
| `operator_experience_yrs` | Numerical | Driver/operator experience (years) |
| `route_type` | Categorical | urban / mixed / long_haul |
| `departure_shift` | Categorical | day / night |

**Key EDA finding:** Urban routes have the highest delay rate (74.5%) vs 57.7% for long-haul — more stops, traffic, and dock congestion. Night departures delay at 72.6% vs 64.9% for day shifts.

---

## 🤖 Model

**Algorithm:** Logistic Regression — `sklearn.linear_model.LogisticRegression`

The choice is deliberate. Before reaching for a complex ensemble, the right question is: *can a simple, interpretable model explain this problem well enough to act on?* In logistics, a model that operations teams can read and trust beats a black-box with 2% higher accuracy every time.

Logistic Regression outputs delay probability (0–1), not just binary pass/fail — enabling risk-based prioritization across the shipment queue.

**Preprocessing:** StandardScaler on numerics, Label Encoding for categoricals, all inside a sklearn Pipeline.

---

## 📈 Key Results

| Metric | Value |
|--------|-------|
| Accuracy | 69.7% |
| Precision (Delay) | 71.9% |
| Recall (Delay) | 91.3% |
| F1 (Delay) | 0.804 |
| AUC-ROC | 0.662 |

**Why Recall matters here:** Missing a delay costs more than a false alarm. 91.3% recall means the model catches 9 out of 10 actual delays before they happen — at the cost of some false alarms on borderline shipments.

---

## 🔍 Top Delay Drivers (Log-Odds Coefficients)

| Feature | Coefficient | Direction |
|---------|-------------|-----------|
| `route_type_urban` | +0.387 | 🔴 Risk factor |
| `num_stops` | +0.375 | 🔴 Risk factor |
| `dock_wait_time_min` | +0.288 | 🔴 Risk factor |
| `operator_experience_yrs` | −0.281 | 🔵 Protective |
| `departure_shift_night` | +0.247 | 🔴 Risk factor |
| `priority` | −0.243 | 🔵 Protective |

Urban routes and stop count are the strongest delay drivers. Experienced operators meaningfully reduce delay probability. Priority flagging works — urgent shipments are 10+ percentage points less likely to be late. These aren't surprises — they're quantified.

---

## 🗂️ Repository Structure

```
Delays_Are_Not_Random/
├── 01_Logistic_Regression_Logistics.ipynb  # Notebook (no outputs)
├── logistics_shipments_data.csv            # Sample dataset (250 rows)
├── README.md
└── requirements.txt
```

> 📦 **Full Project Pack** — complete dataset (1,847 records), notebook with full outputs, presentation deck (PPTX + PDF), and `app.py` simulator available on [Gumroad](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

**Option 1 — Google Colab:** Click the badge above.

**Option 2 — Local:**
```bash
pip install -r requirements.txt
jupyter notebook 01_Logistic_Regression_Logistics.ipynb
```

---

## 💡 Key Learnings

1. **Coefficients are storytelling tools** — they translate model output into language operations managers actually understand.
2. **Recall > Accuracy in logistics** — with 68.0% base delay rate, a naive classifier scores 68% accuracy doing nothing. Recall on the delay class is the real signal.
3. **Dock and route conditions dominate** — urban routes and stop count are the two strongest delay drivers, and they make physical sense: more congestion and more handoff points mean more places for delays to accumulate.
4. **Simple models deployed beat complex models in notebooks** — Logistic Regression is often good enough, and always explainable.
5. **EDA validates model logic** — if the most important features don't make physical sense, the model is learning noise. Here they do.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning  
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
