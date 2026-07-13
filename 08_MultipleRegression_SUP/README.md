# Heat Treatment Cycle Time Intelligence — Multi-Factor Regression for Quench Rack Scheduling

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/MR_Multi_Factor_Process_Impact/blob/main/08_Multi_Factor_Process_Impact.ipynb)

> *"You can't schedule what you can't measure. But you can estimate it."*

---

## 🎯 Business Problem

In heat treatment operations, **time is the invisible raw material**. Every rack that enters a muffle furnace occupies it for a block of time that no other job can reclaim. Getting that estimate wrong — or not having it at all — means scheduling slots built on intuition, buffers padded beyond what physics requires, and planners making decisions with one hand tied behind their back.

The plant in this study runs ~300 part numbers through a single quench rack furnace. For 200 of those PNs, historical cycle records exist. For the remaining **100 low-volume PNs**, no such history was ever captured — making them effectively invisible to the scheduling system.

This project closes that gap with multiple linear regression. A model trained on 500 observed furnace cycles learns how five process variables jointly determine cycle time. That model is then used in two ways: **real-time estimation** before a job starts, and a **planning gap-fill** that gives the scheduling team quantitative cycle time estimates for all 100 unobserved part numbers — for the first time.

---

## 📊 Dataset

- **647 furnace cycle records** captured from the muffle furnace PLC and plant MES layer across two fiscal quarters
- **Target:** `cycle_time_min` — total furnace cycle time per rack (continuous, minutes)
- **Range:** 205.1 – 381.0 min  |  **Mean:** 299.6 min  |  **Std:** 30.2 min
- **Source:** 200 high-volume part numbers with sufficient production history

| Column | Type | Description |
|---|---|---|
| `rack_piece_count` | int | Total pieces loaded in the rack |
| `avg_piece_mass_kg` | float | Weighted average mass per piece (kg) |
| `part_mix_count` | int | Number of distinct part numbers in the rack (1–5) |
| `avg_furnace_temp_c` | float | Mean furnace chamber temperature during cycle (°C) |
| `burner_power_pct` | float | Burner set-point during the hold phase (%) |
| `cycle_time_min` | float | **Target** — total furnace cycle time (minutes) |

### Data Origin (Real-World Perspective)

| Variable(s) | Source System | Notes |
|---|---|---|
| `rack_piece_count` | MES / Production Order | Logged at rack load confirmation before furnace door closes |
| `avg_piece_mass_kg` | ERP / Part Master | Nominal mass from engineering drawing or incoming weighing |
| `part_mix_count` | MES / Work Order Routing | Number of distinct PNs on the rack — derived from the production order |
| `avg_furnace_temp_c` | Furnace PLC / SCADA | Mean thermocouple reading during the hold phase of the cycle |
| `burner_power_pct` | Furnace PLC | Burner set-point captured at cycle initiation |
| `cycle_time_min` | MES / Cycle Log | Timestamp delta: door-close to quench-start, extracted from event log |

> In real-world operations, this dataset does not exist in a single system.
> Rack configuration comes from the MES production order, part mass from the ERP part master, and cycle timestamps from the PLC event log — three separate extractions that require a join on rack ID and shift timestamp before a single row of training data can be assembled.

---

## 🤖 Model

**Algorithm:** Multiple Linear Regression — `sklearn.linear_model.LinearRegression` + `statsmodels OLS`

Linear regression is the right model here for the same reason it was chosen in classical industrial statistics: **when the relationship is additive and each variable measures a distinct physical dimension, OLS is unbiased, interpretable, and statistically verifiable**. The five features in this dataset are nearly orthogonal (VIF < 1.02 for all), meaning each coefficient measures a clean, independent process effect.

Two model representations are used simultaneously: scikit-learn for predictions and the simulator, statsmodels for the full inferential layer — p-values, confidence intervals, F-statistic, and regression diagnostics. Both agree on coefficients; statsmodels adds the statistical accountability that turns a number into an engineering argument.

**Preprocessing:** No scaling required for OLS prediction — coefficients absorb unit differences and express each effect in physical units (minutes per kg, minutes per part type, etc.).
**Split:** 80/20 train/test, `random_state=42`.

---

## 📈 Key Results

| Metric | Value | Operational Meaning |
|---|---|---|
| **R²** | **0.805** | 80.5% of cycle-time variance explained by five process inputs |
| **Adj. R²** | **0.800** | All five features earn their place — penalty for predictors barely changes R² |
| **RMSE** | **13.50 min** | Typical prediction error — scheduler builds ±14 min buffer into slot assignments |
| **MAE** | **11.00 min** | Median absolute error — under 4% of the 300 min mean cycle time |
| **F-statistic** | **530.4 (p ≈ 0)** | Five features are jointly and highly significant |
| **Train / Test** | **517 / 130 cycles** | 80/20 split, `random_state=42` |

### Feature Coefficients — What Each Variable Costs in Minutes

| Feature | Coefficient | Direction | Engineering Interpretation |
|---|---|---|---|
| `avg_piece_mass_kg` | +21.60 min/kg | ↑ Increases cycle | Dominant driver — thermal mass saturates the cycle; +1 kg avg = +22 min |
| `part_mix_count` | +9.59 min/type | ↑ Increases cycle | Metallurgical complexity — tightest alloy window governs the whole rack |
| `burner_power_pct` | −0.37 min/% | ↓ Reduces cycle | Primary controllable reduction lever — 10% more power saves ~3.7 min |
| `rack_piece_count` | +0.29 min/piece | ↑ Increases cycle | Loading density — more pieces means more total mass to heat |
| `avg_furnace_temp_c` | −0.25 min/°C | ↓ Reduces cycle | Secondary lever — running 40°C hotter shaves ~10 min |

Every coefficient carries p < 0.001 and VIF < 1.02. These are not competing signals — they are additive, orthogonal, and directly actionable.

---

## 🗓️ The Planning Gap-Fill

The model's most operationally significant output is not a point prediction — it is the **systematic estimation of cycle times for 100 previously unschedulable part numbers**.

Before this project, those 100 low-volume PNs had no slot estimates. Planners either left them out of the schedule or padded their slots with worst-case buffers. The gap-fill exercise applies the trained model to all 100 PNs under a standard rack condition (120 pieces, 880°C, 85% burner, single PN type), producing a complete planning reference table — with known model uncertainty of ±13.5 min.

This is calibrated inference, not extrapolation. The mass range of the low-volume PNs (0.6–4.0 kg) falls entirely within the training envelope.

---

## 🔧 Simulation & Scenarios

Three operational scenarios demonstrate the model's planning value:

| Scenario | Configuration | Predicted | Notes |
|---|---|---|---|
| **A — Light Rack, High Efficiency** | 120 pcs · 1.2 kg · 1 PN · 900°C · 92% | 239.2 min (3.99 h) | Fastest achievable — light, homogeneous, high temp |
| **B — Heavy Mixed Rack, Risk Cycle** | 200 pcs · 2.8 kg · 4 PNs · 860°C · 78% | 343.6 min (5.73 h) | Worst-case — dense, mixed, cool, under-powered |
| **C — Heavy Rack, Corrected Recipe** | 200 pcs · 2.8 kg · 4 PNs · 900°C · 95% | 328.7 min (5.48 h) | +40°C and +17% power saves 15.0 min (4.4%) |

The B→C recovery shows that temperature and burner power — the two controllable levers — can partially compensate for a difficult rack configuration. The scheduler now has a number to work with, not a guess.

---

## 💡 Key Learnings

1. **Coefficients are the product, not the model.** In industrial regression, the equation itself is the deliverable — each coefficient answers a quantitative question that the process team can act on today without opening a laptop.

2. **Adj. R² barely moving from R² validates feature selection.** When the penalty for adding predictors barely changes the score (0.805 → 0.800), all five variables carry genuine signal. No feature is diluting the model.

3. **Low-volume PNs are not unknowable — they are unmeasured.** The gap-fill exercise turns a planning blindspot into a scheduling table. The difference between "we don't know" and "we haven't measured it yet" matters enormously for operations.

4. **statsmodels alongside scikit-learn is not redundant — it is necessary.** Scikit-learn gives predictions; statsmodels gives accountability. p-values and confidence intervals are what turn a model into an engineering argument that a plant manager will accept.

5. **The scheduler's uncertainty is now quantified.** Before the model, a planner building a slot for a 2.8 kg, 4-PN rack had no reference. After the model, they have a 343.6 min estimate with a ±13.5 min known error band. That is the difference between buffering by intuition and buffering by data.

---

## 🗂️ Repository Structure

```
MR_Multi_Factor_Process_Impact/
├── 08_Multi_Factor_Process_Impact.ipynb   ← Main notebook
├── quench_data.csv                         ← complete data set (GitHub public)
├── requirements.txt                        ← Python dependencies
└── README.md                               ← This file
```

> ✅ **This project is completely free.** Full dataset and simulator included. If this helped you, check out the rest of the portfolio at [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

```bash
git clone https://github.com/LozanoLsa/MR_Multi_Factor_Process_Impact
cd MR_Multi_Factor_Process_Impact
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn scipy jupyter
jupyter notebook 08_Multi_Factor_Process_Impact.ipynb

**Next Option — Run the simulator:**
```bash
python app.py
```

> Or click the badge above to run it directly in Google Colab — no installation needed.

—
## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning  
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
