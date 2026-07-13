# Weld Quality Intelligence — Penetration Prediction & Feature Selection via Lasso Regression

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/Lasso_KDrivers_Welding/blob/main/09_Lasso_Feature_Selection_Process_Drivers.ipynb)

> *"In a 17-variable process, the most valuable question is not 'what predicts quality?' — it's 'what can we stop monitoring?'"*

---

## 🎯 Business Problem

A MIG/MAG welding station in a structural fabrication line captures data from 17 process and material variables every time a bead is deposited: electrical parameters, torch geometry, shielding gas composition, base material conditions, and arc stability readings. The engineering team suspects most of these variables are redundant — only a handful truly drive penetration depth.

The traditional response is to run a Design of Experiments. But with 17 factors, a full factorial DOE is operationally impossible. **Lasso regression offers a data-driven alternative**: it trains a predictive model and simultaneously imposes an L1 penalty that forces irrelevant coefficients to exactly zero. The result is both a predictor and a ranked list of the variables that actually matter.

This project delivers three things: a **prediction model** that estimates penetration before committing to the weld, a **feature selection result** that identifies 6 of 17 variables as genuinely uninformative, and a **contamination quantification** that makes visible what inspectors already suspect — rust and surface grease cancel energy improvements more than any other factor in the process.

---

## 📊 Dataset

- **1,843 weld bead records** captured from the MIG/MAG welding cell PLC, inline sensors, and post-weld inspection logs
- **Target:** `weld_penetration_pct` — effective weld penetration depth (%, continuous)
- **Range:** 0.0 – 94.5%  |  **Mean:** 48.8%  |  **Adequate zone [60–90%]:** 24.4% of welds
- **Candidate features:** 17  |  **Active after Lasso:** 11  |  **Zeroed by L1 penalty:** 6

| Column | Type | Description |
|---|---|---|
| `voltage_v` | float | Arc voltage (V) |
| `current_a` | float | Welding current (A) |
| `wire_feed_speed_mm_s` | float | Wire feed rate (mm/s) |
| `material_temp_c` | float | Base material preheat temperature (°C) |
| `ambient_temp_c` | float | Ambient temperature in welding bay (°C) |
| `travel_speed_mm_s` | float | Torch travel speed (mm/s) |
| `torch_angle_deg` | float | Torch inclination angle (°) — **zeroed by Lasso** |
| `ctwd_mm` | float | Contact tip to work distance (mm) |
| `gas_flow_l_min` | float | Shielding gas flow rate (L/min) |
| `co2_pct` | float | CO₂ content in shielding gas mix (%) — **zeroed by Lasso** |
| `thickness_mm` | float | Base plate thickness (mm) |
| `steel_type` | int | Steel grade category (0, 1, 2) — **zeroed by Lasso** |
| `rust_level` | float | Surface rust score 0–10 (inline vision system) |
| `surface_grease` | float | Surface contamination score 0–10 |
| `wire_diameter_mm` | float | Wire electrode diameter (mm) — **zeroed by Lasso** |
| `roller_wear_pct` | float | Wire feeder roller wear (%) — **zeroed by Lasso** |
| `arc_stability_rms` | float | Arc voltage RMS fluctuation — **zeroed by Lasso** |
| `weld_penetration_pct` | float | **Target** — effective penetration depth (%) |

### Data Origin (Real-World Perspective)

| Variable(s) | Source System | Notes |
|---|---|---|
| `voltage_v`, `current_a` | Welding Power Source / PLC | Logged per bead pass from the inverter controller |
| `wire_feed_speed_mm_s` | Wire Feeder Controller | Encoder reading captured at bead start |
| `material_temp_c` | Contact Pyrometer / Thermocouple | Pre-heat measurement on base material before arc strike |
| `ambient_temp_c` | Environmental Sensor | Bay-level temperature at time of weld |
| `travel_speed_mm_s`, `torch_angle_deg`, `ctwd_mm` | Robotic Arm Controller / SCADA | Kinematic parameters from the robot program for that joint sequence |
| `gas_flow_l_min`, `co2_pct` | Gas Console / Flow Meter | Shielding gas parameters logged at cell startup or per-job changeover |
| `thickness_mm`, `steel_type` | ERP / Part Master | Material specification from the production order |
| `rust_level`, `surface_grease` | Inline Vision System | Pre-weld surface quality scores assigned by the inspection camera |
| `wire_diameter_mm` | Material Cert / Consumable Log | Nominal electrode diameter from the spool label — changes infrequently |
| `roller_wear_pct` | Maintenance System | Roller wear percentage from the last maintenance inspection record |
| `arc_stability_rms` | Welding Power Source | Voltage fluctuation RMS computed from the high-frequency arc waveform |
| `weld_penetration_pct` | Post-Weld Inspection / NDT Log | **TARGET** — penetration score from inline or manual post-weld inspection |

> In real-world operations, this dataset requires joining at least five separate systems: the welding PLC, the robot controller, the inline vision system, the ERP part master, and the maintenance log. The join key is typically a weld sequence ID and shift timestamp — and it rarely exists in a clean format.

---

## 🤖 Model

**Algorithm:** Lasso Regression (L1 regularisation) — `sklearn.linear_model.Lasso` + `LassoCV`

Lasso is not just a regression model — it is a **feature selection mechanism**. The L1 penalty term forces the coefficients of uninformative variables to collapse to exactly zero, not merely shrink. This is the geometric property that distinguishes Lasso from Ridge: the L1 ball has corners, and the OLS solution tends to land on them, producing sparse models.

The critical difference from Project 08 (plain OLS): **scaling is mandatory**. The L1 penalty applies equally to all coefficients — if features are on different scales, variables with larger absolute ranges receive a smaller effective penalty, biasing the feature selection result unfairly. `StandardScaler` makes the competition fair before Lasso decides what stays.

**Alpha tuning:** `LassoCV` evaluated 60 candidate alpha values on a log scale from 0.001 to 100, using 5-fold cross-validation on the training set. The selected alpha (0.1081) is the value that maximises held-out R² without overfitting.

**Preprocessing:** `StandardScaler` on all 17 features, fit on training set only.  
**Split:** 80/20 train/test, `random_state=42`.

---

## 📈 Key Results

| Metric | Value | Operational Meaning |
|---|---|---|
| **R²** | **0.847** | 84.7% of penetration variance explained by 11 active features |
| **RMSE** | **6.27 %pts** | Typical prediction error — sufficient to distinguish weld quality bands |
| **MAE** | **5.08 %pts** | Median miss — roughly 10% relative error on the mean penetration |
| **OLS R²** | **0.847** | All 17 features, same accuracy — Lasso achieves parity with 6 fewer |
| **Train / Test** | **1,474 / 369 welds** | 80/20 split, `random_state=42` |

---

## 🔍 Feature Selection Result (Lasso Coefficients)

| Feature | Coefficient | Direction | Physical Interpretation |
|---|---|---|---|
| `current_a` | +9.14 | ↑ Increases penetration | Primary energy driver — amperage determines heat input per unit length |
| `wire_feed_speed_mm_s` | +7.84 | ↑ Increases penetration | Secondary energy driver — more wire = more filler and arc energy |
| `voltage_v` | +3.78 | ↑ Increases penetration | Arc geometry control — shapes the weld pool width and depth profile |
| `rust_level` | −2.61 | ↓ Reduces penetration | **Strongest negative driver** — surface rust scatters the arc and absorbs energy |
| `surface_grease` | −2.15 | ↓ Reduces penetration | **Second negative driver** — contamination disrupts shielding gas coverage |
| `thickness_mm` | +1.09 | ↑ Increases penetration | Thicker plate absorbs more energy before reaching the heat-affected zone limit |
| `gas_flow_l_min` | +0.83 | ↑ Increases penetration | Shielding quality — adequate flow prevents oxidation and porosity |
| `ctwd_mm` | −0.26 | ↓ Reduces penetration | Longer contact tip distance reduces current density at the arc |
| *(3 more active)* | *(small)* | — | `travel_speed_mm_s`, `material_temp_c`, `ambient_temp_c` — marginal contributions |
| `torch_angle_deg` | **0.000** | — | **Zeroed by Lasso** — angle variation within operating range carries no independent signal |
| `co2_pct` | **0.000** | — | **Zeroed by Lasso** — gas mix captured by `gas_flow_l_min` at observed conditions |
| `steel_type` | **0.000** | — | **Zeroed by Lasso** — grade categories do not separate penetration outcomes in this dataset |
| `wire_diameter_mm` | **0.000** | — | **Zeroed by Lasso** — diameter held nearly constant; no predictive variance |
| `roller_wear_pct` | **0.000** | — | **Zeroed by Lasso** — wear variation within range does not independently affect penetration |
| `arc_stability_rms` | **0.000** | — | **Zeroed by Lasso** — RMS fluctuation captured by other electrical parameters |

The six zeroed variables are not bad measurements — they are **redundant** at the operating conditions in this dataset. The model achieves the same predictive accuracy (R² 0.847) with 11 features as OLS achieves with all 17. Re-investing monitoring bandwidth toward rust and grease scoring will produce better ROI than tracking the zeroed signals.

---

## 🔧 Simulation & Scenarios

Three scenarios demonstrate the model's most important industrial message: **energy determines penetration potential, but contamination decides whether that potential is realised**.

| Scenario | Configuration | Predicted | Quality |
|---|---|---|---|
| **A — Standard Operation** | 24V · 180A · 90 mm/s · rust=3 · grease=3 | 49.1% | ⚠ Under-penetrated |
| **B — Max Energy, Heavy Contamination** | 26V · 220A · 110 mm/s · rust=8 · grease=7 | 44.1% | ⚠ Under-penetrated |
| **C — Max Energy, Clean Surface** | 26V · 220A · 110 mm/s · rust=0.5 · grease=0.5 | 85.0% | ✅ Adequate |

Scenario B vs C is the industrial payoff: the same electrical recipe delivers **40.9 percentage points more penetration** on a clean surface. Even more striking — Scenario B (maximum electrical settings, heavy contamination) produces a *worse* result than Scenario A (standard settings, moderate contamination). Increasing current and voltage cannot overcome a heavily contaminated surface. The pre-weld cleaning step that is sometimes skipped under production pressure carries a quantifiable cost that this model makes visible to anyone who reads it.

---

## 🗂️ Repository Structure

```
Lasso_KDrivers_Welding/
├── 09_Lasso_Feature_Selection_Process_Drivers.ipynb  ← Notebook (no outputs)
├── welding_data.csv                                   ← 250-row sample dataset (GitHub public)
├── README.md
└── requirements.txt
```

> 📦 **Full Project Pack** — complete dataset (1,843 records), notebook with full outputs, presentation deck (PPTX), and `app.py` weld penetration simulator available on [Gumroad](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

**Option 1 — Google Colab:** Click the badge above.

**Option 2 — Local:**
```bash
pip install -r requirements.txt
jupyter notebook 09_Lasso_Feature_Selection_Process_Drivers.ipynb
```

---

## 💡 Key Learnings

1. **Lasso is not just regression — it's DOE by data.** With 17 factors and an operationally impossible full factorial design, Lasso answers the same question a DOE would: which variables matter, and by how much. The L1 penalty replaces the experimental design with a mathematical one.

2. **Scaling is not optional for Lasso.** The L1 penalty shrinks coefficients proportionally. Without `StandardScaler`, features with larger absolute ranges receive smaller effective penalties — the feature selection is biased before it begins. Scale first, always.

3. **LassoCV is the principled tuning method.** Choosing alpha by hand or by a single train/validation split is fragile. 5-fold cross-validation over 60 candidates on a log scale finds the alpha that generalises — reproducibly, without data leakage.

4. **Matching OLS R² with fewer features is the result, not the goal.** Lasso achieving 0.847 vs OLS 0.847 while zeroing 6 features is not a coincidence — it confirms that those variables carry no independent signal. The model is not sacrificing accuracy for parsimony. It is proving parsimony is correct.

5. **Contamination data is process control intelligence.** Rust level and surface grease are the strongest negative drivers — and unlike current or voltage, they are controlled entirely before the arc strikes. A model that quantifies their impact gives the process engineer a business case for a cleaning step, not just a suspicion.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning  
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
