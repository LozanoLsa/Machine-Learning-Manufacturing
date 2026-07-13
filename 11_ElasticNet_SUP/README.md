# Hot Stamping Intelligence — Tensile Strength Prediction via ElasticNet Regression

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/ElasticNet_Hot_Stamping/blob/main/11_ElasticNet_Hot_Stamping.ipynb)

> *"You don't need to know which regularisation to use — you need a framework that finds out for you."*

---

## 🎯 Business Problem

Hot stamping — press hardening a high-strength steel blank above 900°C and quenching it inside the die — is the dominant process for automotive structural safety parts: B-pillars, door reinforcements, bumper beams. The tensile strength of the finished part is determined by a complex interaction of thermal, mechanical, and material variables.

The process engineer faces a choice no textbook resolves cleanly: the data has **some correlated variables** (furnace entry and exit temperatures track each other at r = 0.95) and **some genuinely irrelevant ones**. The right regularisation depends on which problem dominates — and that is not known in advance.

Lasso zeroes redundant features. Ridge stabilises correlated ones. **ElasticNet does both — and lets the data decide the mixture.** By searching a joint grid of penalty strength (α) and L1/L2 mix (l1_ratio) via 5-fold cross-validation, ElasticNetCV selects the optimal combination without requiring the engineer to bet on one regularisation strategy upfront.

In this case, the CV grid confirms the data rewards sparsity: l1_ratio = 1.0 (pure L1) achieves the best score. `furnace_entry_temp_c` is zeroed — its information is fully captured by `furnace_exit_temp_c`. ElasticNet found the correct answer. A naive Ridge model would have kept both sensors active and mislead the process team into monitoring a redundant thermocouple.

---

## 📊 Dataset

- **1,741 hot stamping cycles** captured from the press PLC, furnace SCADA, and inline tensile testing records — spanning three production shifts and two steel grade changeovers
- **Target:** `tensile_strength_mpa` — part tensile strength after press hardening (MPa, continuous)
- **Range:** 591.5 – 1,417.6 MPa  |  **Mean:** 991.0 MPa  |  **Spec:** ≥ 900 MPa  |  **In-spec:** 75.1%
- **Steel grades:** Usibor 1500 MPa class (mean 1,052.6 MPa) · Ductibor 500 MPa class (mean 986.2 MPa) · MBorian 700 MPa class (mean 931.4 MPa)

| Column | Type | Description |
|---|---|---|
| `furnace_entry_temp_c` | float | Blank temperature entering the furnace (°C) — **zeroed by ElasticNet** |
| `furnace_exit_temp_c` | float | Blank temperature exiting the furnace (°C) |
| `die_temp_c` | float | Active die surface temperature (°C) |
| `transfer_time_s` | float | Time between furnace exit and die close (s) |
| `press_force_kn` | float | Maximum press force during forming (kN) |
| `press_speed_mm_s` | float | Ram closing speed (mm/s) |
| `dwell_time_s` | float | Time blank remains in closed die (s) |
| `blank_thickness_mm` | float | Sheet blank thickness (mm) |
| `steel_grade` | int | 1 = Usibor · 2 = Ductibor · 3 = MBorian |
| `active_cooling` | int | 1 = water-cooled die active · 0 = passive |
| `cooling_pressure_bar` | float | Cooling water pressure (bar) |
| `cooling_flow_l_min` | float | Cooling water flow rate (L/min) |
| `tensile_strength_mpa` | float | **Target** — part tensile strength after quench (MPa) |

### Data Origin (Real-World Perspective)

| Variable(s) | Source System | Notes |
|---|---|---|
| `furnace_entry_temp_c`, `furnace_exit_temp_c` | Furnace SCADA / Thermocouple Array | Logged at blank entry and exit portals — two sensors on the same furnace zone |
| `die_temp_c` | Press PLC / Die Thermocouple | Die surface temperature from embedded thermocouple — logged at die close |
| `transfer_time_s` | Press PLC Timer | Timestamp delta between furnace exit signal and die-close signal |
| `press_force_kn` | Press PLC / Load Cell | Peak force recorded during the forming stroke |
| `press_speed_mm_s` | Press PLC / Encoder | Ram velocity at die approach — from the motion controller |
| `dwell_time_s` | Press PLC Timer | Closed die dwell time — programmable parameter logged per stroke |
| `blank_thickness_mm` | ERP / Material Cert | Nominal blank thickness from the coil certificate or incoming inspection |
| `steel_grade` | ERP / Material Master | Grade code from the production order — set at job setup |
| `active_cooling`, `cooling_pressure_bar`, `cooling_flow_l_min` | Cooling Circuit PLC / Flow Meter | Cooling system parameters logged per cycle from the die cooling controller |
| `tensile_strength_mpa` | Inline Tensile Tester / QMS | **TARGET** — destructive or ultrasonic tensile estimate per production lot |

> In real-world hot stamping, this dataset requires joining at least four systems: the furnace SCADA (temperatures), the press PLC (force, speed, dwell, transfer time), the ERP (material spec and grade), and the quality management system (tensile test results). The join key is typically the stroke counter and shift timestamp — and the tensile result may arrive hours after the production event.

---

## 🤖 Model

**Algorithm:** ElasticNet Regression (L1 + L2 combined) — `sklearn.linear_model.ElasticNet` + `ElasticNetCV`

ElasticNet combines both regularisation penalties into a single framework:

$$\text{Loss} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \alpha \left[ \rho \sum_{j}|\beta_j| + \frac{1-\rho}{2} \sum_{j}\beta_j^2 \right]$$

Two hyperparameters are selected jointly: **α** (overall penalty strength) and **ρ / l1_ratio** (L1/L2 mixture). Setting l1_ratio = 0 gives pure Ridge; l1_ratio = 1 gives pure Lasso; values in between blend both behaviours.

**Why ElasticNet over Ridge or Lasso alone?** Because the structure of the hot stamping data was not known upfront. The furnace temperature collinearity (r = 0.95) suggested Ridge; the presence of at least one redundant sensor suggested Lasso. ElasticNet searches the full mixture landscape and lets the data answer. In this case, the CV confirmed l1_ratio = 1.0 — the data is Lasso-like. ElasticNet found the right answer without requiring a prior guess.

**Alpha and l1_ratio tuning:** `ElasticNetCV` searched 60 × 8 = 480 hyperparameter combinations (60 alpha values on a log scale × 8 l1_ratio candidates) using 5-fold cross-validation. Selected: **alpha = 0.1597 · l1_ratio = 1.00**.

**Preprocessing:** `StandardScaler` on all 12 features — mandatory for both L1 and L2 penalties.  
**Split:** 80/20 train/test, `random_state=42`.

---

## 📈 Key Results

| Metric | ElasticNet | Ridge | Lasso | Operational Meaning |
|---|---|---|---|---|
| **R²** | **0.893** | 0.891 | 0.893 | 89.3% of tensile variance explained — realistic ceiling under multi-shift process variability |
| **RMSE** | **43.24 MPa** | 44.1 MPa | 43.3 MPa | 4.4% relative error on 900 MPa spec — adequate for recipe classification and process window guidance |
| **MAE** | **34.64 MPa** | 35.2 MPa | 34.7 MPa | Median absolute miss across 349 test cycles |
| **Features active** | **10/12** | 12/12 | 10/12 | ElasticNet and Lasso zero entry temp and dwell time; Ridge keeps all |
| **Train / Test** | **1,392 / 349 cycles** | — | — | 80/20 split, `random_state=42` |

All three regularisations converge on accuracy within margin. **The value of ElasticNet is not higher R² — it is that the framework selected the correct penalty automatically.** Ridge would have kept `furnace_entry_temp_c` and `dwell_time_s` active with small misleading coefficients. ElasticNet zeroed both and told the truth.

---

## 🔍 ElasticNet Coefficients — Standardised (MPa per σ)

| Feature | Coefficient | Direction | Metallurgical Interpretation |
|---|---|---|---|
| `press_force_kn` | +113.87 | ↑ Increases strength | Dominant driver — forming pressure determines martensite transformation completeness |
| `steel_grade` | −48.27 | ↓ Reduces strength | Grade encoding: Grade 3 (MBorian) < Grade 2 (Ductibor) < Grade 1 (Usibor) by design |
| `die_temp_c` | −27.22 | ↓ Reduces strength | Hotter die slows quench — reduces martensite fraction — primary operator adjustment lever |
| `blank_thickness_mm` | −10.65 | ↓ Reduces strength | Thicker blanks cool more slowly — lower quench rate — lower martensite content |
| `active_cooling` | +6.68 | ↑ Increases strength | Water-cooled die enforces faster quench — direct strength gain |
| `furnace_exit_temp_c` | +7.22 | ↑ Increases strength | Higher austenising temperature → more complete transformation potential |
| `transfer_time_s` | −3.22 | ↓ Reduces strength | Longer transfer = more cooling before die close = less martensite |
| `cooling_flow_l_min` | +3.18 | ↑ Increases strength | Higher flow rate → faster heat extraction → stronger part |
| `cooling_pressure_bar` | +1.75 | ↑ Increases strength | Pressure ensures flow uniformity across die channels |
| `press_speed_mm_s` | +0.57 | ↑ Increases strength | Faster closure reduces cooling time in open air before quench |
| `furnace_entry_temp_c` | **0.000** | — | **Zeroed by ElasticNet** — fully redundant with furnace_exit_temp_c (r = 0.92) |
| `dwell_time_s` | **0.000** | — | **Zeroed by ElasticNet** — at observed operating range (3–20 s), dwell variation does not produce measurable tensile differences once die temperature and cooling configuration are controlled |

Two features zeroed, not one. `furnace_entry_temp_c` is a redundant sensor; `dwell_time_s` is a non-contributing parameter at the operational window captured in this dataset. ElasticNet eliminated both without being told which one was the problem.

---

## 🔧 Simulation & Scenarios

| Scenario | Configuration | Predicted | Status |
|---|---|---|---|
| **A — Usibor, Optimised** | Grade 1 · 920°C · 1,350 kN · die 140°C · active cooling · 4.0 s transfer | **1,224 MPa** | ✅ Pass (+324 MPa margin) |
| **B — MBorian, Passive Cooling** | Grade 3 · 870°C · 1,050 kN · die 165°C · passive · 6.5 s transfer | **734 MPa** | ⚠ Below 900 MPa spec |
| **C — MBorian, Corrected Recipe** | Grade 3 · 870°C · 1,050 kN · die 145°C · active cooling · 4.5 s transfer | **812 MPa** | ⚠ Below 900 MPa spec |

**Important context on Scenarios B and C:** MBorian is a 700 MPa target class designed for controlled ductility, not maximum strength. Its own product specification does not require 900 MPa. The simulator correctly shows that even an optimised recipe (Scenario C) reaches only 812 MPa for this grade — a physically accurate result, not a model failure. Activating cooling recovers **+77.6 MPa** vs Scenario B, confirming the cooling system is the primary lever for any grade.

The 2D response surface (Press Force × Die Temperature for Usibor with active cooling) extends this into a full process window map, showing every force/temperature combination that meets the 900 MPa spec.

---

## 🗂️ Repository Structure

```
ElasticNet_Hot_Stamping/
├── 11_ElasticNet_Hot_Stamping.ipynb   ← Notebook (no outputs)
├── hot_stamping_data.csv               ← Complete dataset (1,741 cycles)
├── README.md
└── requirements.txt
```

> ✅ **This project is completely free.** Full dataset (1,741 cycles) and simulator included. If this helped you, check out the rest of the portfolio at [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

**Option 1 — Google Colab:** Click the badge above.

**Option 2 — Local:**
```bash
pip install -r requirements.txt
jupyter notebook 11_ElasticNet_Hot_Stamping.ipynb
```

---

## 💡 Key Learnings

1. **ElasticNet is not a compromise — it is a search strategy.** Choosing between Ridge and Lasso requires knowing the data structure in advance. ElasticNetCV runs both and finds the answer. When the data is clearly Lasso-like (l1_ratio = 1.0), ElasticNet confirms it. When it is clearly Ridge-like (l1_ratio = 0.0), it confirms that too. The framework earns its place before you know which regularisation is right.

2. **A zeroed coefficient is a process insight.** `furnace_entry_temp_c` eliminated means the entry thermocouple adds zero predictive value once exit temperature is known. This is not a data quality finding — it is an instrumentation decision. If the sensor is expensive to maintain, the model just justified decommissioning it.

3. **R² = 0.893 under multi-shift variability is operationally useful.** Hot stamping across three production shifts, two steel grade changeovers, and real thermocouple drift does not produce laboratory-clean data. An R² of 0.893 means the model explains the large majority of tensile variance while honestly reflecting the remaining process scatter — which no recipe can fully eliminate.

4. **Press force dominates because physics dominates.** A coefficient of +114 MPa/σ on press_force_kn is not a modelling artefact — it reflects the metallurgical reality that forming pressure determines martensite transformation completeness. When a model's hierarchy matches first-principles engineering, the model is doing its job.

5. **Steel grade is a variable, not a constant.** The grade coefficient (−49.60 per σ) captures the metallurgical hierarchy across Usibor, Ductibor, and MBorian in a single parameter. A model that predicts tensile strength across three grade families simultaneously is a process library, not just a process predictor.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning  
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
