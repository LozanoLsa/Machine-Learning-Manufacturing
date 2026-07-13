# ⚙️ Riveting Process Intelligence — Head Diameter Prediction via Linear Regression

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/LR_Cycle_Time_Estimation/blob/main/07_LR_Cycle_Time_Riveting.ipynb)

**Project 07 · LozanoLsa Industrial ML Portfolio**
**Algorithm:** Multiple Linear Regression (OLS · Ridge · Lasso · ElasticNet)
**Domain:** Automotive & Aerospace Manufacturing — Joining & Assembly

---

## Project Overview

Every rivet head is a negotiation between material, force, and time. Get the combination wrong and the joint is either too weak to carry load or too large to fit the adjacent panel — and you find out after the press has fired, not before.

This project builds a linear regression model trained on 1,763 press cycles spanning two production shifts and three rivet size families that predicts the formed rivet head diameter from six process inputs. The model is deliberately interpretable: every coefficient maps to an engineering lever an operator can actually touch. No black box. Just physics, quantified.

No magic. Just coefficients.

---

## Problem Statement

In high-volume riveting — automotive body panels, aerospace fuselage skins, HVAC ductwork — the formed head of a rivet must land inside a 0.4 mm specification window. Too small and the bearing area is insufficient for structural integrity. Too large and the head interferes with adjacent components or triggers costly rework.

The traditional loop is reactive: run production, pull samples, send to CMM, wait for the report. By the time a dimensional deviation is confirmed, dozens of non-conforming parts are already in the queue. Worse, changeovers between rivet sizes — say, switching from a 4 mm to a 5 mm shank — are handled with the same press recipe until the CMM report says otherwise.

This project interrupts that loop. The model lets an engineer query the expected head diameter before committing to a recipe — and, critically, before the press fires.

---

## Objective 🎯

- Predict the formed rivet head diameter (mm) from six controllable and environmental process variables using an interpretable linear model.
- Enable pre-production recipe validation: identify whether a force/stroke combination will produce an in-spec head, and by what margin.
- This project is intentionally a foundation for analysis, not a production-ready solution. The model covers the training envelope [25–60 kN force, 3–6 mm stroke] and assumes first-order linear behavior — both valid conditions for standard riveting operations.

---

## Dataset Description 📊

### Features (X)

| Feature | Description | Units |
|---|---|---|
| `press_force_kn` | Applied press force | kN |
| `press_stroke_mm` | Ram travel distance | mm |
| `rivet_diameter_mm` | Nominal shank diameter (3, 4, or 5 mm) | mm |
| `rivet_length_mm` | Rivet shank length | mm |
| `temperature_c` | Ambient temperature in press bay | °C |
| `hold_time_ms` | Dwell time at full press force | ms |

### Target Variable (Y)

| Variable | Description |
|---|---|
| `head_diameter_mm` | Formed rivet head diameter measured by inline CMM (mm) |

Spec window: **[5.0 – 5.4] mm**. The auxiliary column `head_diameter_ok` (1 = in-spec, 0 = NG) is excluded from modeling to prevent data leakage.

### Data Origin (Real-World Perspective)

| Variable(s) | Source System | Notes |
|---|---|---|
| `press_force_kn`, `press_stroke_mm` | Press PLC | Recorded per cycle by the servo controller and ram encoder |
| `hold_time_ms` | Press PLC Timer | Dwell time logged at cycle completion |
| `rivet_diameter_mm`, `rivet_length_mm` | Material Cert / ERP | Nominal dimensions from incoming inspection certificate |
| `temperature_c` | Environmental Sensor | Ambient temperature at the press bay, logged at cycle start |
| `head_diameter_mm` | Inline CMM Station | Measured formed head dimension — the TARGET variable |

> In real-world operations, this dataset does not exist in a single system.
> PLC logs, CMM exports, and ERP material certificates must be joined — and that integration effort is typically 80% of the engineering work.

---

## Modeling Approach 🧠

Riveting head formation is, to first order, a linear physical process. More press force displaces more material. A larger rivet shank provides more material to form. The relationship between these inputs and the output diameter is additive and nearly proportional — which makes linear regression not a compromise, but the right model.

Four variants were benchmarked to test whether regularisation adds value when the underlying relationship is genuinely linear:

| Model | Penalty | Result |
|---|---|---|
| OLS Linear Regression | None | Best — no regularisation benefit needed |
| Ridge (α = 1.0) | L2 | Identical performance to OLS |
| Lasso (α = 0.01) | L1 | No feature zeroed out — all features carry signal |
| ElasticNet (α = 0.01, ρ = 0.5) | L1 + L2 | Same convergence |

When Ridge, Lasso, and ElasticNet all converge to OLS performance, the features are honest. Every variable in the model carries genuine, non-redundant signal. That is worth knowing.

OLS was selected as the production model. Its coefficients are unbiased, directly interpretable in engineering units, and require no scaling — making them a natural tool for process engineers who speak in kN and mm, not standardised z-scores.

---

## Key Results 📈

| Metric | Value | Operational Meaning |
|---|---|---|
| **R²** | **0.909** | 90.9% of head-diameter variance explained by six process inputs |
| **RMSE** | **0.081 mm** | Typical prediction error — 8.1% of the 1.0 mm spec window |
| **MAE** | **0.065 mm** | Median absolute error — within most press measurement system resolution |
| **Baseline R²** | −0.011 | Naïve mean model performs no better than chance |
| **Train / Test** | 1,410 / 353 cycles | 80/20 split, `random_state=42` |

### Feature Coefficients — Engineering Levers

| Feature | Coefficient | Direction | Engineering Interpretation |
|---|---|---|---|
| `rivet_diameter_mm` | +0.186 mm/mm | ↑ Increases diameter | Dominant driver — more shank material → larger formed head |
| `press_stroke_mm` | +0.075 mm/mm | ↑ Increases diameter | Primary controllable lever for recipe tuning |
| `press_force_kn` | +0.019 mm/kN | ↑ Increases diameter | Secondary lever — fine-tune force within recipe |
| `rivet_length_mm` | +0.004 mm/mm | ↑ Increases diameter | Small but consistent at high volume |
| `temperature_c` | +0.003 mm/°C | ↑ Increases diameter | Thermal expansion compounds across thousands of cycles |
| `hold_time_ms` | +0.0002 mm/ms | ↑ Increases diameter | Minimal — creep effect at extreme dwell times only |

The coefficient structure confirms what metallurgical intuition would suggest: the rivet shank diameter governs the result, stroke and force are the tunable levers, and environmental variables leave a small but measurable fingerprint.

---

## Simulation & Scenarios

The model becomes an engineering tool in Section 10. Three scenarios demonstrate predictive power:

| Scenario | Setup | Predicted | Status |
|---|---|---|---|
| **A — Standard Operation** | 4 mm rivet, 42 kN, 4.5 mm stroke | 5.236 mm | ✅ OK (+0.164 mm margin) |
| **B — Unannounced Batch Change** | 5 mm rivet, same press recipe | 5.419 mm | ❌ NG (−0.019 mm over spec) |
| **C — Model-Guided Adjustment** | 5 mm rivet, 32 kN, 3.6 mm stroke | 5.147 mm | ✅ OK (+0.147 mm margin) |

The 2D response surface (Section 8) extends this into a full process design map: given a rivet size and a target diameter, the optimal force/stroke combinations are identified by grid search — without a single physical trial.

---

## Business Insights

1. **Rivet diameter is the dominant variable.** Switching shank size without recalculating the recipe is the most common source of out-of-spec heads. The model makes this miscalibration visible before the press fires.

2. **50.7% overall spec compliance hides a 5 mm problem.** The 4 mm rivet family achieves 55.0% in-spec; the 5 mm family drops to 38.8%. The current process is not calibrated for larger rivets.

3. **Force and stroke are the levers.** Combined coefficient of +0.075 (stroke) and +0.019 (force) means a 1 mm stroke increase raises head diameter by roughly the same as a ~4 kN force increase — useful for recipe trade-off decisions.

4. **Environmental drift is quantifiable.** A 10°C temperature swing adds 0.027 mm to predicted head diameter. Across a summer production ramp, this is not negligible — and the model gives the SPC team a physical basis for tightening control limits seasonally.

5. **OLS outperforming nothing and matching regularisation means the model is well-posed.** There is no noise to regularise and no redundant features to shrink. Nothing revolutionary. Just quantified.

---

## 🗂️ Repository Structure

```
LR_Cycle_Time_Estimation/
├── 07_LR_Cycle_Time_Riveting.ipynb   ← Main notebook (12 sections)
├── lr_riveting_data.csv              ← Complete 1,763-cycle dataset
├── requirements.txt                   ← Python dependencies
├── 07_LR_Riveting_Head_Diameter.pdf  ← Slide deck (PDF export)
└── README.md                          ← This file
```

> 📦 **Full Project Pack** — complete 1,763-cycle dataset, notebook with full outputs, presentation deck (PPTX + PDF), and `app.py` pre-cycle recipe simulator available on [Gumroad](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

```bash
git clone https://github.com/LozanoLsa/LR_Cycle_Time_Estimation
cd LR_Cycle_Time_Estimation
pip install pandas numpy scikit-learn matplotlib seaborn scipy statsmodels jupyter
jupyter notebook 07_LR_Cycle_Time_Riveting.ipynb
```

> Or click the badge above to run it directly in Google Colab — no installation needed.

**Notebook structure:**

| Section | Content |
|---|---|
| 1 | Setup & imports |
| 2 | Load data |
| 3 | Sanity checks |
| 4 | Exploratory data analysis |
| 5 | Preprocessing & train/test split |
| 6 | Model training (OLS · Ridge · Lasso · ElasticNet) |
| 7 | Model evaluation (metrics · predicted vs measured · residuals) |
| 8 | Interpretability (coefficients · SHAP · 2D response surface) |
| 9 | Statistical assumption validation (normality · DW · GQ · VIF) |
| 10 | Process simulator (3 scenarios · optimal recipe grid search) |
| 11 | Final reflection |

---

## 💡 Key Learnings

1. **Interpretability is a design goal, not a consolation prize.** A coefficient of +0.183 mm/mm is not a "simple" result — it is a direct translation of physics into a number an engineer can act on.

2. **When regularisation adds nothing, the features are honest.** Ridge, Lasso, and ElasticNet all converge to OLS performance here. This means every input variable carries genuine, non-redundant signal worth keeping.

3. **R² of 0.909 is strong for a multi-shift physical process.** It means the six press parameters explain 90.9% of the variation in head diameter across two production shifts and three rivet families. The remaining 9.1% represents genuine process scatter — inter-shift tooling variation, material batch effects, and CMM measurement noise that no recipe model can eliminate.

4. **The 2D response surface turns a model into a tool.** The ability to visually read the in-spec operating region for any rivet size transforms regression from a passive estimator into an active process design instrument.

5. **Prediction before the press fires is the entire value proposition.** The RMSE of 0.081 mm — 8.1% of the 1.0 mm spec window — is operationally useful at the point where recipe decisions are made, even if it does not reach CMM-level precision.

---

## Next Steps 🚀

1. **Add Bayesian or conformal prediction intervals.** OLS gives a point estimate: 5.236 mm. A risk-based manufacturing decision needs uncertainty bounds: "95% of cycles with this recipe will land between 5.174 and 5.298 mm." That is a different conversation — and a more powerful one.

2. **Extend to a nonlinear model for extreme parameter ranges.** The linear model is valid within the training envelope. At very high forces or very long hold times, material hardening and saturation effects introduce nonlinearity. A polynomial or Gaussian process regression would capture those edges.

3. **Build a real-time PLC integration prototype.** The `predict_diameter()` function is already production-ready as a callable. The next step is connecting it to the press controller — so the model runs before each cycle, not just as a notebook demo.

4. **Apply the response surface to recipe standardisation across shifts.** Different shifts often run slightly different recipes for the same rivet size. The model provides a single, physics-grounded reference recipe that engineering teams can align around.

---

—
## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning  
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
