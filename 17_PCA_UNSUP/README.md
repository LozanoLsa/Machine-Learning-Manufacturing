# 17 · Principal Component Analysis — Predictive Maintenance for Industrial Motors

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/PCA_Predictive_Maintenance/blob/main/17_PCA_Predictive_Maintenance.ipynb)

> *"Ten sensors produce noise. Three principal components produce understanding. The distance in that reduced space tells you what the machine cannot say out loud."*

---

## 🎯 Ten Sensors, One Truth

A maintenance engineer sits in front of a dashboard showing 10 trend charts.

Vibration RMS: 4.2 mm/s — elevated but below the 5.0 alarm. Bearing temperature: 78°C — within bounds. Lubrication pressure: 2.8 bar — acceptable. High-frequency vibration: 0.9g — fine. Current: 22A — normal range. He looks at all ten charts and decides to wait until the next scheduled inspection window.

Seventy-two hours later, the motor fails mid-shift.

The information was there. It was hiding in the correlation structure between the ten variables — not in any single reading, but in the pattern that all ten were drawing together. Vibration was climbing. Temperature was climbing with it. Lubrication pressure was dropping in exact proportion. Current was rising. Power factor was deteriorating. Ten sensors, each telling a fraction of the same story: **this motor is on a degradation trajectory.**

Principal Component Analysis does not watch ten charts. It watches the geometry. It compresses ten correlated sensor readings into a low-dimensional **health space** — a map where every motor occupies a position determined by all its signals simultaneously. In that map, a degrading motor does not look like ten slightly elevated readings. It looks like a point drifting steadily away from where healthy motors live.

This project applies PCA to **1,687 motor condition snapshots** from 10 industrial sensors. PC1 — the Mechanical Degradation Axis — captures **54.4% of all variance** in the dataset by itself. It is not a statistical abstraction. It is the degradation trajectory, extracted from the correlation structure of the data. A single number — the **Reliability Index (IR)** — combines reconstruction error (SPE) and distance from the origin (Hotelling T²) into a health score that collapses ten dimensions into one operational decision.

> 📦 **Full Project Pack** — complete dataset, notebook with full outputs,
> presentation deck (PPTX + PDF), and `app.py` simulator available on
> [Gumroad](https://lozanolsa.gumroad.com).

---

## 📊 Dataset

**File:** `mtto_data.csv` — 1,687 motor condition snapshots from 10 continuous sensors across three operating states.

| Column | Unit | Description | PCA Role |
|--------|------|-------------|----------|
| `vibration_rms_mm_s` | mm/s | RMS vibration amplitude at motor housing | Mechanical health |
| `vibration_hf_g` | g | High-frequency vibration — bearing wear indicator | Mechanical health |
| `bearing_temp_c` | °C | Bearing housing temperature | Thermal health |
| `motor_current_a` | A | Motor phase current | Electrical load |
| `voltage_v` | V | Supply voltage | Electrical supply |
| `power_factor_pct` | % | Power factor | Electrical quality |
| `speed_rpm` | rpm | Shaft rotational speed | Mechanical load |
| `lube_flow_l_min` | L/min | Lubricant flow rate | Lubrication system |
| `lube_pressure_bar` | bar | Lubricant line pressure | Lubrication system |
| `internal_humidity_pct` | % | Internal motor cavity humidity | Environmental |
| `op_status` | label | Normal / Alert / Critical | **Visualization only — not used in PCA** |

> ⚠️ `op_status` is used **only for EDA coloring and external validation.** PCA operates in fully unsupervised mode on the 10 sensor readings.

**Status Distribution:**

| Status | Count | Share |
|--------|-------|-------|
| Normal | 998 | 59.2% |
| Alert | 517 | 30.6% |
| Critical | 172 | 10.2% |

**Data Origin — Real-World Source Systems:**

| Feature | Source System | Instrument |
|---------|--------------|------------|
| `vibration_rms_mm_s`, `vibration_hf_g` | SCADA / CbM System | Triaxial accelerometer at motor housing — RMS and HF band extracted |
| `bearing_temp_c` | DCS / PLC | RTD or IR pyrometer at bearing housing |
| `motor_current_a`, `voltage_v`, `power_factor_pct` | Power monitoring relay (PMR) | CT-based current measurement + voltage tap per phase |
| `speed_rpm` | DCS | Proximity probe or encoder on shaft |
| `lube_flow_l_min`, `lube_pressure_bar` | SCADA / Lubrication system | Inline flow meter and pressure transducer on lube circuit |
| `internal_humidity_pct` | SCADA | Capacitive humidity sensor inside motor enclosure |

**Key EDA Findings:**

The sensor array is highly correlated — not by accident, but by physics. Vibration drives bearing temperature through friction. Bearing wear reduces lubrication efficiency, dropping flow and pressure. Higher friction and heat degrade insulation, increasing current draw and deteriorating power factor. Humidity rises as seals degrade. The degradation cascade is a coordinated multi-sensor event, which is why PCA — not any single threshold — is the right tool to detect it.

The degradation signature from Normal to Critical state confirms this:

| Sensor | Normal | Critical | Change |
|--------|--------|---------|--------|
| Vibration RMS (mm/s) | 1.99 | 6.17 | **+210%** |
| HF Vibration (g) | 0.36 | 2.09 | **+488%** |
| Bearing Temp (°C) | 65.0 | 93.8 | **+44%** |
| Motor Current (A) | 17.3 | 28.6 | **+66%** |
| Lube Pressure (bar) | 3.69 | 2.05 | **−44%** |
| Lube Flow (L/min) | 35.0 | 24.0 | **−31%** |
| Power Factor (%) | 95.4 | 86.0 | **−10%** |

Voltage and speed are effectively constant across status groups — they load onto PC2 and PC3, not onto the mechanical degradation axis PC1.

---

## 🤖 Model

### Why PCA — not DBSCAN, not a classifier?

DBSCAN (Project 16) answers: *does this observation belong to the normal population?* The answer is binary. It works well when the process has a well-defined density boundary and anomalies appear as isolated points.

PCA answers a different question: *where is this motor on the degradation continuum, and how far has it traveled from healthy?* The output is not a binary label — it is a coordinate in a health space, a reconstruction error, and a Mahalanobis distance. Together they form a continuous health index that can track gradual drift, identify the dominant failure mode through loadings, and trigger tiered alerts at calibrated thresholds.

The right tool for predictive maintenance is the one that provides **early warning before the anomaly is clear** — and that requires a continuous score, not a boundary crossing.

### Variance Decomposition: The Scree Analysis

Running PCA on all 10 components reveals the variance structure:

| Components | Variance Explained | Interpretation |
|------------|-------------------|----------------|
| PC1 alone | **54.4%** | The mechanical degradation axis dominates |
| 3 PCs | **74.0%** | Sufficient for health space visualization and IR |
| 6 PCs | **91.6%** | Used for SPE reconstruction error computation |

PC1 capturing 54.4% in a single component is not a coincidence — it reflects the strong multi-collinearity of the degradation cascade. Vibration, temperature, current, and lubrication all move together because they share the same root cause. PCA found that cause and named it PC1.

### Physical Interpretation of Principal Components

| Component | Variance | Physical Interpretation | Key Loadings |
|-----------|----------|------------------------|--------------|
| **PC1** | 54.4% | **Mechanical Degradation Axis** | vibration_hf (+0.401), vibration_rms (+0.394), bearing_temp (+0.388), current (+0.362) · lube_pressure (−0.360), power_factor (−0.334), lube_flow (−0.328) |
| **PC2** | 10.0% | **Electrical Supply Axis** | voltage_v (+0.977) |
| **PC3** | 9.6% | **Speed Axis** | speed_rpm (+0.932) |

PC1 is the model. High PC1 score means: elevated vibration and temperature, reduced lubrication, degraded power factor — the mechanical failure cascade written in one number.

### The Reliability Index (IR)

Two complementary metrics quantify how abnormal a motor snapshot is:

**SPE — Squared Prediction Error:** how poorly the 3-PC model can reconstruct the original 10-sensor reading. A high SPE means the observation contains patterns outside the normal operating manifold — the motor is doing something the model has not seen before.

**Hotelling T²:** the Mahalanobis distance from the origin in PC space. A high T² means the motor is far from its normal operating position in the health map.

Both are normalized and combined:

```
IR = 1 / (1 + SPE_norm + T²_norm)   ∈ (0, 1]
```

| IR Range | Status | Mean by Known State |
|----------|--------|---------------------|
| ≥ 0.86 | 🟢 Normal | Normal motors: mean IR = 0.838 |
| 0.74 – 0.86 | 🟡 Alert | Alert motors: mean IR = 0.750 |
| 0.62 – 0.74 | 🟠 Severe | — |
| < 0.62 | 🔴 Critical | Critical motors: mean IR = 0.587 |

The IR thresholds are not arbitrary — they were calibrated to the known status distribution. Alert motors average IR = 0.750, which falls in the 0.74–0.86 band. Critical motors average IR = 0.587, which falls below 0.62. The index aligns with ground truth without ever using that ground truth in its computation.

---

## 📈 Key Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **PC1 variance** | 54.4% | Single dominant axis — the mechanical degradation continuum |
| **3-PC total variance** | 74.0% | Health space captures 3/4 of all sensor information |
| **SPE UCL (95th pct.)** | 6.55 | Reconstruction error upper control limit |
| **T² UCL (95th pct.)** | 9.08 | Mahalanobis distance upper control limit |
| **IR · Normal motors** | 0.837 ± 0.069 | High and compact — normal operating band is well-defined |
| **IR · Alert motors** | 0.751 ± 0.088 | Centered in the Alert band — threshold alignment confirmed |
| **IR · Critical motors** | 0.590 ± 0.088 | Below 0.62 threshold — critical classification confirmed |

**IR Classification Distribution (full dataset):**

| IR Category | Count | Share |
|-------------|-------|-------|
| Normal (≥ 0.86) | 471 | 27.9% |
| Alert (0.74–0.86) | 740 | 43.9% |
| Severe (0.62–0.74) | 316 | 18.7% |
| Critical (< 0.62) | 160 | 9.5% |

---

## 🔍 The Health Map

The PC1–PC2 projection of 1,687 motor snapshots reveals the structure of motor degradation visually. Normal motors (green) cluster near the origin — low PC1, low reconstruction error. Alert motors (amber) drift rightward along PC1. Critical motors (red) occupy the high-PC1 region, pulled by the degradation cascade: high vibration, high temperature, low lubrication, elevated current.

PC2 (voltage) and PC3 (speed) are orthogonal to degradation — they reflect supply and load conditions that vary independently. A motor can have low PC2 (supply voltage issue) without being mechanically degraded, and vice versa. The three-axis health space captures both dimensions of the problem.

---

## 🗂️ Repository Structure

```
PCA_Predictive_Maintenance/
├── 17_PCA_Predictive_Maintenance.ipynb   # Full educational notebook
├── mtto_data.csv                          # 250-row sample dataset
├── requirements.txt
└── README.md
```

> 📦 **Full Project Pack** — complete dataset (1,687 records), notebook with full outputs,
> presentation deck (PPTX + PDF), and `app.py` simulator available on
> [Gumroad](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

**Option 1 — Google Colab (no installation):**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/PCA_Predictive_Maintenance/blob/main/17_PCA_Predictive_Maintenance.ipynb)

**Option 2 — Local:**

```bash
git clone https://github.com/LozanoLsa/PCA_Predictive_Maintenance.git
cd PCA_Predictive_Maintenance
pip install -r requirements.txt
jupyter notebook 17_PCA_Predictive_Maintenance.ipynb
```

**Option 3 — Streamlit Simulator:**

```bash
streamlit run app.py
```

---

## 💡 Key Learnings

1. **PC1 is not a statistic — it is a physical axis.** When the loadings of vibration, temperature, current, and lubrication all converge into a single component that explains 55% of variance, that component is not an artifact of the math. It is the degradation cascade that every motor engineer already knows exists. PCA gave it coordinates.

2. **The correlation structure is the signal.** Individual sensors rising or falling can have many explanations. When six sensors move in a correlated, physically consistent direction simultaneously, there is one explanation: the mechanical state of the motor is changing. PCA extracts that joint movement in a single score.

3. **SPE and T² are complementary, not redundant.** SPE flags observations that are strange in a way the model cannot reconstruct — unexpected failure modes, sensor faults, operating conditions outside the training range. T² flags observations that are far from normal within the known degradation space. A motor approaching end-of-life will show high T². A motor with a novel failure mode may show high SPE first. Running both in parallel provides full coverage.

4. **The IR threshold is a calibration decision, not a discovery.** The thresholds (0.86, 0.74, 0.62) were selected by examining the IR distribution against known status labels. In a real deployment without pre-labeled data, the thresholds would be set by process knowledge — what False Alarm Rate is acceptable, what inspection cost justifies an Alert trigger. The model provides the score; the engineer sets the threshold.

5. **Dimensionality reduction is not information loss — it is information extraction.** Moving from 10 sensors to 3 PCs does not discard 74.0% of the data — it discards 26.0% of variance, most of which is noise and orthogonal variation (voltage supply fluctuations, load changes) that carries no degradation signal. The signal became cleaner, not weaker.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
