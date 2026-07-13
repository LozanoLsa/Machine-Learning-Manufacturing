# 16 · DBSCAN — Anomaly Detection in CNC Machining Operations

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/DBSCAN_Anomaly_CNC_Detection/blob/main/16_DBSCAN_Anomaly_CNC_Detection.ipynb)

> *"K-Means looks for what you expect. DBSCAN finds what's actually happening — and flags what doesn't belong."*

---

## 🎯 The Wrong Question

Most anomaly detection systems ask the wrong question.

They ask: *"Is this reading above the limit?"* And when the answer is no — when vibration is at 4.1 mm/s against a 5.0 limit, temperature at 52°C against a 65° cutoff, dimensional deviation at 10 μm against a 15 μm tolerance — the system concludes that everything is fine. No alarm. No work order. No intervention.

Meanwhile, the tool is 40 hours from catastrophic failure.

The right question is not whether any individual reading crossed a line someone drew years ago. The right question is: **does this machining cycle belong to the normal operating population, or not?** That is a question about geometry, not thresholds. It requires knowing the shape of normality in the full multivariate space — and then asking whether a new observation falls inside that shape or outside it.

That is precisely what DBSCAN does, and it is why density-based anomaly detection is fundamentally different from every alarm system currently running on most factory floors.

This project applies DBSCAN to **1,724 CNC machining cycles** from a precision manufacturing cell, using three process signals: vibration amplitude, spindle temperature, and dimensional deviation from nominal. The algorithm maps the dense core of normal operation, then flags **149 anomalous cycles (8.6%)** — a tool-wear signature that sits below every individual threshold but is unmistakable in the three-dimensional process space.

> 📦 **Full Project Pack** — complete dataset, notebook with full outputs,
> presentation deck (PPTX + PDF), and `app.py` simulator available on
> [Gumroad](https://lozanolsa.gumroad.com).

---

## 📊 Dataset

**File:** `CNC_data.csv` — 1,724 machining cycle records from a CNC production cell process historian.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `vibration_mm_s` | float | mm/s RMS | Vibration amplitude at spindle housing — primary tool-wear indicator |
| `spindle_temp_c` | float | °C | Spindle bearing temperature — rises with friction as lubrication degrades |
| `dim_deviation_um` | float | μm | Dimensional deviation of finished part from nominal — quality output signal |

> No target variable. DBSCAN is fully unsupervised — it discovers the normal operating envelope from the data itself, without being told which cycles are anomalous.

**Data Origin — Real-World Source Systems:**

| Feature | Source System | Instrument |
|---------|--------------|------------|
| `vibration_mm_s` | SCADA / CNC controller | Piezoelectric accelerometer on spindle housing, RMS over cycle duration |
| `spindle_temp_c` | DCS / Machine tool controller | RTD or thermocouple at spindle bearing — logged at end-of-cycle |
| `dim_deviation_um` | SPC / CMM or in-line gauge | Coordinate Measuring Machine or air-gauge post-machining inspection |

**Key EDA Findings:**

- The normal operating cloud is compact and well-defined: vibration 2.68 mm/s, spindle temp 45.4°C, dimensional deviation 8.12 μm. The K-distance plot reveals a sharp elbow at eps = 0.45, confirming that the boundary between normal and anomalous is geometrically clear in scaled space.
- The 149 anomalous cycles show a consistent multivariate signature: vibration elevated by 42% (3.80 vs 2.68 mm/s), temperature elevated by 16% (52.7 vs 45.4°C), and dimensional deviation elevated by 39% (11.32 vs 8.12 μm). No single variable alone exceeds its individual control limit across all 149 anomalous cycles.
- PCA projects 86.6% of variance into two components, confirming that the three process signals carry a coherent signal — the anomaly cluster is visible and separated in the PCA projection.

---

## 🤖 Model

### Why DBSCAN, not a threshold and not K-Means?

K-Means (Project 15) answered the question: *how many operating modes does this process have?* It needed to be told k and it found the centroids of each mode.

DBSCAN answers a different question: *what is the shape of the normal operating region, and what falls outside it?* It requires no prior knowledge of how many clusters exist, and it produces no centroid to interpret — only a boundary. Points inside the dense core are normal. Points isolated from that core are noise. Noise is the anomaly.

This distinction matters operationally. In a CNC cell, the process engineer does not know in advance how many anomalous patterns exist — tool wear, spindle imbalance, coolant starvation, and vibration resonance all produce different signatures at different stages. DBSCAN does not need to know the names of those failure modes. It only needs to know where normal ends.

The threshold alternative fails for the same reason it always fails in multivariate processes: a vibration of 4.0 mm/s at 44°C spindle temperature is a very different signal from 4.0 mm/s at 56°C. The number means different things in different contexts. DBSCAN encodes that context automatically — the neighborhood radius `eps` operates in the full three-dimensional scaled space, not on each variable in isolation.

### Hyperparameter Selection

DBSCAN has two parameters: `eps` (neighborhood radius) and `min_samples` (minimum neighbors to form a core point).

The K-distance plot — computed for k = 4 (= `min_samples − 1`) — reveals the elbow at `eps = 0.45` in standardized units. Below this value, points are in the dense core; above it, points are isolated from any core neighborhood. Sensitivity analysis confirms the selection is stable: at eps = 0.40 the anomaly rate rises to 10.2%, at eps = 0.50 it drops to 6.6%, bracketing the 8.6% result that aligns with domain expectations for a well-maintained precision machining cell.

`min_samples = 5` means a point needs at least 5 neighbors within eps to become a core point — sufficient to distinguish genuine process clusters from random scatter in a 1,724-record dataset.

### Preprocessing

StandardScaler brings all three variables to zero mean and unit variance before distance computation. Without scaling, dimensional deviation in micrometers would dominate Euclidean distances over temperature in degrees Celsius, and the density map would be distorted accordingly. The scaler is fitted on historical data and applied to any new observation at inference time.

No train/test split — DBSCAN maps the density landscape of the full historical window. For real-time deployment, new machining cycles are scaled with the fitted scaler and their distance to core samples is computed; cycles that fall outside all core neighborhoods receive label −1.

---

## 📈 Key Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **eps** | 0.45 | Neighborhood radius in standardized units — set at K-distance plot elbow |
| **min_samples** | 5 | Minimum neighbors for core point — density threshold |
| **Normal cycles (C0)** | 1,570 (91.1%) | Dense core — within normal operating envelope |
| **Borderline (C1)** | 5 (0.3%) | Micro-cluster at periphery of normal region |
| **Anomalies (−1)** | 149 (8.6%) | Isolated from all core neighborhoods — tool-wear signature |
| **Core points** | 1,513 | Points with ≥ 5 neighbors within eps — the backbone of the normal region |
| **PCA variance (2D)** | 86.6% | Three signals carry coherent structure — anomaly separation visible in 2D |

**Anomaly vs. Normal Process Signatures:**

| Signal | Normal (C0) | Anomaly (−1) | Delta |
|--------|-------------|--------------|-------|
| Vibration (mm/s) | 2.68 | 3.80 | **+42%** |
| Spindle Temp (°C) | 45.4 | 52.7 | **+16%** |
| Dim. Deviation (μm) | 8.12 | 11.32 | **+39%** |

The anomaly signature is consistent: elevated vibration and temperature co-occur with degraded dimensional output. The correlation is not coincidental — a worn cutting tool generates more friction (heat), more mechanical instability (vibration), and less precise geometry (deviation). DBSCAN caught the pattern without being told any of this.

---

## 🔍 What the Anomalies Actually Are

DBSCAN does not name failure modes — it identifies points that do not belong. The process engineer's job is to close the loop by correlating the 120 flagged cycles with maintenance records, tool change logs, and production reports.

The multivariate signature of the anomalous cycles points toward a **progressive tool wear trajectory**: vibration climbing as the cutting edge degrades, spindle temperature rising as friction increases with a worn tool-workpiece interface, and dimensional deviation widening as the tool loses its ability to hold tolerance. This is a continuous drift, not a sudden event — which is exactly why threshold-based alarms miss it.

The 6 borderline cycles in the micro-cluster (C1) are particularly informative: they occupy the boundary between C0 and the anomalous region. In a real-time deployment, these cycles deserve a yellow flag rather than a red one — not an immediate stop, but a signal to inspect at the next scheduled break.

---

## 🗂️ Repository Structure

```
DBSCAN_Anomaly_CNC_Detection/
├── 16_DBSCAN_Anomaly_CNC_Detection.ipynb   # Full educational notebook
├── CNC_data.csv                             # 250-row sample dataset
├── requirements.txt
└── README.md
```

> 📦 **Full Project Pack** — complete dataset (1,724 records), notebook with full outputs,
> presentation deck (PPTX + PDF), and `app.py` simulator available on
> [Gumroad](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

**Option 1 — Google Colab (no installation):**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/DBSCAN_Anomaly_CNC_Detection/blob/main/16_DBSCAN_Anomaly_CNC_Detection.ipynb)

**Option 2 — Local:**

```bash
git clone https://github.com/LozanoLsa/DBSCAN_Anomaly_CNC_Detection.git
cd DBSCAN_Anomaly_CNC_Detection
pip install -r requirements.txt
jupyter notebook 16_DBSCAN_Anomaly_CNC_Detection.ipynb
```

**Option 3 — Streamlit Simulator:**

```bash
streamlit run app.py
```

---

## 💡 Key Learnings

1. **The right question changes the tool.** Threshold-based control asks whether a reading exceeds a limit. Density-based detection asks whether an observation belongs to the normal population. These are different questions with different answers — and the second one catches what the first one misses.

2. **The K-distance plot is not a formality.** Choosing `eps` from the elbow of the sorted K-distance plot is the mechanistic link between the data's geometry and the algorithm's sensitivity. Skip this step and `eps` becomes a guess. Run it and `eps` becomes an observation.

3. **Sensitivity analysis before deployment.** The anomaly rate at `eps = 0.45` is 8.6%. At `eps = 0.35` it rises to 13.1%. At `eps = 0.55` it drops to 5.0%. That range should be evaluated against domain knowledge — for a precision CNC cell, 8–9% aligns with expected tool-wear frequency. For a different process, the calibration point changes. Always validate the rate against what the process engineer considers operationally reasonable.

4. **DBSCAN produces a boundary, not a diagnosis.** The 120 flagged cycles are not labeled "tool wear" by the algorithm. They are labeled "not normal." Turning that into a maintenance action requires a human closing the loop — correlating with tool change records, measuring against tolerance history, and deciding what the pattern means in this specific process context.

5. **The borderline cluster is the early warning.** The 5 cycles in C1 (0.3%) did not cross into the anomalous region, but they are no longer in the dense core either. In a real-time system, these are the most valuable observations — they are the process drifting toward the boundary before it crosses it. Building an alert tier for borderline cases is often more operationally valuable than the anomaly label itself.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
