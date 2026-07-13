# Industrial ML Portfolio — 24 Projects Across 9 Algorithm Families

> *"In manufacturing, every process has a Y and a set of Xs. We've been solving that problem with DOE for decades. Machine learning is DOE with a hundred factors — same question, larger scale, faster answer."*

---

## Why This Portfolio Exists

Most ML courses teach on clean, generic datasets — iris flowers, housing prices, titanic survivors. This portfolio teaches on **manufacturing problems**: riveting presses, quench furnaces, laser welds, hot stamping cells, chemical reactors.

Every dataset here started as a real operational question: *Why does the head diameter go out of spec mid-shift? Which employees are about to resign? Is this weld sound or not?* The data was engineered to reflect what actually lives in PLCs, SCADA systems, MES platforms, and CMM exports — not what lives in Kaggle.

The goal is not to show that ML works. The goal is to show **when** each algorithm earns its complexity, and **what it tells an engineer that a control chart can't**.

---

## Who This Is For

### Lean Six Sigma Engineers Exploring ML
You already know DOE. You know what R², residuals, and process capability mean. You've run a gauge R&R and you understand why measurement error matters. This portfolio speaks your language. Each project starts with the engineering problem, not the algorithm. The math is explained in terms you already know — coefficients as process levers, clusters as process families, anomaly scores as out-of-control signals.

**You don't need to become a data scientist. You need to know when to call one — and what to ask for.**

### Data Scientists Learning Manufacturing
You know Python, scikit-learn, cross-validation, and hyperparameter tuning. What you may not know is what a *quench furnace* is, why *transfer time* matters in hot stamping, or what an engineer means when they say a rivet head "went NG." This portfolio closes that gap. The domain context is deliberate — because a model that's technically correct but operationally nonsensical is useless on a production floor.

**Your feedback is the most valuable in this portfolio. If the engineering narrative doesn't make sense to you, that's a gap worth closing.**

---

## One Portfolio. 24 Individual Repos. Two Ways to Navigate.

This repository exists to tell the full story — the arc from logistic regression on labeled defects to a model-based planner that simulates reactor futures. That story only makes sense when you see all 24 projects together: the learning path, the DOE analogy, the progression from supervised to unsupervised to reinforcement learning.

**But if you only want one project**, each one also lives as a standalone repository. Clone just what you need, without pulling the rest.

| # | Project | Algorithm | Domain | Standalone Repo |
|---|---|---|---|---|
| 01 | Delays Are Not Random | Logistic Regression | Logistics | [Delays_Are_Not_Random](https://github.com/LozanoLsa/Delays_Are_Not_Random) |
| 02 | Visual Defects Are Not Random | Naïve Bayes | Manufacturing QC | [Visual_Defects_Are_Not_Random](https://github.com/LozanoLsa/Visual_Defects_Are_Not_Random) |
| 03 | Motor Failure Prediction | KNN | Predictive Maintenance | [Motor_Failures_Prediction](https://github.com/LozanoLsa/Motor_Failures_Prediction) |
| 04 | HR Risk Analytics | SVM | Human Resources | [HR_Risk_SVM_Prediction](https://github.com/LozanoLsa/HR_Risk_SVM_Prediction) |
| 05 | Process Decisions Optimization | Decision Tree | Process Engineering | [Process_Decisions_Optimization](https://github.com/LozanoLsa/Process_Decisions_Optimization) |
| 06 | pH Adjustment Syrup | Random Forest | Food & Beverage | [PH_Adjustment_Syrup](https://github.com/LozanoLsa/PH_Adjustment_Syrup) |
| 07 | Rivet Head Dimension Control | Linear Regression | Automotive Assembly | [LR_Cycle_Time_Estimation](https://github.com/LozanoLsa/LR_Cycle_Time_Estimation) |
| 08 | Furnace Cycle Time Optimization | Multiple Regression | Heat Treatment | [MR_Multi_Factor_Process_Impact](https://github.com/LozanoLsa/MR_Multi_Factor_Process_Impact) |
| 09 | Weld Penetration Optimization | Lasso (L1) | Welding | [Lasso_KDrivers_Welding](https://github.com/LozanoLsa/Lasso_KDrivers_Welding) |
| 10 | Laser Cutting Surface Quality | Ridge (L2) | Laser Processing | [Ridge_Stable_Process_Modeling](https://github.com/LozanoLsa/Ridge_Stable_Process_Modeling) |
| 11 | Steel Strength Optimization | ElasticNet | Hot Stamping | [ElasticNet_Hot_Stamping](https://github.com/LozanoLsa/ElasticNet_Hot_Stamping) |
| 12 | Advanced Scrap Prediction | XGBoost | Process Quality | [XGBoost_Advanced_Scrap_Prediction](https://github.com/LozanoLsa/XGBoost_Advanced_Scrap_Prediction) |
| 13 | Employee Attrition Risk | Self-Training | HR Semi-Supervised | [SelfT_HR_Turnover_Prediction](https://github.com/LozanoLsa/SelfT_HR_Turnover_Prediction) |
| 14 | Laser Weld Defect Detection | Co-Training | QC Semi-Supervised | [CoTraining_Welding](https://github.com/LozanoLsa/CoTraining_Welding) |
| 15 | Reaction Process Clustering | K-Means | Chemical Processing | [Kmeans_Runaway_Risk](https://github.com/LozanoLsa/Kmeans_Runaway_Risk) |
| 16 | CNC Anomaly Detection | DBSCAN | CNC Machining | [DBSCAN_Anomaly_CNC_Detection](https://github.com/LozanoLsa/DBSCAN_Anomaly_CNC_Detection) |
| 17 | Predictive Maintenance PCA | PCA | Equipment Health | [PCA_Predictive_Maintenance](https://github.com/LozanoLsa/PCA_Predictive_Maintenance) |
| 18 | Chemical Source Separation | ICA | Pharma / Chemical | [ICA_Pharma_Features](https://github.com/LozanoLsa/ICA_Pharma_Features) |
| 19 | Employee Pattern Mining | Apriori | HR Analytics | [Apriori_HR_People](https://github.com/LozanoLsa/Apriori_HR_People) |
| 20 | Statistical Anomaly Detection | Z-Score | Process Monitoring | [ZScore_Anomaly_Detection](https://github.com/LozanoLsa/ZScore_Anomaly_Detection) |
| 21 | Advanced Anomaly Detection | Isolation Forest | Press / Equipment | [IsoForest_Anomaly_Detection](https://github.com/LozanoLsa/IsoForest_Anomaly_Detection) |
| 22 | AGV Path Optimization | Q-Learning | Autonomous Vehicles | [QLearning_AGV](https://github.com/LozanoLsa/QLearning_AGV) |
| 23 | Assembly Line Policy Optimization | Policy Optimization | Production Lines | [PolicyOpt_Assembly](https://github.com/LozanoLsa/PolicyOpt_Assembly) |
| 24 | pH Reactor Control | Model-Based RL | Chemical Reactor | [ModelBased_pH](https://github.com/LozanoLsa/ModelBased_pH) |

---

## The Data Problem This Portfolio Solves

Every project includes a pre-built, ready-to-run CSV dataset. No scraping. No API keys. No data cleaning required.

| What you get | Why it matters |
|---|---|
| Pre-engineered CSV per project | In real operations, dataset integration is 80% of the work — PLCs, CMM exports, and ERP certs must be joined before a single model trains. This portfolio skips that barrier so you can focus on the algorithm. |
| Consistent file structure | notebook + app.py + CSV + data guide + README + requirements + PDF — same layout across all 24 projects |
| Process-realistic feature sets | Features reflect what sensors actually log, what operators can control, and what measurement systems can measure |
| Domain context in every README | You know what the Y is, who measures it, why it drifts, and what fixing it is worth |

---

## What Is Inside Each Project Folder

```
XX_Algorithm_Domain/
├── XX_AlgorithmName.ipynb          ← Main notebook (10–12 sections, cleared outputs)
├── app.py                           ← Streamlit simulator — run the model without coding
├── dataset_name.csv                 ← Complete dataset, ready to load
├── data_sources_and_features.txt    ← Feature dictionary: source system, units, notes
├── Presentation_Name.pdf            ← Slide deck: problem → model → results → action
├── README.md                        ← Project README with key results and context
├── requirements.txt                 ← Python dependencies for this project
└── cover.png                        ← Project cover image
```

---

## The 9-Phase Learning Path

The folders are numbered 01–24 but the learning path is not strictly sequential by number. Navigate by algorithm family:

| Phase | Family | Folders | What You Learn |
|---|---|---|---|
| **1** | Supervised Classifiers | 01, 02, 03, 04, 05, 06 | Logistic Regression, Naïve Bayes, KNN, SVM, Decision Tree, Random Forest |
| **2** | Supervised Regression | 07, 08 | Linear Regression, Multiple Regression |
| **3** | Regularised Regression | 09, 10, 11 | Lasso (L1), Ridge (L2), ElasticNet (L1+L2) |
| **4** | Boosting | 12 | XGBoost — when ensemble trees beat single trees |
| **5** | Semi-Supervised | 13, 14 | Self-Training, Co-Training — learning with few labels |
| **6** | Unsupervised Clustering | 15, 16 | K-Means, DBSCAN — grouping without labels |
| **7** | Dimensionality Reduction | 17, 18 | PCA, ICA — finding structure in high-dimensional data |
| **8** | Association Rules | 19 | Apriori — market basket analysis for process defects |
| **9** | Anomaly Detection | 20, 21 | Z-Score, Isolation Forest — catching what doesn't belong |
| **10** | Reinforcement Learning | 22, 23, 24 | Q-Learning, Policy Optimization, Model-Based RL |

**Suggested starting points:**
- New to ML? Start at **Phase 1** (folder 01)
- Already know supervised learning? Jump to **Phase 5** (folder 13)
- Process engineer focused on monitoring? Go to **Phase 9** (folder 20)
- Want to see the full arc? **Phase 10** (folder 24) shows where all 24 projects point

---

## ML as DOE — The Mental Model

If you've run a Design of Experiments, you already understand machine learning at its core:

| DOE | Machine Learning |
|---|---|
| Identify the Y | Define the target variable |
| Select the Xs | Feature selection / engineering |
| Design the experiment | Collect and structure the dataset |
| Fit the model | Train the algorithm |
| Analyse main effects and interactions | Read coefficients, feature importances, SHAP values |
| Predict response at new factor levels | Inference on new observations |
| Validate residuals | Check model assumptions |

The difference: DOE works with 4–8 carefully designed factors at 2–4 levels. ML works with 10–100 factors at continuous levels, learning the response surface from observational data instead of designed experiments.

**Same question. Same rigour. Larger scale.**

---

## How to Download and Run

### Option 1 — Clone the entire portfolio (recommended)

```bash
# 1. Clone this repository
git clone https://github.com/LozanoLsa/Machine-Learning-Manufacturing
cd Machine-Learning-Manufacturing

# 2. Navigate to the project you want to run
cd 01_Logistic_Regression_SUP

# 3. Create a virtual environment (recommended)
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the notebook
jupyter notebook

# 6. Or run the interactive app (no coding needed)
streamlit run app.py
```

### Option 2 — Clone a single project repo

Each project has its own standalone repository (see the table above). Clone only what you need:

```bash
git clone https://github.com/LozanoLsa/Delays_Are_Not_Random
cd Delays_Are_Not_Random
pip install -r requirements.txt
streamlit run app.py
```

### Option 3 — Run in Google Colab (no installation needed)

Each notebook has a "Open in Colab" badge at the top. Click it, upload the CSV when prompted, and run all cells.

### First-time setup guide (no terminal experience)

1. Install [Python 3.10+](https://www.python.org/downloads/) — check "Add to PATH" during install
2. Open **Command Prompt** (Windows) or **Terminal** (Mac)
3. Type `pip install jupyter streamlit` and press Enter
4. Navigate to the project folder: `cd path\to\01_Logistic_Regression_SUP`
5. Run `jupyter notebook` — your browser will open automatically
6. For the simulator: run `streamlit run app.py` instead

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| pandas, numpy | Data manipulation |
| scikit-learn | Most algorithm implementations |
| XGBoost | Project 12 |
| statsmodels | OLS regression diagnostics (Projects 07, 08) |
| matplotlib, seaborn, plotly | Visualisation |
| Streamlit | Interactive app.py simulators |
| mlxtend | Apriori association rules (Project 19) |

---

## Clone It. Fork It. Improve It.

Found a dataset that fits better? A better hyperparameter set? A domain error in the engineering narrative? Open a pull request. The most valuable contributions are the ones that make the manufacturing context more accurate — better than anyone else, process engineers and production data scientists know what the real data looks like.

---

## Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning  
GitHub: [LozanoLsa](https://github.com/LozanoLsa)

For more ML tutorials, algorithm deep-dives, and manufacturing use cases:  
**Website: [lozanolsa.github.io](https://lozanolsa.github.io/)**

---

*Turning Operations into Predictive Systems — one dataset at a time.*
