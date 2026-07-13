# 19 · Apriori — Association Rules in People Analytics

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/Apriori_HR_People/blob/main/19_Apriori_HR_People.ipynb)

> *"Correlation is what dashboards show you. Association rules show you what conditions tend to occur together — and what outcome tends to follow. That's the difference between a report and a policy."*

---

## 🎯 15.2% Leave. That's a Fact. The Policy Question Is: Which Profile?

The HR dashboard says: attrition rate = 15.2%.

The HR manager nods and writes in the quarterly report: *employee retention remains a challenge.* Action item: improve employee engagement. Owner: HR. Due date: end of quarter.

The problem is not the engagement program. The problem is that 15.2% is an average — and averages are the most effective way to hide the structure of a problem. The employees who leave are not a random 15.2% sample. They are a specific profile. Night shift. No recent training. No remote work access. Low manager rating. When those four conditions co-occur, the attrition rate is not 15.2%. It is 22.4% — **48% above the organizational baseline.**

That number is not in the dashboard. It is in the data, waiting for an algorithm that knows how to find it.

**Association rule mining** is the only unsupervised method in this portfolio that produces its output in the form of a policy. Not a cluster label. Not a health score. Not a root cause attribution. A conditional statement: *when these HR conditions co-occur in an employee's profile, this outcome is 1.46× more likely than baseline.* That conditional statement is directly actionable in a way that no descriptive statistic is.

This project applies the **Apriori algorithm** to **1,748 employee records** across 8 HR dimensions: department, shift, seniority, contract type, remote work, training status, overtime, and manager rating. From 3,861 frequent itemsets, it derives **2,148 HR-critical association rules** — organized by outcome (attrition, performance, absenteeism) and ranked by lift. The output is a **People Risk Analyzer** that takes any employee profile and surfaces the relevant rules: which risks apply, which protective factors are present, and which interventions have the strongest evidence behind them.

> 📦 **Full Project Pack** — complete dataset, notebook with full outputs,
> presentation deck (PPTX + PDF), and `app.py` simulator available on
> [Gumroad](https://lozanolsa.gumroad.com).

---

## 📊 Dataset

**File:** `people_data_analysis.csv` — 1,748 employee records across 8 HR dimensions and 3 outcome variables.

**HR Profile Variables (antecedents in rules):**

| Column | Values | Description |
|--------|--------|-------------|
| `department` | Operations, Quality, Logistics, Production, Admin | Organizational unit |
| `shift` | Day, Night | Work schedule |
| `seniority` | Junior, Mid, Senior | Experience level |
| `contract_type` | Permanent, Temporary | Employment type |
| `remote_work` | Yes, No | Remote work access |
| `training_status` | Completed, Partial, None | Last training cycle status |
| `overtime` | Yes, No | Regular overtime requirement |
| `manager_rating` | High, Medium, Low | Direct manager assessment score |

**Outcome Variables (consequents in rules):**

| Column | Distribution | Baseline Rate |
|--------|-------------|---------------|
| `attrition` | No: 1,483 (84.8%) · Yes: 265 (15.2%) | attrition=Yes: **15.2%** |
| `performance` | High: 510 (29.2%) · Medium: 873 (49.9%) · Low: 365 (20.9%) | performance=Low: **20.9%** |
| `absenteeism` | Low: 775 (44.3%) · Medium: 652 (37.3%) · High: 321 (18.4%) | absenteeism=High: **18.4%** |

> ⚠️ **Unlike supervised ML, all columns — including outcomes — are items.** Apriori discovers which HR profile conditions co-occur with which outcomes, without specifying a fixed dependent variable. Rules can run in both directions: profile → outcome and outcome → profile.

**Why Apriori handles this data naturally:**

Every other algorithm in this portfolio required StandardScaler — a transformation that assumes numerical features measured in compatible units. This dataset has no numerical features. It is entirely categorical — discrete HR conditions, each either present or absent in an employee record. Apriori was designed for exactly this structure: binary transaction matrices where each row is a basket (here, an employee profile) and each column is an item (here, `shift=Night`, `training_status=None`, `attrition=Yes`). No encoding, no scaling, no information lost to binning.

---

## 🤖 Model

### Why Apriori and not a classifier?

A classifier (Projects 01–06) would predict `attrition=Yes` or `attrition=No` for each employee. It would be trained on labeled data and produce a probability score. That is useful for individual risk prediction.

Apriori answers a different question at a different level: **which combinations of HR conditions systematically produce which outcomes?** The output is not a prediction for a single employee — it is a policy library for the organization. Rule 1 says: when `manager_rating=Low` AND `remote_work=No` co-occur, `absenteeism=High` has lift=1.665. That means the HR team should look at every employee matching that profile — not because the algorithm predicted one outcome, but because the combination has a structural, statistically validated association with the worst absenteeism outcome.

This is the difference between a model and a policy. Supervised classifiers produce individual predictions. Association rules produce organizational insights.

### Transaction Encoding and Thresholds

Each of the 1,748 employee records becomes a basket of 11 items (`department=Operations`, `shift=Night`, `training_status=None`, etc.). The `TransactionEncoder` converts these to a binary matrix of shape (1,748 × 30) — 30 unique items across all 11 columns.

**Threshold selection** was determined by sweeping `min_support` from 0.04 to 0.20:

| min_support | Itemsets | HR Rules | Max Lift |
|-------------|----------|----------|----------|
| 0.04 | 3,861 | 2,148 | **1.665** |
| 0.06 | 2,092 | 1,192 | 1.600 |
| 0.10 | 811 | 459 | 1.543 |
| 0.15 | 323 | 169 | 1.258 |

`min_support = 0.04` (appearing in at least 70 of 1,748 records) was selected: enough rules to cover all HR risk profiles, while excluding spurious combinations that appear in fewer than 4% of the workforce. `min_confidence = 0.60` filters for rules where the consequent occurs in at least 60% of antecedent cases.

### The Three Metrics That Matter

**Support** = how common is this pattern in the workforce? A rule with support=0.042 affects 73 employees out of 1,748. Useful to know for prioritizing interventions at scale.

**Confidence** = given this profile, how often does the outcome occur? A rule with confidence=0.306 on `absenteeism=High` means 30.6% of employees matching the antecedent show high absenteeism. Compare to the 18.4% baseline.

**Lift** = confidence / baseline. This is the primary ranking metric. A lift of 1.665 on `absenteeism=High` means employees matching this profile are 66.5% more likely to show high absenteeism than a randomly selected employee. Lift < 1 means the profile is **protective** — the outcome is less likely than baseline.

---

## 📈 Key Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Frequent itemsets** | 3,861 | Item combinations appearing in ≥ 4% of records |
| **Total rules** | 7,561 | Directional if-then rules with lift ≥ 1.0 |
| **HR-critical rules** | 2,148 | Rules with outcome as consequent |
| **Lift range** | 1.000 – 1.665 | Max: absenteeism=High rules |
| **Confidence range** | 0.103 – 1.000 | Max: 100% confidence on attrition=No retention profiles |
| **Support range** | 0.040 – 0.607 | Min threshold to max common item |

**HR-Critical Rules by Outcome:**

| Outcome | Rules Found | Baseline Rate |
|---------|-------------|---------------|
| `attrition=Yes` | 21 | 15.2% |
| `attrition=No` | 793 | 84.8% |
| `performance=High` | 193 | 29.2% |
| `performance=Low` | 55 | 20.9% |
| `absenteeism=High` | 67 | 18.4% |
| `absenteeism=Low` | 329 | 44.3% |

---

## 🔍 Top Rules by Outcome Category

### 🔴 Attrition Risk (attrition=Yes · baseline 15.2%)

**Rule 1:** `training_status=None` → `attrition=Yes`
Support: 0.045 · Confidence: 22.4% · **Lift: 1.480**
Employees with no training in the last cycle are 48% more likely to leave than baseline. Affects ~79 employees. This is the single highest-impact individual HR variable for attrition.

**Rule 2:** `contract_type=Temporary + remote_work=No` → `attrition=Yes`
Support: 0.043 · Confidence: 21.3% · **Lift: 1.405**
Temporary-contract employees without remote work access show 40% elevated attrition risk. The combination is the signal — either condition alone has a weaker effect.

**Rule 3:** `overtime=Yes + remote_work=No` → `attrition=Yes`
Support: 0.055 · Confidence: 21.0% · **Lift: 1.383**
Mandatory overtime without flexibility is a consistent attrition driver independent of shift or seniority.

### ✅ Retention Profile (attrition=No · confidence up to 100%)

**Rule 1:** `manager_rating=High + remote_work=Yes + training_status=Completed` → `attrition=No`
Support: 0.054 · Confidence: **100.0%** · Lift: 1.179
No employee matching this three-condition profile left the organization in this dataset. The three interventions that reliably retain employees are all organizational — manager quality, flexibility, and development investment.

### ⚡ High Absenteeism (absenteeism=High · baseline 18.4% · max lift 1.665)

**Rule 1:** `manager_rating=Low + remote_work=No` → `absenteeism=High`
Support: 0.042 · Confidence: 30.6% · **Lift: 1.665** ← highest lift in the full ruleset
Poor management combined with no flexibility produces the strongest HR risk signal in the dataset — 66% above baseline absenteeism.

**Rule 2:** `performance=Medium + remote_work=No + seniority=Mid` → `absenteeism=High`
Support: 0.040 · Confidence: 26.8% · **Lift: 1.460**

### 🏆 High Performance (performance=High · baseline 29.2%)

**Rule 1:** `overtime=No + remote_work=Yes + training_status=Completed` → `performance=High`
Support: 0.044 · Confidence: 43.8% · **Lift: 1.500**
The highest-performance profile is not the one working most hours. It is the one with flexibility, completed development, and no overtime load.

---

## 🗂️ Repository Structure

```
Apriori_HR_People/
├── 19_Apriori_HR_People.ipynb    # Full educational notebook
├── people_data_analysis.csv       # 250-row sample dataset
├── requirements.txt
└── README.md
```

> 📦 **Full Project Pack** — complete dataset (1,748 records), notebook with full outputs,
> presentation deck (PPTX + PDF), and `app.py` simulator available on
> [Gumroad](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

**Option 1 — Google Colab (no installation):**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/Apriori_HR_People/blob/main/19_Apriori_HR_People.ipynb)

**Option 2 — Local:**

```bash
git clone https://github.com/LozanoLsa/Apriori_HR_People.git
cd Apriori_HR_People
pip install -r requirements.txt
jupyter notebook 19_Apriori_HR_People.ipynb
```

**Option 3 — Streamlit Simulator:**

```bash
streamlit run app.py
```

---

## 💡 Key Learnings

1. **Lift is the operative metric — support and confidence are supporting evidence.** A rule with support=0.042 and lift=1.665 is more actionable than a rule with support=0.400 and lift=1.050. The first affects 73 employees with a 66% above-baseline risk. The second is a trivially common pattern with almost no predictive value. Always sort by lift, filter by support for scale.

2. **Combinations produce non-linear risk amplification.** `training_status=None` alone has lift=1.480 on attrition. `remote_work=No` alone has a smaller individual effect. Together with `manager_rating=Low`: lift rises to 1.665 on absenteeism — substantially above either variable in isolation. The combination is not additive — it is multiplicative. This is exactly what univariate HR reporting cannot detect.

3. **High confidence does not imply high lift — and vice versa.** The `attrition=No` retention rules have confidence up to 100% but lift only up to 1.179 — because 84.8% of employees already don't leave. The `absenteeism=High` rules have confidence of 30.6% but lift of 1.665 — because the baseline is only 18.4%. Both are informative, but for different reasons. Understanding the baseline is mandatory for interpreting either metric.

4. **The most powerful retention profile is entirely policy-driven.** The rule with 100% confidence on `attrition=No` requires: high manager rating, remote work access, completed training. None of those are employee characteristics. They are organizational decisions. Association rules on HR data have a tendency to produce conclusions that are inconvenient for managers — the risk lives in the policy, not the person.

5. **Protective rules (lift < 1) are as valuable as risk rules (lift > 1).** A rule where `remote_work=Yes + training_status=Completed` reduces `attrition=Yes` probability to below baseline is not just an absence of risk — it is an evidence-based retention investment. The intervention with the strongest evidence is the one that appears most consistently on the protective side of the lift distribution.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
