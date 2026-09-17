# Hospital Readmission Prediction

An academic Machine Learning case study predicting 30-day unplanned hospital readmission risk for diabetic inpatients using electronic health records (EHR) and Logistic Regression with L2 Regularization.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Objective](#objective)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Machine Learning Approach](#machine-learning-approach)
- [Data Preprocessing](#data-preprocessing)
- [Logistic Regression with L2 Regularization](#logistic-regression-with-l2-regularization)
- [Evaluation Metrics](#evaluation-metrics)
- [False Positive vs False Negative Clinical Considerations](#false-positive-vs-false-negative-clinical-considerations)
- [Results](#results)
- [Limitations](#limitations)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)

---

## Problem Statement

Unplanned hospital readmissions within 30 days of discharge are a major indicator of healthcare quality, care transition efficacy, and operational efficiency:
* **Patient Well-Being:** Readmissions frequently stem from premature discharge, incomplete disease resolution, adverse medication reactions, or inadequate outpatient transition coordination.
* **Financial & Regulatory Penalties:** Under regulatory frameworks such as the U.S. Centers for Medicare & Medicaid Services (CMS) **Hospital Readmissions Reduction Program (HRRP)**, hospitals face financial penalties if their 30-day readmission rates exceed expected benchmarks.
* **Healthcare Capacity:** High readmission rates cause overcrowding in inpatient units and emergency departments, straining nursing staff and clinical resources.

---

## Objective

Develop an end-to-end, interpretable machine learning solution to:
1. Predict whether a hospitalized diabetic patient will experience an **unplanned readmission within 30 days** of discharge ($Y=1$) or not ($Y=0$).
2. Utilize **Logistic Regression with L2 Regularization** ($C=1.0$), strictly adhering to academic case study requirements.
3. Evaluate model discrimination using **ROC-AUC** as the primary performance metric.
4. Analyze the asymmetric clinical and economic trade-offs of **False Negatives vs. False Positives** across different decision thresholds.
5. Recover and interpret regression coefficients to understand feature associations with readmission log-odds.

---

## Dataset

* **Source:** [UCI Machine Learning Repository — Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
* **Citation:** Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*. BioMed Research International, 2014.
* **Original Dimensions:** 101,766 clinical encounters across 71,518 unique diabetic patients, with 50 features.
* **Key Features:**
  * **Demographics:** `race`, `gender`, `age` (10-year brackets).
  * **Admission & Stay Characteristics:** `admission_type_id`, `discharge_disposition_id`, `admission_source_id`, `time_in_hospital` (days).
  * **Prior Utilization (Visits):** `number_outpatient`, `number_emergency`, `number_inpatient` in the preceding year.
  * **Clinical Tests & Vitals:** `num_lab_procedures`, `num_procedures`, `num_medications`, `number_diagnoses`, `max_glu_serum`, `A1Cresult`.
  * **Diagnoses:** `diag_1`, `diag_2`, `diag_3` (primary, secondary, and tertiary ICD-9 codes).
  * **Medications:** 24 diabetes medications (e.g., `insulin`, `metformin`, `glipizide`) tracking dosage changes.
* **Target Transformation:**
  The raw target column `readmitted` contains `<30`, `>30`, and `NO`. As required:
  * `<30` $\rightarrow$ **1** (Readmitted within 30 days — positive class, ~8.98% prevalence).
  * `>30` and `NO` $\rightarrow$ **0** (Not readmitted within 30 days — negative class, ~91.02%).

---

## Technologies Used

* **Python 3.10+ / 3.13**
* **scikit-learn** (Pipeline, ColumnTransformer, LogisticRegression, StandardScaler, OneHotEncoder, metrics)
* **pandas** (Data manipulation and cleaning)
* **numpy** (Numerical computing)
* **matplotlib & seaborn** (Clinical data visualization)
* **Jupyter Notebook & nbconvert** (Interactive data exploration and reporting)

---

## Machine Learning Approach

```
Raw Patient Records (101,766 encounters)
    │
    ▼
Data Cleaning & Exclusion
  ├─ Remove deceased / hospice patients (discharge_disposition_id in {11, 13, 14, 19, 20, 21})
  ├─ Deduplicate to first (index) admission per patient (patient_nbr) -> 69,987 unique patient encounters
  ├─ Drop non-predictive identifiers (encounter_id, patient_nbr)
  ├─ Drop high-missing features (>95% missing: weight; administrative: payer_code)
  └─ Map ICD-9 codes (diag_1, diag_2, diag_3) into 9 clinical categories
    │
    ▼
Stratified Train/Test Split (80% Train : 55,989 | 20% Test : 13,998)
    │
    ▼
scikit-learn Pipeline (Zero Data Leakage)
  ├─ Numerical: StandardScaler()
  ├─ Categorical: OneHotEncoder(handle_unknown='ignore', sparse_output=False)
  └─ Classifier: LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
    │
    ▼
Evaluation & Clinical Trade-Off Analysis
  ├─ ROC-AUC (0.6474) & ROC Curve
  ├─ Confusion Matrix & Classification Report
  ├─ Sensitivity vs. Specificity across decision thresholds
  └─ Coefficient Interpretation (Odds Ratios)
```

---

## Data Preprocessing

1. **Ineligible Patient Filtering:** 2,423 encounters where patients expired or were discharged to hospice care were excluded because they are clinically ineligible for readmission.
2. **Mitigating Patient Clustering & Leakage:** Retaining only the first (index) admission per patient eliminates repeat-encounter correlation for the same individual and strictly prevents the same patient from appearing in both training and test splits (which would cause data leakage). While this eliminates patient-level repeated measures, it does not guarantee that observational EHR records are strictly independent and identically distributed (i.i.d.), as hospital-level admission policies, regional practice variations, and secular trends across the 10-year study window remain.
3. **ICD-9 Clinical Grouping:** Diagnosis codes were mapped into standard clinical categories:
   * Circulatory (390–459, 785)
   * Respiratory (460–519, 786)
   * Digestive (520–579, 787)
   * Diabetes (250.xx)
   * Injury (800–999)
   * Musculoskeletal (710–739)
   * Genitourinary (580–629, 788)
   * Neoplasms (140–239)
   * Other
4. **Leakage Prevention:** `ColumnTransformer` fits scalers and encoders **exclusively** on training data within the `Pipeline`.

---

## Logistic Regression with L2 Regularization

The case study explicitly mandates Logistic Regression with L2 regularization:

$$\ln \left( \frac{p}{1 - p} \right) = \mathbf{w}^T \mathbf{x} + b$$

$$P(Y=1 \mid \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

### L2 Regularization (Ridge Penalty)
Minimizes the regularized negative log-likelihood:

$$J(\mathbf{w}, b) = - \frac{1}{N} \sum_{i=1}^N \left[ y_i \ln p_i + (1 - y_i) \ln (1 - p_i) \right] + \frac{1}{2C} \|\mathbf{w}\|_2^2$$

* **$\|\mathbf{w}\|_2^2 = \sum_{j=1}^D w_j^2$:** Shrinks regression coefficients toward zero, preventing individual features or one-hot sparse levels from dominating predictions.
* **$C = 1.0$:** The inverse regularization parameter ($C = 1/\lambda$). Sets a balanced trade-off between bias and variance.
* **Solver:** `lbfgs` with `max_iter=1000` to guarantee convergence.

---

## Evaluation Metrics

* **ROC-AUC (Primary Metric):** Evaluates the model's ability to rank readmitted patients above non-readmitted patients across all potential classification thresholds. Computed directly from `predict_proba()`.
* **Accuracy:** Proportion of correct predictions. (Deceptive in imbalanced settings).
* **Precision (PPV):** Of patients flagged by the model as high-risk, what fraction was actually readmitted?
* **Recall / Sensitivity:** Of patients who actually suffered readmission, what fraction did the model successfully identify?
* **F1-Score:** Harmonic mean of precision and recall.
* **Confusion Matrix:** True Negatives (TN), False Positives (FP), False Negatives (FN), True Positives (TP).

---

## False Positive vs False Negative Clinical Considerations

The cost of classification errors in clinical healthcare is asymmetric:

### 1. False Negative (FN)
* **Clinical Scenario:** A patient who will be readmitted within 30 days is predicted as low risk ($\hat{Y}=0$) and discharged routinely.
* **Impact:** 
  * Missed opportunity for high-touch post-discharge interventions (e.g., transition nurse calls, home health visits, medication reconciliation).
  * Disease complications may escalate unmonitored at home, leading to emergency department presentation, ICU admission, or preventable mortality.
  * The hospital faces regulatory financial penalties under HRRP.

### 2. False Positive (FP)
* **Clinical Scenario:** A patient who will not be readmitted within 30 days is predicted as high risk ($\hat{Y}=1$).
* **Impact:**
  * Hospital resources are deployed unnecessarily (care coordinator outreach, post-discharge calls).
  * Potential for **alert fatigue** among clinical staff if false positive volume is overwhelming.
  * Mild patient anxiety regarding perceived health frailty.

### Trade-Off Analysis: Why "False Negatives Are Not Always Worse"
The optimal decision threshold depends on clinical and economic context:
* If the intervention is **low-cost and non-invasive** (e.g., an automated follow-up phone call or SMS check-in), the cost of an FP is negligible. The hospital should lower the decision threshold to maximize Recall (minimizing dangerous FNs).
* If the intervention is **high-cost, scarce, or invasive** (e.g., intensive home nursing or specialized medications with side-effects), high FPs drain hospital resources and cause alert fatigue.
* **Threshold Tuning:** At the default 0.50 cutoff, the model predicts almost no readmissions due to class imbalance (~9% positive). Adjusting the threshold to the empirical prevalence (~0.09) elevates recall to **53.38%**, catching over half of all readmissions.

---

## Results

> **Note:** These are actual results calculated on the held-out test split (13,998 samples). No numbers are fabricated.

### 1. Primary Model Performance
| Metric | Value | Interpretation & Clinical Significance |
| :--- | :--- | :--- |
| **ROC-AUC (Primary Metric)** | **0.6474** | Evaluates ranking discrimination across all thresholds; consistent with published literature. |
| Accuracy (Threshold = 0.50) | 90.97% | **Deceptive:** Driven almost entirely by predicting the negative majority class (~91.02%). |
| Precision (Threshold = 0.50) | 26.67% | Only 4 out of 15 flagged patients were actually readmitted. |
| Recall (Threshold = 0.50) | **0.32%** | **Critically low:** The model misses 99.68% of readmitted patients (1,253 False Negatives). |
| F1-Score (Threshold = 0.50) | 0.0063 | Reflects severe sensitivity deficit at default threshold. |

> [!WARNING]
> **The Accuracy Paradox in Imbalanced Readmission Data:**
> The nominal 90.97% accuracy must **not** be mistaken for high clinical efficacy. In an imbalanced cohort with an ~8.98% positive rate, a "dummy" model that predicts zero readmissions for every single encounter achieves 91.02% accuracy while offering zero clinical utility. At the default 0.50 cutoff, the Logistic Regression model predicts almost all patients as negative (1,253 False Negatives out of 1,257 actual readmissions). Therefore, **ROC-AUC (0.6474)** is the primary metric of clinical discrimination, and **threshold calibration** is clinically non-negotiable.

### 2. Confusion Matrix at Default Threshold (0.50)
| | Predicted: No Readmission (0) | Predicted: Readmitted <30 Days (1) | Total |
| :--- | :---: | :---: | :---: |
| **Actual: No Readmission (0)** | **12,730 (TN)** | **11 (FP)** | 12,741 |
| **Actual: Readmitted <30 Days (1)** | **1,253 (FN)** | **4 (TP)** | 1,257 |

### 3. Threshold Calibration Analysis
| Decision Threshold | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) | Recall (Sensitivity) | Precision |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.5000 (Default)** | 4 | 11 | 1,253 | 12,730 | 0.32% | 26.67% |
| **0.2000** | 173 | 448 | 1,084 | 12,293 | 13.76% | 27.86% |
| **0.1500** | 281 | 1,037 | 976 | 11,704 | 22.35% | 21.32% |
| **0.1000** | 563 | 3,219 | 694 | 9,522 | 44.79% | 14.89% |
| **0.0898 (Prevalence)** | **671** | **4,102** | **586** | **8,639** | **53.38%** | **14.06%** |

* Calibrating the threshold to the cohort prevalence (~0.0898) raises Recall from **0.32% to 53.38%**, successfully capturing **671 at-risk patients** (reducing FNs from 1,253 down to 586).

### 4. Key Predictor Coefficients (Log-Odds)
* **Strongest Continuous Positive Predictors (Higher Readmission Risk):**
  * Prior Inpatient Visits (`number_inpatient`): **+0.1905** (OR = 1.21)
  * Number of Diagnoses (`number_diagnoses`): **+0.0759** (OR = 1.08)
  * Prior Emergency Visits (`number_emergency`): **+0.0525** (OR = 1.05)
  * Length of Hospital Stay (`time_in_hospital`): **+0.0503** (OR = 1.05)
* **Key Protective Predictors (Lower Readmission Risk):**
  * Discharged directly to home (`discharge_disposition_id_1`): **-0.6134** (OR = 0.54)
  * Pediatric age group (`age_[0-10)`): **-0.3867** (OR = 0.68)

> [!NOTE]
> **Statistical Association vs. Medical Causation:**
> Logistic regression coefficients represent the estimated change in log-odds of readmission associated with a one-unit change in a feature, holding all other features constant. These weights reflect observational statistical associations within the trained model and do **NOT** establish clinical causation. Unmeasured illness severity, socioeconomic factors, and provider practice patterns heavily influence these empirical coefficients.

---

## Limitations

1. **Historical Dataset (1999–2008):** Medical standards and clinical protocols for type 2 diabetes management have evolved significantly over the past 15+ years.
2. **Administrative / Billing Data:** The dataset records ICD-9 billing codes and medication changes, but lacks fine-grained bedside telemetry, real-time lab trends, and socioeconomic indicators.
3. **Class Imbalance:** With an ~8.98% readmission rate, threshold calibration is essential for clinical utility.
4. **Clinical Decision Support Only:** This algorithm is an assistive clinical decision support (CDS) tool. It must never replace clinical assessment by physicians and nurses.

---

## Project Structure

```
Hospital_Readmission_ML_CA/
├── data/
│   ├── diabetic_data.csv          # UCI clinical encounters dataset
│   └── IDS_mapping.csv            # ID code description mappings
│
├── notebooks/
│   └── hospital_readmission_prediction.ipynb   # 17-section executed Jupyter notebook
│
├── src/
│   └── model.py                   # Standalone, reusable Python pipeline
│
├── .gitignore                     # Standard Git ignore rules
├── README.md                      # Comprehensive academic project documentation
└── requirements.txt               # Required Python packages
```

---

## How to Run

### 1. Environment Setup
Install dependencies using pip:
```bash
pip install -r requirements.txt
```

### 2. Run the Standalone Python Pipeline
To load the data, train the L2 Logistic Regression model, and output all evaluation metrics directly in the terminal:
```bash
python src/model.py
```

### 3. Run the Jupyter Notebook
To open and interact with the complete, executed notebook:
```bash
jupyter notebook notebooks/hospital_readmission_prediction.ipynb
```
or
```bash
jupyter lab
```
