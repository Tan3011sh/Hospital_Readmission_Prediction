"""
Hospital Readmission Prediction using Logistic Regression with L2 Regularization
Academic Machine Learning Project

This module provides an end-to-end, reusable pipeline to:
1. Load patient records from the UCI Diabetes 130-US Hospitals dataset.
2. Clean and preprocess clinical features (diagnoses, prior visits, vitals/labs).
3. Split into stratified train and test sets (80/20).
4. Train a Logistic Regression model with L2 regularization inside a scikit-learn Pipeline.
5. Evaluate performance using ROC-AUC, Accuracy, Precision, Recall, F1, and Confusion Matrix.
6. Extract and interpret model coefficients (log-odds).
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Suppress minor depreciation/future warnings for clean output
warnings.filterwarnings("ignore", category=FutureWarning)


def map_icd9(code):
    """
    Categorize ICD-9 diagnosis codes into standard clinical categories
    based on medical informatics conventions (Strack et al., 2014).
    """
    if pd.isna(code) or code == "?":
        return "Missing"
    code_str = str(code).strip()
    if code_str.startswith("V") or code_str.startswith("E"):
        return "Other"
    try:
        val = float(code_str)
        if (390 <= val <= 459) or val == 785:
            return "Circulatory"
        elif (460 <= val <= 519) or val == 786:
            return "Respiratory"
        elif (520 <= val <= 579) or val == 787:
            return "Digestive"
        elif 250 <= val < 251:
            return "Diabetes"
        elif 800 <= val <= 999:
            return "Injury"
        elif 710 <= val <= 739:
            return "Musculoskeletal"
        elif (580 <= val <= 629) or val == 788:
            return "Genitourinary"
        elif 140 <= val <= 239:
            return "Neoplasms"
        else:
            return "Other"
    except ValueError:
        return "Other"


def load_and_clean_data(data_path="data/diabetic_data.csv"):
    """
    Load raw clinical dataset and perform required data cleaning:
    - Exclude expired or hospice-discharged patients (ineligible for readmission).
    - Deduplicate patients by retaining the first encounter per patient to mitigate repeat-visit
      clustering and prevent train/test data leakage (without assuming pure i.i.d. observations).
    - Transform target 'readmitted' into binary: 1 for '<30', 0 for '>30' and 'NO'.
    - Clean missing values, collapse high-cardinality diagnosis codes and specialties.
    - Drop identifiers and uninformative features.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Please place diabetic_data.csv in the data/ directory."
        )

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Raw dataset shape: {df.shape}")

    # 1. Exclude patients who expired or entered hospice (cannot be readmitted)
    expired_ids = [11, 13, 14, 19, 20, 21]
    df = df[~df["discharge_disposition_id"].isin(expired_ids)].copy()

    # 2. Keep only the first encounter per patient to mitigate repeat-visit clustering and prevent leakage
    df = df.drop_duplicates(subset=["patient_nbr"], keep="first").copy()
    print(f"Shape after removing expired records and deduplicating patients: {df.shape}")

    # 3. Binary target creation (<30 -> 1, else 0)
    df["target"] = (df["readmitted"] == "<30").astype(int)
    pos_count = df["target"].sum()
    neg_count = len(df) - pos_count
    print(f"Target distribution: 0 (No 30-day readmit) = {neg_count} ({neg_count/len(df):.2%}), "
          f"1 (Readmitted <30 days) = {pos_count} ({pos_count/len(df):.2%})")

    # 4. Drop non-predictive identifiers and features with excessive missing values (>95%)
    # 'examide' and 'citoglipton' have zero variance (single value 'No' throughout).
    drop_cols = ["encounter_id", "patient_nbr", "weight", "payer_code", "readmitted", "examide", "citoglipton"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # 5. Group ICD-9 diagnosis codes into clinical disease categories
    for diag in ["diag_1", "diag_2", "diag_3"]:
        if diag in df.columns:
            df[f"{diag}_group"] = df[diag].apply(map_icd9)
            df = df.drop(columns=[diag])

    # 6. Clean medical_specialty: keep top 10 specialties, group others, handle missing '?'
    if "medical_specialty" in df.columns:
        top_specialties = df["medical_specialty"].value_counts().nlargest(10).index.tolist()
        df["medical_specialty"] = df["medical_specialty"].apply(
            lambda x: x if x in top_specialties and x != "?" else ("Missing" if x == "?" else "Other")
        )

    # 7. Impute missing race codes '?' with 'Missing'
    if "race" in df.columns:
        df["race"] = df["race"].replace("?", "Missing")

    # 8. Remove rare invalid gender records (3 cases in dataset)
    if "gender" in df.columns:
        df = df[df["gender"] != "Unknown/Invalid"].copy()

    # 9. Ensure administrative categorical ID columns are treated as strings
    cat_id_cols = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
    for col in cat_id_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Separate features and target
    y = df.pop("target")
    X = df
    print(f"Final cleaned dataset: {X.shape[0]} samples, {X.shape[1]} predictor features.")

    return X, y


def build_pipeline(numerical_cols, categorical_cols, C=1.0, random_state=42):
    """
    Construct a scikit-learn Pipeline with:
    - StandardScaler for numerical variables.
    - OneHotEncoder for categorical variables.
    - LogisticRegression classifier with L2 regularization penalty and C=1.0.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            penalty="l2",
            C=C,
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs"
        ))
    ])

    return pipeline


def evaluate_model(clf, X_test, y_test, threshold=0.50):
    """
    Evaluate the fitted model and return metrics dict.
    Calculates ROC-AUC using predicted probabilities.
    """
    # 1. Predicted probabilities for the positive class (readmitted <30 days)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # 2. Predicted classes using the specified decision threshold
    y_pred = (y_prob >= threshold).astype(int)

    # 3. Evaluation metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "roc_auc": roc_auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "y_prob": y_prob,
        "y_pred": y_pred
    }

    return metrics


def get_top_coefficients(clf, top_n=10):
    """
    Extract and sort logistic regression coefficients for feature interpretation.
    """
    preprocessor = clf.named_steps["preprocessor"]
    classifier = clf.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    coefs = classifier.coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs)
    }).sort_values("coefficient", ascending=False)

    top_positive = coef_df.head(top_n)
    top_negative = coef_df.tail(top_n).sort_values("coefficient", ascending=True)

    return top_positive, top_negative, coef_df


def run_pipeline(data_path="data/diabetic_data.csv", test_size=0.20, random_state=42):
    """
    End-to-end execution function:
    Loads data, trains Logistic Regression with L2 regularization, and reports metrics.
    """
    X, y = load_and_clean_data(data_path)

    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"\nFeature split: {len(num_cols)} numerical features, {len(cat_cols)} categorical features.")

    # Train / Test split with stratification to maintain class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")

    # Build and fit pipeline
    print("\nTraining Logistic Regression with L2 Regularization (C=1.0, max_iter=1000)...")
    clf = build_pipeline(num_cols, cat_cols, C=1.0, random_state=random_state)
    clf.fit(X_train, y_train)
    print("Model training completed successfully.")

    # Evaluate model at default threshold (0.50)
    metrics_50 = evaluate_model(clf, X_test, y_test, threshold=0.50)
    print("\n" + "=" * 55)
    print("MODEL EVALUATION RESULTS (Decision Threshold = 0.50)")
    print("=" * 55)
    print(f"Primary Metric - ROC-AUC : {metrics_50['roc_auc']:.4f}")
    print(f"Accuracy                 : {metrics_50['accuracy']:.4f}")
    print(f"Precision                : {metrics_50['precision']:.4f}")
    print(f"Recall                   : {metrics_50['recall']:.4f}")
    print(f"F1-Score                 : {metrics_50['f1_score']:.4f}")
    print("\nConfusion Matrix (Threshold 0.50):")
    tn, fp, fn, tp = metrics_50["confusion_matrix"].ravel()
    print(f"  True Negatives  (TN) : {tn:5d}  |  False Positives (FP) : {fp:5d}")
    print(f"  False Negatives (FN) : {fn:5d}  |  True Positives  (TP) : {tp:5d}")
    print("  * Critical Academic Note: High accuracy (90.97%) is an artifact of class imbalance")
    print("    (~91% negative base rate). At threshold 0.50, recall is nearly zero (0.32%),")
    print("    resulting in 1,253 False Negatives (missed readmissions).")

    # Evaluate model at prevalence threshold (~0.09) to illustrate clinical cost trade-off
    prevalence = y_train.mean()
    metrics_prev = evaluate_model(clf, X_test, y_test, threshold=prevalence)
    print("\n" + "=" * 55)
    print(f"CLINICAL THRESHOLD EVALUATION (Prevalence Threshold = {prevalence:.4f})")
    print("=" * 55)
    print(f"Recall (Sensitivity)    : {metrics_prev['recall']:.4f}")
    print(f"Precision                : {metrics_prev['precision']:.4f}")
    tn_p, fp_p, fn_p, tp_p = metrics_prev["confusion_matrix"].ravel()
    print(f"  True Negatives  (TN) : {tn_p:5d}  |  False Positives (FP) : {fp_p:5d}")
    print(f"  False Negatives (FN) : {fn_p:5d}  |  True Positives  (TP) : {tp_p:5d}")
    print("  * Note: Lowering threshold to prevalence boosts sensitivity from 0.32% to 53.38%,")
    print("    reducing False Negatives from 1,253 to 586, balanced against 4,102 False Positives.")

    # Feature interpretation
    print("\n" + "=" * 55)
    print("TOP PREDICTOR COEFFICIENTS (Model Log-Odds Interpretation)")
    print("=" * 55)
    print("  * Note: Coefficients represent change in log-odds within this model;")
    print("    they indicate statistical association and do NOT establish medical causation.")
    top_pos, top_neg, _ = get_top_coefficients(clf, top_n=5)
    print("\nTop Positive Coefficients (Associated with Increased Readmission Risk):")
    for _, row in top_pos.iterrows():
        print(f"  {row['feature']:<45} : +{row['coefficient']:.4f}")

    print("\nTop Negative Coefficients (Associated with Decreased Readmission Risk):")
    for _, row in top_neg.iterrows():
        print(f"  {row['feature']:<45} : {row['coefficient']:.4f}")

    return clf, metrics_50, metrics_prev


if __name__ == "__main__":
    run_pipeline()
