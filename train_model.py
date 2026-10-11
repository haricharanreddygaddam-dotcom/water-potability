"""
Water Potability Prediction Model Trainer
Uses Random Forest Classifier on Water Quality Data
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Feature definitions and WHO standards
FEATURE_SPECS = {
    "ph": {
        "name": "pH Level",
        "unit": "pH",
        "min": 0.0,
        "max": 14.0,
        "mean": 7.08,
        "std": 1.59,
        "who_min": 6.5,
        "who_max": 8.5,
        "description": "Acid-base balance. WHO recommends 6.5 - 8.5."
    },
    "Hardness": {
        "name": "Hardness",
        "unit": "mg/L",
        "min": 47.0,
        "max": 323.0,
        "mean": 196.4,
        "std": 32.95,
        "who_min": 60.0,
        "who_max": 200.0,
        "description": "Dissolved calcium and magnesium salts."
    },
    "Solids": {
        "name": "Total Dissolved Solids (TDS)",
        "unit": "ppm",
        "min": 320.0,
        "max": 61227.0,
        "mean": 22014.0,
        "std": 8768.6,
        "who_min": 100.0,
        "who_max": 1000.0,
        "description": "Total dissolved minerals and organic matter."
    },
    "Chloramines": {
        "name": "Chloramines",
        "unit": "ppm",
        "min": 0.35,
        "max": 13.1,
        "mean": 7.12,
        "std": 1.58,
        "who_min": 0.5,
        "who_max": 4.0,
        "description": "Disinfectant compounds. Safe limit up to 4 ppm."
    },
    "Sulfate": {
        "name": "Sulfate",
        "unit": "mg/L",
        "min": 129.0,
        "max": 481.0,
        "mean": 333.77,
        "std": 36.14,
        "who_min": 50.0,
        "who_max": 250.0,
        "description": "Naturally occurring mineral salts. WHO target < 250 mg/L."
    },
    "Conductivity": {
        "name": "Conductivity",
        "unit": "μS/cm",
        "min": 181.0,
        "max": 753.0,
        "mean": 426.21,
        "std": 80.82,
        "who_min": 100.0,
        "who_max": 400.0,
        "description": "Electrical conductivity; reflects dissolved ions."
    },
    "Organic_carbon": {
        "name": "Total Organic Carbon (TOC)",
        "unit": "ppm",
        "min": 2.2,
        "max": 28.3,
        "mean": 14.28,
        "std": 3.31,
        "who_min": 1.0,
        "who_max": 4.0,
        "description": "Organic carbon compound concentration."
    },
    "Trihalomethanes": {
        "name": "Trihalomethanes (THMs)",
        "unit": "μg/L",
        "min": 0.7,
        "max": 124.0,
        "mean": 66.4,
        "std": 16.18,
        "who_min": 10.0,
        "who_max": 80.0,
        "description": "Disinfection byproducts. Safe limit < 80 μg/L."
    },
    "Turbidity": {
        "name": "Turbidity",
        "unit": "NTU",
        "min": 1.45,
        "max": 6.73,
        "mean": 3.97,
        "std": 0.78,
        "who_min": 0.5,
        "who_max": 5.0,
        "description": "Water cloudiness / light scattering. Ideal < 1.0 NTU."
    }
}

FEATURES = list(FEATURE_SPECS.keys())

def generate_water_dataset(n_samples=3276, random_state=42):
    """
    Generates realistic synthetic dataset adhering to Kaggle Water Potability
    statistical parameters, distributions, and domain potability rules.
    """
    np.random.seed(random_state)
    
    data = {}
    for feat, specs in FEATURE_SPECS.items():
        # Truncated normal distribution clamped to realistic min and max
        values = np.random.normal(specs["mean"], specs["std"], n_samples)
        values = np.clip(values, specs["min"], specs["max"])
        data[feat] = np.round(values, 2)
        
    df = pd.DataFrame(data)
    
    # Calculate Potability based on water quality chemistry and realistic noise
    # Water is potable when key parameters are closer to safe thresholds
    ph_score = np.exp(-((df["ph"] - 7.3) / 1.5) ** 2)
    solids_penalty = np.clip(df["Solids"] / 25000.0, 0, 1.5)
    chloramines_penalty = np.where(df["Chloramines"] > 8.0, (df["Chloramines"] - 8.0) / 4.0, 0)
    sulfate_penalty = np.where(df["Sulfate"] > 350.0, (df["Sulfate"] - 350.0) / 100.0, 0)
    thms_penalty = np.where(df["Trihalomethanes"] > 80.0, (df["Trihalomethanes"] - 80.0) / 30.0, 0)
    turbidity_penalty = np.where(df["Turbidity"] > 4.5, (df["Turbidity"] - 4.5) / 2.0, 0)
    organic_penalty = np.where(df["Organic_carbon"] > 16.0, (df["Organic_carbon"] - 16.0) / 6.0, 0)
    hardness_penalty = np.clip(np.abs(df["Hardness"] - 180) / 100.0, 0, 1)

    potability_latent = (
        1.6 * ph_score
        - 0.8 * solids_penalty
        - 0.9 * chloramines_penalty
        - 0.7 * sulfate_penalty
        - 0.8 * thms_penalty
        - 0.6 * turbidity_penalty
        - 0.6 * organic_penalty
        - 0.3 * hardness_penalty
        + np.random.normal(0, 0.45, n_samples) # stochastic real-world variance
    )
    
    # Threshold chosen to match ~39% potable class balance as in Kaggle dataset
    threshold = np.percentile(potability_latent, 61.0)
    df["Potability"] = (potability_latent >= threshold).astype(int)
    
    return df

def train_and_evaluate(df):
    """
    Trains Random Forest Classifier and computes performance metrics.
    """
    X = df[FEATURES]
    y = df["Potability"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Tuned Random Forest model
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    # Predict
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    # Feature Importances
    importances = [
        {"feature": feat, "importance": round(float(imp), 4), "name": FEATURE_SPECS[feat]["name"]}
        for feat, imp in zip(FEATURES, rf.feature_importances_)
    ]
    importances.sort(key=lambda x: x["importance"], reverse=True)
    
    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc), 4),
        "confusion_matrix": cm,
        "n_samples": len(df),
        "potable_count": int(y.sum()),
        "non_potable_count": int((y == 0).sum()),
        "feature_importances": importances
    }
    
    return rf, metrics, X_test, y_test

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "water_potability.csv")
    
    if os.path.exists(csv_path):
        print(f"Loading existing dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        # Impute missing values with mean if any
        for col in FEATURES:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
    else:
        print("Synthesizing realistic Kaggle-aligned Water Potability dataset (3,276 samples)...")
        df = generate_water_dataset()
        df.to_csv(csv_path, index=False)
        print(f"Saved dataset to {csv_path}")
        
    print("Training Random Forest Classifier...")
    model, metrics, X_test, y_test = train_and_evaluate(df)
    
    # Save artifacts
    model_path = os.path.join(base_dir, "water_potability_rf.joblib")
    metrics_path = os.path.join(base_dir, "model_metrics.json")
    
    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print("\n--- Training Results ---")
    print(f"Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"F1 Score:  {metrics['f1_score'] * 100:.2f}%")
    print(f"ROC-AUC:   {metrics['roc_auc'] * 100:.2f}%")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")
    print("\nTop 3 Important Features:")
    for item in metrics["feature_importances"][:3]:
        print(f"  - {item['name']}: {item['importance'] * 100:.2f}%")
        
    print(f"\nModel saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()
