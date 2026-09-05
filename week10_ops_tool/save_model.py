"""
Run this script ONCE from your Week 9 folder to save the trained model
and feature column list for the Streamlit dashboard.

Usage:
    python save_model.py

Place the output files (xgb_model.joblib, feature_cols.joblib)
in the same folder as app.py before launching the dashboard.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

RANDOM_STATE = 42
CSV_PATH     = "synthetic_predictive_maintenance_data.csv"

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)

# ── FEATURE ENGINEERING — must match app.py exactly ──────────────────────────
df['vib_torque_stress']      = df['vibration'] * df['torque']
df['wear_cycle_degradation'] = df['tool_wear'] * df['cycle']
df['temp_speed_ratio']       = df['process_temp'] / (df['rotational_speed'] + 1e-6)
df['pressure_noise_ratio']   = df['pressure'] / (df['noise'] + 1e-6)

FEATURE_COLS = [c for c in df.columns if c not in ['asset_id', 'failure']]
X = df[FEATURE_COLS]
y = df['failure']

print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Class distribution: Normal={sum(y==0)}, Failure={sum(y==1)}")

# ── TRAIN ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print("Applying SMOTE...")
smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_res, y_res = smote.fit_resample(X_train, y_train)

neg = int(np.bincount(y_train)[0])
pos = int(np.bincount(y_train)[1])
spw = neg // pos
print(f"scale_pos_weight = {spw}")

model = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=spw,
    eval_metric='logloss', random_state=RANDOM_STATE,
    n_jobs=-1, verbosity=0
)
print("Training XGBoost...")
model.fit(X_res, y_res)

# ── QUICK VALIDATION ──────────────────────────────────────────────────────────
from sklearn.metrics import recall_score, roc_auc_score
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(f"Test Recall : {recall_score(y_test, y_pred):.4f}")
print(f"Test AUC    : {roc_auc_score(y_test, y_proba):.4f}")

# ── SAVE MODEL + FEATURE LIST ─────────────────────────────────────────────────
joblib.dump(model,        "xgb_model.joblib")
joblib.dump(FEATURE_COLS, "feature_cols.joblib")

print("\n✅ Saved:")
print("   xgb_model.joblib    — trained XGBoost model")
print("   feature_cols.joblib — exact feature column order")
print("\nCopy both files into your week10_ops_tool/ folder before running the dashboard.")