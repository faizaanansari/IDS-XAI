import joblib
import pandas as pd
import numpy as np

OUTPUT_DIR = "models/"

# Load the SCALER and feature names
scaler        = joblib.load(f"{OUTPUT_DIR}scaler.pkl")
feature_names = joblib.load(f"{OUTPUT_DIR}feature_names.pkl")
X_test        = joblib.load(f"{OUTPUT_DIR}X_test.pkl")
y_test        = joblib.load(f"{OUTPUT_DIR}y_test.pkl")

# Reset index
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# INVERSE TRANSFORM to get original unscaled values
X_test_original = pd.DataFrame(
    scaler.inverse_transform(X_test),
    columns=feature_names
)

attacks = X_test_original[y_test == 1].head(5)
benigns = X_test_original[y_test == 0].head(5)

print("\n🔴 REAL ATTACK SAMPLES — enter these in dashboard:")
for idx, (_, row) in enumerate(attacks.iterrows()):
    print(f"\n--- Attack Sample {idx+1} ---")
    for feat in feature_names[:16]:
        print(f"  {feat}: {row[feat]:.4f}")

print("\n\n🟢 REAL BENIGN SAMPLES — enter these in dashboard:")
for idx, (_, row) in enumerate(benigns.iterrows()):
    print(f"\n--- Benign Sample {idx+1} ---")
    for feat in feature_names[:16]:
        print(f"  {feat}: {row[feat]:.4f}")