import joblib
import numpy as np
import pandas as pd

OUTPUT_DIR = "models/"

scaler        = joblib.load(f"{OUTPUT_DIR}scaler.pkl")
feature_names = joblib.load(f"{OUTPUT_DIR}feature_names.pkl")
X_test        = joblib.load(f"{OUTPUT_DIR}X_test.pkl")
y_test        = joblib.load(f"{OUTPUT_DIR}y_test.pkl")
ensemble      = joblib.load(f"{OUTPUT_DIR}ensemble_model.pkl")

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Find samples the model CORRECTLY predicts as ATTACK with high confidence
y_pred  = ensemble.predict(X_test)
y_proba = ensemble.predict_proba(X_test)[:, 1]

# Correctly predicted attacks with highest confidence
correct_attacks = (y_test == 1) & (y_pred == 1)
attack_indices  = X_test[correct_attacks].index
attack_probas   = y_proba[attack_indices]

# Sort by confidence descending
sorted_idx = attack_indices[np.argsort(attack_probas)[::-1]]
top_attacks = sorted_idx[:5]

# Inverse transform to get real values
X_original = pd.DataFrame(
    scaler.inverse_transform(X_test),
    columns=feature_names
)

print("\n🔴 HIGH CONFIDENCE ATTACK SAMPLES (model verified):")
for rank, idx in enumerate(top_attacks):
    prob = y_proba[idx]
    print(f"\n--- Attack Sample {rank+1} | Model confidence: {prob*100:.1f}% ATTACK ---")
    for feat in feature_names[:16]:
        print(f"  {feat}: {X_original.loc[idx, feat]:.4f}")

# Also find high confidence benign
correct_benign = (y_test == 0) & (y_pred == 0)
benign_indices = X_test[correct_benign].index
benign_probas  = 1 - y_proba[benign_indices]
sorted_ben_idx = benign_indices[np.argsort(benign_probas)[::-1]]
top_benigns    = sorted_ben_idx[:3]

print("\n\n🟢 HIGH CONFIDENCE BENIGN SAMPLES (model verified):")
for rank, idx in enumerate(top_benigns):
    prob = 1 - y_proba[idx]
    print(f"\n--- Benign Sample {rank+1} | Model confidence: {prob*100:.1f}% BENIGN ---")
    for feat in feature_names[:16]:
        print(f"  {feat}: {X_original.loc[idx, feat]:.4f}")