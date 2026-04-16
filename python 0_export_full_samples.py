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

y_pred  = ensemble.predict(X_test)
y_proba = ensemble.predict_proba(X_test)[:, 1]

X_original = pd.DataFrame(
    scaler.inverse_transform(X_test),
    columns=feature_names
)

# Best attack
correct_attacks = (y_test == 1) & (y_pred == 1)
attack_idx = X_test[correct_attacks].index
best_attack = attack_idx[np.argmax(y_proba[attack_idx])]

# Best benign
correct_benign = (y_test == 0) & (y_pred == 0)
benign_idx = X_test[correct_benign].index
best_benign = benign_idx[np.argmin(y_proba[benign_idx])]

print("ATTACK_SAMPLE = {")
for f in feature_names:
    print(f"    '{f}': {X_original.loc[best_attack, f]},")
print("}")

print("\nBENIGN_SAMPLE = {")
for f in feature_names:
    print(f"    '{f}': {X_original.loc[best_benign, f]},")
print("}")