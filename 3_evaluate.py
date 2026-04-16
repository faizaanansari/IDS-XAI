import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score
)

OUTPUT_DIR = "models/"
PLOT_DIR   = "outputs/"

# ── LOAD ─────────────────────────────────────────────────
X_test  = joblib.load(f"{OUTPUT_DIR}X_test.pkl")
y_test  = joblib.load(f"{OUTPUT_DIR}y_test.pkl")
ensemble = joblib.load(f"{OUTPUT_DIR}ensemble_model.pkl")

models = {
    "Random Forest": joblib.load(f"{OUTPUT_DIR}rf_model.pkl"),
    "XGBoost":       joblib.load(f"{OUTPUT_DIR}xgb_model.pkl"),
    "LightGBM":      joblib.load(f"{OUTPUT_DIR}lgbm_model.pkl"),
    "Ensemble":      ensemble,
}

# ── METRICS ──────────────────────────────────────────────
print("\n" + "="*60)
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    print(f"\n📊 {name}")
    print(f"   Accuracy : {acc:.4f} | AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["BENIGN","ATTACK"]))
print("="*60)

# ── CONFUSION MATRIX (Ensemble) ──────────────────────────
y_pred = ensemble.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["BENIGN","ATTACK"],
            yticklabels=["BENIGN","ATTACK"])
plt.title("Confusion Matrix — Ensemble Model")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}confusion_matrix.png", dpi=150)
plt.show()
print("[✓] Saved confusion_matrix.png")

# ── ROC CURVES ───────────────────────────────────────────
plt.figure(figsize=(8,6))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — All Models")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}roc_curves.png", dpi=150)
plt.show()
print("[✓] Saved roc_curves.png")