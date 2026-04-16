import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import time

OUTPUT_DIR = "models/"

print("[*] Loading preprocessed data...")
X_train = joblib.load(f"{OUTPUT_DIR}X_train.pkl")
y_train = joblib.load(f"{OUTPUT_DIR}y_train.pkl")
X_test  = joblib.load(f"{OUTPUT_DIR}X_test.pkl")
y_test  = joblib.load(f"{OUTPUT_DIR}y_test.pkl")

# ── MODELS ───────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

lgbm = LGBMClassifier(
    n_estimators=200,
    max_depth=12,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm)],
    voting='soft',
    weights=[1, 1, 1],
    n_jobs=-1
)

# ── TRAIN ────────────────────────────────────────────────
print("[*] Training ensemble (this will take several minutes)...")
start = time.time()
ensemble.fit(X_train, y_train)
elapsed = time.time() - start
print(f"[✓] Training complete in {elapsed:.1f}s")

# ── EVALUATE EACH MODEL ──────────────────────────────────
print("\n📊 Individual Model Performance on Test Set:")
print("-" * 50)
names = ["Random Forest", "XGBoost", "LightGBM"]
for name, est in zip(names, ensemble.estimators_):
    y_pred = est.predict(X_test)
    y_prob = est.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"  {name:20s} → Accuracy: {acc:.4f} | AUC: {auc:.4f}")

y_pred_ens = ensemble.predict(X_test)
y_prob_ens = ensemble.predict_proba(X_test)[:, 1]
acc_ens = accuracy_score(y_test, y_pred_ens)
auc_ens = roc_auc_score(y_test, y_prob_ens)
print(f"  {'Ensemble':20s} → Accuracy: {acc_ens:.4f} | AUC: {auc_ens:.4f}")
print("-" * 50)

# ── SAVE ─────────────────────────────────────────────────
joblib.dump(ensemble,                f"{OUTPUT_DIR}ensemble_model.pkl")
joblib.dump(ensemble.estimators_[0], f"{OUTPUT_DIR}rf_model.pkl")
joblib.dump(ensemble.estimators_[1], f"{OUTPUT_DIR}xgb_model.pkl")
joblib.dump(ensemble.estimators_[2], f"{OUTPUT_DIR}lgbm_model.pkl")
print("[✓] All models saved.")