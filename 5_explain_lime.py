import joblib
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

OUTPUT_DIR = "models/"
PLOT_DIR   = "outputs/"

# ── LOAD ─────────────────────────────────────────────────
X_train       = joblib.load(f"{OUTPUT_DIR}X_train.pkl")
X_test        = joblib.load(f"{OUTPUT_DIR}X_test.pkl")
y_test        = joblib.load(f"{OUTPUT_DIR}y_test.pkl")
feature_names = joblib.load(f"{OUTPUT_DIR}feature_names.pkl")
ensemble      = joblib.load(f"{OUTPUT_DIR}ensemble_model.pkl")

# ── LIME EXPLAINER ────────────────────────────────────────
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=["BENIGN", "ATTACK"],
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

# ── EXPLAIN 3 INSTANCES ──────────────────────────────────
for i in [0, 1, 2]:
    instance = X_test.iloc[i].values
    actual   = y_test.iloc[i]
    predicted = ensemble.predict([instance])[0]
    label = "ATTACK" if predicted == 1 else "BENIGN"

    print(f"\n[*] Explaining instance {i} | Actual: {'ATTACK' if actual else 'BENIGN'} | Predicted: {label}")

    exp = explainer.explain_instance(
        instance,
        ensemble.predict_proba,
        num_features=15,
        top_labels=1
    )

    # Show in matplotlib
    fig = exp.as_pyplot_figure(label=predicted)
    plt.title(f"LIME Explanation — Instance {i} | Predicted: {label}")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}lime_explanation_{i}.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[✓] Saved lime_explanation_{i}.png")

    # Save HTML
    exp.save_to_file(f"{PLOT_DIR}lime_explanation_{i}.html")
    print(f"[✓] Saved lime_explanation_{i}.html")