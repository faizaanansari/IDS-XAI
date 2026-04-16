import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = "models/"
PLOT_DIR   = "outputs/"

X_test        = joblib.load(f"{OUTPUT_DIR}X_test.pkl")
feature_names = joblib.load(f"{OUTPUT_DIR}feature_names.pkl")
rf_model      = joblib.load(f"{OUTPUT_DIR}rf_model.pkl")
xgb_model     = joblib.load(f"{OUTPUT_DIR}xgb_model.pkl")
lgbm_model    = joblib.load(f"{OUTPUT_DIR}lgbm_model.pkl")

X_sample = X_test.sample(500, random_state=42).reset_index(drop=True)

def get_shap_values(explainer, data):
    """Always returns a 2D array for class 1 (attack)."""
    sv = explainer.shap_values(data)
    if isinstance(sv, list):
        return np.array(sv[1])   # binary list → take class 1
    if sv.ndim == 3:
        return sv[:, :, 1]       # 3D array → take class 1 slice
    return sv                    # already 2D

def save_shap_plots(model, model_name, file_prefix):
    print(f"[*] Computing SHAP for {model_name}...")
    explainer = shap.TreeExplainer(model)
    sv = get_shap_values(explainer, X_sample)

    # Summary (beeswarm)
    plt.figure()
    shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary — {model_name}", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}{file_prefix}_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved {file_prefix}_summary.png")

    # Bar (mean |SHAP|)
    plt.figure()
    shap.summary_plot(sv, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance — {model_name}", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}{file_prefix}_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved {file_prefix}_importance.png")

    return explainer, sv

# ── All 3 models ──────────────────────────────────────────
rf_exp,   rf_sv   = save_shap_plots(rf_model,   "Random Forest", "shap_rf")
xgb_exp,  xgb_sv  = save_shap_plots(xgb_model,  "XGBoost",       "shap_xgb")
lgbm_exp, lgbm_sv = save_shap_plots(lgbm_model, "LightGBM",      "shap_lgbm")

# ── Waterfall for each model ──────────────────────────────
for model, name, prefix in [
    (rf_model,   "Random Forest", "shap_rf"),
    (xgb_model,  "XGBoost",       "shap_xgb"),
    (lgbm_model, "LightGBM",      "shap_lgbm"),
]:
    try:
        exp2     = shap.Explainer(model, X_sample)
        shap_exp = exp2(X_sample.iloc[:1])

        # Handle multi-output shape
        if shap_exp.values.ndim == 3:
            import shap as shap_lib
            single = shap_lib.Explanation(
                values        = shap_exp.values[0, :, 1],
                base_values   = shap_exp.base_values[0, 1],
                data          = shap_exp.data[0],
                feature_names = feature_names
            )
        else:
            single = shap_exp[0]

        plt.figure()
        shap.plots.waterfall(single, show=False)
        plt.title(f"SHAP Waterfall — {name}", fontsize=13)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}{prefix}_waterfall.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[✓] Saved {prefix}_waterfall.png")
    except Exception as e:
        print(f"[!] Waterfall skipped for {name}: {e}")

# ── Cross-model comparison chart ──────────────────────────
print("[*] Generating cross-model SHAP comparison...")

# Ensure all sv arrays are 2D and 1D mean
mean_rf   = np.abs(np.array(rf_sv)).mean(axis=0).flatten()
mean_xgb  = np.abs(np.array(xgb_sv)).mean(axis=0).flatten()
mean_lgbm = np.abs(np.array(lgbm_sv)).mean(axis=0).flatten()

# Align lengths (safety check)
min_len = min(len(mean_rf), len(mean_xgb), len(mean_lgbm), len(feature_names))
mean_rf   = mean_rf[:min_len]
mean_xgb  = mean_xgb[:min_len]
mean_lgbm = mean_lgbm[:min_len]
feats     = feature_names[:min_len]

comp_df = pd.DataFrame({
    "Feature":       feats,
    "Random Forest": mean_rf,
    "XGBoost":       mean_xgb,
    "LightGBM":      mean_lgbm,
}).set_index("Feature")

top15   = comp_df.sum(axis=1).nlargest(15).index
comp_df = comp_df.loc[top15]

fig, ax = plt.subplots(figsize=(12, 7))
comp_df.plot(kind='barh', ax=ax, color=['#3b82f6', '#f59e0b', '#10b981'])
ax.set_title("SHAP Feature Importance Comparison — All 3 Models (Top 15)", fontsize=13)
ax.set_xlabel("Mean |SHAP Value|")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}shap_comparison_all_models.png", dpi=150, bbox_inches='tight')
plt.close()
print("[✓] Saved shap_comparison_all_models.png")

print("\n[✓] All SHAP plots complete!")