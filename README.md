# IDS-XAI: Explainable AI for Intrusion Detection Systems

An end-to-end machine learning pipeline for network intrusion detection, combined with Explainable AI (XAI) techniques to make model predictions interpretable. Built on the CICIDS2017 dataset, this project trains classification models to detect malicious network traffic and uses SHAP and LIME to explain *why* the model flags a given sample as an attack.

## Overview

Intrusion Detection Systems (IDS) powered by machine learning are often "black boxes" — they can flag suspicious traffic but rarely explain their reasoning. This makes them harder to trust and debug in real security operations. This project addresses that gap by:

- Training ML models (Random Forest, XGBoost, LightGBM, and an ensemble) to classify network traffic as benign or malicious
- Applying **SHAP** (SHapley Additive exPlanations) for global and local feature-importance analysis
- Applying **LIME** (Local Interpretable Model-agnostic Explanations) for per-sample explanations
- Providing an interactive dashboard to explore predictions and their explanations

## Features

- Multi-model training pipeline (Random Forest, XGBoost, LightGBM, Ensemble)
- SHAP-based global feature importance and summary plots
- LIME-based local explanations for individual predictions
- Confident sample selection for highlighting clear-cut model decisions
- Interactive dashboard for exploring model behavior
- Modular, script-based pipeline that's easy to re-run end-to-end

## Project Structure

```
IDS-XAI/
├── 0_find_confident_samples.py   # Identify high-confidence predictions for explanation
├── 0_get_sample_values.py        # Extract feature values for selected samples
├── 0_export_full_samples.py      # Export full sample records for inspection
├── 1_preprocess.py               # Data cleaning, encoding, and feature preparation
├── 2_train.py                    # Model training (RF, XGBoost, LightGBM, ensemble)
├── 3_evaluate.py                 # Model evaluation (accuracy, precision, recall, F1, etc.)
├── 4_explain_shap.py             # SHAP explainability (global + local)
├── 5_explain_lime.py             # LIME explainability (local, per-sample)
├── 6_dashboard.py                # Interactive dashboard for results & explanations
├── 67_dashboard_extra.py         # Additional dashboard components/visualizations
├── outputs/                      # Generated plots, HTML explanations, reports
├── data/                         # Raw/processed datasets (excluded from repo, see below)
├── models/                       # Trained model files (excluded from repo, see below)
├── requirements.txt              # Python dependencies
└── README.md
```

## Data & Models

The raw datasets and trained model files are **not included in this repository** due to their size (several GB). They are excluded via `.gitignore`.

**Dataset:** This project uses the [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) dataset (Canadian Institute for Cybersecurity), which includes labeled benign and attack traffic (DDoS, PortScan, Web Attacks, Infiltration, etc.) captured over multiple days.

To reproduce this project:
1. Download the CICIDS2017 CSVs from the [official source](https://www.unb.ca/cic/datasets/ids-2017.html) (or Kaggle mirror)
2. Place them in a `data/` folder in the project root
3. Run the pipeline scripts in order (see **Usage** below) to regenerate models and outputs

> If you'd like to share your trained models/datasets with others, consider hosting them on Google Drive, Kaggle, or Hugging Face Datasets and linking here.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/faizaanansari/IDS-XAI.git
   cd IDS-XAI
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Add the dataset to a `data/` folder as described above.

## Usage

Run the pipeline scripts in order:

```bash
# 1. Preprocess raw data
python 1_preprocess.py

# 2. Train models
python 2_train.py

# 3. Evaluate model performance
python 3_evaluate.py

# 4. Generate SHAP explanations
python 4_explain_shap.py

# 5. Generate LIME explanations
python 5_explain_lime.py

# Optional: select and export confident/representative samples
python 0_find_confident_samples.py
python 0_get_sample_values.py
python 0_export_full_samples.py

# Launch the interactive dashboard
python 6_dashboard.py
```

## Explainability Outputs

Results are saved to the `outputs/` folder, including:
- SHAP summary plots (e.g. `shap_rf_summary.png`) showing global feature importance
- LIME explanation reports (e.g. `lime_explanation_0.html`) showing per-sample reasoning
- Evaluation metrics and visualizations

## Tech Stack

- **Language:** Python
- **ML Models:** Scikit-learn (Random Forest), XGBoost, LightGBM
- **Explainability:** SHAP, LIME
- **Dashboard:** (add framework here, e.g. Streamlit/Dash — update once confirmed)
- **Data Handling:** Pandas, NumPy

## Author

**Faizaan Ansari**
GitHub: [@faizaanansari](https://github.com/faizaanansari)
