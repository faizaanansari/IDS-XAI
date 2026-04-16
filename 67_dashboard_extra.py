from flask import Flask, render_template_string, request, jsonify
import joblib
import numpy as np
import shap
import os
import base64

app = Flask(__name__)

OUTPUT_DIR = "models/"
PLOT_DIR   = "outputs/"

print("[*] Loading models...")
ensemble      = joblib.load(f"{OUTPUT_DIR}ensemble_model.pkl")
feature_names = joblib.load(f"{OUTPUT_DIR}feature_names.pkl")
rf_model      = joblib.load(f"{OUTPUT_DIR}rf_model.pkl")
xgb_model     = joblib.load(f"{OUTPUT_DIR}xgb_model.pkl")
lgbm_model    = joblib.load(f"{OUTPUT_DIR}lgbm_model.pkl")
scaler        = joblib.load(f"{OUTPUT_DIR}scaler.pkl")

print("[*] Loading SHAP explainers (takes ~30s)...")
shap_rf   = shap.TreeExplainer(rf_model)
shap_xgb  = shap.TreeExplainer(xgb_model)
shap_lgbm = shap.TreeExplainer(lgbm_model)
print("[✓] All models and explainers ready.")

def img_to_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None

def extract_shap_1d(explainer, vals):
    sv = np.array(explainer.shap_values(vals))
    if sv.ndim == 4:
        sv = sv[1, 0, :, 0]
    elif sv.ndim == 3:
        if sv.shape[0] == 2:
            sv = sv[1, 0, :]
        elif sv.shape[2] == 2:
            sv = sv[0, :, 1]
        else:
            sv = sv[0, 0, :]
    elif sv.ndim == 2:
        sv = sv[1, :] if sv.shape[0] == 2 else sv[0, :]
    return sv.flatten()

# Pre-compute dataset means for default values
print("[*] Computing feature means for defaults...")
X_test = joblib.load(f"{OUTPUT_DIR}X_test.pkl")
scaled_means = X_test.mean().values  # means in scaled space = ~0
raw_means = scaler.inverse_transform(X_test.mean().values.reshape(1,-1))[0]
feature_defaults = {feature_names[i]: float(raw_means[i]) for i in range(len(feature_names))}
print("[✓] Defaults ready.")

# Known attack sample (verified 100% confidence)
import random

def make_random_attack():
    """Generate a randomised but realistic attack-leaning traffic sample."""
    random.seed()
    return {
        'Destination Port': random.choice([80, 443, 8080, 22, 23, 3389, 445]),
        'Flow Duration': random.randint(50, 500000),
        'Total Fwd Packets': random.randint(1, 10),
        'Total Backward Packets': random.randint(0, 3),
        'Total Length of Fwd Packets': random.uniform(0, 500),
        'Total Length of Bwd Packets': random.uniform(5000, 15000),
        'Fwd Packet Length Max': random.uniform(0, 400),
        'Fwd Packet Length Min': 0.0,
        'Fwd Packet Length Mean': random.uniform(0, 200),
        'Fwd Packet Length Std': random.uniform(0, 200),
        'Bwd Packet Length Max': random.uniform(4000, 10000),
        'Bwd Packet Length Min': 0.0,
        'Bwd Packet Length Mean': random.uniform(1000, 4000),
        'Bwd Packet Length Std': random.uniform(1000, 5000),
        'Flow Bytes/s': random.uniform(50, 500),
        'Flow Packets/s': random.uniform(0.05, 0.5),
        'Flow IAT Mean': random.uniform(5000000, 20000000),
        'Flow IAT Std': random.uniform(10000000, 40000000),
        'Flow IAT Max': random.uniform(50000000, 100000000),
        'Flow IAT Min': random.uniform(1, 10),
        'Fwd IAT Total': random.uniform(50000000, 100000000),
        'Fwd IAT Mean': random.uniform(10000000, 40000000),
        'Fwd IAT Std': random.uniform(20000000, 60000000),
        'Fwd IAT Max': random.uniform(50000000, 100000000),
        'Fwd IAT Min': random.uniform(1, 10),
        'Bwd IAT Total': random.uniform(50000, 300000),
        'Bwd IAT Mean': random.uniform(10000, 100000),
        'Bwd IAT Std': random.uniform(20000, 100000),
        'Bwd IAT Max': random.uniform(80000, 200000),
        'Bwd IAT Min': random.uniform(100, 500),
        'Fwd PSH Flags': 0.0, 'Bwd PSH Flags': 0.0,
        'Fwd URG Flags': 0.0, 'Bwd URG Flags': 0.0,
        'Fwd Header Length': random.choice([80, 100, 120, 140]),
        'Bwd Header Length': random.choice([120, 140, 160, 180]),
        'Fwd Packets/s': random.uniform(0.02, 0.2),
        'Bwd Packets/s': random.uniform(0.03, 0.2),
        'Min Packet Length': 0.0,
        'Max Packet Length': random.uniform(4000, 10000),
        'Packet Length Mean': random.uniform(800, 2000),
        'Packet Length Std': random.uniform(1500, 4000),
        'Packet Length Variance': random.uniform(2000000, 15000000),
        'FIN Flag Count': random.choice([0, 1]),
        'SYN Flag Count': 0.0, 'RST Flag Count': 0.0,
        'PSH Flag Count': 0.0, 'ACK Flag Count': 0.0,
        'URG Flag Count': 0.0, 'CWE Flag Count': 0.0, 'ECE Flag Count': 0.0,
        'Down/Up Ratio': random.uniform(0.8, 1.2),
        'Average Packet Size': random.uniform(800, 1800),
        'Avg Fwd Segment Size': random.uniform(50, 200),
        'Avg Bwd Segment Size': random.uniform(1000, 4000),
        'Fwd Header Length.1': random.choice([80, 100, 120, 140]),
        'Fwd Avg Bytes/Bulk': 0.0, 'Fwd Avg Packets/Bulk': 0.0, 'Fwd Avg Bulk Rate': 0.0,
        'Bwd Avg Bytes/Bulk': 0.0, 'Bwd Avg Packets/Bulk': 0.0, 'Bwd Avg Bulk Rate': 0.0,
        'Subflow Fwd Packets': random.randint(1, 10),
        'Subflow Fwd Bytes': random.uniform(0, 500),
        'Subflow Bwd Packets': random.randint(0, 5),
        'Subflow Bwd Bytes': random.uniform(5000, 15000),
        'Init_Win_bytes_forward': random.choice([0, 64, 128, 256]),
        'Init_Win_bytes_backward': random.uniform(100, 500),
        'act_data_pkt_fwd': random.randint(1, 3),
        'min_seg_size_forward': 20.0,
        'Active Mean': 0.0, 'Active Std': 0.0, 'Active Max': 0.0, 'Active Min': 0.0,
        'Idle Mean': random.uniform(40000000, 90000000),
        'Idle Std': 0.0,
        'Idle Max': random.uniform(40000000, 90000000),
        'Idle Min': random.uniform(40000000, 90000000),
    }

def make_random_benign():
    """Generate a randomised but realistic benign-leaning traffic sample."""
    random.seed()
    return {
        'Destination Port': random.choice([53, 443, 80, 8080, 993, 587]),
        'Flow Duration': random.randint(100, 300),
        'Total Fwd Packets': random.randint(2, 4),
        'Total Backward Packets': random.randint(2, 4),
        'Total Length of Fwd Packets': random.uniform(40, 120),
        'Total Length of Bwd Packets': random.uniform(60, 160),
        'Fwd Packet Length Max': random.uniform(20, 60),
        'Fwd Packet Length Min': random.uniform(20, 40),
        'Fwd Packet Length Mean': random.uniform(20, 50),
        'Fwd Packet Length Std': random.uniform(0, 5),
        'Bwd Packet Length Max': random.uniform(30, 80),
        'Bwd Packet Length Min': random.uniform(30, 60),
        'Bwd Packet Length Mean': random.uniform(30, 65),
        'Bwd Packet Length Std': random.uniform(0, 5),
        'Flow Bytes/s': random.uniform(700000, 1200000),
        'Flow Packets/s': random.uniform(15000, 35000),
        'Flow IAT Mean': random.uniform(30, 80),
        'Flow IAT Std': random.uniform(50, 120),
        'Flow IAT Max': random.uniform(100, 200),
        'Flow IAT Min': random.uniform(1, 5),
        'Fwd IAT Total': random.uniform(1, 5),
        'Fwd IAT Mean': random.uniform(1, 5),
        'Fwd IAT Std': 0.0,
        'Fwd IAT Max': random.uniform(1, 5),
        'Fwd IAT Min': random.uniform(1, 5),
        'Bwd IAT Total': random.uniform(1, 5),
        'Bwd IAT Mean': random.uniform(1, 5),
        'Bwd IAT Std': 0.0,
        'Bwd IAT Max': random.uniform(1, 5),
        'Bwd IAT Min': random.uniform(1, 5),
        'Fwd PSH Flags': 0.0, 'Bwd PSH Flags': 0.0,
        'Fwd URG Flags': 0.0, 'Bwd URG Flags': 0.0,
        'Fwd Header Length': random.choice([32, 40, 48]),
        'Bwd Header Length': random.choice([32, 40, 48]),
        'Fwd Packets/s': random.uniform(10000, 20000),
        'Bwd Packets/s': random.uniform(10000, 20000),
        'Min Packet Length': random.uniform(20, 40),
        'Max Packet Length': random.uniform(40, 80),
        'Packet Length Mean': random.uniform(30, 55),
        'Packet Length Std': random.uniform(5, 15),
        'Packet Length Variance': random.uniform(30, 200),
        'FIN Flag Count': 0.0, 'SYN Flag Count': 0.0, 'RST Flag Count': 0.0,
        'PSH Flag Count': 0.0, 'ACK Flag Count': 0.0,
        'URG Flag Count': 0.0, 'CWE Flag Count': 0.0, 'ECE Flag Count': 0.0,
        'Down/Up Ratio': random.uniform(0.9, 1.1),
        'Average Packet Size': random.uniform(35, 60),
        'Avg Fwd Segment Size': random.uniform(20, 50),
        'Avg Bwd Segment Size': random.uniform(30, 65),
        'Fwd Header Length.1': random.choice([32, 40, 48]),
        'Fwd Avg Bytes/Bulk': 0.0, 'Fwd Avg Packets/Bulk': 0.0, 'Fwd Avg Bulk Rate': 0.0,
        'Bwd Avg Bytes/Bulk': 0.0, 'Bwd Avg Packets/Bulk': 0.0, 'Bwd Avg Bulk Rate': 0.0,
        'Subflow Fwd Packets': random.randint(2, 4),
        'Subflow Fwd Bytes': random.uniform(40, 120),
        'Subflow Bwd Packets': random.randint(2, 4),
        'Subflow Bwd Bytes': random.uniform(60, 160),
        'Init_Win_bytes_forward': -1.0,
        'Init_Win_bytes_backward': -1.0,
        'act_data_pkt_fwd': 1.0,
        'min_seg_size_forward': 20.0,
        'Active Mean': 0.0, 'Active Std': 0.0, 'Active Max': 0.0, 'Active Min': 0.0,
        'Idle Mean': 0.0, 'Idle Std': 0.0, 'Idle Max': 0.0, 'Idle Min': 0.0,
    }

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>XAI Intrusion Detection System</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }
    header {
      background: #1e293b; padding: 18px 40px;
      border-bottom: 2px solid #334155;
      display: flex; align-items: center; gap: 14px;
    }
    header h1 { font-size: 1.7rem; color: #38bdf8; }
    header p  { color: #94a3b8; font-size: 0.85rem; margin-top: 3px; }
    .container { max-width: 1200px; margin: 32px auto; padding: 0 24px; }
    .tabs { display: flex; gap: 8px; margin-bottom: 24px; flex-wrap: wrap; }
    .tab-btn {
      padding: 10px 20px; border-radius: 8px; border: 1px solid #334155;
      background: #1e293b; color: #94a3b8; cursor: pointer;
      font-size: 0.9rem; font-weight: 600; transition: all 0.2s;
    }
    .tab-btn:hover  { background: #334155; color: #e2e8f0; }
    .tab-btn.active { background: #0ea5e9; color: white; border-color: #0ea5e9; }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    .card {
      background: #1e293b; border-radius: 12px; padding: 26px;
      margin-bottom: 22px; border: 1px solid #334155;
    }
    .card h2 {
      color: #38bdf8; margin-bottom: 16px; font-size: 0.95rem;
      text-transform: uppercase; letter-spacing: 1px;
    }
    .demo-btns { display: flex; gap: 10px; margin-bottom: 16px; }
    .demo-btn {
      padding: 8px 20px; border-radius: 8px; border: none;
      font-weight: 700; cursor: pointer; font-size: 0.88rem; transition: opacity 0.2s;
    }
    .demo-btn:hover { opacity: 0.85; }
    .demo-btn.attack { background: #dc2626; color: white; }
    .demo-btn.benign { background: #16a34a; color: white; }
    .demo-btn.reset  { background: #334155; color: #94a3b8; }
    .feature-grid {
      display: grid; grid-template-columns: repeat(4, 1fr);
      gap: 10px; max-height: 420px; overflow-y: auto;
      padding-right: 6px; margin-bottom: 16px;
    }
    .feature-grid::-webkit-scrollbar { width: 6px; }
    .feature-grid::-webkit-scrollbar-track { background: #0f172a; }
    .feature-grid::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    .feat-item label { display: block; margin-bottom: 3px; color: #94a3b8; font-size: 0.72rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .feat-item input {
      width: 100%; padding: 6px 8px; background: #0f172a;
      border: 1px solid #475569; border-radius: 5px;
      color: #e2e8f0; font-size: 0.82rem;
    }
    .feat-item input:focus { outline: none; border-color: #38bdf8; }
    .feat-item input.highlighted { border-color: #f59e0b; background: #1c1708; }
    .btn-predict {
      width: 100%; padding: 14px; background: #0ea5e9;
      border: none; border-radius: 8px; color: white;
      font-size: 1rem; font-weight: 700; cursor: pointer; transition: background 0.2s;
    }
    .btn-predict:hover    { background: #0284c7; }
    .btn-predict:disabled { background: #334155; cursor: not-allowed; }
    #result { display: none; }
    .result-top { display: flex; align-items: center; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }
    .badge { display: inline-block; padding: 10px 28px; border-radius: 20px; font-weight: 800; font-size: 1.3rem; }
    .badge.attack { background: #dc2626; }
    .badge.benign { background: #16a34a; }
    .prob-txt { color: #94a3b8; font-size: 0.9rem; line-height: 1.8; }
    .prob-txt strong { color: #e2e8f0; }
    .grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
    .vote-card { background: #0f172a; border-radius: 10px; padding: 16px; border: 1px solid #334155; text-align: center; }
    .vote-card .model-name { font-size: 0.8rem; color: #94a3b8; margin-bottom: 8px; text-transform: uppercase; }
    .vote-card .vote-badge { display: inline-block; padding: 5px 16px; border-radius: 12px; font-weight: 700; font-size: 0.9rem; }
    .vote-card .vote-badge.attack { background: #7f1d1d; color: #fca5a5; }
    .vote-card .vote-badge.benign { background: #14532d; color: #86efac; }
    .vote-card .vote-probs { font-size: 0.78rem; color: #64748b; margin-top: 6px; }
    .shap-panel { background: #0f172a; border-radius: 10px; padding: 18px; border: 1px solid #334155; }
    .shap-panel h3 { font-size: 0.85rem; color: #94a3b8; margin-bottom: 14px; text-transform: uppercase; }
    .bar-wrap { margin: 7px 0; }
    .bar-label { font-size: 0.73rem; color: #94a3b8; margin-bottom: 2px; display: flex; justify-content: space-between; }
    .bar-outer { background: #1e293b; border-radius: 4px; height: 15px; }
    .bar-inner { height: 15px; border-radius: 4px; transition: width 0.5s; min-width: 3px; }
    .bar-pos { background: linear-gradient(90deg,#f87171,#dc2626); }
    .bar-neg { background: linear-gradient(90deg,#4ade80,#16a34a); }
    .legend { display: flex; gap: 16px; margin-bottom: 14px; font-size: 0.78rem; }
    .legend-dot { width: 10px; height: 10px; border-radius: 2px; display: inline-block; margin-right: 5px; }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .img-card { background: #0f172a; border-radius: 10px; padding: 14px; border: 1px solid #334155; text-align: center; }
    .img-card h3 { color: #94a3b8; font-size: 0.78rem; margin-bottom: 10px; text-transform: uppercase; }
    .img-card img { width: 100%; border-radius: 6px; }
    .img-card .no-img { color: #475569; font-size: 0.82rem; padding: 30px 0; }
    .info-box { background: #0f172a; border-left: 3px solid #38bdf8; padding: 11px 15px; border-radius: 6px; margin-bottom: 18px; font-size: 0.85rem; color: #94a3b8; line-height: 1.6; }
    .info-box strong { color: #38bdf8; }
    hr { border: none; border-top: 1px solid #334155; margin: 18px 0; }
    .spinner { display:none; text-align:center; padding:10px; color:#38bdf8; font-size:0.9rem; }
    .error-box { background:#7f1d1d; color:#fca5a5; padding:12px 16px; border-radius:8px; margin-top:12px; font-size:0.85rem; display:none; }
    @media(max-width:900px){ .feature-grid { grid-template-columns: repeat(2,1fr); } }
    @media(max-width:600px){ .feature-grid { grid-template-columns: 1fr; } .grid3,.grid2 { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<header>
  <div style="font-size:2rem">🛡️</div>
  <div>
    <h1>XAI Intrusion Detection System</h1>
    <p>Ensemble ML: Random Forest + XGBoost + LightGBM &nbsp;|&nbsp; Explainability: SHAP + LIME</p>
  </div>
</header>

<div class="container">
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('predict',this)">🔍 Predict & Explain</button>
    <button class="tab-btn" onclick="switchTab('lime',this)">🧪 LIME Analysis</button>
    <button class="tab-btn" onclick="switchTab('shap-global',this)">📊 Global SHAP</button>
    <button class="tab-btn" onclick="switchTab('comparison',this)">⚖️ Model Comparison</button>
  </div>

  <!-- TAB 1: PREDICT -->
  <div id="tab-predict" class="tab-content active">
    <div class="card">
      <h2>⚙️ Network Flow Features — All {{ n_features }} Features</h2>
      <div class="info-box">
        <strong>Quick Demo:</strong> Click a button below to auto-fill verified sample values,
        then click Analyse Traffic.
        All 78 features are shown — unspecified ones use dataset mean as default.
      </div>
      <div class="demo-btns">
  <button class="demo-btn attack" onclick="fillSample('attack')">🔴 Generate Attack Traffic</button>
  <button class="demo-btn benign" onclick="fillSample('benign')">🟢 Generate Benign Traffic</button>
  <button class="demo-btn reset"  onclick="location.reload()">🔄 New Random Sample</button>
    </div>
      <div class="feature-grid" id="feature-inputs"></div>
      <button class="btn-predict" id="predict-btn" onclick="predict()">🔍 Analyse Traffic</button>
      <div class="spinner" id="spinner">⏳ Running ensemble + SHAP for all 3 models...</div>
      <div class="error-box" id="error-box"></div>
    </div>

    <div id="result">
      <div class="card">
        <h2>🎯 Ensemble Prediction</h2>
        <div class="result-top">
          <span id="pred-badge" class="badge"></span>
          <div class="prob-txt">
            Confidence: <strong id="confidence"></strong><br>
            Benign: <strong id="prob-benign"></strong> &nbsp;|&nbsp;
            Attack: <strong id="prob-attack"></strong>
          </div>
        </div>
        <hr>
        <h2>🗳️ Individual Model Votes</h2>
        <div class="grid3" id="vote-cards"></div>
      </div>

      <div class="card">
        <h2>🔬 Live SHAP Explanations — All 3 Models</h2>
        <div class="info-box">
          <strong>How to read:</strong> Each panel shows how that model reached its decision.
          🔴 Red = pushes toward ATTACK &nbsp;|&nbsp; 🟢 Green = pushes toward BENIGN.
          Longer bar = stronger influence. Top 8 features shown per model.
        </div>
        <div class="legend">
          <span><span class="legend-dot" style="background:#dc2626"></span>→ ATTACK</span>
          <span><span class="legend-dot" style="background:#16a34a"></span>→ BENIGN</span>
        </div>
        <div class="grid3" id="shap-panels"></div>
      </div>
    </div>
  </div>

  <!-- TAB 2: LIME -->
  <div id="tab-lime" class="tab-content">
    <div class="card">
      <h2>🧪 LIME — Local Interpretable Model-agnostic Explanations</h2>
      <div class="info-box">
        <strong>What is LIME?</strong> LIME explains individual predictions by perturbing input
        features and observing how the ensemble output changes. It builds a simple linear model
        locally around each prediction to show which features drove the decision.
        Pre-generated on 3 real test samples from CIC-IDS-2017.
      </div>
      <div class="grid2" id="lime-images"></div>
    </div>
  </div>

  <!-- TAB 3: GLOBAL SHAP -->
  <div id="tab-shap-global" class="tab-content">
    <div class="card">
      <h2>📊 Global SHAP — All 3 Models</h2>
      <div class="info-box">
        <strong>Global vs Local:</strong> These plots show which features matter most
        across the entire test set. Beeswarm = importance + direction.
        Bar = mean absolute SHAP value.
      </div>
      <div class="grid2" id="shap-global-images"></div>
    </div>
  </div>

  <!-- TAB 4: COMPARISON -->
  <div id="tab-comparison" class="tab-content">
    <div class="card">
      <h2>⚖️ Cross-Model SHAP Feature Importance</h2>
      <div class="info-box">
        <strong>Why compare?</strong> Features important across all 3 models are most trustworthy.
        Disagreements reveal model-specific behaviour.
      </div>
      <div id="comparison-images"></div>
    </div>
  </div>
</div>

<script>
const features     = {{ features | tojson }};
const defaults     = {{ defaults | tojson }};
const attackSample = {{ attack_sample | tojson }};
const benignSample = {{ benign_sample | tojson }};

// Build feature grid
const inputArea = document.getElementById('feature-inputs');
features.forEach(f => {
  const safe = f.replace(/[^a-zA-Z0-9]/g,'_');
  const val  = (defaults[f] !== undefined ? defaults[f] : 0).toFixed(4);
  inputArea.innerHTML += `<div class="feat-item">
    <label title="${f}">${f}</label>
    <input type="number" id="f_${safe}" value="${val}" step="any">
  </div>`;
});

function switchTab(name, btn) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  btn.classList.add('active');
}

function fillSample(type) {
  const sample = type === 'attack' ? attackSample : type === 'benign' ? benignSample : null;
  features.forEach(f => {
    const safe  = f.replace(/[^a-zA-Z0-9]/g,'_');
    const input = document.getElementById('f_'+safe);
    if (!input) return;
    if (sample && sample[f] !== undefined) {
      input.value = sample[f];
      input.classList.add('highlighted');
    } else {
      input.value = (defaults[f] !== undefined ? defaults[f] : 0).toFixed(4);
      input.classList.remove('highlighted');
    }
  });
}

function renderBars(containerId, shapData) {
  const div = document.getElementById(containerId);
  if (!div) return;
  const maxVal = Math.max(...shapData.map(s => Math.abs(s.value)), 0.0001);
  div.innerHTML = shapData.map(s => {
    const pct = (Math.abs(s.value)/maxVal*100).toFixed(1);
    const cls = s.value > 0 ? 'bar-pos' : 'bar-neg';
    const dir = s.value > 0 ? '→ATK' : '→BEN';
    const col = s.value > 0 ? '#f87171' : '#4ade80';
    return `<div class="bar-wrap">
      <div class="bar-label">
        <span>${s.feature}</span>
        <span style="color:${col}">${dir} (${s.value.toFixed(3)})</span>
      </div>
      <div class="bar-outer">
        <div class="bar-inner ${cls}" style="width:${pct}%"></div>
      </div>
    </div>`;
  }).join('');
}

async function predict() {
  const btn    = document.getElementById('predict-btn');
  const spinner= document.getElementById('spinner');
  const errBox = document.getElementById('error-box');
  btn.disabled = true; spinner.style.display = 'block';
  errBox.style.display = 'none';
  document.getElementById('result').style.display = 'none';

  const vals = {};
  features.forEach(f => {
    const safe = f.replace(/[^a-zA-Z0-9]/g,'_');
    const v = parseFloat(document.getElementById('f_'+safe)?.value);
    vals[f] = isNaN(v) ? (defaults[f] || 0) : v;
  });

  try {
    const res  = await fetch('/predict', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(vals)
    });
    const text = await res.text();
    let data;
    try { data = JSON.parse(text); }
    catch(e) {
      errBox.textContent = 'Server error — check terminal.';
      errBox.style.display = 'block';
      btn.disabled = false; spinner.style.display = 'none';
      return;
    }
    if (data.error) {
      errBox.textContent = 'Error: ' + data.error;
      errBox.style.display = 'block';
      btn.disabled = false; spinner.style.display = 'none';
      return;
    }

    document.getElementById('result').style.display = 'block';
    const badge = document.getElementById('pred-badge');
    badge.textContent = data.prediction;
    badge.className   = 'badge ' + (data.prediction==='ATTACK'?'attack':'benign');
    document.getElementById('confidence').textContent  = (data.confidence*100).toFixed(1)+'%';
    document.getElementById('prob-benign').textContent = (data.prob_benign*100).toFixed(1)+'%';
    document.getElementById('prob-attack').textContent = (data.prob_attack*100).toFixed(1)+'%';

    // Votes
    const voteDiv = document.getElementById('vote-cards');
    voteDiv.innerHTML = data.votes.map(v => `
      <div class="vote-card">
        <div class="model-name">${v.model}</div>
        <div class="vote-badge ${v.prediction==='ATTACK'?'attack':'benign'}">${v.prediction}</div>
        <div class="vote-probs">Benign: ${(v.prob_benign*100).toFixed(1)}% | Attack: ${(v.prob_attack*100).toFixed(1)}%</div>
      </div>`).join('');

    // SHAP
    const shapDiv = document.getElementById('shap-panels');
    shapDiv.innerHTML = data.shap_all.map(m => {
      const pid = 'shap_'+m.model.replace(/[^a-zA-Z0-9]/g,'_');
      return `<div class="shap-panel"><h3>${m.model}</h3><div id="${pid}"></div></div>`;
    }).join('');
    setTimeout(() => {
      data.shap_all.forEach(m => {
        renderBars('shap_'+m.model.replace(/[^a-zA-Z0-9]/g,'_'), m.shap);
      });
    }, 50);

  } catch(e) {
    errBox.textContent = 'Network error: '+e.message;
    errBox.style.display = 'block';
  }
  btn.disabled = false; spinner.style.display = 'none';
}

// LIME
const limeDiv = document.getElementById('lime-images');
{{ lime_images | tojson }}.forEach((b64,i) => {
  limeDiv.innerHTML += b64
    ? `<div class="img-card"><h3>LIME — Instance ${i+1}</h3><img src="data:image/png;base64,${b64}"></div>`
    : `<div class="img-card"><h3>LIME — Instance ${i+1}</h3><div class="no-img">Run 5_explain_lime.py first</div></div>`;
});

// Global SHAP
const shapGlobalDiv = document.getElementById('shap-global-images');
const globalImgs    = {{ shap_global_images | tojson }};
const globalTitles  = {{ shap_global_titles | tojson }};
globalImgs.forEach((b64,i) => {
  shapGlobalDiv.innerHTML += b64
    ? `<div class="img-card"><h3>${globalTitles[i]}</h3><img src="data:image/png;base64,${b64}"></div>`
    : `<div class="img-card"><h3>${globalTitles[i]}</h3><div class="no-img">Run 4_explain_shap.py first</div></div>`;
});

// Comparison
const compImg = {{ comparison_image | tojson }};
document.getElementById('comparison-images').innerHTML = compImg
  ? `<img src="data:image/png;base64,${compImg}" style="width:100%;border-radius:8px">`
  : `<div style="color:#475569;padding:40px;text-align:center">Run 4_explain_shap.py first</div>`;
</script>
</body>
</html>
"""

@app.route("/")
def index():
    lime_images = [img_to_base64(f"{PLOT_DIR}lime_explanation_{i}.png") for i in range(3)]
    shap_global_images = [
        img_to_base64(f"{PLOT_DIR}shap_rf_summary.png"),
        img_to_base64(f"{PLOT_DIR}shap_rf_importance.png"),
        img_to_base64(f"{PLOT_DIR}shap_xgb_summary.png"),
        img_to_base64(f"{PLOT_DIR}shap_xgb_importance.png"),
        img_to_base64(f"{PLOT_DIR}shap_lgbm_summary.png"),
        img_to_base64(f"{PLOT_DIR}shap_lgbm_importance.png"),
        img_to_base64(f"{PLOT_DIR}shap_rf_waterfall.png"),
        img_to_base64(f"{PLOT_DIR}shap_xgb_waterfall.png"),
        img_to_base64(f"{PLOT_DIR}shap_lgbm_waterfall.png"),
    ]
    shap_global_titles = [
        "RF — SHAP Summary",       "RF — Feature Importance",
        "XGBoost — SHAP Summary",  "XGBoost — Feature Importance",
        "LightGBM — SHAP Summary", "LightGBM — Feature Importance",
        "RF — Waterfall",          "XGBoost — Waterfall",
        "LightGBM — Waterfall",
    ]
    return render_template_string(
        HTML,
        features=feature_names,
        n_features=len(feature_names),
        defaults=feature_defaults,
        attack_sample=make_random_attack(),   # fresh random every load
        benign_sample=make_random_benign(),   # fresh random every load
        lime_images=lime_images,
        shap_global_images=shap_global_images,
        shap_global_titles=shap_global_titles,
        comparison_image=img_to_base64(f"{PLOT_DIR}shap_comparison_all_models.png"),
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        # Build full feature vector with all 78 features
        raw  = np.array([data.get(f, feature_defaults[f]) for f in feature_names]).reshape(1, -1)
        vals = scaler.transform(raw)

        pred  = ensemble.predict(vals)[0]
        proba = ensemble.predict_proba(vals)[0]

        models_info = [
            ("Random Forest", rf_model,   shap_rf),
            ("XGBoost",       xgb_model,  shap_xgb),
            ("LightGBM",      lgbm_model, shap_lgbm),
        ]

        votes    = []
        shap_all = []

        for name, model, explainer in models_info:
            p  = model.predict(vals)[0]
            pr = model.predict_proba(vals)[0]
            votes.append({
                "model":       name,
                "prediction":  "ATTACK" if p == 1 else "BENIGN",
                "prob_benign": float(pr[0]),
                "prob_attack": float(pr[1]),
            })
            sv = extract_shap_1d(explainer, vals)
            n  = min(len(feature_names), len(sv))
            pairs = sorted(
                [{"feature": feature_names[i], "value": float(sv[i])} for i in range(n)],
                key=lambda x: abs(x["value"]), reverse=True
            )[:8]
            shap_all.append({"model": name, "shap": pairs})

        return jsonify({
            "prediction":  "ATTACK" if pred == 1 else "BENIGN",
            "confidence":  float(proba[pred]),
            "prob_benign": float(proba[0]),
            "prob_attack": float(proba[1]),
            "votes":       votes,
            "shap_all":    shap_all,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("[✓] Dashboard at http://127.0.0.1:5000")
    app.run(debug=True)