from flask import Flask, render_template_string, request, jsonify
import joblib
import numpy as np
import pandas as pd
import shap
import random
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

# Pre-load real samples at startup for realistic demo
print("[*] Loading real samples for demo...")
X_test_raw = joblib.load(f"{OUTPUT_DIR}X_test.pkl").reset_index(drop=True)
y_test_raw = joblib.load(f"{OUTPUT_DIR}y_test.pkl").reset_index(drop=True)
X_test_original = pd.DataFrame(
    scaler.inverse_transform(X_test_raw),
    columns=feature_names
)

# Get correctly classified attacks and benigns at 75-95% confidence
y_pred_all  = ensemble.predict(X_test_raw)
y_proba_all = ensemble.predict_proba(X_test_raw)[:, 1]

# Attacks: correctly predicted, confidence between 80-99%
attack_mask = (
    (y_test_raw == 1) & 
    (y_pred_all == 1) & 
    (y_proba_all >= 0.80) & 
    (y_proba_all <= 0.99)
)
attack_pool = X_test_original[attack_mask].reset_index(drop=True)

# Benigns: correctly predicted, confidence between 80-99%
benign_mask = (
    (y_test_raw == 0) & 
    (y_pred_all == 0) & 
    (y_proba_all <= 0.20) & 
    (y_proba_all >= 0.01)
)
benign_pool = X_test_original[benign_mask].reset_index(drop=True)

print(f"[✓] Attack pool: {len(attack_pool)} samples | Benign pool: {len(benign_pool)} samples")

def make_random_attack():
    """Pick a real attack sample randomly from the pool."""
    row = attack_pool.sample(1, random_state=random.randint(0, 99999)).iloc[0]
    return {f: float(row[f]) for f in feature_names}

def make_random_benign():
    """Pick a real benign sample randomly from the pool."""
    row = benign_pool.sample(1, random_state=random.randint(0, 99999)).iloc[0]
    return {f: float(row[f]) for f in feature_names}

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>XAI Intrusion Detection System</title>

  <style>
    * { transition: all 0.25s ease-in-out;box-sizing: border-box; margin: 0; padding: 0; }
    body {
  font-family: 'Segoe UI', sans-serif;
  background: #f5efe6;   /* soft beige */
  color: #2c2c2c;        /* charcoal text */
  min-height: 100vh;
}
 header {
  background: #f5efe6;
  padding: 28px 40px;
  border-bottom: 1px solid rgba(0,0,0,0.08);
  display: flex;
  align-items: center;
  gap: 16px;
}

header div {
  display: flex;
  flex-direction: column;
  justify-content: center;
}

    header h1 { font-size: 1.7rem; color: #1a1a1a; }
    header p  { color: #6b6b6b; font-size: 0.85rem; margin-top: 3px; }
    .container {
  max-width: 1100px;          /* slightly tighter = more premium */
  margin: 60px auto;          /* more top-bottom space */
  padding: 0 32px;            /* more side breathing */
}
    .tabs {
  display: flex;
  gap: 12px;                  /* more spacing */
  margin-bottom: 36px;        /* more separation */
  flex-wrap: wrap;
}
    .tab-btn {
  padding: 10px 20px;
  border-radius: 999px;
  border: 1px solid rgba(0,0,0,0.1);
  background: transparent;
  color: #555;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 600;
  transition: all 0.25s ease;
}
    .tab-btn:hover  { background: #334155; color: #e2e8f0;transform: scale(1.05); }
    .tab-btn.active {
  background: linear-gradient(135deg, #c8a96a, #a3844f);
  color: #fff;
  border: none;
  box-shadow: 0 6px 20px rgba(200,169,106,0.3);
}
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    .card {
  background: #ffffff;
  border-radius: 16px;
  padding: 34px;
  margin-bottom: 32px;
  border: 1px solid rgba(0,0,0,0.05);
  box-shadow: 0 10px 30px rgba(0,0,0,0.06);
  transition: all 0.3s ease;
}
.card:hover {
  transform: translateY(-6px);
  box-shadow: 0 20px 50px rgba(0,0,0,0.12);
}
.card > *:not(:last-child) {
  margin-bottom: 14px;
}
    .card h2 {
  color: #2a2a2a;                 /* deep charcoal */
  font-size: 1.2rem;              /* stronger presence */
  font-weight: 600;
  margin-bottom: 18px;
  letter-spacing: 0.4px;
  text-transform: none;           /* remove uppercase */
}
    .demo-btns {
  display: flex;
  gap: 14px;
  margin-bottom: 24px;
}
    .demo-btn {
      padding: 8px 20px; border-radius: 8px; border: none;
      font-weight: 700; cursor: pointer; font-size: 0.88rem; transition: all 0.25s ease;
    }
    .demo-btn:hover { opacity: 0.85;transform: scale(1.05); }
    .demo-btn.attack {
  background: #3e2f24;   /* deep brown */
  color: white;
}
    .demo-btn.benign {
  background: #6b8e23;   /* muted green */
  color: white;
}
    .demo-btn.reset {
  background: transparent;
  border: 1px solid #ccc;
  color: #555;
}
    .feature-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;                  /* increased */
  max-height: 420px;
  overflow-y: auto;
  padding-right: 8px;
  margin-bottom: 22px;
}
    .feature-grid::-webkit-scrollbar { width: 6px; }
    .feature-grid::-webkit-scrollbar-track { background: #0f172a; }
    .feature-grid::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    .feat-item label { display: block; margin-bottom: 3px; color: #94a3b8; font-size: 0.72rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .feat-item input {
  background: #fafafa;
  border: 1px solid #ddd;
  color: #333;
}
    .feat-item input:focus {
  outline: none;
  border-color: #c8a96a;
  box-shadow: 0 0 0 3px rgba(200,169,106,0.2);
}
.feat-item input:hover {
  border-color: #c8a96a;
}
    .feat-item input.highlighted {
  border-color: #c8a96a;                    /* gold accent */
  background: rgba(200, 169, 106, 0.12);    /* soft gold tint */
  color: #2c2c2c;
}
    .btn-predict {
  width: 100%;
  padding: 14px;
  background: linear-gradient(135deg, #3e2f24, #1a1a1a);
  border: none;
  border-radius: 999px;
  color: white;
  font-size: 1rem;
  font-weight: 700;
}
    .btn-predict:hover    { background: #0284c7; }
    .btn-predict:disabled { background: #334155; cursor: not-allowed; }
    #result { display: none; }
    .result-top {
  display: flex;
  align-items: center;
  gap: 28px;                  /* more breathing */
  flex-wrap: wrap;
  margin-bottom: 26px;
}
    .badge { display: inline-block; padding: 10px 28px; border-radius: 20px; font-weight: 800; font-size: 1.3rem; }
    .badge.attack {
  background: #2a1a1a;
  color: #f5c2c2;
}
    .badge.benign {
  background: #1f2a1f;
  color: #c2f5c2;
}
    .prob-txt { color: #94a3b8; font-size: 0.9rem; line-height: 1.8; }
    .prob-txt strong { color: #e2e8f0; }
    .grid3 {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 22px;                  /* more space between cards */
}
    .vote-card {
  background: #ffffff;
  border-radius: 14px;
  padding: 16px;
  border: 1px solid rgba(0,0,0,0.05);
  text-align: center;
  transition: all 0.25s ease;
}
    .vote-card .model-name { font-size: 0.8rem; color: #94a3b8; margin-bottom: 8px; text-transform: uppercase; }
    .vote-card .vote-badge { display: inline-block; padding: 5px 16px; border-radius: 12px; font-weight: 700; font-size: 0.9rem; }
    .vote-card .vote-badge.attack { background: #7f1d1d; color: #fca5a5; }
    .vote-card .vote-badge.benign { background: #14532d; color: #86efac; }
    .vote-card .vote-probs { font-size: 0.78rem; color: #64748b; margin-top: 6px; }
    .shap-panel {
  background: #ffffff;
  border-radius: 14px;
  padding: 18px;
  border: 1px solid rgba(0,0,0,0.05);
  transition: all 0.25s ease;
}

.vote-card:hover,
.shap-panel:hover,
.img-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 30px rgba(0,0,0,0.08);
}


    .shap-panel h3 { font-size: 0.85rem; color: #94a3b8; margin-bottom: 14px; text-transform: uppercase; }
    .bar-wrap { margin: 7px 0; }
    .bar-label { font-size: 0.73rem; color: #94a3b8; margin-bottom: 2px; display: flex; justify-content: space-between; }
    .bar-outer { background: #1e293b; border-radius: 4px; height: 15px; }
    .bar-inner { height: 15px; border-radius: 4px; transition: width 0.5s; min-width: 3px; }
    .bar-pos { background: linear-gradient(90deg,#f87171,#dc2626); }
    .bar-neg { background: linear-gradient(90deg,#4ade80,#16a34a); }
    .legend { display: flex; gap: 16px; margin-bottom: 14px; font-size: 0.78rem; }
    .legend-dot { width: 10px; height: 10px; border-radius: 2px; display: inline-block; margin-right: 5px; }
    .grid2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 22px;
}
    .img-card {
  background: #ffffff;
  border-radius: 14px;
  padding: 14px;
  border: 1px solid rgba(0,0,0,0.05);
  text-align: center;
  transition: all 0.25s ease;
}
    .img-card h3 { color: #94a3b8; font-size: 0.78rem; margin-bottom: 10px; text-transform: uppercase; }
    .img-card img { width: 100%; border-radius: 6px; }
    .img-card .no-img { color: #475569; font-size: 0.82rem; padding: 30px 0; }
    .info-box {
  background: #faf6f0;
  border-left: 3px solid #c8a96a;
  padding: 14px 18px;
  border-radius: 8px;
  margin-bottom: 22px;
  font-size: 0.85rem;
  color: #5a5a5a;
}
    .info-box strong { color: #38bdf8; }
    hr { border: none; border-top: 1px solid #334155; margin: 18px 0; }
    .spinner { display:none; text-align:center; padding:10px; color:#38bdf8; font-size:0.9rem; }
    .error-box { background:#7f1d1d; color:#fca5a5; padding:12px 16px; border-radius:8px; margin-top:12px; font-size:0.85rem; display:none; }
    @media(max-width:900px){ .feature-grid { grid-template-columns: repeat(2,1fr); } }
    @media(max-width:600px){ .feature-grid { grid-template-columns: 1fr; } .grid3,.grid2 { grid-template-columns:1fr; } }


/* ===== COLLAPSIBLE ===== */
.collapsible-header {
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
  padding: 12px 0;
  color: #2a2a2a;
}

.collapsible-header span {
  font-size: 0.9rem;
  color: #888;
}

.collapsible-content {
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.35s ease;
}

.collapsible.open .collapsible-content {
  max-height: 1000px;
}

.collapsible-header:hover {
  color: #c8a96a;
}


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
  <button class="demo-btn reset"  onclick="randomSample()">🔄 New Random Sample</button>
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
        <div class="collapsible open">
  <div class="collapsible-header" onclick="toggleCollapse(this)">
    🗳️ Individual Model Votes
    <span>▼</span>
  </div>

  <div class="collapsible-content">
    <div class="grid3" id="vote-cards"></div>
  </div>
</div>
      </div>

      <div class="card">
        <div class="collapsible">
  <div class="collapsible-header" onclick="toggleCollapse(this)">
    🔬 Live SHAP Explanations — All 3 Models
    <span>▼</span>
  </div>

  <div class="collapsible-content">

    <div class="info-box">
      <strong>How to read:</strong> Each panel shows how that model reached its decision.
      🔴 Red = pushes toward ATTACK &nbsp;|&nbsp; 🟢 Green = pushes toward BENIGN.
    </div>

    <div class="legend">
      <span><span class="legend-dot" style="background:#dc2626"></span>→ ATTACK</span>
      <span><span class="legend-dot" style="background:#16a34a"></span>→ BENIGN</span>
    </div>

    <div class="grid3" id="shap-panels"></div>

  </div>
</div>
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

async function fillSample(type) {

  const endpoint =
    type === 'attack'
      ? '/random_attack'
      : '/random_benign';

  const res = await fetch(endpoint);
  const sample = await res.json();

  features.forEach(f => {
    const safe  = f.replace(/[^a-zA-Z0-9]/g,'_');
    const input = document.getElementById('f_'+safe);

    if (!input) return;

    if (sample[f] !== undefined) {
      input.value = sample[f];
      input.classList.add('highlighted');
    }
  });
}


async function randomSample() {

  const endpoint =
    Math.random() < 0.5
      ? '/random_attack'
      : '/random_benign';

  const res = await fetch(endpoint);
  const sample = await res.json();

  features.forEach(f => {
    const safe  = f.replace(/[^a-zA-Z0-9]/g,'_');
    const input = document.getElementById('f_'+safe);

    if (!input) return;

    if (sample[f] !== undefined) {
      input.value = sample[f];
      input.classList.add('highlighted');
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

function toggleCollapse(el) {
  const parent = el.parentElement;
  parent.classList.toggle('open');

  const arrow = el.querySelector('span');
  arrow.textContent = parent.classList.contains('open') ? '▼' : '▶';
}


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


@app.route("/random_attack")
def random_attack():
    return jsonify(make_random_attack())

@app.route("/random_benign")
def random_benign():
    return jsonify(make_random_benign())



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



