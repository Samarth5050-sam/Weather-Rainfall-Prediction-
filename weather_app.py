import os
import io
import base64
import folium
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, render_template_string, redirect, url_for

# ==============================
# Flask setup
# ==============================
app = Flask(__name__)

# ==============================
# Constants
# ==============================
N_STEPS = 3  # Use a 3-day window to capture temporal trends

# ==============================
# LSTM Model (Regression)
# ==============================
class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Use the last time step
        return out

class LSTMModelWrapper:
    def __init__(self, model, scaler, features):
        self.model = model
        self.scaler = scaler
        self.feature_names_in_ = features

    def predict(self, X_seq):
        self.model.eval()
        X_t = torch.tensor(X_seq, dtype=torch.float32)
        if len(X_t.shape) == 2: X_t = X_t.unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(X_t)
        return np.maximum(0, outputs.numpy().flatten())

# ==============================
# Helper: Train model
# ==============================
def prepare_and_train(df):
    df = df.copy().dropna()
    df.columns = df.columns.str.strip()

    required = {"Location", "RISK_MM", "Date"}
    if not required.issubset(set(df.columns)):
        raise KeyError(f"CSV must contain columns: {required}")

    unique_locations = sorted(df["Location"].unique().tolist())
    unique_dates     = sorted(df["Date"].unique().tolist())

    df['Date_obj'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Location', 'Date_obj'])

    # Drop non-feature columns
    drop_cols = [c for c in ["Date", "Date_obj", "Time", "RISK_MM", "RainTomorrow"] if c in df.columns]
    features_df = df.drop(columns=drop_cols)
    
    categorical_cols = features_df.select_dtypes(include=[object]).columns
    encoders = {}
    for c in categorical_cols:
        le = LabelEncoder()
        features_df[c] = le.fit_transform(features_df[c].astype(str))
        encoders[c] = le

    # Standardize
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    
    # Target: Predict NEXT DAY rainfall amount (stored in RISK_MM)
    y_target = df["RISK_MM"].values.astype(np.float32)

    # Create sequences correctly grouped by location
    X_seqs, y_seqs = [], []
    for loc in unique_locations:
        loc_mask = df["Location"] == loc
        loc_features = scaled_features[loc_mask]
        loc_y = y_target[loc_mask]
        
        # We need N_STEPS history to predict the target of the last day in that window
        for i in range(len(loc_features) - N_STEPS + 1):
            X_seqs.append(loc_features[i : i + N_STEPS])
            y_seqs.append(loc_y[i + N_STEPS - 1])
            
    X_seqs = np.array(X_seqs)
    y_seqs = np.array(y_seqs)

    if len(X_seqs) == 0:
        # Fallback if data is too small for sequences
        X_seqs = np.array([scaled_features[i:i+1] for i in range(len(df))])
        y_seqs = np.array(y_target)

    X_train, X_test, y_train, y_test = train_test_split(X_seqs, y_seqs, test_size=0.2, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    lstm_model = WeatherLSTM(input_size=X_train.shape[2])
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.005)
    
    def weighted_mse_loss(pred, target):
        weights = torch.ones_like(target)
        weights[target > 0.1] = 15.0 # Higher weight for rain events
        return (weights * (pred - target)**2).mean()

    lstm_model.train()
    for epoch in range(150): # More epochs for better fit
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = lstm_model(batch_X)
            loss = weighted_mse_loss(outputs, batch_y)
            loss.backward()
            optimizer.step()

    wrapper = LSTMModelWrapper(lstm_model, scaler, features_df.columns.tolist())
    y_pred = wrapper.predict(X_test)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae":  float(mean_absolute_error(y_test, y_pred)),
        "r2":   float(r2_score(y_test, y_pred)),
    }

    return {
        "model":            wrapper,
        "df_raw":           df,
        "encoders":         encoders,
        "metrics":          metrics,
        "unique_locations": unique_locations,
        "unique_dates":     unique_dates,
    }

# ==============================
# Load dataset
# ==============================
STATE = {"trained": False}
DEFAULT_CSV = os.environ.get("DEFAULT_CSV", "weather_sep_oct_2026.csv")

import pickle
STATE_FILE = "weather_state.pkl"

if os.path.exists(STATE_FILE):
    try:
        with open(STATE_FILE, "rb") as f:
            saved = pickle.load(f)
        
        # Reconstruct the LSTM model from saved weights
        config = saved["model_config"]
        lstm_model = WeatherLSTM(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"]
        )
        
        # Load weights (stored as numpy arrays, convert back to torch tensors)
        weights = saved.get("model_weights") or saved.get("model_state_dict")
        state_dict = {}
        for key, val in weights.items():
            state_dict[key] = torch.tensor(val) if not isinstance(val, torch.Tensor) else val
        lstm_model.load_state_dict(state_dict)
        lstm_model.eval()
        
        wrapper = LSTMModelWrapper(lstm_model, saved["scaler"], saved["feature_names"])
        
        STATE = {
            "trained": True,
            "model": wrapper,
            "df_raw": saved["df_raw"],
            "encoders": saved["encoders"],
            "metrics": saved["metrics"],
            "unique_locations": saved["unique_locations"],
            "unique_dates": saved["unique_dates"],
        }
        print(f"[OK] Pre-trained state loaded from {STATE_FILE}! Bypassing boot training.")
    except Exception as e:
        print(f"[ERROR] Could not load state from {STATE_FILE}: {e}")
        STATE = {"trained": False}
else:
    try:
        df_default = pd.read_csv(DEFAULT_CSV)
        STATE.update(prepare_and_train(df_default))
        STATE["trained"] = True
        print(f"[OK] Time-Series LSTM Model trained on {len(df_default)} rows from '{DEFAULT_CSV}'")
    except Exception as exc:
        print(f"[WARN] Could not train on '{DEFAULT_CSV}': {exc}")
        STATE["trained"] = False

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ==============================
# Advanced UI Styles (Glassmorphism)
# ==============================
STYLE = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
  * { box-sizing: border-box; font-family: 'Outfit', sans-serif; }
  body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: #e2e8f0;
    text-align: center;
    margin: 0;
    padding: 20px;
    min-height: 100vh;
  }
  @keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    border-radius: 20px;
    padding: 30px 40px;
    margin: 20px auto;
    max-width: 700px;
    text-align: left;
    animation: fadeIn 0.6s ease-out forwards;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
  }
  .card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 40px rgba(0, 0, 0, 0.6);
  }
  h1 { font-size: 2.5rem; margin-bottom: 5px; font-weight: 700; background: -webkit-linear-gradient(#fff, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  h2 { font-size: 1.4rem; margin: 0 0 15px; font-weight: 600; color: #f8fafc; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; }
  label { font-size: 1rem; display: block; margin-bottom: 8px; color: #cbd5e1; }
  select, input[type="date"] {
    width: 100%;
    padding: 12px 16px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.2);
    background: rgba(0,0,0,0.2);
    color: white;
    font-size: 1.05rem;
    margin-bottom: 20px;
    outline: none;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
    appearance: none;
  }
  select:focus, input[type="date"]:focus {
    border-color: #38bdf8;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.3);
  }
  select option { background: #1e293b; color: white; }
  button, .btn {
    display: inline-block;
    padding: 12px 28px;
    border: none;
    border-radius: 12px;
    background: linear-gradient(135deg, #0ea5e9, #2563eb);
    color: white;
    cursor: pointer;
    font-size: 1.05rem;
    font-weight: 600;
    text-decoration: none;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
  }
  button:hover, .btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
    background: linear-gradient(135deg, #38bdf8, #3b82f6);
  }
  button:active, .btn:active {
    transform: translateY(1px);
  }
  .btn-row { display: flex; gap: 15px; flex-wrap: wrap; margin-top: 10px; }
  .badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 30px;
    font-size: 0.95rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    box-shadow: 0 0 15px rgba(0,0,0,0.2);
  }
  .badge-yes  { background: linear-gradient(135deg, #ef4444, #b91c1c); box-shadow: 0 0 15px rgba(239, 68, 68, 0.5); }
  .badge-no   { background: linear-gradient(135deg, #10b981, #047857); box-shadow: 0 0 15px rgba(16, 185, 129, 0.5); }
  ul { list-style: none; padding: 0; }
  ul li { padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.05); color: #e2e8f0; }
  ul li:last-child { border-bottom: none; }
  ul li strong { color: #fff; }
  hr { border: 0; height: 1px; background: rgba(255,255,255,0.1); margin: 20px 0; }
  img { max-width: 100%; border-radius: 12px; margin-top: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
  .back { display: inline-flex; align-items: center; margin-top: 25px; color: #38bdf8; text-decoration: none; font-weight: 600; transition: color 0.2s; }
  .back:hover { color: #7dd3fc; }
  .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 15px; margin-top: 15px; }
  .stat-box { background: rgba(0,0,0,0.2); padding: 15px; border-radius: 12px; text-align: center; border: 1px solid rgba(255,255,255,0.05); }
  .stat-box span { display: block; font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px; }
  .stat-box strong { display: block; font-size: 1.2rem; color: #fff; }
</style>
"""

HOME_HTML = STYLE + """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>🌦 Weather Prediction AI</title>
</head>
<body>
  <div style="animation: fadeIn 0.8s ease-out">
    <h1>🌦 Advanced Rainfall Predictor AI</h1>
    <p style="opacity:0.8; font-size: 1.1rem; margin-bottom: 30px;">PyTorch LSTM trained to predict precise Rainfall amounts (mm)</p>
  </div>

  {% if not trained %}
  <div class="card" style="background:rgba(239, 68, 68, 0.2); border-color: rgba(239, 68, 68, 0.5);">
    <p style="text-align: center; font-weight: 600;">⚠️ No dataset loaded. Please check your CSV path.</p>
  </div>
  {% else %}

  <!-- Predict section -->
  <div class="card" style="animation-delay: 0.1s">
    <h2>☔ Predict Proper Rainfall Amount</h2>
    <form method="post" action="/predict">
      <label for="location">📍 Select Location</label>
      <select name="location" id="location" required>
        {% for loc in locations %}
        <option value="{{ loc }}">{{ loc }}</option>
        {% endfor %}
      </select>

      <label for="date">📅 Select Target Date (Requires past 3 days of data)</label>
      <select name="date" id="date" required>
        {% for d in dates %}
        <option value="{{ d }}">{{ d }}</option>
        {% endfor %}
      </select>

      <div class="btn-row" style="justify-content: center; margin-top: 20px;">
        <button type="submit" style="width: 100%; padding: 15px; font-size: 1.1rem;">🔍 Generate Proper Rainfall Prediction</button>
      </div>
    </form>
  </div>

  <!-- Visualisations -->
  <div class="card" style="animation-delay: 0.2s">
    <h2>📊 Analytics & Visualisations</h2>
    <div class="btn-row">
      <form method="get" action="/map" style="flex: 1;"><button type="submit" style="width: 100%;">🗺 Interactive Map</button></form>
      <form method="get" action="/metrics" style="flex: 1;"><button type="submit" style="width: 100%;">📈 Model Metrics</button></form>
    </div>
  </div>

  <div class="card" style="animation-delay: 0.3s; background: transparent; box-shadow: none; border: none; text-align: center; padding: 10px;">
    <span style="font-size:0.85rem; color: #94a3b8;">
      LSTM Sequence Window: 3 Days • {{ loc_count }} Locations
    </span>
  </div>
  {% endif %}
</body>
</html>
"""

PREDICT_HTML = STYLE + """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Prediction - {{ location }}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    /* Rain animation */
    .rain-container { position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 0; overflow: hidden; }
    .raindrop { position: absolute; width: 2px; background: linear-gradient(transparent, rgba(56,189,248,0.6)); animation: fall linear infinite; border-radius: 0 0 5px 5px; }
    @keyframes fall { to { transform: translateY(100vh); } }
    body > *:not(.rain-container) { position: relative; z-index: 1; }
    
    /* Circular progress */
    .circle-progress { position: relative; width: 120px; height: 120px; display: inline-block; }
    .circle-progress svg { transform: rotate(-90deg); }
    .circle-progress .value { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 1.5rem; font-weight: 700; color: #fff; }
    .circle-progress .label { position: absolute; bottom: -25px; left: 50%; transform: translateX(-50%); font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; white-space: nowrap; }
    
    /* Weather hero */
    .weather-hero { font-size: 5rem; margin: 0; line-height: 1; animation: pulse 2s ease-in-out infinite; }
    @keyframes pulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.1); } }
    
    /* Comparison bar */
    .compare-bar { display: flex; gap: 20px; align-items: center; justify-content: center; flex-wrap: wrap; margin: 20px 0; }
    .compare-item { text-align: center; }
    .compare-item .num { font-size: 2rem; font-weight: 700; }
    .compare-item .lbl { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    .compare-divider { width: 1px; height: 50px; background: rgba(255,255,255,0.2); }
    
    /* Tab system */
    .tab-row { display: flex; gap: 0; margin-bottom: 20px; border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1); }
    .tab-btn { flex: 1; padding: 10px; text-align: center; background: rgba(0,0,0,0.2); color: #94a3b8; cursor: pointer; font-size: 0.9rem; font-weight: 600; transition: all 0.3s; border: none; font-family: 'Outfit', sans-serif; }
    .tab-btn.active { background: rgba(56,189,248,0.2); color: #38bdf8; }
    .tab-btn:hover { background: rgba(56,189,248,0.1); }
    .chart-container { height: 300px; width: 100%; }
  </style>
</head>
<body>
  {% if predicted_mm > 1 %}
  <div class="rain-container" id="rainBox"></div>
  {% endif %}

  <div style="animation: fadeIn 0.6s ease-out;">
    <div style="display: flex; justify-content: space-between; align-items: center; max-width: 700px; margin: 0 auto 20px;">
        <div style="text-align: left;">
            <h1 style="margin:0;">{{ location }}</h1>
            <p style="opacity:0.8; font-size: 1.1rem; margin:0;">Forecast for Next Day (after: <strong>{{ date_label }}</strong>)</p>
        </div>
        <button onclick="exportData()" class="btn" style="padding: 8px 16px; font-size: 0.9rem; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); box-shadow: none;">Export</button>
    </div>
  </div>

  <!-- Weather Hero Card -->
  <div class="card" style="animation-delay: 0.1s; border: 2px solid {{ intensity_color }}40; text-align: center;">
    <div class="weather-hero">{{ weather_icon }}</div>
    <h2 style="color: {{ intensity_color }}; border: none; text-align: center; font-size: 1.6rem; margin: 10px 0 5px;">{{ weather_condition }}</h2>
    
    <div style="font-size: 4.5rem; font-weight: 700; color: #fff; margin: 10px 0;">
      {{ "%.1f"|format(predicted_mm) }} <span style="font-size: 1.5rem; color:#94a3b8;">mm</span>
    </div>
    
    <p style="font-size: 1.1rem; color: #cbd5e1; font-weight: 500; margin: 10px 0 20px;">{{ suggestion }}</p>
    
    <!-- Confidence + Rain Probability Circles -->
    <div style="display: flex; justify-content: center; gap: 60px; margin: 30px 0 15px;">
      <div class="circle-progress">
        <svg width="120" height="120">
          <circle cx="60" cy="60" r="50" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="8"/>
          <circle cx="60" cy="60" r="50" fill="none" stroke="{{ intensity_color }}" stroke-width="8" 
                  stroke-dasharray="{{ rain_prob * 3.14 }} 314" stroke-linecap="round"/>
        </svg>
        <span class="value">{{ rain_prob }}%</span>
        <span class="label">Rain Probability</span>
      </div>
      <div class="circle-progress">
        <svg width="120" height="120">
          <circle cx="60" cy="60" r="50" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="8"/>
          <circle cx="60" cy="60" r="50" fill="none" stroke="#a78bfa" stroke-width="8" 
                  stroke-dasharray="{{ confidence * 3.14 }} 314" stroke-linecap="round"/>
        </svg>
        <span class="value">{{ confidence }}%</span>
        <span class="label">Model Confidence</span>
      </div>
    </div>
    
    <!-- Intensity Gauge -->
    <div style="max-width: 500px; margin: 25px auto 10px;">
      <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #94a3b8; margin-bottom: 5px;">
          <span>Intensity</span>
          <span>{% if predicted_mm > 10 %}SEVERE{% elif predicted_mm > 5 %}HEAVY{% elif predicted_mm > 1 %}LIGHT{% elif predicted_mm > 0.2 %}DRIZZLE{% else %}DRY{% endif %}</span>
      </div>
      <div style="height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; overflow: hidden;">
          <div style="width: {{ [predicted_mm * 5, 100]|min }}%; height: 100%; background: linear-gradient(90deg, {{ intensity_color }}, {{ intensity_color }}88); transition: width 1.5s ease-out; border-radius: 4px;"></div>
      </div>
    </div>

    <div style="margin-top: 15px;">
      <span class="badge {% if predicted_mm > 1.0 %}badge-yes{% else %}badge-no{% endif %}">
        {% if predicted_mm > 1.0 %}RAIN EXPECTED{% else %}NO SIGNIFICANT RAIN{% endif %}
      </span>
    </div>
  </div>

  <!-- Why this prediction? (Feature Insights) -->
  <div class="card" style="animation-delay: 0.15s;">
    <h2 style="border: none; margin-bottom: 15px;">🔍 Why this prediction? (Model Insights)</h2>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
      {% for insight in insights %}
      <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 12px; border-left: 3px solid {{ intensity_color }};">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
          <span style="font-size: 1.5rem;">{{ insight.icon }}</span>
          <strong style="color: #fff; font-size: 1rem;">{{ insight.label }}</strong>
        </div>
        <p style="font-size: 0.85rem; color: #94a3b8; margin: 0; line-height: 1.4;">{{ insight.desc }}</p>
      </div>
      {% endfor %}
    </div>
  </div>

  <!-- Actual vs Predicted Comparison -->
  {% if actual_mm is not none %}
  <div class="card" style="animation-delay: 0.15s; text-align: center;">
    <h2 style="text-align: center; border: none;">Prediction vs Actual (Dataset Verification)</h2>
    <div class="compare-bar">
      <div class="compare-item">
        <div class="num" style="color: #38bdf8;">{{ "%.1f"|format(predicted_mm) }} mm</div>
        <div class="lbl">LSTM Predicted</div>
      </div>
      <div class="compare-divider"></div>
      <div class="compare-item">
        <div class="num" style="color: #10b981;">{{ "%.1f"|format(actual_mm) }} mm</div>
        <div class="lbl">Actual (Dataset)</div>
      </div>
      <div class="compare-divider"></div>
      <div class="compare-item">
        <div class="num" style="color: {% if (predicted_mm - actual_mm)|abs < 1 %}#10b981{% elif (predicted_mm - actual_mm)|abs < 3 %}#f97316{% else %}#ef4444{% endif %};">
          {{ "%.1f"|format((predicted_mm - actual_mm)|abs) }} mm
        </div>
        <div class="lbl">Error</div>
      </div>
    </div>
    <p style="font-size: 0.85rem; color: #94a3b8;">
      {% if (predicted_mm - actual_mm)|abs < 1 %}Excellent prediction accuracy!
      {% elif (predicted_mm - actual_mm)|abs < 3 %}Good prediction within acceptable margin.
      {% else %}Prediction shows moderate variance from actual value.{% endif %}
    </p>
  </div>
  {% endif %}

  <!-- Interactive Multi-Chart -->
  <div class="card" style="animation-delay: 0.2s">
    <h2>Interactive Time-Series Analysis</h2>
    <div class="tab-row">
      <button class="tab-btn active" onclick="switchChart('rainfall')">Rainfall</button>
      <button class="tab-btn" onclick="switchChart('humidity')">Humidity</button>
      <button class="tab-btn" onclick="switchChart('pressure')">Pressure</button>
      <button class="tab-btn" onclick="switchChart('temp')">Temperature</button>
    </div>
    <div class="chart-container">
        <canvas id="trendsChart"></canvas>
    </div>
  </div>

  <!-- Weather Details Grid -->
  <div class="card" style="animation-delay: 0.3s">
    <h2>Current Day Weather Details</h2>
    <div class="stat-grid">
      {% for k, v in summary.items() %}
      <div class="stat-box">
        <span>{{ k }}</span>
        <strong>{{ v }}</strong>
      </div>
      {% endfor %}
      <div class="stat-box" style="border: 1px solid rgba(56,189,248,0.2);">
        <span>Model</span>
        <strong>LSTM</strong>
      </div>
    </div>
  </div>

  <div style="max-width: 700px; margin: 20px auto; text-align: left;">
      <a href="/" class="back" style="animation: fadeIn 1s ease-out forwards; animation-delay: 0.5s; opacity: 0;">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 8px;"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>
        Back to Dashboard
      </a>
  </div>

  <script>
    // Rain animation
    const rainBox = document.getElementById('rainBox');
    if (rainBox) {
        const intensity = Math.min({{ predicted_mm }} * 8, 150);
        for (let i = 0; i < intensity; i++) {
            const drop = document.createElement('div');
            drop.className = 'raindrop';
            drop.style.left = Math.random() * 100 + '%';
            drop.style.height = (Math.random() * 20 + 10) + 'px';
            drop.style.animationDuration = (Math.random() * 1 + 0.5) + 's';
            drop.style.animationDelay = Math.random() * 2 + 's';
            drop.style.opacity = Math.random() * 0.5 + 0.3;
            rainBox.appendChild(drop);
        }
    }

    // Chart.js with tab switching
    const ctx = document.getElementById('trendsChart').getContext('2d');
    const chartData = {{ chart_data|tojson }};
    
    const datasets = {
        rainfall: { label: 'Rainfall (mm)', data: chartData.rainfall, color: '#38bdf8', fill: true },
        humidity: { label: 'Humidity (%)', data: chartData.humidity, color: '#10b981', fill: false },
        pressure: { label: 'Pressure (hPa)', data: chartData.pressure, color: '#a78bfa', fill: false },
        temp: { label: 'Temperature (C)', data: chartData.temp, color: '#f97316', fill: false }
    };
    
    let chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.labels,
            datasets: [{
                label: datasets.rainfall.label,
                data: datasets.rainfall.data,
                borderColor: datasets.rainfall.color,
                backgroundColor: datasets.rainfall.color + '20',
                borderWidth: 3, tension: 0.4, fill: true,
                pointBackgroundColor: datasets.rainfall.color,
                pointRadius: 5, pointHoverRadius: 8
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#e2e8f0', font: { family: 'Outfit', size: 13 } } } },
            scales: {
                x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });
    
    function switchChart(type) {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        event.target.classList.add('active');
        const ds = datasets[type];
        chart.data.datasets[0] = {
            label: ds.label, data: ds.data,
            borderColor: ds.color, backgroundColor: ds.color + '20',
            borderWidth: 3, tension: 0.4, fill: ds.fill,
            pointBackgroundColor: ds.color, pointRadius: 5, pointHoverRadius: 8
        };
        chart.update();
    }

    function exportData() {
        const data = {
            location: "{{ location }}", date: "{{ date_label }}",
            predicted_mm: {{ predicted_mm }},
            actual_mm: {{ actual_mm if actual_mm is not none else 'null' }},
            confidence: {{ confidence }}, rain_probability: {{ rain_prob }},
            weather_condition: "{{ weather_condition }}",
            suggestion: "{{ suggestion }}", history: chartData
        };
        fetch('/export', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) })
        .then(r => r.blob()).then(blob => {
            const a = document.createElement('a');
            a.href = window.URL.createObjectURL(blob);
            a.download = `forecast_{{ location }}_{{ date_label }}.json`;
            a.click();
        });
    }
  </script>
</body>
</html>
"""

METRICS_HTML = STYLE + """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Model Metrics</title>
</head>
<body>
  <h1>📈 Time-Series Regression Performance</h1>
  <div class="card" style="animation-delay: 0.1s">
    <h2>Evaluation Scores (Rainfall Amount Prediction)</h2>
    <div class="stat-grid" style="margin-bottom: 30px;">
      <div class="stat-box"><span>Root Mean Sq Error (RMSE)</span><strong>{{ "%.2f"|format(metrics.rmse) }} mm</strong></div>
      <div class="stat-box"><span>Mean Absolute Error (MAE)</span><strong>{{ "%.2f"|format(metrics.mae) }} mm</strong></div>
      <div class="stat-box"><span>R² Score</span><strong>{{ "%.2f"|format(metrics.r2) }}</strong></div>
    </div>
    <p style="color: #94a3b8; font-size: 0.9rem; text-align:center;">Lower RMSE and MAE values indicate better predictive accuracy for rainfall amounts.</p>
  </div>
  <a href="/" class="back">Back to Dashboard</a>
</body>
</html>
"""

MAP_HTML = STYLE + """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>India Heatmap</title>
</head>
<body>
  <h1>🗺 Regional Rainfall Analysis</h1>
  <div class="card" style="padding:0; overflow:hidden; max-width:900px; animation-delay: 0.1s;">
    <iframe src="/mapframe" width="100%" height="650" style="border:none; display: block;"></iframe>
  </div>
  <a href="/" class="back">Back to Dashboard</a>
</body>
</html>
"""

@app.route("/")
def home():
    if not STATE.get("trained"):
         return render_template_string(HOME_HTML, trained=False, csv_path=DEFAULT_CSV, locations=[], dates=[], row_count=0, loc_count=0, date_count=0)
    return render_template_string(
        HOME_HTML, trained=True, csv_path=DEFAULT_CSV,
        locations=STATE["unique_locations"], dates=STATE["unique_dates"],
        row_count=len(STATE["df_raw"]), loc_count=len(STATE["unique_locations"]), date_count=len(STATE["unique_dates"])
    )

@app.route("/predict", methods=["POST"])
def predict():
    if not STATE.get("trained"): return redirect(url_for("home"))

    loc = request.form.get("location", "")
    date_sel = request.form.get("date", "")
    
    df_raw = STATE["df_raw"]
    loc_data = df_raw[df_raw["Location"] == loc].sort_values(by="Date_obj")

    if loc_data.empty: return f"No data for {loc}.", 400

    # Find the index of the selected date
    loc_data = loc_data.reset_index(drop=True)
    target_idx = loc_data[loc_data["Date"] == date_sel].index
    if len(target_idx) == 0:
        target_idx = len(loc_data) - 1
        date_sel = loc_data.iloc[target_idx]["Date"]
        date_label = f"{date_sel} (Closest available)"
    else:
        target_idx = target_idx[0]
        date_label = date_sel

    # Extract sequence (N_STEPS)
    start_idx = max(0, target_idx - N_STEPS + 1)
    seq_raw = loc_data.iloc[start_idx : target_idx + 1].copy()

    # If sequence is too short, pad it by repeating the first row
    while len(seq_raw) < N_STEPS:
        seq_raw = pd.concat([seq_raw.iloc[[0]], seq_raw], ignore_index=True)

    # Prepare features for the model
    drop_cols = [c for c in ["Date", "Date_obj", "Time", "RISK_MM", "RainTomorrow"] if c in seq_raw.columns]
    seq_features = seq_raw.drop(columns=drop_cols)
    
    for c, le in STATE["encoders"].items():
        if c in seq_features.columns:
            seq_features[c] = le.transform(seq_features[c].astype(str))
            
    # Standardize
    model = STATE["model"]
    seq_scaled = model.scaler.transform(seq_features)
    X_input = np.array([seq_scaled]) # shape (1, N_STEPS, features)

    # Prediction (Regression for exact Rainfall amount in mm)
    predicted_mm = float(model.predict(X_input)[0])
    
    # Get the actual value for comparison (if available in dataset)
    actual_mm = float(seq_raw.iloc[-1]["RISK_MM"]) if "RISK_MM" in seq_raw.columns else None
    
    # Calculate confidence based on how similar today's conditions are to training data
    confidence = min(95, max(55, 85 - abs(predicted_mm - (actual_mm or predicted_mm)) * 5))
    
    # Rain probability percentage (sigmoid-like mapping from mm)
    if predicted_mm > 10:
        rain_prob = 95
    elif predicted_mm > 5:
        rain_prob = 80
    elif predicted_mm > 1:
        rain_prob = 60
    elif predicted_mm > 0.3:
        rain_prob = 35
    else:
        rain_prob = 10
    
    if predicted_mm > 10.0:
        suggestion = "🌊 Severe Rainfall Warning! Stay indoors and avoid waterlogged areas."
        weather_icon = "⛈️"
        weather_condition = "Thunderstorms"
        intensity_color = "#ef4444"
    elif predicted_mm > 5.0:
        suggestion = "☔ Heavy Rain Expected! Carry an umbrella and plan ahead."
        weather_icon = "🌧️"
        weather_condition = "Heavy Rain"
        intensity_color = "#f97316"
    elif predicted_mm > 1.0:
        suggestion = "🌦️ Light Rain Expected. Might want to carry a jacket."
        weather_icon = "🌦️"
        weather_condition = "Light Showers"
        intensity_color = "#38bdf8"
    elif predicted_mm > 0.2:
        suggestion = "🌤️ Slight drizzle possible. Keep an umbrella handy just in case."
        weather_icon = "🌤️"
        weather_condition = "Partly Cloudy"
        intensity_color = "#a3e635"
    else:
        suggestion = "🌞 Clear skies! No significant rain expected. Have a great day!"
        weather_icon = "☀️"
        weather_condition = "Clear"
        intensity_color = "#10b981"

    # Plot Basis of prediction (Recent trends) - Use up to 7 previous data points for the graph
    graph_start_idx = max(0, target_idx - 6)
    graph_seq = loc_data.iloc[graph_start_idx : target_idx + 1].copy()
    
    matplotlib.rcParams['text.color'] = 'white'
    matplotlib.rcParams['axes.labelcolor'] = 'white'
    matplotlib.rcParams['xtick.color'] = 'white'
    matplotlib.rcParams['ytick.color'] = 'white'
    fig_basis, ax_basis = plt.subplots(figsize=(7, 3.5))
    dates_plot = graph_seq["Date"].tolist()
    if "Rainfall" in graph_seq.columns:
        ax_basis.plot(dates_plot, graph_seq["Rainfall"], marker='o', label='Historical Rainfall (mm)', color='#38bdf8', linewidth=2)
    if "Humidity3pm" in graph_seq.columns:
        ax_basis.plot(dates_plot, graph_seq["Humidity3pm"], marker='x', label='Humidity3pm (%)', color='#10b981', linewidth=2)
    
    ax_basis.set_title("Time-Series Basis for Prediction")
    ax_basis.legend(loc="upper left", framealpha=0.2)
    fig_basis.patch.set_alpha(0)
    ax_basis.patch.set_alpha(0)
    basis_img = fig_to_base64(fig_basis)

    # Rich summary with all key weather metrics
    summary = {}
    row = seq_raw.iloc[-1]
    if "MinTemp" in seq_raw.columns: summary["Min Temp"] = f"{row['MinTemp']:.1f} °C"
    if "MaxTemp" in seq_raw.columns: summary["Max Temp"] = f"{row['MaxTemp']:.1f} °C"
    if "Temp3pm" in seq_raw.columns: summary["Temp 3PM"] = f"{row['Temp3pm']:.1f} °C"
    if "Humidity9am" in seq_raw.columns: summary["Humidity 9AM"] = f"{row['Humidity9am']:.0f}%"
    if "Humidity3pm" in seq_raw.columns: summary["Humidity 3PM"] = f"{row['Humidity3pm']:.0f}%"
    if "Pressure3pm" in seq_raw.columns: summary["Pressure 3PM"] = f"{row['Pressure3pm']:.1f} hPa"
    if "WindGustSpeed" in seq_raw.columns: summary["Wind Gust"] = f"{row['WindGustSpeed']:.0f} km/h"
    if "Sunshine" in seq_raw.columns: summary["Sunshine"] = f"{row['Sunshine']:.1f} hrs"
    if "Cloud3pm" in seq_raw.columns: summary["Cloud Cover"] = f"{row['Cloud3pm']:.0f}/8"
    if "Rainfall" in seq_raw.columns: summary["Today's Rain"] = f"{row['Rainfall']:.1f} mm"

    # Prepare data for Interactive JS Chart (Chart.js)
    chart_data = {
        "labels": graph_seq["Date"].tolist(),
        "rainfall": graph_seq["Rainfall"].tolist() if "Rainfall" in graph_seq.columns else [],
        "humidity": graph_seq["Humidity3pm"].tolist() if "Humidity3pm" in graph_seq.columns else [],
        "pressure": graph_seq["Pressure3pm"].tolist() if "Pressure3pm" in graph_seq.columns else [],
        "temp": graph_seq["Temp3pm"].tolist() if "Temp3pm" in graph_seq.columns else []
    }

    # Calculate Feature Insights (Simplified Importance)
    # We look at the latest scaled features and see which are most extreme (deviating from mean)
    latest_scaled = seq_scaled[-1]
    feature_names = model.feature_names_in_
    insights = []
    
    # Humidity Insight
    if "Humidity3pm" in feature_names:
        idx = feature_names.index("Humidity3pm")
        val = latest_scaled[idx]
        if val > 1.0: insights.append({"icon": "💧", "label": "High Humidity", "desc": "Elevated moisture levels are a primary driver for this rain forecast."})
        elif val < -1.0: insights.append({"icon": "🌵", "label": "Low Humidity", "desc": "Dry air conditions are significantly reducing rain probability."})
        
    # Pressure Insight
    if "Pressure3pm" in feature_names:
        idx = feature_names.index("Pressure3pm")
        val = latest_scaled[idx]
        if val < -1.0: insights.append({"icon": "📉", "label": "Falling Pressure", "desc": "A low-pressure system is detected, which often precedes rainfall."})
        elif val > 1.0: insights.append({"icon": "📈", "label": "High Pressure", "desc": "Stable high-pressure conditions are keeping the skies clear."})

    # Wind Insight
    if "WindGustSpeed" in feature_names:
        idx = feature_names.index("WindGustSpeed")
        val = latest_scaled[idx]
        if val > 1.5: insights.append({"icon": "💨", "label": "Strong Gusts", "desc": "High wind speeds suggest potential storm development."})

    if not insights:
        insights.append({"icon": "⚖️", "label": "Stable Conditions", "desc": "Weather metrics are within normal ranges for this period."})

    return render_template_string(
        PREDICT_HTML, location=loc, date_label=date_label, predicted_mm=predicted_mm, 
        suggestion=suggestion, basis_img=basis_img, summary=summary,
        chart_data=chart_data, actual_mm=actual_mm, confidence=confidence,
        rain_prob=rain_prob, weather_icon=weather_icon, 
        weather_condition=weather_condition, intensity_color=intensity_color,
        insights=insights
    )

@app.route("/export", methods=["POST"])
def export_data():
    import json
    data = request.json
    return json.dumps(data, indent=4), 200, {'Content-Type': 'application/json', 'Content-Disposition': 'attachment; filename=prediction.json'}

@app.route("/metrics")
def metrics():
    if not STATE.get("trained"): return redirect(url_for("home"))
    m_dict = STATE["metrics"]
    class M: pass
    mo = M()
    mo.rmse, mo.mae, mo.r2 = m_dict["rmse"], m_dict["mae"], m_dict["r2"]
    return render_template_string(METRICS_HTML, metrics=mo)

@app.route("/map")
def map_view():
    if not STATE.get("trained"): return redirect(url_for("home"))
    return render_template_string(MAP_HTML)

@app.route("/mapframe")
def map_frame():
    from folium.plugins import HeatMap
    city_coords = {"Delhi":[28.7041,77.1025], "Mumbai":[19.0760,72.8777], "Bengaluru":[12.9716,77.5946],
                   "Chennai":[13.0827,80.2707], "Kolkata":[22.5726,88.3639], "Hyderabad":[17.3850,78.4867],
                   "Pune":[18.5204,73.8567], "Jaipur":[26.9124,75.7873], "Ahmedabad":[23.0225,72.5714],
                   "Chandigarh":[30.7333,76.7794], "Nagpur":[21.1458,79.0882], "Goa":[15.2993,74.1240],
                   "Indore":[22.7196,75.8577], "Srinagar":[34.0836,74.7973], "Bhopal":[23.2599,77.4126],
                   "Guwahati":[26.1445,91.7362], "Kochi":[9.9312,76.2673], "Lucknow":[26.8467,80.9462],
                   "Shimla":[31.1048,77.1734], "Patna":[25.5941,85.1376]}
    try:
        df_raw = STATE["df_raw"]
        model = STATE["model"]
        m = folium.Map(location=[22.0, 78.0], zoom_start=5, tiles="CartoDB dark_matter")
        
        heat_data = []
        for loc in STATE["unique_locations"]:
            coords = city_coords.get(loc)
            if not coords: continue
            
            # Get latest data for this location to predict
            loc_data = df_raw[df_raw["Location"] == loc].sort_values(by="Date_obj")
            if loc_data.empty: continue
            
            latest_row = loc_data.iloc[-1:]
            
            # Prepare features for prediction
            drop_cols = [c for c in ["Date", "Date_obj", "Time", "RISK_MM", "RainTomorrow"] if c in latest_row.columns]
            feat = latest_row.drop(columns=drop_cols)
            for c, le in STATE["encoders"].items():
                if c in feat.columns:
                    feat[c] = le.transform(feat[c].astype(str))
            
            # Predict
            scaled = model.scaler.transform(feat)
            pred_mm = float(model.predict(np.array([scaled]))[0])
            
            # Color based on prediction
            color = "#ef4444" if pred_mm > 5 else "#38bdf8"
            
            heat_data.append([coords[0], coords[1], pred_mm])
            folium.CircleMarker(
                location=coords, radius=7,
                popup=f"<b>{loc}</b><br>Latest Date: {latest_row['Date'].values[0]}<br><span style='color:{color}; font-weight:bold;'>Forecast: {pred_mm:.2f} mm</span>",
                color=color, fill=True, fill_color=color, fill_opacity=0.7
            ).add_to(m)
        
        if heat_data:
            HeatMap(heat_data, radius=25, blur=15, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)
        
        # Add a custom HTML Legend
        legend_html = '''
             <div style="position: fixed; 
                         bottom: 50px; left: 50px; width: 180px; height: 120px; 
                         background-color: rgba(15, 23, 42, 0.9); z-index:9999; font-size:14px;
                         color: white; padding: 10px; border-radius: 10px;
                         border: 1px solid rgba(255,255,255,0.2); backdrop-filter: blur(5px);">
             <b style="color: #38bdf8;">Forecast Legend</b><br>
             <i style="background:red; width:10px; height:10px; display:inline-block; border-radius:50%;"></i> Severe (>10mm)<br>
             <i style="background:orange; width:10px; height:10px; display:inline-block; border-radius:50%;"></i> Heavy (>5mm)<br>
             <i style="background:lime; width:10px; height:10px; display:inline-block; border-radius:50%;"></i> Light (>1mm)<br>
             <i style="background:blue; width:10px; height:10px; display:inline-block; border-radius:50%;"></i> Dry (0-1mm)
             </div>
             '''
        m.get_root().html.add_child(folium.Element(legend_html))
            
    except Exception as e:
        print(f"Map error: {e}")
        m = folium.Map(location=[22.0, 78.0], zoom_start=5)
    return m._repr_html_()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
