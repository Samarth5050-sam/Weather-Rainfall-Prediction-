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
N_STEPS = 3  # Time series lookback window (e.g. use past 3 days)

# ==============================
# LSTM Model (Regression)
# ==============================
class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        # Regression outputs exactly 1 value (Rainfall in mm)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Take last sequence output
        # Apply ReLU because rainfall cannot be negative
        out = torch.relu(out)
        return out

class LSTMModelWrapper:
    def __init__(self, model, scaler, features):
        self.model = model
        self.scaler = scaler
        self.feature_names_in_ = features

    def predict(self, X_seq):
        self.model.eval()
        # X_seq should be (batch, seq, features) numpy array
        X_t = torch.tensor(X_seq, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_t)
        return outputs.numpy().flatten()

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
    
    # Target: Predict EXACT rainfall amount (Regression)
    y_target = df["RISK_MM"].values.astype(np.float32)

    # Create sequences
    X_seqs = []
    y_seqs = []
    
    # To keep track of raw df indices to construct sequences for evaluation
    locations_arr = df['Location'].values
    
    for i in range(N_STEPS - 1, len(df)):
        if locations_arr[i] == locations_arr[i - N_STEPS + 1]: # check if same location
            X_seqs.append(scaled_features[i - N_STEPS + 1 : i + 1])
            y_seqs.append(y_target[i])

    if not X_seqs:
        # Fallback if dataset is too small
        X_seqs = np.array([scaled_features[i:i+1] for i in range(len(df))])
        y_seqs = np.array(y_target)
    else:
        X_seqs = np.array(X_seqs)
        y_seqs = np.array(y_seqs)

    X_train, X_test, y_train, y_test = train_test_split(X_seqs, y_seqs, test_size=0.2, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    lstm_model = WeatherLSTM(input_size=X_train.shape[2])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.005)

    lstm_model.train()
    for epoch in range(15): # 15 epochs
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = lstm_model(batch_X)
            loss = criterion(outputs, batch_y)
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

import dill
STATE_FILE = "weather_state.pkl"

if os.path.exists(STATE_FILE):
    try:
        with open(STATE_FILE, "rb") as f:
            STATE = dill.load(f)
        print(f"[OK] Pre-trained state loaded directly from {STATE_FILE}! Bypassing boot training.")
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
  <title>Prediction – {{ location }}</title>
</head>
<body>
  <div style="animation: fadeIn 0.6s ease-out;">
    <h1>📍 {{ location }}</h1>
    <p style="opacity:0.8; font-size: 1.1rem; margin-bottom: 20px;">Forecast for: <strong>{{ date_label }}</strong></p>
  </div>

  <div class="card" style="animation-delay: 0.1s; border: 2px solid rgba(56, 189, 248, 0.5);">
    <h2 style="color: #38bdf8;">Proper Rainfall Prediction</h2>
    <div style="text-align: center; margin: 30px 0;">
      <span style="font-size: 3.5rem; font-weight: 700; color: #fff;">
        {{ "%.1f"|format(predicted_mm) }} <span style="font-size: 1.5rem; color:#94a3b8;">mm</span>
      </span>
      <p style="font-size:1.2rem; margin-top:15px; color: #cbd5e1;">{{ suggestion }}</p>
      
      <div style="margin-top: 15px;">
        <span class="badge {% if predicted_mm > 1.0 %}badge-yes{% else %}badge-no{% endif %}">
          {% if predicted_mm > 1.0 %}RAIN EXPECTED{% else %}NO SIGNIFICANT RAIN{% endif %}
        </span>
      </div>
    </div>
  </div>

  <div class="card" style="animation-delay: 0.2s">
    <h2>Basis of Prediction: Time-Series Trends</h2>
    <p style="font-size:0.9rem; color:#94a3b8; margin-top:-10px;">The LSTM model analysed these exact historical features over the preceding days to calculate the rainfall amount.</p>
    <div style="text-align: center; background: rgba(0,0,0,0.2); border-radius: 12px; padding: 10px;">
      <img src="data:image/png;base64,{{ basis_img }}" alt="Basis of prediction chart" style="width: 100%; max-width: 600px; box-shadow: none;">
    </div>
  </div>

  <div class="card" style="animation-delay: 0.3s">
    <h2>Historical Weather Summary for this Window</h2>
    <div class="stat-grid">
      {% for k, v in summary.items() %}
      <div class="stat-box">
        <span>{{ k }}</span>
        <strong>{{ v }}</strong>
      </div>
      {% endfor %}
    </div>
  </div>

  <a href="/" class="back" style="animation: fadeIn 1s ease-out forwards; animation-delay: 0.5s; opacity: 0;">
    Back to Dashboard
  </a>
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

    # Prepare features
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
    
    if predicted_mm > 5.0:
        suggestion = "☔ Heavy Rain Expected! Carry an umbrella and plan ahead."
    elif predicted_mm > 1.0:
        suggestion = "🌧️ Light Rain Expected. Might want to carry a jacket."
    else:
        suggestion = "🌞 No significant rain expected. Have a great day!"

    # Plot Basis of prediction (Recent trends)
    matplotlib.rcParams['text.color'] = 'white'
    matplotlib.rcParams['axes.labelcolor'] = 'white'
    matplotlib.rcParams['xtick.color'] = 'white'
    matplotlib.rcParams['ytick.color'] = 'white'
    fig_basis, ax_basis = plt.subplots(figsize=(7, 3.5))
    dates_plot = seq_raw["Date"].tolist()
    if "Rainfall" in seq_raw.columns:
        ax_basis.plot(dates_plot, seq_raw["Rainfall"], marker='o', label='Historical Rainfall (mm)', color='#38bdf8', linewidth=2)
    if "Humidity3pm" in seq_raw.columns:
        ax_basis.plot(dates_plot, seq_raw["Humidity3pm"], marker='x', label='Humidity3pm (%)', color='#10b981', linewidth=2)
    
    ax_basis.set_title("Time-Series Basis for Prediction")
    ax_basis.legend(loc="upper left", framealpha=0.2)
    fig_basis.patch.set_alpha(0)
    ax_basis.patch.set_alpha(0)
    basis_img = fig_to_base64(fig_basis)

    # Summary
    summary = {}
    if "Temp3pm" in seq_raw.columns: summary["Avg Temp 3PM"] = f"{seq_raw['Temp3pm'].mean():.1f} °C"
    if "Humidity3pm" in seq_raw.columns: summary["Avg Humidity 3PM"] = f"{seq_raw['Humidity3pm'].mean():.1f}%"
    if "Pressure3pm" in seq_raw.columns: summary["Avg Pressure"] = f"{seq_raw['Pressure3pm'].mean():.1f} hPa"

    return render_template_string(
        PREDICT_HTML, location=loc, date_label=date_label, predicted_mm=predicted_mm, 
        suggestion=suggestion, basis_img=basis_img, summary=summary
    )

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
    city_coords = {"Delhi":[28.7041,77.1025], "Mumbai":[19.0760,72.8777], "Bengaluru":[12.9716,77.5946],
                   "Chennai":[13.0827,80.2707], "Kolkata":[22.5726,88.3639], "Hyderabad":[17.3850,78.4867],
                   "Pune":[18.5204,73.8567], "Jaipur":[26.9124,75.7873], "Ahmedabad":[23.0225,72.5714],
                   "Chandigarh":[30.7333,76.7794], "Nagpur":[21.1458,79.0882], "Goa":[15.2993,74.1240],
                   "Indore":[22.7196,75.8577], "Srinagar":[34.0836,74.7973], "Bhopal":[23.2599,77.4126],
                   "Guwahati":[26.1445,91.7362], "Kochi":[9.9312,76.2673], "Lucknow":[26.8467,80.9462],
                   "Shimla":[31.1048,77.1734], "Patna":[25.5941,85.1376]}
    try:
        df_raw = STATE["df_raw"]
        agg = df_raw.groupby("Location")["Rainfall"].mean().reset_index() if "Rainfall" in df_raw.columns else pd.DataFrame(columns=["Location", "Rainfall"])
        m = folium.Map(location=[22.0, 78.0], zoom_start=5, tiles="CartoDB dark_matter")
        for _, row in agg.iterrows():
            coords = city_coords.get(row["Location"])
            if coords:
                folium.CircleMarker(location=coords, radius=max(5, min(row["Rainfall"] / 2, 35)),
                                    popup=f"<b>{row['Location']}</b><br>Avg Rainfall: {row['Rainfall']:.2f} mm",
                                    color="#38bdf8", fill=True, fill_color="#0ea5e9", fill_opacity=0.7).add_to(m)
    except:
        m = folium.Map(location=[22.0, 78.0], zoom_start=5)
    return m._repr_html_()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
