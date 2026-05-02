import os

file_path = "c:\\Users\\HP\\OneDrive\\Desktop\\weather_app\\weather_app.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

import re

# We will use regex to find the blocks STYLE, HOME_HTML, PREDICT_HTML, METRICS_HTML, MAP_HTML
# and replace them all.

new_ui = '''# ==============================
# Shared CSS / Style
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
  img { max-width: 100%; border-radius: 12px; margin-top: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.05); }
  .back { display: inline-flex; align-items: center; margin-top: 25px; color: #38bdf8; text-decoration: none; font-weight: 600; transition: color 0.2s; }
  .back:hover { color: #7dd3fc; }
  .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 15px; margin-top: 15px; }
  .stat-box { background: rgba(0,0,0,0.2); padding: 15px; border-radius: 12px; text-align: center; border: 1px solid rgba(255,255,255,0.05); }
  .stat-box span { display: block; font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px; }
  .stat-box strong { display: block; font-size: 1.2rem; color: #fff; }
</style>
"""

# ==============================
# Home template
# ==============================
HOME_HTML = STYLE + """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>🌦 Weather Prediction AI</title>
</head>
<body>
  <div style="animation: fadeIn 0.8s ease-out">
    <h1>🌦 Next-Gen Weather AI</h1>
    <p style="opacity:0.8; font-size: 1.1rem; margin-bottom: 30px;">Deep Learning powered predictions for India</p>
  </div>

  {% if not trained %}
  <div class="card" style="background:rgba(239, 68, 68, 0.2); border-color: rgba(239, 68, 68, 0.5);">
    <p style="text-align: center; font-weight: 600;">⚠️ No dataset loaded. Please check your CSV path (<code>{{ csv_path }}</code>).</p>
  </div>
  {% else %}

  <!-- Predict section -->
  <div class="card" style="animation-delay: 0.1s">
    <h2>☔ Predict Rain Tomorrow</h2>
    <form method="post" action="/predict">
      <label for="location">📍 Select Location</label>
      <select name="location" id="location" required>
        {% for loc in locations %}
        <option value="{{ loc }}">{{ loc }}</option>
        {% endfor %}
      </select>

      <label for="date">📅 Select Date</label>
      <select name="date" id="date">
        <option value="">— Use Historical Average —</option>
        {% for d in dates %}
        <option value="{{ d }}">{{ d }}</option>
        {% endfor %}
      </select>

      <div class="btn-row" style="justify-content: center; margin-top: 20px;">
        <button type="submit" style="width: 100%; padding: 15px; font-size: 1.1rem;">🔍 Generate AI Prediction</button>
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
      LSTM Neural Network • {{ row_count }} Data Points • {{ loc_count }} Locations
    </span>
  </div>
  {% endif %}
</body>
</html>
"""

# ==============================
# Predict template
# ==============================
PREDICT_HTML = STYLE + """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Prediction – {{ location }}</title>
</head>
<body>
  <div style="animation: fadeIn 0.6s ease-out;">
    <h1>📍 {{ location }}</h1>
    <p style="opacity:0.8; font-size: 1.1rem; margin-bottom: 20px;">Forecast for: <strong>{{ date_label }}</strong></p>
  </div>

  <div class="card" style="animation-delay: 0.1s">
    <h2>AI Prediction: Rain Tomorrow?</h2>
    <div style="text-align: center; margin: 30px 0;">
      <span class="badge {% if tomorrow == 'Yes' %}badge-yes{% else %}badge-no{% endif %}" style="font-size: 1.8rem; padding: 10px 30px;">
        {{ tomorrow }}
      </span>
      <p style="font-size:1.2rem; margin-top:15px; color: #cbd5e1;">{{ suggestion }}</p>
    </div>
  </div>

  <div class="card" style="animation-delay: 0.2s">
    <h2>Probability Analysis</h2>
    <div style="text-align: center; background: rgba(0,0,0,0.2); border-radius: 12px; padding: 10px;">
      <img src="data:image/png;base64,{{ bar_img }}" alt="Tomorrow bar chart" style="box-shadow: none; border: none; background: transparent; width: 100%; max-width: 300px;">
    </div>
  </div>

  <div class="card" style="animation-delay: 0.3s">
    <h2>Extended Forecast ({{ location }})</h2>
    <div class="stat-grid" style="margin-bottom: 20px;">
      {% for d, p in extended.items() %}
      <div class="stat-box">
        <span>{{ d }} Days</span>
        <strong>{{ (p*100)|round(1) }}%</strong>
      </div>
      {% endfor %}
    </div>
    <div style="text-align: center; background: rgba(0,0,0,0.2); border-radius: 12px; padding: 10px;">
      <img src="data:image/png;base64,{{ ext_img }}" alt="Extended forecast chart" style="box-shadow: none; border: none; background: transparent;">
    </div>
  </div>

  <div class="card" style="animation-delay: 0.4s">
    <h2>Historical Weather Summary</h2>
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
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 8px;">
      <path d="M19 12H5M12 19l-7-7 7-7"/>
    </svg>
    Back to Dashboard
  </a>
</body>
</html>
"""

# ==============================
# Metrics template
# ==============================
METRICS_HTML = STYLE + """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Model Metrics</title>
</head>
<body>
  <h1>📈 Model Performance</h1>
  <p style="opacity:0.8; margin-bottom: 20px;">LSTM Classification Metrics</p>

  <div class="card" style="animation-delay: 0.1s">
    <h2>Evaluation Scores</h2>
    <div class="stat-grid" style="margin-bottom: 30px;">
      <div class="stat-box"><span>Accuracy</span><strong>{{ "%.2f"|format(metrics.accuracy*100) }}%</strong></div>
      <div class="stat-box"><span>Precision</span><strong>{{ "%.2f"|format(metrics.precision*100) }}%</strong></div>
      <div class="stat-box"><span>Recall</span><strong>{{ "%.2f"|format(metrics.recall*100) }}%</strong></div>
      <div class="stat-box"><span>F1-Score</span><strong>{{ "%.2f"|format(metrics.f1*100) }}%</strong></div>
    </div>
    
    <h2>Confusion Matrix</h2>
    <div style="text-align: center; background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px;">
      <img src="data:image/png;base64,{{ cm_img }}" alt="Confusion matrix" style="box-shadow: none; border: none; max-width: 400px;">
    </div>
  </div>

  <a href="/" class="back">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 8px;">
      <path d="M19 12H5M12 19l-7-7 7-7"/>
    </svg>
    Back to Dashboard
  </a>
</body>
</html>
"""

# ==============================
# Map template
# ==============================
MAP_HTML = STYLE + """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>India Heatmap</title>
</head>
<body>
  <h1>🗺 Regional Rainfall Analysis</h1>
  <p style="opacity:0.8; margin-bottom: 20px;">Interactive geographic visualisation</p>

  <div class="card" style="padding:0; overflow:hidden; max-width:900px; animation-delay: 0.1s; border: 1px solid rgba(255,255,255,0.2);">
    <iframe src="/mapframe" width="100%" height="650" style="border:none; border-radius:20px; display: block;"></iframe>
  </div>

  <a href="/" class="back">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 8px;">
      <path d="M19 12H5M12 19l-7-7 7-7"/>
    </svg>
    Back to Dashboard
  </a>
</body>
</html>
'''

import re
# Find the start of STYLE
start_idx = content.find("# ==============================\n# Shared CSS / Style\n# ==============================")
# Find the end of MAP_HTML
end_match = re.search(r'MAP_HTML = STYLE \+ """[\s\S]*?</html>\n"""', content[start_idx:])
if end_match:
    end_idx = start_idx + end_match.end()
else:
    # fallback to old string
    end_match = re.search(r'MAP_HTML = STYLE \+ """[\s\S]*?</a>\n"""', content[start_idx:])
    end_idx = start_idx + end_match.end()

new_content = content[:start_idx] + new_ui + content[end_idx:]

with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print("UI successfully updated in weather_app.py")
