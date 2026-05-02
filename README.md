<h1 align="center">🌦️ Advanced Time-Series Weather Predictor AI</h1>

<p align="center">
  A highly advanced, full-stack predictive weather application powered by a <b>PyTorch Long Short-Term Memory (LSTM) Neural Network</b>. It uses time-series sequence modeling to predict exact proper rainfall amounts (in millimeters) based on historical meteorological data.
</p>

## ✨ Key Features

- **🧠 True Time-Series LSTM Architecture**: Instead of basic snapshot classification, the model extracts a sliding sequence window (past 3 days) of weather data to calculate trends and temporal dependencies before making predictions.
- **☔ Regression-Based Predictions**: Explicitly calculates the exact mathematical amount of rainfall (`mm`) expected, rather than a generic "Yes/No".
- **📊 Predictive Basis Visualization**: A dynamic graphing engine plots the historical *Rainfall* and *Humidity* directly on the frontend UI, visually explaining the exact sequence of data the LSTM evaluated to make its decision.
- **🗺️ Interactive Heatmap**: Generates a live interactive geographical heatmap using `Folium`, clustering and visualizing rainfall severity across different regions in India.
- **✨ Premium Glassmorphism UI**: A stateless, responsive, state-of-the-art frontend completely generated via Flask using custom animations and CSS gradients.

## 🛠️ Technology Stack

* **Deep Learning Framework:** PyTorch (`torch`, `nn.LSTM`)
* **Backend Server:** Flask
* **Data Processing Pipeline:** Pandas, NumPy, Scikit-learn (`StandardScaler`, `LabelEncoder`)
* **Data Visualization:** Matplotlib (headless Base64 generation), Folium (interactive maps)
* **Model Serialization:** Dill (Robust architecture mapping)
* **Frontend:** Vanilla HTML/CSS (Glassmorphism design, Outfit Font)

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Samarth5050-sam/Weather-Rainfall-Prediction-.git
cd Weather-Rainfall-Prediction-
```

### 2. Install Dependencies
Make sure you have Python 3.9+ installed. Then run:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the Flask development server:
```bash
python weather_app.py
```

### 4. Access the UI
Open your browser and navigate to:
```text
http://localhost:8050/
```

## 🧠 Model Pipeline Overview

1. **Data Ingestion**: The app actively scans and cleans the target dataset (`weather_sep_oct_2026.csv`), dropping any broken rows or corrupted NaN sequences.
2. **Sequential Windowing**: It isolates unique locations, sorts them strictly by date, and builds `(Sequence_Length=3)` batches. 
3. **Training & Regularization**: Fed into an optimized dual-layer PyTorch LSTM with a `dropout` regularization of 0.2. Loss is calculated using `nn.MSELoss()`.
4. **Metrics Tracking**: Evaluates model performance post-training generating custom `Root Mean Square Error (RMSE)` and `Mean Absolute Error (MAE)` scores dynamically.

---
*Built by Samarth.*
