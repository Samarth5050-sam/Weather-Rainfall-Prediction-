# 🌦️ Weather Rainfall Prediction — Group Presentation Notes

---

## 🎯 One-Line Summary
> We built a **Deep Learning web application** that predicts **exact rainfall (in mm)** for the next day across **20 Indian cities** using an **LSTM Neural Network** trained on real weather data.

---

## 📌 Problem Statement
- Rainfall prediction is critical for agriculture, disaster management, and daily planning
- Traditional methods rely on expensive radar/satellite systems
- **Our goal:** Build an accessible, web-based tool that predicts rainfall using basic weather measurements

---

## 📊 Dataset
| Detail | Value |
|---|---|
| File | `weather_sep_oct_2026.csv` |
| Records | **366 rows** |
| Cities | **20** (Mumbai, Delhi, Bengaluru, Chennai, Kolkata, etc.) |
| Date Range | Sept–Oct 2026 |
| Features | **21 weather parameters** (temp, humidity, pressure, wind, clouds, etc.) |
| Target | **RISK_MM** — tomorrow's rainfall in mm |

---

## 🧠 Model — How It Works (Simple Explanation)

```
TODAY'S WEATHER DATA → LSTM Neural Network → TOMORROW'S RAINFALL (mm)
```

**3 Steps:**

1. **Input:** We feed 21 weather features (temperature, humidity, pressure, wind speed, cloud cover, etc.) for a selected city and date
2. **Process:** The LSTM (Long Short-Term Memory) neural network finds hidden patterns between these features and rainfall
3. **Output:** A single number — predicted rainfall in millimeters (e.g., **5.4 mm**)

---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| AI Model | **PyTorch LSTM** |
| Backend | **Flask (Python)** |
| Frontend | **HTML/CSS + Chart.js** |
| Maps | **Folium + HeatMap** |
| Data Processing | **Pandas, Scikit-learn** |
| Deployment | **Gunicorn + Render/Vercel** |

---

## ⚡ Key Features

1. **🧠 LSTM Rainfall Prediction** — Predicts exact mm of rain, not just Yes/No
2. **📊 Interactive Charts** — Hoverable Chart.js graphs showing historical trends
3. **🗺️ Live HeatMap** — India map with real-time predictions for all 20 cities
4. **🌡️ Intensity Gauge** — Visual bar showing Dry / Light / Heavy / Severe
5. **📥 Data Export** — Download any forecast as a JSON file
6. **☁️ Cloud Deployable** — Optimized for hosting with CPU-only PyTorch

---

## 🏗️ Architecture Flow

```
CSV Dataset (366 rows)
        ↓
   Data Cleaning (drop NaN)
        ↓
   Encode Text → Numbers (LabelEncoder)
        ↓
   Normalize Features (StandardScaler)
        ↓
   Train LSTM (100 epochs, weighted loss)
        ↓
   Save to weather_state.pkl
        ↓
   Flask Web App loads pkl → Serves predictions
```

---

## 💡 Smart Design Decisions

| Challenge | Our Solution |
|---|---|
| 72% of days have zero rain → model always predicts 0 | **Weighted Loss** — rainy days get 10x importance during training |
| PyTorch is 800MB → too large for free hosting | **CPU-only PyTorch** (~150MB) |
| Training takes 30 seconds → server times out | **Pre-trained state file** — loads in <1 second |
| Standard pickle can't save custom classes | **Dill library** for serialization |

---

## 📈 Model Performance

| Metric | Value | Meaning |
|---|---|---|
| **RMSE** | 3.78 mm | Average prediction error |
| **MAE** | 2.00 mm | Average absolute error |
| **R²** | 0.10 | Model explains 10% of variance |

> **Note:** R² is low because we only have ~18 data points per city. Professional weather models use millions of satellite observations. Despite this, our model successfully predicts non-zero values for actual rain events.

---

## 🖥️ Live Demo Points

When presenting, show these 4 pages:

1. **Dashboard** (`/`) — Select city and date, see dataset stats
2. **Prediction** (`/predict`) — Show the rainfall number, intensity gauge, and interactive chart
3. **Map** (`/map`) — Click different cities on the HeatMap to see their forecasts
4. **Metrics** (`/metrics`) — Show RMSE, MAE, R² to demonstrate model evaluation

---

## 🎤 Sample Talking Script (2 Minutes)

> *"Our project is a Weather Rainfall Prediction system built using Deep Learning.*
>
> *We have a dataset of 366 weather observations across 20 Indian cities from September-October 2026. Each row contains 21 features like temperature, humidity, pressure, wind speed, and cloud cover.*
>
> *We trained an LSTM Neural Network — which is a type of AI that can learn patterns from sequential data — to predict how many millimeters of rain will fall tomorrow, based on today's weather conditions.*
>
> *A key challenge was that 72% of our data had zero rainfall, so the model kept predicting zero. We solved this by using a weighted loss function that gives 10x more importance to rainy days during training.*
>
> *The web app is built with Flask and features interactive Chart.js visualizations, a live HeatMap of India showing predictions for all cities, and the ability to export data as JSON.*
>
> *The entire system is deployment-ready on cloud platforms like Render, with the model pre-trained and saved to a .pkl file for instant loading."*

---

## ❓ Anticipated Questions & Answers

**Q: Why LSTM and not Random Forest?**
> LSTM can learn temporal patterns — that yesterday's weather affects today's. Random Forest treats each data point independently.

**Q: Why is accuracy low?**
> We have only ~18 data points per city. Real weather prediction uses millions of satellite data points. For our dataset size, the model performs well.

**Q: Can it predict for cities not in the dataset?**
> No, the model is trained only on the 20 cities in the CSV. Adding more cities requires more training data.

**Q: What's the most important feature for prediction?**
> Humidity at 3PM, atmospheric pressure, cloud cover, and whether it rained today are the strongest indicators.
