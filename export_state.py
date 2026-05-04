"""
Export trained model state as pure Python data types.
No external module dependencies - any pickle viewer can open this.
"""
import os
import sys
import pickle
import numpy as np

# Remove old state to force retraining
if os.path.exists("weather_state.pkl"):
    os.remove("weather_state.pkl")
    print("Removed old weather_state.pkl...")

# Import triggers training
import weather_app

if weather_app.STATE.get("trained"):
    print("Training finished! Exporting...")
    
    wrapper = weather_app.STATE["model"]
    df = weather_app.STATE["df_raw"]
    
    # Convert EVERYTHING to plain Python types
    # Model weights: convert torch tensors to numpy arrays
    state_dict_numpy = {}
    for key, tensor in wrapper.model.state_dict().items():
        state_dict_numpy[key] = tensor.cpu().numpy()
    
    # DataFrame: convert to dict of lists
    df_dict = {}
    for col in df.columns:
        df_dict[col] = df[col].tolist()
    
    # Encoders: extract the classes_ arrays as plain lists
    encoders_plain = {}
    for col_name, le in weather_app.STATE["encoders"].items():
        encoders_plain[col_name] = le.classes_.tolist()
    
    # Scaler: extract mean_ and scale_ as plain lists
    scaler = wrapper.scaler
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "n_features": int(scaler.n_features_in_),
    }
    
    export_data = {
        "trained": True,
        "model_weights": state_dict_numpy,       # dict of numpy arrays
        "model_config": {
            "input_size": int(wrapper.model.lstm.input_size),
            "hidden_size": int(wrapper.model.lstm.hidden_size),
            "num_layers": int(wrapper.model.lstm.num_layers),
        },
        "scaler_data": scaler_data,               # dict of plain lists
        "scaler": scaler,                          # sklearn object (for app loading)
        "feature_names": list(wrapper.feature_names_in_),
        "df_raw_dict": df_dict,                    # dict of lists (for viewer)
        "df_raw": df,                              # pandas DataFrame (for app loading)
        "encoders_plain": encoders_plain,          # dict of lists (for viewer)
        "encoders": weather_app.STATE["encoders"], # sklearn objects (for app loading)
        "metrics": {k: float(v) for k, v in weather_app.STATE["metrics"].items()},
        "unique_locations": list(weather_app.STATE["unique_locations"]),
        "unique_dates": list(weather_app.STATE["unique_dates"]),
    }
    
    with open("weather_state.pkl", "wb") as f:
        pickle.dump(export_data, f, protocol=4)
    
    size = os.path.getsize("weather_state.pkl")
    print(f"Export successful! Size: {size:,} bytes")
    print(f"Keys: {list(export_data.keys())}")
    
    # Verify it loads cleanly
    with open("weather_state.pkl", "rb") as f:
        verify = pickle.load(f)
    print(f"Verification: Loaded {len(verify)} keys successfully!")
    print(f"Metrics: {verify['metrics']}")
    print(f"Locations: {verify['unique_locations']}")
else:
    print("Failed to train model.")
