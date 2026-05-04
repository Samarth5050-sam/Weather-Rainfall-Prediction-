"""
Export the trained model state to weather_state.pkl
Saves as standard Python types (no custom classes) so any pickle viewer can open it.
"""
import os
import sys
import pickle
import torch

# Force fresh training by removing old state
if os.path.exists("weather_state.pkl"):
    os.remove("weather_state.pkl")
    print("Removed old weather_state.pkl to force retraining...")

# Import triggers training
import weather_app

if weather_app.STATE.get("trained"):
    print("Training finished! Exporting state...")
    
    wrapper = weather_app.STATE["model"]
    
    # Save as plain Python types (no custom classes)
    export_data = {
        "trained": True,
        "model_state_dict": wrapper.model.state_dict(),  # PyTorch state dict (plain dict)
        "model_config": {
            "input_size": wrapper.model.lstm.input_size,
            "hidden_size": wrapper.model.lstm.hidden_size,
            "num_layers": wrapper.model.lstm.num_layers,
        },
        "scaler": wrapper.scaler,                        # sklearn object (pickle-safe)
        "feature_names": wrapper.feature_names_in_,       # list of strings
        "df_raw": weather_app.STATE["df_raw"],            # pandas DataFrame
        "encoders": weather_app.STATE["encoders"],        # dict of LabelEncoders
        "metrics": weather_app.STATE["metrics"],          # dict of floats
        "unique_locations": weather_app.STATE["unique_locations"],  # list of strings
        "unique_dates": weather_app.STATE["unique_dates"],          # list of strings
    }
    
    with open("weather_state.pkl", "wb") as f:
        pickle.dump(export_data, f)
    
    size = os.path.getsize("weather_state.pkl")
    print(f"Export successful! Size: {size:,} bytes")
    print(f"Saved {len(export_data)} keys: {list(export_data.keys())}")
else:
    print("Failed to train model.")
