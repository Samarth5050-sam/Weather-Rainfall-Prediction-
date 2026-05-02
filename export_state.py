import dill
import weather_app

# weather_app.py already trains the model on import
if weather_app.STATE.get("trained"):
    print("Training finished! Exporting complete STATE to weather_state.pkl...")
    with open("weather_state.pkl", "wb") as f:
        dill.dump(weather_app.STATE, f)
    print("Export successful. Size:", f.tell(), "bytes.")
else:
    print("Failed to train model.")
