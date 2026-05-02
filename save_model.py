import dill
import weather_app

if weather_app.STATE.get("trained"):
    with open("weather_lstm_model.pkl", "wb") as f:
        dill.dump(weather_app.STATE["model"], f)
    print("Model successfully saved to weather_lstm_model.pkl")
else:
    print("Failed to train model, cannot save.")
