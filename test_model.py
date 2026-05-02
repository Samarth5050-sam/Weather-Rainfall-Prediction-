import dill

# Load the binary file back into a Python object
with open('weather_lstm_model.pkl', 'rb') as f:
    my_model = dill.load(f)

# Now you can inspect it!
print("Model loaded successfully!")
print("Features used:", my_model.feature_names_in_)
print("PyTorch Architecture:\n", my_model.model)
