import traceback
import weather_app
import pandas as pd

df = pd.read_csv('weather_sep_oct_2026.csv')
try:
    weather_app.prepare_and_train(df)
    print('SUCCESS!')
except Exception as e:
    traceback.print_exc()
