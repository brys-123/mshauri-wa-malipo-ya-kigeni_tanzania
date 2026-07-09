import logging
logging.basicConfig(level=logging.DEBUG)

from prophet import Prophet
import pandas as pd

df = pd.DataFrame({
    'ds': pd.date_range('2024-01-01', periods=40),
    'y': range(40)
})
m = Prophet()
m.fit(df)
print("SUCCESS")