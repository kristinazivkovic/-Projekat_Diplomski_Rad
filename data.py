from dotenv import load_dotenv
import os
from lse import LSE
import pandas as pd


load_dotenv()
api_key = os.getenv("LSE_KEY")
client = LSE(api_key=api_key)

df = client.candles("AAPL", start="2010-01-01", end="2010-02-01", timeframe="5m")
df = pd.DataFrame(df)
print(len(df))
print(df['high'].describe())
print(df.head())