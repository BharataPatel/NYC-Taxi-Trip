import pandas as pd
import requests
from io import BytesIO

url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"

print("Downloading dataset...")

response = requests.get(url)

print("Loading data...")

df = pd.read_parquet(BytesIO(response.content))

print("Rows:", len(df))
print("Columns:", len(df.columns))

df.to_csv("NYC_Taxi_2024.csv", index=False)

print("CSV created successfully!")
print("File: NYC_Taxi_2024.csv")