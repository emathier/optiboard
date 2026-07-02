import requests
import streamlit as st
import polars as pl
from logging_config import get_logger
import duckdb as dd
from contextlib import contextmanager
import time

@contextmanager
def time_this(label):
    start = time.time()  
    yield
    log.debug(f"{label} took {time.time() - start:.4f}s")

log = get_logger("fetchWeatherForecast")
log.debug("Starting to fetch weather forecast data")

# Fetch forecast for the next 7 days
request = "https://api.open-meteo.com/v1/forecast?latitude=47.3667&longitude=8.55&forecast_days=7&hourly=cloud_cover,apparent_temperature,wind_speed_10m,precipitation&timezone=Europe%2FBerlin"

with time_this("HTTP request"):
    response = requests.get(request)
      
log.debug(f"Received response with status code {response.status_code}")

# Run request
response.raise_for_status()  
log.debug("Response status code is OK, proceeding to parse JSON")
data = response.json()
log.info(f"Number of hourly data points: {len(data['hourly']['time'])}")

# Convert to Polars DataFrame
df = pl.DataFrame(data['hourly'])

# Convert time column to datetime with correct timezone.
# Data from request is in Europe/Berlin timezone. E.g 2026-01-04T00:00 
df = df.with_columns(
    pl.col("time")
    .str.to_datetime()
    .dt.replace_time_zone("Europe/Berlin", non_existent="null", ambiguous="null")
)

print(df.head())

#write to file
df.write_parquet("data-files/weather_forecast.parquet")
log.info("Saved weather forecast data to weather_forecast.parquet")