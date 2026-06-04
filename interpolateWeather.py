import polars as pl
from logging_config import get_logger
import duckdb as dd

log = get_logger("interpolateWeather")
log.debug("Starting to interpolate weather data")

# Load data
weather = dd.read_parquet("historical_weather.parquet").pl()
log.debug(f"Loaded weather data with shape {weather.shape}")
occupancy = dd.read_parquet("city-occupancy.parquet").pl()
log.debug(f"Loaded occupancy data with shape {occupancy.shape}")

# Ensure weather is sorted by time
weather = weather.sort("time")

# Combine all unique timestamps from both datasets
all_timestamps = (
    pl.concat([
        weather.select(pl.col("time").alias("timestamp")),
        occupancy.select(pl.col("timestamp")),
    ])
    .sort("timestamp")
)

# Left join weather data onto the full timeline
weather_cols = ["cloud_cover", "apparent_temperature", "wind_speed_10m", "precipitation"]
weather_full = all_timestamps.join(
    weather,
    left_on="timestamp",
    right_on="time",
    how="left",
).sort("timestamp")

# Linearly interpolate weather columns between hourly data points
weather_full = weather_full.with_columns([
    pl.col(col).interpolate(method="linear").backward_fill().forward_fill()
    for col in weather_cols
])

# Bind interpolated weather to each occupancy measurement
occupancy_weather = occupancy.join(weather_full, on="timestamp", how="left")

log.info(f"Occupancy with interpolated weather: {occupancy_weather.shape}")

# Null diagnostics table — before vs after for weather cols, final for all cols
null_diag = pl.DataFrame({
    "column": occupancy_weather.columns,
    "total": [occupancy_weather.height] * len(occupancy_weather.columns),
    "nulls": [occupancy_weather[c].null_count() for c in occupancy_weather.columns],
}).with_columns(
    (pl.col("nulls").cast(pl.Float64) / pl.col("total").cast(pl.Float64) * 100).round(2).alias("null_pct"),
    (pl.col("total") - pl.col("nulls")).alias("non_null"),
).with_columns(
    ((pl.col("non_null").cast(pl.Float64) / pl.col("total").cast(pl.Float64)) * 100).round(2).alias("non_null_pct"),
).select([
    pl.col("column"),
    pl.col("total"),
    pl.col("non_null"),
    pl.col("nulls"),
    pl.col("non_null_pct"),
    pl.col("null_pct"),
])

# Save result
occupancy_weather.write_parquet("weather-occupancy-weather.parquet")

# Print readable diagnostics table
pl.Config.set_tbl_rows(len(null_diag))
pl.Config.set_tbl_cols(len(null_diag.columns))
log.debug(f"\n{null_diag}")
log.info("Saved occupancy with interpolated weather to weather-occupancy-weather.parquet")