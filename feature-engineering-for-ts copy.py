import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    # Imports
    import marimo as mo
    import polars as pl
    import plotly.graph_objects as go
    import lightgbm as lgb
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime,timedelta
    import numpy as np

    TARGET = ["hallenbad_city"]
    COVARIATES = ["cloud_cover", "apparent_temperature", "wind_speed_10m", "precipitation"]
    UNWANTED = ['date']

    # Settings
    LAGS = False
    UID = True

    # Data import
    df = pl.read_parquet("data-files/occupancy-weather.parquet").with_columns(
        pl.col('timestamp').dt.round("5m")
    ).sort(by="timestamp").upsample(time_column="timestamp", every="5m")


    df = df.with_columns(
        (
            pl.col("timestamp").dt.hour() + 
            (pl.col("timestamp").dt.minute() / 60) + 
            (pl.col("timestamp").dt.second() / 3600)
        ).alias("hour_of_day"),
        pl.col('timestamp').dt.weekday().alias("weekday"),
        pl.col('timestamp').dt.day().alias("day_of_month"),
        pl.col('timestamp').dt.ordinal_day().alias("day_of_year"),
        pl.col('timestamp').dt.month().alias("month"),
        pl.col('timestamp').dt.year().alias('year'),
        pl.col('timestamp').dt.epoch().alias('epoch'),
        pl.col('timestamp').dt.date().alias('date')
    )

    if LAGS: 
        lags = [
            ("1h", 12),
            ("8h", 12 * 8),
            ("24h", 12 * 24),
            ("1week", 12 * 24 * 7),
            *[(f"t-{t}", -t) for t in np.arange(1,10)]
        ]
        # Time lags
        df = df.with_columns(
            [pl.col(TARGET).shift(lag[1]).alias(f"lag_{lag[0]}_{TARGET[0]}") for lag in lags]
        )

    
    if UID:
        df = df.with_columns(
            pl.lit("single_series").alias("unique_id")
        )



    # 1. Get the max date scalar and extract its time zone
    max_date = df["date"].max()
    tz = df["timestamp"].dtype.time_zone

    # 2. Convert directly to local 8:00 AM datetimes using Polars expressions
    val_cutoff = pl.select((max_date - pl.duration(days=2)).cast(pl.Datetime).dt.offset_by("8h").dt.replace_time_zone(tz)).item()
    test_cutoff = pl.select((max_date - pl.duration(days=1)).cast(pl.Datetime).dt.offset_by("8h").dt.replace_time_zone(tz)).item()

    df = df.drop_nulls(subset=TARGET)
    # 3. Split efficiently
    train_df = df.filter(pl.col("timestamp") < val_cutoff).drop(UNWANTED)
    val_df = df.filter(pl.col("timestamp").is_between(val_cutoff, test_cutoff, closed="left")).drop(UNWANTED)
    test_df = df.filter(pl.col("timestamp") >= test_cutoff).drop(UNWANTED)


    train_df.write_parquet(f"data-files/lagged-train_df.parquet")
    val_df.write_parquet(f"data-files/lagged-val_df.parquet")
    test_df.write_parquet(f"data-files/lagged-test_df.parquet")
    return test_df, train_df


@app.cell
def _(train_df):
    train_df
    return


@app.cell
def _(test_df):
    test_df
    return


if __name__ == "__main__":
    app.run()
