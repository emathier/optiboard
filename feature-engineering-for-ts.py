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

    TARGET = ["hallenbad_city"]
    COVARIATES = ["cloud_cover", "apparent_temperature", "wind_speed_10m", "precipitation"]

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

    # Time lags
    df = df.with_columns(
        pl.col(TARGET).shift(12*8).alias(f"lag_8h_{TARGET[0]}"),
        pl.col(TARGET).shift(12*24).alias(f"lag_24h_{TARGET[0]}"),
        pl.col(TARGET).shift(12*24*7).alias(f"lag_1week_{TARGET[0]}"),
    )

    df
    return COVARIATES, TARGET, df, go, lgb, mo, pl, sns


@app.cell
def _(TARGET, df, pl, sns):
    # plot a day
    date_to_plot = pl.date(2026,7,1)
    df_plot = df.filter(pl.col('date') == date_to_plot)
    sns.relplot(data=df_plot, x = "hour_of_day", y = TARGET[0], kind="line")
    return


@app.cell
def _(df, pl):
    # 1. Get the max date scalar and extract its time zone
    max_date = df["date"].max()
    tz = df["timestamp"].dtype.time_zone

    # 2. Convert directly to local 8:00 AM datetimes using Polars expressions
    val_cutoff = pl.select((max_date - pl.duration(days=2)).cast(pl.Datetime).dt.offset_by("8h").dt.replace_time_zone(tz)).item()
    test_cutoff = pl.select((max_date - pl.duration(days=1)).cast(pl.Datetime).dt.offset_by("8h").dt.replace_time_zone(tz)).item()

    # 3. Split efficiently
    train_df = df.filter(pl.col("timestamp") < val_cutoff)
    val_df = df.filter(pl.col("timestamp").is_between(val_cutoff, test_cutoff, closed="left"))
    test_df = df.filter(pl.col("timestamp") >= test_cutoff)
    return test_df, train_df, val_df


@app.cell
def _():
    return


@app.cell
def _(COVARIATES, TARGET, go, lgb, mo, test_df, train_df, val_df):
    FEATURES = COVARIATES + ["hour_of_day", "weekday", "month", f"lag_8h_{TARGET[0]}", f"lag_24h_{TARGET[0]}", f"lag_1week_{TARGET[0]}"]

    train_data = lgb.Dataset(train_df[FEATURES].to_numpy(), label=train_df[TARGET[0]].to_numpy())
    val_data = lgb.Dataset(val_df[FEATURES].to_numpy(), label=val_df[TARGET[0]].to_numpy(), reference=train_data)

    model = lgb.train({"objective":"regression","metric":"mae","verbosity":-1,"random_state":42}, 
                      train_data, num_boost_round=1000, valid_sets=val_data, callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    plots = []
    for name, d in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        d_plot = d if len(d) < 5000 else d.sample(n=5000, seed=42)
        pred = model.predict(d_plot[FEATURES].to_numpy())
        fig = go.Figure()
        fig.add_scatter(x=d_plot["timestamp"], y=d_plot[TARGET[0]], name="actual", mode="lines+markers", marker=dict(size=2))
        fig.add_scatter(x=d_plot["timestamp"], y=pred, name="pred", opacity=.7, mode="lines+markers", marker=dict(size=2))
        fig.update_layout(title=name, width=900, height=350)
        plots.append(mo.vstack([mo.md(f"**{name}**"), mo.ui.plotly(fig)]))

    mo.vstack(plots)
    return


if __name__ == "__main__":
    app.run()
