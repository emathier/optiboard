import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():

    import marimo as mo
    import polars as pl
    import numpy as np

    from darts import TimeSeries
    from darts.models import RegressionModel, SKLearnModel
    from darts.dataprocessing.transformers import Scaler
    from darts.metrics import mape, smape

    TARGET = "hallenbad_city"
    COVARIATES = ["cloud_cover", "apparent_temperature", "wind_speed_10m", "precipitation"]

    # Load data in wide format: timestamp, pools, weather
    raw = pl.read_parquet('data-files/occupancy-weather.parquet')

    # Prepare: localize timezone, round to 5min, aggregate
    df = (
        raw.select(["timestamp", TARGET] + COVARIATES)
        .with_columns(pl.col("timestamp").dt.replace_time_zone(None))
        .sort("timestamp")
        .with_columns(pl.col("timestamp").dt.round("5m"))
        .group_by('timestamp')
        .agg(pl.col('*').mean())
        .drop_nulls(subset=[TARGET])
    )

    # Create aligned TimeSeries
    ts_target = TimeSeries.from_dataframe(df, time_col="timestamp", fill_missing_dates= False, value_cols=TARGET, freq="5min")
    ts_cov = TimeSeries.from_dataframe(df, time_col="timestamp", fill_missing_dates= False, value_cols=COVARIATES, freq="5min")



    # Train/val/test split (80/10/10)
    n = len(ts_target)
    train, temp = ts_target.split_before(int(0.8 * n))
    val, test = temp.split_before(int(0.5 * len(temp)))
    cov_train, cov_temp = ts_cov.split_before(int(0.8 * n))
    cov_val, cov_test = cov_temp.split_before(int(0.5 * len(cov_temp)))



    mo.md(f"""**Target:** {TARGET}  
    **Covariates:** {', '.join(COVARIATES)}  
    **Data:** {df.height} rows → **TS:** {len(ts_target)} steps (5-min)  
    **Train:** {len(train)} | **Val:** {len(val)} | **Test:** {len(test)}""")
    return SKLearnModel, cov_train, cov_val, mape, mo, smape, train, val


@app.cell
def _(train):
    train.drop_nulls()
    return


@app.cell
def _(
    SKLearnModel,
    cov_train,
    cov_val,
    mape,
    mo,
    scaler_target,
    smape,
    train,
    val,
):

    import lightgbm as lgb



    model = SKLearnModel(
        lags=96,                    # 8 hours lookback on target
        lags_past_covariates=96,    # 8 hours lookback on weather
        output_chunk_length=12,     # 1 hour forecast
        model=lgb.LGBMRegressor(
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=500,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        ),
        random_state=42,
    )

    model.fit(train, past_covariates=cov_train)

    # Forecast 1 hour ahead from val start
    forecast = model.predict(n=12, series=val, past_covariates=cov_val)
    forecast_orig = scaler_target.inverse_transform(forecast)
    actual_orig = scaler_target.inverse_transform(val[:12])

    mo.hstack([
        mo.md(f"**SMAPE:** {smape(actual_orig, forecast_orig):.2f}%  **MAPE:** {mape(actual_orig, forecast_orig):.2f}%"),
        forecast_orig.plot(label="Forecast"),
        actual_orig.plot(label="Actual"),
    ])

    return


if __name__ == "__main__":
    app.run()
