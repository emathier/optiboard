import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor




    return TimeSeriesDataFrame, TimeSeriesPredictor, pl


@app.cell
def _(TimeSeriesDataFrame, pl):
    # 1. Read and clean the base data
    df = pl.read_parquet('data-files/occupancy-weather.parquet').with_columns(
        pl.col('timestamp').cast(pl.Datetime).dt.round('5m')
    )

    df_ag = df.select([
        pl.lit("hallenbad_city").alias("item_id"),
        pl.col("timestamp"),
        pl.col("hallenbad_city").alias("value"),  # Your target
        pl.col("cloud_cover"),
        pl.col("apparent_temperature"),
        pl.col("wind_speed_10m"),
        pl.col("precipitation"),
    ]).drop_nulls(subset=["value"]) # Drop rows where target itself is missing

    # 4. Convert to AutoGluon TimeSeriesDataFrame
    train_data = TimeSeriesDataFrame.from_data_frame(
        df_ag.to_pandas(),
        id_column="item_id",
        timestamp_column="timestamp"
    ).convert_frequency('5min')
    return df, train_data


@app.cell
def _(TimeSeriesPredictor, train_data):
    predictor = TimeSeriesPredictor(
        prediction_length=6*12,
        target="value",
        eval_metric="WQL",
    )

    predictor.fit(
        train_data,
        presets="medium_quality",
        time_limit=600,
    )
    return (predictor,)


@app.cell
def _(predictor, train_data):
    import matplotlib.pyplot as plt

    # 1. Generate predictions on your actual training data (or a local test set if you have one)
    predictions = predictor.predict(train_data)

    # 2. Plot using your local train_data, which actually contains the 'value' column
    predictor.plot(
        data=train_data, 
        predictions=predictions, 
        quantile_levels=[0.1, 0.9], 
        max_history_length=200, 
        max_num_item_ids=4
    )

    plt.show() # Standard practice to display the matplotlib figure in notebooks
    return


@app.cell
def _(df):
    df
    return


if __name__ == "__main__":
    app.run()
