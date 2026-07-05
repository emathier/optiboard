import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import polars as pl
    import numba as nb
    from mlforecast import MLForecast
    from mlforecast.target_transforms import Differences
    from window_ops.rolling import rolling_mean
    from utilsforecast.plotting import plot_series
    from numba import njit
    import lightgbm as lgb
    import datetime
    from datetime import timedelta
    from mlforecast.auto import (
        AutoLightGBM,
        AutoMLForecast,
        AutoModel,
        AutoRidge,
        ridge_space,
    )

    COVARIATES = ["cloud_cover", "apparent_temperature", "wind_speed_10m", "precipitation"]

    # Import data
    df = pl.read_parquet('data-files/occupancy-weather.parquet').unpivot(index=['timestamp'] + COVARIATES).filter(pl.col('variable').is_in(['hallenbad_oerlikon', 'hallenbad_city'])).rename({
        'timestamp' : 'ds',
        'variable' : 'unique_id',
        'value' : 'y'
    }
    ).drop_nulls()

    df = df.sort('ds').drop_nans().drop_nulls().with_columns(
        # Round and convert to naive datetime
        pl.col('ds').dt.round('5m').dt.replace_time_zone(None)
    ).upsample(
        time_column='ds', 
        every='5m', 
        by='unique_id'  # Upsample each pool individually
    ).fill_null(strategy='forward').drop_nans()

    max_date = df.select(pl.col('ds').max())[0,0]
    df_train = df.filter(pl.col('ds') < (max_date - timedelta(days=2))).to_pandas()
    df_test = df.filter(pl.col('ds') >= (max_date - timedelta(days=2))).to_pandas()
    return AutoLightGBM, AutoMLForecast, df_test, df_train, plot_series


@app.cell
def _(df_train, plot_series):
    plot_series(df_train, max_insample_length=12*24*7)
    return


@app.cell
def _(AutoLightGBM, AutoMLForecast):
    def decimal_hour(dates):
        """Converts a pandas datetime Index or Series to hours with decimals for mins/secs"""
        dt = dates.dt if hasattr(dates, "dt") else dates
        return dt.hour + dt.minute / 60.0 + dt.second / 3600.0

    # Setup automated hyperparameter search framework
    fcst = AutoMLForecast(
        models={'lgb': AutoLightGBM()},
        freq='5min',
        season_length=24 * 12,  # 288 ticks per day (5-min intervals)
        init_config=lambda trial: {
            'lags': [1, 12, 24 * 12, 24 * 12 * 7],
            'date_features': [decimal_hour, 'weekday', 'dayofyear']
        }
    )
    return (fcst,)


@app.cell
def _(df_train, fcst):
    # Perform cross-validation parameter tuning and fit the optimal model
    fcst.fit(
        df_train,
        n_windows=2, 
        h=24 * 12, 
        num_samples=5,  # Number of search trials (increase for a broader search)
    )
    return


@app.cell
def _(df_test, fcst):
    # Predict over the test horizon using the optimized model pipeline
    preds = fcst.predict(h=24 * 12, X_df=df_test)
    return (preds,)


@app.cell
def _(df_test, plot_series, preds):
    plot_series(df_test, preds, max_insample_length=96*12)
    return


if __name__ == "__main__":
    app.run()
