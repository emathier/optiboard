import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


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
    import optuna

    COVARIATES = ["cloud_cover", "apparent_temperature", "wind_speed_10m", "precipitation"]

    # Import data
    df = pl.read_parquet('data-files/occupancy-weather.parquet').unpivot(index=['timestamp'] + COVARIATES).filter(pl.col('variable').is_in(['hallenbad_city'])).rename({
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
        group_by='unique_id'  # <-- CRITICAL: Upsample each pool individually
    ).fill_null(strategy='forward').drop_nans()

    max_date = df.select(pl.col('ds').max())[0,0]
    df_train = df.filter(pl.col('ds') < (max_date - timedelta(days=2))).to_pandas()
    df_test = df.filter(pl.col('ds') >= (max_date - timedelta(days=2))).to_pandas()
    return (
        AutoMLForecast,
        AutoModel,
        MLForecast,
        df_test,
        df_train,
        lgb,
        optuna,
        plot_series,
    )


@app.cell
def _(df_train, plot_series):
    plot_series(df_train, max_insample_length=12*24*7)
    return


@app.cell
def _(MLForecast, df_train, lgb, plot_series):
    lgb_params = {
        'verbosity': -1,
        'num_leaves': 512,
    }


    def decimal_hour(dates):
        """Converts a pandas datetime Index or Series to hours with decimals for mins/secs"""
        # If it's a Series, use .dt accessor; if it's an Index, use it directly
        dt = dates.dt if hasattr(dates, "dt") else dates

        return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


    fcst = MLForecast(
        models={
            'avg': lgb.LGBMRegressor(**lgb_params),
            'q75': lgb.LGBMRegressor(**lgb_params, objective='quantile', alpha=0.75),
            'q25': lgb.LGBMRegressor(**lgb_params, objective='quantile', alpha=0.25),
        },
        freq='5min',  # our series have integer timestamps, so we'll just add 1 in every timestep
        target_transforms=[],
        lags=[1,12,24*12, 24*12*7],
        date_features=[decimal_hour, 'weekday','dayofyear']

    )
    prep = fcst.preprocess(df_train,static_features=[])
    prep
    plot_series(prep,max_insample_length=12*24*7)
    return decimal_hour, fcst


@app.cell
def _():
    #fcst.fit(df_train,static_features=[])
    return


@app.cell
def _(df_test, fcst):

    preds = fcst.predict(h=24 * 12, X_df=df_test)
    return (preds,)


@app.cell
def _(df_test, fcst):
    print(fcst.get_missing_future(72*12,  df_test))
    return


@app.cell
def _(df_test, plot_series, preds):
    plot_series(df_test, preds, max_insample_length=96*12)
    return


@app.cell
def _(AutoMLForecast, AutoModel, decimal_hour, lgb, optuna):


    def get_lgb_space(objective="l2", alpha=None):
        def tune_config(trial: optuna.Trial):
            cfg = {
                "verbosity": -1,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 512),
                "objective": objective,
                "n_jobs": -1,  # <-- LightGBM will utilize all available cores per trial
            }
            if alpha is not None:
                cfg["alpha"] = alpha
            return cfg
        return tune_config

    auto_fcst = AutoMLForecast(
        models={
            'avg': AutoModel(model=lgb.LGBMRegressor(), config=get_lgb_space('l2')),
            'q75': AutoModel(model=lgb.LGBMRegressor(), config=get_lgb_space('quantile', 0.75)),
            'q25': AutoModel(model=lgb.LGBMRegressor(), config=get_lgb_space('quantile', 0.25)),
        },
        freq='5min',
        init_config=lambda trial: {
            'lags': [1, 12, 24 * 12, 24 * 12 * 7],
            'date_features': [decimal_hour, 'weekday', 'dayofyear'],
        },
        fit_config=lambda trial: {
            'static_features': [] 
        }
    )
    return (auto_fcst,)


@app.cell
def _(auto_fcst, df_train):
    auto_fcst.fit(
        df_train,
        n_windows=3, 
        h=24 * 12, 
        num_samples=100
    )
    return


@app.cell
def _(df_test, fcst):
    # Predict over the test horizon using the optimized quantile models
    auto_preds = fcst.predict(h=24 * 12, X_df=df_test)
    return (auto_preds,)


@app.cell
def _(auto_preds, df_test, plot_series):
    # Visualize actuals side by side with your dynamic predictions
    plot_series(df_test, auto_preds, max_insample_length=96*12)
    return


@app.cell
def _(auto_fcst):
    auto_fcst.models['avg']

    return


@app.cell
def _(auto_fcst):
    import json

    # 1. Extract the best configuration dictionary for each tuned target
    best_hyperparameters = {}
    for model_name, study in auto_fcst.results_.items():
        # Retrieve the best trial's user configuration dictionary
        best_hyperparameters[model_name] = study.best_trial.user_attrs['config']

    # 2. Save the structured parameters to a local JSON file
    output_path = 'data-files/optimized_hyperparameters.json'
    with open(output_path, 'w') as f:
        json.dump(best_hyperparameters, f, indent=4, default=str)
    
    print(f"Successfully saved tuned parameters to {output_path}!")
    return


if __name__ == "__main__":
    app.run()
