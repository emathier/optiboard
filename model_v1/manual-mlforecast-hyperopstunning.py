import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import marimo as mo
    import pandas as pd
    import polars as pl
    import numba as nb
    from mlforecast import MLForecast
    from utilsforecast.plotting import plot_series
    from utilsforecast.evaluation import evaluate
    from utilsforecast.losses import rmse, mape
    import lightgbm as lgb
    from datetime import timedelta
    import optuna
    import matplotlib.pyplot as plt
    import time
    import timeout_decorator
    import multiprocessing


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
        group_by='unique_id' 
    ).fill_null(strategy='forward').drop_nans()

    max_date = df.select(pl.col('ds').max())[0,0]
    df_train = df.filter(pl.col('ds') < (max_date - timedelta(days=2))).to_pandas()
    df_test = df.filter(pl.col('ds') >= (max_date - timedelta(days=2))).to_pandas()
    mo.vstack([plot_series(df_train, max_insample_length=12*24*7), plot_series(df_test, max_insample_length=48*12)])


    def decimal_hour(dates):
        """Converts a pandas datetime Index or Series to hours with decimals for mins/secs"""
        # If it's a Series, use .dt accessor; if it's an Index, use it directly
        dt = dates.dt if hasattr(dates, "dt") else dates

        return dt.hour + dt.minute / 60.0 + dt.second / 3600.0

    return (
        MLForecast,
        decimal_hour,
        df_train,
        evaluate,
        lgb,
        mape,
        mo,
        multiprocessing,
        optuna,
        rmse,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optuna Stack for avg predection.
    Try different
    """)
    return


@app.cell
def _(MLForecast, decimal_hour, df_train, evaluate, lgb, mape, q, rmse):
    def objective(trial, return_obj):
        try:

            # 1. The Largest Plausible Parameter Space


            lgb_params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",

                # Structural Parameters
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "num_leaves": trial.suggest_int("num_leaves", 20, 3000), 
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),

                # Learning Rate & Trees
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 3000),

                # Sampling & Regularization (to prevent overfitting)
                "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                'verbosity': -1,
                'n_jobs' : 1
            }

            # 2. Add Dynamic Constraints (Crucial for leaf-wise growth)
            # Ensure num_leaves does not exceed theoretical maximum for a given depth
            max_leaves_theoretical = 2 ** lgb_params["max_depth"]
            if lgb_params["num_leaves"] > max_leaves_theoretical:
                lgb_params["num_leaves"] = max_leaves_theoretical

            fcst = MLForecast(
                models=lgb.LGBMRegressor(**lgb_params),
                freq='5min',
                target_transforms=[],
                lags=[1,12,24*12, 24*12],
                date_features=[decimal_hour, 'weekday','dayofyear'],

            )

            cv_df = fcst.cross_validation(
                df=df_train,
                h=24*12,
                n_windows=3,
                static_features=[]
            )

            cv_results = evaluate(
                cv_df.drop(columns='cutoff'),
                metrics=[rmse,mape],
                agg_fn='mean',
            )

            score = cv_results.values[0, 1]

            # Put the raw float into the queue for the parent process
            q.put(score)
            return score

        except Exception as e:
            # Pass the error string back to the main process so you can read it!
            q.put(("ERROR", str(e)))


    return (objective,)


@app.cell
def _(multiprocessing, objective, optuna):
    def objective_with_timout(trial):
            # Use a Queue instead of Value to pass data reliably across processes
            q = multiprocessing.Queue()

            # We pass the queue 'q' instead of return_obj
            p1 = multiprocessing.Process(target=objective, args=(trial, q))
            p1.start()

            # Wait up to 7 seconds for the process to finish
            p1.join(timeout=10.0)

            if p1.is_alive():
                p1.terminate()
                p1.join()
                raise optuna.TrialPruned("Trial exceeded the 7-second execution limit.")

            # If the queue is empty, the objective function crashed or failed silently
            if q.empty():
                raise optuna.TrialPruned("Trial process completed but returned no results.")

            return q.get()

    return (objective_with_timout,)


@app.cell
def _(objective_with_timout, optuna):


    # 1. Setup Database
    db_file = "model_v1/optuna_study_avg.db"
    storage_url = f"sqlite:///{db_file}"

    # 2. Create the study inside the SQL database
    sampler = optuna.samplers.TPESampler(n_startup_trials=20)
    study = optuna.create_study(
        study_name="mlforecast_lgbm",
        direction="minimize", 
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True
    )

    study.optimize(
        objective_with_timout,           
        n_trials=1,         
        n_jobs=1,  
        show_progress_bar=True
    )
    return (study,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    lgb_params = {
        'verbosity': -1,
        'num_leaves': 50,
        'n_jobs' : 4
    }


    fcst = MLForecast(
        models={
            'avg': lgb.LGBMRegressor(**lgb_params, objective='quantile', alpha=0.5),
            'q75': lgb.LGBMRegressor(**lgb_params, objective='quantile', alpha=0.75),
            'q25': lgb.LGBMRegressor(**lgb_params, objective='quantile', alpha=0.25),
        },
        freq='5min',  # our series have integer timestamps, so we'll just add 1 in every timestep
        target_transforms=[],
        lags=[1,12,24*12, 24*12],
        date_features=[decimal_hour, 'weekday','dayofyear'],

    )

    cv_df = fcst.cross_validation(
        df=df_train,
        h=24*12,
        n_windows=3,
        static_features=[]
    )

    cv_results = evaluate(
        cv_df.drop(columns='cutoff'),
        metrics=[rmse,mape],
        agg_fn='mean',
    )
    """)
    return


@app.cell
def _(study):
    study.best_params
    return


@app.cell
def _():
    {'max_depth': 5, 'num_leaves': 1527, 'min_child_samples': 72, 'learning_rate': 0.07218777418692918, 'n_estimators': 1002, 'subsample': 0.6451303292416953, 'colsample_bytree': 0.9108368933561712, 'reg_alpha': 1.1812973236692408e-07, 'reg_lambda': 0.6058357708882873}
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RMSE prediction best.
    {
      "max_depth": 5,
      "num_leaves": 1527,
      "min_child_samples": 72,
      "learning_rate": 0.07218777418692918,
      "n_estimators": 1002,
      "subsample": 0.6451303292416953,
      "colsample_bytree": 0.9108368933561712,
      "reg_alpha": 1.1812973236692408e-07,
      "reg_lambda": 0.6058357708882873
    }
    """)
    return


app._unparsable_cell(
    r"""
    study.visualization.
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
