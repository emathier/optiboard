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
    return (
        MLForecast,
        df_train,
        evaluate,
        lgb,
        mape,
        mo,
        optuna,
        plt,
        rmse,
        time,
    )


@app.cell
def _(plt):
    def decimal_hour(dates):
        """Converts a pandas datetime Index or Series to hours with decimals for mins/secs"""
        # If it's a Series, use .dt accessor; if it's an Index, use it directly
        dt = dates.dt if hasattr(dates, "dt") else dates

        return dt.hour + dt.minute / 60.0 + dt.second / 3600.0

    def plot_cv(df, df_cv,  fname = None,uid = 'hallenbad_city', last_n=24 * 12):
        cutoffs = df_cv.query('unique_id == @uid')['cutoff'].unique()
        fig, ax = plt.subplots(nrows=len(cutoffs), ncols=1, figsize=(14, 6), gridspec_kw=dict(hspace=0.8))
        for cutoff, axi in zip(cutoffs, ax.flat):
            df.query('unique_id == @uid').tail(last_n).set_index('ds').plot(ax=axi, title=uid, y='y')
            df_cv.query('unique_id == @uid & cutoff == @cutoff').set_index('ds').plot(ax=axi, title=uid, y='LGBMRegressor')
        return fig

    return decimal_hour, plot_cv


@app.cell
def _(MLForecast, decimal_hour, df_train, evaluate, lgb, mape, rmse):



    lgb_params = {
        'verbosity': -1,
        'num_leaves': 60,
        'n_jobs' : 4
    }


    fcst = MLForecast(
        models=lgb.LGBMRegressor(**lgb_params, objective='quantile', alpha=0.5),
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



    return cv_df, cv_results


@app.cell
def _(cv_df, cv_results, df_train, mo, plot_cv):
    mo.hstack([cv_results, plot_cv(df_train, cv_df,last_n=24*12*3)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optuna Stack for avg predection.
    Try different
    """)
    return


@app.cell
def _(
    MLForecast,
    decimal_hour,
    df_train,
    evaluate,
    lgb,
    mape,
    optuna,
    rmse,
    time,
):
    def objective(trial):
        # 1. The Largest Plausible Parameter Space

        start_time = time.time()

        # Clean 3-line time-checking callback function
        def time_checker(env):
            if time.time() - start_time > 5: 
                raise optuna.TrialPruned("Trial exceeded 7-second runtime limit.")


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
            'n_jobs' : 4,
            'callbacks': [time_checker]
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

        return cv_results.values[0,1]


    return (objective,)


@app.cell
def _(objective, optuna):
    import signal

    sampler = optuna.samplers.TPESampler(n_startup_trials=20)

    # 1. Create the In-Memory Study
    study = optuna.create_study(
        direction="minimize", 
        sampler=sampler
    )

    # 2. Hard kill wrapper via UNIX Signal Alarms
    def stable_objective_with_timeout(trial):
        # Define a custom handler to raise an exception when the alarm fires
        def alarm_handler(signum, frame):
            raise TimeoutError("Trial hard-killed: Exceeded 7-second time limit.")

        # Register the handler
        signal.signal(signal.SIGALRM, alarm_handler)
    
        # Set the alarm for 7 seconds
        signal.alarm(10)
    
        try:
            result = objective(trial)
            return result
        except TimeoutError as e:
            raise optuna.TrialPruned(str(e))
        except Exception as e:
            raise optuna.TrialPruned(f"Trial failed due to internal error: {e}")
        finally:
            # IMPORTANT: Cancel the alarm so it doesn't interrupt subsequent code
            signal.alarm(0)

    # 3. Optimize 
    study.optimize(
        stable_objective_with_timeout,           
        n_trials=400,         
        n_jobs=1,  
        show_progress_bar=True
    )

    # 4. View Results Safely
    try:
        print("Best Trial Metrics:", study.best_value)
        print("Best Hyperparameters:", study.best_params)
    except ValueError:
        print("All trials were pruned/killed within the 7-second limit.")
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
