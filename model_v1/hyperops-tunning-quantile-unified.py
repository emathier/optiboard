from datetime import timedelta
import multiprocessing
import os
import time
import lightgbm as lgb
import optuna
import pandas as pd
import polars as pl
from mlforecast import MLForecast
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mape, rmse, quantile_loss
from functools import partial
from sklearn.metrics import mean_pinball_loss


def decimal_hour(dates):
    """Converts a pandas datetime Index or Series to hours with decimals for mins/secs"""
    dt = dates.dt if hasattr(dates, "dt") else dates
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


def prepare_data():
    """Loads and prepares the dataset using Polars, returning pandas DataFrames."""
    COVARIATES = [
        "cloud_cover",
        "apparent_temperature",
        "wind_speed_10m",
        "precipitation",
    ]

    # 1. Read, unpivot, filter, and cast to explicit datetime
    df = (
        pl.read_parquet("data-files/occupancy-weather.parquet")
        .unpivot(index=["timestamp"] + COVARIATES)
        .filter(pl.col("variable").is_in(["hallenbad_city"]))
        .rename({"timestamp": "ds", "variable": "unique_id", "value": "y"})
        .drop_nulls()
        .with_columns(pl.col("ds").dt.round("5m").dt.replace_time_zone(None))
    )

    # 2. Extract unique rows and sort for upsampling (Keep covariates)
    df = df.unique(subset=["unique_id", "ds"]).sort(["unique_id", "ds"])

    # 4. Upsample and fill missing values for ALL columns seamlessly
    df = (
        df.upsample(time_column="ds", every="5m", group_by="unique_id")
        .with_columns(pl.all().forward_fill().backward_fill())
    )

    # 5. Split train/test cleanly
    max_date = df.select(pl.col("ds").max())[0, 0]
    df_train = df.filter(pl.col("ds") < (max_date - timedelta(days=2))).to_pandas()

    return df_train


def objective_worker(trial_params, df_train, q):
    """Isolated worker process that handles LightGBM model building and evaluation."""
    try:
        # Enforce the dynamic constraint on num_leaves based on max_depth
        max_leaves_theoretical = 2 ** trial_params["max_depth"]
        if trial_params["num_leaves"] > max_leaves_theoretical:
            trial_params["num_leaves"] = max_leaves_theoretical

        # Create three separate models with the same hyperparameters but different alphas
        params_q25 = trial_params.copy()
        params_q25["alpha"] = 0.25

        params_q50 = trial_params.copy()
        params_q50["alpha"] = 0.50

        params_q75 = trial_params.copy()
        params_q75["alpha"] = 0.75

        fcst = MLForecast(
            models={
                "q25": lgb.LGBMRegressor(**params_q25),
                "q50": lgb.LGBMRegressor(**params_q50),
                "q75": lgb.LGBMRegressor(**params_q75),
            },
            freq="5min",
            target_transforms=[],
            lags=[1, 12, 24 * 12],
            date_features=[decimal_hour, "weekday", "dayofyear"],
        )

        cv_df = fcst.cross_validation(
            df=df_train, 
            h=24 * 12, 
            n_windows=7, 
            static_features=[]
        )

        loss_q25 = mean_pinball_loss(cv_df['y'], cv_df['q25'], alpha=0.25)
        loss_q50 = mean_pinball_loss(cv_df['y'], cv_df['q50'], alpha=0.50)
        loss_q75 = mean_pinball_loss(cv_df['y'], cv_df['q75'], alpha=0.75)
        score = loss_q25 + loss_q50 + loss_q75

        q.put(("SUCCESS", (score, loss_q25, loss_q50, loss_q75)))

    except Exception as e:
        q.put(("ERROR", str(e)))


def objective_with_timeout(trial, df_train):
    """Suggests parameters and tracks the execution window using a process-safe Queue."""
    # 1. Define the parameter space safely on the main thread
    trial_params = {
        "objective": "quantile",
        "alpha": 0.75,
        "boosting_type": "gbdt",
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "num_leaves": trial.suggest_int("num_leaves", 20, 8000),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-3, 0.3, log=True
        ),
        "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-16, 100.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "verbosity": -1,
        "n_jobs": 3,
    }

    q = multiprocessing.Queue()

    # Pass parameters and training data directly to avoid module-level pickling failures
    p1 = multiprocessing.Process(
        target=objective_worker, args=(trial_params, df_train, q)
    )

    start_time = time.time()
    p1.start()

    # Wait up to 70 seconds for the evaluation to finish
    p1.join(timeout=70)
    elapsed_time = time.time() - start_time
    trial.set_user_attr("runtime", elapsed_time)

    if p1.is_alive():
        p1.terminate()
        p1.join()
        raise optuna.TrialPruned("Trial exceeded the 70-second execution limit.")

    if q.empty():
        raise optuna.TrialPruned(
            "Trial process completed but returned no data status."
        )

    status, value = q.get()

    if status == "ERROR":
        print(f"-> Trial collapsed internally: {value}")
        raise optuna.TrialPruned(f"Child process error: {value}")

    score, loss_q25, loss_q50, loss_q75 = value
    trial.set_user_attr("loss_q25", loss_q25)
    trial.set_user_attr("loss_q50", loss_q50)
    trial.set_user_attr("loss_q75", loss_q75)

    return score


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run a small scale test")
    args = parser.parse_args()

    # Ensure folder structure exists
    os.makedirs("model_v1", exist_ok=True)

    print("Loading data...")
    df_train = prepare_data()

    if args.test:
        print("Running in test mode (small scale)...")
        # Keep only the last 15 days of training data to speed up the test
        df_train['ds'] = pd.to_datetime(df_train['ds'])
        max_ds = df_train['ds'].max()
        df_train = df_train[df_train['ds'] > (max_ds - timedelta(days=15))].copy()
        print(f"Test train size: {len(df_train)}")
        
        db_file = "optuna_study_unified_test.db"
        if os.path.exists(db_file):
            os.remove(db_file)
            
        n_trials = 2
        n_jobs = 1
        study_name = "mlforecast_lgbm_unified_test"
    else:
        db_file = "optuna_study_unified.db"
        n_trials = 5000
        n_jobs = 30
        study_name = "mlforecast_lgbm_unified"

    storage_url = f"sqlite:///{db_file}"

    sampler = optuna.samplers.TPESampler(n_startup_trials=1000 if not args.test else 1)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True,
    )

    print("Beginning hyperparameter optimization...")
    study.optimize(
        lambda trial: objective_with_timeout(trial, df_train),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    print("\nOptimization Complete!")
    if args.test:
        print("Trial attributes (logged losses):")
        for trial in study.trials:
            print(f"Trial {trial.number}: State={trial.state}, Value={trial.value}")
            print(f"  User attrs: {trial.user_attrs}")
        # Clean up the test database
        if os.path.exists(db_file):
            os.remove(db_file)
            print(f"Cleaned up test database: {db_file}")
    else:
        print("Best Parameters found:")
        print(study.best_params)