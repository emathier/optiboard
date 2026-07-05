import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import lightgbm as gbm
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    TARGET = "hallenbad_city"
    from mlforecast import MLForecast
    import lightgbm as lgb
    from sklearn.model_selection import ParameterGrid
    from sklearn.metrics import mean_squared_error, mean_absolute_error as mae
    import optuna
    import concurrent.futures


    from sklearn.experimental import enable_halving_search_cv  # Required!
    from sklearn.model_selection import HalvingGridSearchCV, TimeSeriesSplit, GridSearchCV

    train_df = pl.read_parquet(f"data-files/lagged-train_df.parquet")
    val_df = pl.read_parquet(f"data-files/lagged-val_df.parquet")
    test_df = pl.read_parquet(f"data-files/lagged-test_df.parquet")
    return (
        MLForecast,
        TARGET,
        concurrent,
        lgb,
        mae,
        mean_squared_error,
        optuna,
        plt,
        sns,
        test_df,
        train_df,
        val_df,
    )


@app.cell
def _(TARGET, test_df, train_df, val_df):
    OTHER_POOLS = [
        'adliswil', 'bern_marzili', 'bern_wylerbad', 'bern_weissenstein', 'enge',
        'entfelden', 'flussbad_oberer_letten', 'flussbad_unterer_letten',
        'flussbad_unterer_letten_flussteil', 'frauenbad_stadthausquai', 'freibad_allenmoos',
        'freibad_auhof', 'freibad_heuried', 'freibad_letzigraben', 'freibad_seebach',
        'freibad_zwischen_den_hoelzern', 'hallenbad_altstetten', 'hallenbad_blaesi',
        'hallenbad_bungertwies', 'hallenbad_leimbach', 'hallenbad_oerlikon', 'josel_areal',
        'luzern', 'rotkreuz', 'seebad_utoquai', 'strandbad_mythenquai',
        'strandbad_tiefenbrunnen', 'strandbad_wollishofen', 'waermebad_kaeferberg',
        'wengen', 'zug', 'bern'
    ]

    WEATHER_FEATURES = ['cloud_cover', 'apparent_temperature', 'wind_speed_10m', 'precipitation']
    TIME_FEATURES = ['hour_of_day', 'weekday', 'day_of_month', 'day_of_year', 'month', 'year', 'epoch','timestamp', 'unique_id']

    train = (
            train_df.drop(OTHER_POOLS)
            .upsample(every="5m", time_column="timestamp")
            .fill_null(strategy="forward")
        )
    val = val_df.drop(OTHER_POOLS)
    test = test_df.drop(OTHER_POOLS)

    know_at_inference_time = val_df.select(WEATHER_FEATURES + TIME_FEATURES).upsample(every="5m", time_column="timestamp").fill_null(strategy="forward")

    print(f"Training features: {[c for c in train.columns if c not in ['unique_id', 'timestamp', TARGET]]}")
    return know_at_inference_time, train


@app.cell
def _(
    MLForecast,
    TARGET,
    concurrent,
    know_at_inference_time,
    lgb,
    mae,
    mean_squared_error,
    optuna,
    train,
):

    optuna.logging.set_verbosity(optuna.logging.INFO) 
    train_pd = train.to_pandas()

    def objective(trial):
        # 1. Added learning_rate (log=True is best practice for LR)
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'colsample_bytree': 0.8, 
            'subsample': 0.8,
            'min_child_samples': 20, 
            'random_state': 42,
            'verbose': -1,
            'force_row_wise': True,
            'n_jobs': -1  
        }
    
        print(f"\n--> Trial {trial.number} [Trees: {params['n_estimators']}, LR: {params['learning_rate']:.3f}]")
    
        # 2. Wrap the cross-validation in a standalone function
        def run_evaluation():
            fcst = MLForecast(models=[lgb.LGBMRegressor(**params)], freq='5min', lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 96, 288, 2016])
            # Set n_windows=1 to give it a fighting chance to finish in 5 seconds
            cv_res = fcst.cross_validation(df=train_pd, h=72, n_windows=7, step_size=72, time_col='timestamp', target_col=TARGET, static_features=[])
            print(mae(cv_res[TARGET], cv_res['LGBMRegressor']),end="")
            return mean_squared_error(cv_res[TARGET], cv_res['LGBMRegressor'])

        # 3. Enforce the 5-second hard limit
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_evaluation)
            try:
                # Wait exactly 5 seconds for a result
                mse = future.result(timeout=30.0) 
                print(f"<-- Trial {trial.number} Finished | MSE: {mse:.4f}")
                return mse
            except concurrent.futures.TimeoutError:
                # If it hits 5 seconds, cut it off and tell Optuna to skip it
                print(f"❌ Trial {trial.number} exceeded 30s timeout. Pruning!")
                raise optuna.TrialPruned()

    # 4. Run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    print(f"\n✅ Best Params: {study.best_params} | Best MSE: {study.best_value:.4f}")

    # 5. Train final model
    final_params = {
        **study.best_params, 
        'verbose': -1, 
        'force_row_wise': True, 
        'n_jobs': 4
    }

    final_fcst = MLForecast(models=[lgb.LGBMRegressor(**final_params)], freq='5min', lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 96, 288, 2016])
    final_fcst.fit(train_pd, time_col='timestamp', target_col=TARGET, static_features=[])

    predictions = final_fcst.predict(h=72, X_df=know_at_inference_time.to_pandas())
    return (predictions,)


@app.cell
def _(TARGET, plt, predictions, sns, val_df):

    # Set a clean visual theme
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6))

    # Convert validation data to pandas for plotting
    val_pd = val_df.to_pandas()

    # Merge actuals with predictions on timestamp to align them perfectly
    merged_df = predictions.merge(
        val_pd[['timestamp', TARGET]], 
        on='timestamp', 
        how='inner'
    )

    # Plot the actual validation data
    ax.plot(
        merged_df['timestamp'], 
        merged_df[TARGET], 
        label="Actual (Validation)", 
        color="#1f77b4", 
        linewidth=2
    )

    # Plot the forecast (MLForecast defaults the column name to the model's class name)
    ax.plot(
        merged_df['timestamp'], 
        merged_df['LGBMRegressor'], 
        label="Forecast (LightGBM)", 
        color="#ff7f0e", 
        linestyle="--", 
        linewidth=2
    )

    # Formatting
    clean_title = TARGET.replace('_', ' ').title()
    ax.set_title(f"{clean_title} - Forecast vs Actuals", fontsize=16, pad=15)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Occupancy", fontsize=12)
    ax.legend(fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Returning the figure tells Marimo to render it automatically in the UI
    fig
    return


if __name__ == "__main__":
    app.run()
