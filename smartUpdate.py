import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from google.cloud import bigquery
    import requests
    from mlforecast import MLForecast
    import lightgbm as lgb
    import pandas as pd
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pickle
    from datetime import date

    COVARIATES = [
        "cloud_cover",
        "apparent_temperature",
        "wind_speed_10m",
        "precipitation",
    ]


    UPDATE_OCCUPANCY = True
    UPDATE_HISTORICAL_WEATHER = True
    RETRAIN_MODEL = True
    return (
        COVARIATES,
        MLForecast,
        RETRAIN_MODEL,
        UPDATE_HISTORICAL_WEATHER,
        UPDATE_OCCUPANCY,
        bigquery,
        date,
        go,
        lgb,
        mo,
        pickle,
        pl,
        requests,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fetch occupancy and save to parquet.
    """)
    return


@app.cell
def _(UPDATE_OCCUPANCY, bigquery, pl):
    # get newest date in data
    df = pl.read_parquet('data-files/occupancy.parquet')
    newest_occupancy = df.select(pl.col('timestamp').max()).item()

    if UPDATE_OCCUPANCY:


        # Setup connection to bq
        client = bigquery.Client.from_service_account_json('creds.json')

        # Setup query
        query = """
            SELECT * FROM `optiswim-scraper.badi_data.currentfill`
            WHERE timestamp > @last_timestamp
        """

        # handle datetime fun
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("last_timestamp", "TIMESTAMP", newest_occupancy)
            ]
        )
        query_job = client.query(query, job_config=job_config)  # API request
        results = pl.from_arrow(query_job.result().to_arrow()).with_columns(
            pl.col('timestamp').dt.convert_time_zone('Europe/Zurich')
        )
        print(f'Found {len(results)} new occupancy data points. Adding them to occupancy.parquet')


        # Saving new file
        df = df.vstack(results).sort(by='timestamp', descending=True)
        df.unique(subset='timestamp').write_parquet('data-files/occupancy.parquet')
        df.head()
    return df, newest_occupancy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fetch historical weather and save to parquet
    """)
    return


@app.cell
def _(UPDATE_HISTORICAL_WEATHER, newest_occupancy, pl, requests):
    # Boundries
    df_weather = pl.read_parquet('data-files/historical_weather.parquet').drop_nulls()
    newest_weather = df_weather.select(pl.col('timestamp').max()).item()

    if UPDATE_HISTORICAL_WEATHER:
        start_date = newest_weather.strftime("%Y-%m-%d")
        end_date = newest_occupancy.strftime("%Y-%m-%d")

        # Request
        request = f"https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=47.3667&longitude=8.55&start_date={start_date}&end_date={end_date}&hourly=cloud_cover,apparent_temperature,wind_speed_10m,precipitation&timezone=Europe%2FBerlin"

        # Execute
        response = requests.get(request)
        new_weather_data = pl.DataFrame(response.json()['hourly']).with_columns(
            pl.col("time")
            .str.to_datetime()
            .dt.replace_time_zone("Europe/Zurich", non_existent="null", ambiguous="null")
        ).sort(by='time', descending=True).rename({'time' : 'timestamp'})

        print(f'Found {len(new_weather_data)} new weather data points. Adding them to historical-weather.parquet')
        df_weather = df_weather.vstack(new_weather_data).sort(by='timestamp', descending=True)
        df_weather.unique(subset='timestamp').write_parquet('data-files/historical_weather.parquet')
        df_weather.head()
    return (df_weather,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Merge Dataframes
    """)
    return


@app.cell
def _(df, df_weather, pl):
    df_occupancy = df.with_columns(
        pl.col('timestamp').dt.round('5m')
    )

    df_weather_upsampeled = df_weather.sort('timestamp').upsample(every='5m', time_column='timestamp').interpolate()
    df_occupancy_weather = df_occupancy.sort('timestamp').join(df_weather_upsampeled ,on='timestamp', how='left').sort('timestamp', descending=True).unique(subset='timestamp')

    df_occupancy_weather.write_parquet('data-files/occupancy-weather.parquet')
    return (df_occupancy_weather,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fetch weather forecast
    """)
    return


@app.cell
def _(df_occupancy_weather, newest_occupancy, pl, requests):
    newest_occupancy

    request_forecast = "https://api.open-meteo.com/v1/forecast?latitude=47.3667&longitude=8.55&forecast_days=3&hourly=cloud_cover,apparent_temperature,wind_speed_10m,precipitation&timezone=Europe%2FBerlin"

    response_forecast = requests.get(request_forecast)

    forecast_data = pl.DataFrame(response_forecast.json()['hourly']).with_columns(
        pl.col("time")
        .str.to_datetime()
        .dt.replace_time_zone("Europe/Zurich", non_existent="null", ambiguous="null")
    ).sort(by='time', descending=True).rename({'time' : 'timestamp'})

    # Upsample
    forecast_data_upsampeled = forecast_data.sort('timestamp').upsample(every='5m',time_column='timestamp').interpolate()

    # Merge into full dataset
    df_combined = pl.concat(
        [df_occupancy_weather, forecast_data_upsampeled], 
        how="diagonal"
    )

    # Clean up any overlapping timestamps and sort
    df_final = (
        df_combined
        .unique(subset=["timestamp"], keep="first") # Keeps historical over forecast if they overlap
        .sort("timestamp", descending=True)
    )

    df_final.write_parquet('data-files/occupancy-weather-forecast.parquet')
    return (df_final,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Train models on new data
    Parameters from hyper-param opt euler-r1
    """)
    return


@app.cell
def _(mo):
    avg_params = {'objective': 'regression',"metric": "rmse",'max_depth': 6, 'num_leaves': 64, 'min_child_samples': 81, 'learning_rate': 0.1461850400837278, 'n_estimators': 48, 'subsample': 0.6187186502554506, 'colsample_bytree': 0.7418257791996054, 'reg_alpha': 18.37278169756974, 'reg_lambda': 2.9154791826478093e-05,"n_jobs": 3, 'force_row_wise':'true', 'verbosity' : -1}

    q25_params = {"objective": "quantile","alpha": 0.25,'max_depth': 9, 'num_leaves': 512, 'min_child_samples': 88, 'learning_rate': 0.07389856738099874, 'n_estimators': 84, 'subsample': 0.9333086686035081, 'colsample_bytree': 0.5090952964299004, 'reg_alpha': 0.017336162465283883, 'reg_lambda': 0.0019493228509678015, "n_jobs": 3, 'force_row_wise':'true', 'verbosity' : -1}

    q75_params = {"objective": "quantile","alpha": 0.75,'max_depth': 6, 'num_leaves': 64, 'min_child_samples': 62, 'learning_rate': 0.023925349245515218, 'n_estimators': 243, 'subsample': 0.5512803618937144, 'colsample_bytree': 0.5271164964005131, 'reg_alpha': 1.322055949009689e-08, 'reg_lambda': 1.012740564311525e-06, "n_jobs": 3, 'force_row_wise':'true', 'verbosity' : -1}

    mo.hstack([avg_params, q25_params, q75_params])
    return avg_params, q25_params, q75_params


@app.cell
def _(
    COVARIATES,
    MLForecast,
    RETRAIN_MODEL,
    avg_params,
    df_final,
    lgb,
    pickle,
    pl,
    q25_params,
    q75_params,
):
    df_ml = (
        df_final
        .unpivot(index=["timestamp"] + COVARIATES)
        .filter(pl.col("variable").is_in(["hallenbad_oerlikon"]))
        .rename({"timestamp": "ds", "variable": "unique_id", "value": "y"})
        .drop_nulls()
        .with_columns(pl.col("ds").dt.replace_time_zone(None))
    )

    df_ml = df_ml.unique(subset=["unique_id", "ds"]).sort(["unique_id", "ds"])

    df_ml = (
        df_ml.upsample(time_column="ds", every="5m", group_by="unique_id")
        .with_columns(pl.all().forward_fill().backward_fill())
    )


    def decimal_hour(dates):
        """Converts a pandas datetime Index or Series to hours with decimals for mins/secs"""
        dt = dates.dt if hasattr(dates, "dt") else dates
        return dt.hour + dt.minute / 60.0 + dt.second / 3600.0

    if RETRAIN_MODEL:
        fcst = MLForecast(
            models={'rmse' : lgb.LGBMRegressor(**avg_params),
                   'q25' : lgb.LGBMRegressor(**q25_params),
                   'q75' : lgb.LGBMRegressor(**q75_params)},
            freq="5min",
            target_transforms=[],
            lags=[1, 12, 24 * 12],
            date_features=[decimal_hour, "weekday", "day_of_year"],
        )

        fcst.fit(df_ml.to_pandas(), static_features=[])

        pickle.dump(fcst, open("model_v1/model1.pkl", "wb"))
    else: 
        fcst = pickle.load(open('model_v1/model1.pkl', 'rb'))
    
    return (fcst,)


@app.cell
def _(COVARIATES, df_final, fcst, newest_occupancy, pl):
    X_df = (
        df_final
        .filter(pl.col('timestamp') > newest_occupancy)
        .unpivot(index=["timestamp"] + COVARIATES)
        .filter(pl.col("variable").is_in(["hallenbad_oerlikon"]))
        .rename({"timestamp": "ds", "variable": "unique_id", "value": "y"})
        .with_columns(pl.col("ds").dt.replace_time_zone(None))
        .drop('y')
    )

    pred = pl.from_pandas(fcst.predict(h =48 * 12, X_df= X_df.to_pandas())).rename({'ds' : 'timestamp'}).with_columns(
        pl.col('timestamp').dt.replace_time_zone('Europe/Zurich')
    )

    df_pred = df_final.join(pred, how='left', on='timestamp')
    return (df_pred,)


@app.cell
def _(date, df_pred, go, pl):


    # Pivot back to wide format to handle shading easily
    df_w = df_pred.sort(
        "timestamp"
    ).filter(pl.col('timestamp') > date.today())

    fig = go.Figure(
        [
            # Shading bounds
            go.Scatter(x=df_w["timestamp"], y=df_w["q25"], line_width=0, showlegend=False),
            go.Scatter(
                x=df_w["timestamp"],
                y=df_w["q75"],
                fill="tonexty",
                fillcolor="rgba(255,165,0,0.2)",
                name="Q25-Q75",
                line_width=0,
            ),
            # Lines
            go.Scatter(
                x=df_w["timestamp"], y=df_w["hallenbad_oerlikon"], name="Actual"
            ),
            go.Scatter(
                x=df_w["timestamp"], y=df_w["rmse"], name="RMSE", line_dash="dash"
            ),
        ]
    )

    fig.update_layout(template="plotly_white", hovermode="x unified")
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
