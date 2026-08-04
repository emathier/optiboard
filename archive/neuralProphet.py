import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pandas as pd
    from google.cloud import bigquery
    import requests
    import plotly.graph_objects as go
    import pickle
    from datetime import date
    from tqdm import tqdm
    import torch

    # Monkeypatch torch.load to avoid weights_only error in PyTorch 2.6+ with older PyTorch Lightning
    original_load = torch.load
    def patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load

    from neuralprophet import NeuralProphet

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
        NeuralProphet,
        RETRAIN_MODEL,
        UPDATE_HISTORICAL_WEATHER,
        UPDATE_OCCUPANCY,
        bigquery,
        date,
        go,
        mo,
        pd,
        pickle,
        pl,
        requests,
        torch,
        tqdm,
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
def _(COVARIATES, df_occupancy_weather, newest_occupancy, pl, requests):
    newest_occupancy

    request_forecast = "https://api.open-meteo.com/v1/forecast?latitude=47.3667&longitude=8.55&forecast_days=4&hourly=cloud_cover,apparent_temperature,wind_speed_10m,precipitation&timezone=Europe%2FBerlin"

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

    # Clean up any overlapping timestamps, sort and upsample to fill gaps
    df_final = (
        df_combined
        .unique(subset=["timestamp"], keep="first") # Keeps historical over forecast if they overlap
        .sort("timestamp")
        .upsample(every="5m", time_column="timestamp")
        .with_columns(pl.col(COVARIATES).interpolate())
        .sort("timestamp", descending=True)
    ).drop(pl.col('bern'))

    # Fill any null weather covariates using the forecast data (since the historical weather might not be available yet for the last 5-min intervals)
    df_final = df_final.join(
        forecast_data_upsampeled,
        on="timestamp",
        how="left",
        suffix="_forecast"
    )
    for col in COVARIATES:
        df_final = df_final.with_columns(
            pl.col(col).fill_null(pl.col(f"{col}_forecast"))
        )
    df_final = df_final.drop([f"{col}_forecast" for col in COVARIATES])

    df_final.write_parquet('data-files/occupancy-weather-forecast.parquet')
    return (df_final,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Train NeuralProphet models on new data
    """)
    return


@app.cell
def _(
    COVARIATES,
    RETRAIN_MODEL,
    df_final,
    pl,
    pd,
    torch,
    NeuralProphet,
    tqdm,
):
    import os
    from neuralprophet import load as np_load
    from neuralprophet import save as np_save

    # Prepare training dataframe
    df_ml = (
        df_final
        .unpivot(index=["timestamp"] + COVARIATES)
        .rename({"timestamp": "ds", "variable": "unique_id", "value": "y"})
        .drop_nulls()
        .with_columns(pl.col("ds").dt.replace_time_zone(None))
    )

    df_ml = df_ml.unique(subset=["unique_id", "ds"]).sort(["unique_id", "ds"])

    df_ml = (
        df_ml.upsample(time_column="ds", every="5m", group_by="unique_id")
        .with_columns(pl.all().forward_fill().backward_fill())
    )

    # Future inputs dataframe (where y is null)
    last_dates = df_ml.group_by("unique_id").agg(pl.col("ds").max().alias("last_ds"))

    X_df = (
        df_final
        .unpivot(index=["timestamp"] + COVARIATES)
        .rename({"timestamp": "ds", "variable": "unique_id", "value": "y"})
        .with_columns(pl.col("ds").dt.replace_time_zone(None))
        .drop('y')
    )

    X_df = (
        X_df
        .join(last_dates, on="unique_id", how="left")
        .filter(pl.col("ds") > pl.col("last_ds"))
        .drop("last_ds")
    )

    preds = []
    unique_ids = df_ml["unique_id"].unique().to_list()
    os.makedirs("model_v2", exist_ok=True)

    for uid in tqdm(unique_ids):
        # Filter data for this facility
        group = df_ml.filter(pl.col("unique_id") == uid)
        if len(group) == 0:
            continue
            
        df_train = group.to_pandas()[["ds", "y"] + COVARIATES]
        
        # Prepare future covariates
        group_future = X_df.filter(pl.col("unique_id") == uid)
        if len(group_future) == 0:
            continue
            
        df_future_cov = group_future.to_pandas()[["ds"] + COVARIATES]
        
        # Train or load model
        model_path = f"model_v2/np_model_{uid}.np"
        
        if RETRAIN_MODEL:
            m = NeuralProphet(
                growth="off",
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=True,
                quantiles=[0.25, 0.5, 0.75]
            )
            for col in COVARIATES:
                m = m.add_future_regressor(name=col)
            m = m.add_country_holidays("CH")
            
            # For speed, use the last 20,000 rows for training
            df_train_sub = df_train.tail(20000)
            
            # Fit model
            m.fit(df_train_sub, freq="5min", epochs=10)
            np_save(m, model_path)
        else:
            if os.path.exists(model_path):
                m = np_load(model_path)
            else:
                print(f"Model path {model_path} not found. Skipping {uid}.")
                continue
        
        # Predict
        periods = len(df_future_cov)
        future = m.make_future_dataframe(df=df_train.tail(20000), regressors_df=df_future_cov, periods=periods)
        
        # Keep only the columns that the model expects (some might be dropped if constant)
        keep_cols = ["ds", "y"]
        if m.config_regressors is not None:
            keep_cols.extend(list(m.config_regressors.regressors.keys()))
        future = future[[col for col in future.columns if col in keep_cols]]
        
        forecast = m.predict(future)
        
        # Format the predictions to match the project schema
        forecast_df = forecast[['ds', 'yhat1', 'yhat1 25.0%', 'yhat1 75.0%']].copy()
        forecast_df = forecast_df.rename(columns={
            'yhat1 25.0%': 'q25',
            'yhat1': 'q50',
            'yhat1 75.0%': 'q75'
        })
        forecast_df['unique_id'] = uid
        preds.append(pl.from_pandas(forecast_df))

    pred = pl.concat(preds).rename({'ds' : 'timestamp'}).with_columns(
        pl.col('timestamp').dt.replace_time_zone('Europe/Zurich')
    )

    product = df_final.join(pred.pivot(index='timestamp', on='unique_id'), on='timestamp', how='left')
    product.write_parquet('data-files/inference.parquet')
    return (product,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plotting
    """)
    return


@app.cell
def _(date, go, pl, product):


    # Pivot back to wide format to handle shading easily
    df_w = product.sort(
        "timestamp"
    ).filter(pl.col('timestamp') > date.today())

    fig = go.Figure(
        [
            # Shading bounds
            go.Scatter(x=df_w["timestamp"], y=df_w["q25_hallenbad_oerlikon"], line_width=0, showlegend=False),
            go.Scatter(
                x=df_w["timestamp"],
                y=df_w["q75_hallenbad_oerlikon"],
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
                x=df_w["timestamp"], y=df_w["q50_hallenbad_oerlikon"], name="q50", line_dash="dash"
            ),
        ]
    )

    fig.update_layout(template="plotly_white", hovermode="x unified")
    fig
    return


if __name__ == "__main__":
    app.run()
