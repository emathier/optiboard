# 🏊 Optiboard: Public Pool Occupancy Forecast Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://optiboard.streamlit.app)

Optiboard is a predictive modeling system and interactive dashboard designed to forecast public pool occupancy in Zurich and Bern. By scraping turnstile data and combining it with meteorological conditions, the system trains an autoregressive LightGBM model to predict visitor crowds up to 48 hours in advance, allowing swimmers to optimize their visits.

---

## 🏗️ System Architecture & Workflow

1. **Scraping & Ingestion**: Real-time turnstile data is scraped from public pool sites and ingested into Google BigQuery. Weather history is retrieved via Open-Meteo APIs.
2. **Feature Engineering**: Calculates autoregressive lags, time-of-day/day-of-week calendar features, and aligns hourly weather forecasts.
3. **ML Training & Tuning**: Utilizes **MLForecast** to train a multi-output **LightGBM** model. Hyperparameters are tuned using **Optuna** to optimize for RMSE and pinball loss (quantiles).
4. **Inference & UQ**: Predicts expected occupancy and quantiles ($Q_{25}$, $Q_{75}$) for Uncertainty Quantification (UQ) using quantile regression.
5. **Frontend Dashboard**: Built using **Streamlit** to visualize predictions, weather context, and historical occupancy.

---

## 📂 Repository File Index

### 🌐 Frontend (Streamlit Dashboard)
Located in the `streamlit/` directory.

* **[streamlit/Predictions.py](file:///Users/etienne/optiboard/streamlit/Predictions.py)**: The main landing page of the dashboard. Displays interactive pool occupancy forecasts (with quantile ranges) and weather conditions, featuring summary cards like *Latest Actual Occupancy*, *Peak Predicted Tomorrow*, *Raining Tomorrow?*, and *Last Update Time*.
* **[streamlit/pages/Data_Explorer.py](file:///Users/etienne/optiboard/streamlit/pages/Data_Explorer.py)**: A polished multi-pool historical data explorer. Offers custom date/time range selection (with localized boundaries) and tabs for comparing occupancy trends, weather subplots, and raw data export.
* **[streamlit/pages/Data_Availability.py](file:///Users/etienne/optiboard/streamlit/pages/Data_Availability.py)**: A diagnostic heatmap dashboard showing which calendar days contain historical scraper records.
* **[streamlit/logging_config.py](file:///Users/etienne/optiboard/streamlit/logging_config.py)**: Configures standard formatted logger instances for dashboard operations.

### ⚙️ Orchestration & Notebooks
* **[smartUpdate.py](file:///Users/etienne/optiboard/smartUpdate.py)**: A reactive **Marimo Notebook** script serving as the primary pipeline orchestrator. It checks for new turnstile records from BigQuery, fetches Open-Meteo archived weather, retrains the model if configured, runs batch inference, and outputs the result files to `data-files/`.
* **[creds.json](file:///Users/etienne/optiboard/creds.json)**: Google Cloud service account key credentials for accessing BigQuery turnstile databases (Git-ignored/secret).

### 📊 Scripts (Utility & Ingestion)
Located in the `scripts/` directory.

* **[scripts/fetchOccupancy.py](file:///Users/etienne/optiboard/scripts/fetchOccupancy.py)**: Pulls recent raw turnstile occupancy records from Google BigQuery.
* **[scripts/fetchHistoricalWeather.py](file:///Users/etienne/optiboard/scripts/fetchHistoricalWeather.py)**: Connects to Open-Meteo to download historical weather data corresponding to scraper timestamps.
* **[scripts/fetchForecast.py](file:///Users/etienne/optiboard/scripts/fetchForecast.py)**: Downloads fresh 48-hour weather forecasts (temp, rain, cloud cover, wind) for inference.
* **[scripts/interpolateWeather.py](file:///Users/etienne/optiboard/scripts/interpolateWeather.py)**: Interpolates hourly meteorological logs into 5-minute ticks to match turnstile resolution.
* **[scripts/logging_config.py](file:///Users/etienne/optiboard/scripts/logging_config.py)**: Logging configuration utility for standalone script operations.

### 🧠 Model Training & Hyperparameter Tuning
Located in the `model_v1/` directory.

* **[model_v1/model1.pkl](file:///Users/etienne/optiboard/model_v1/model1.pkl)**: The trained serialized MLForecast LightGBM model dictionary.
* **[model_v1/optuna_study_avg.db](file:///Users/etienne/optiboard/model_v1/optuna_study_avg.db)**: SQLite database containing Optuna hyperparameter trials and logs.
* **[model_v1/manual-mlforecast-hyperopstunning.py](file:///Users/etienne/optiboard/model_v1/manual-mlforecast-hyperopstunning.py)**: Helper script to test and log manual hyperparameter tweaks.
* **[model_v1/hyperops-tunning-rmse.py](file:///Users/etienne/optiboard/model_v1/hyperops-tunning-rmse.py)**: Optuna study script targeting RMSE optimization for the expected forecast.
* **[model_v1/hyperops-tunning-quantile-25.py](file:///Users/etienne/optiboard/model_v1/hyperops-tunning-quantile-25.py)**: Optuna study script optimizing pinball loss for the lower ($Q_{25}$) confidence bound.
* **[model_v1/hyperops-tunning-quantile-75.py](file:///Users/etienne/optiboard/model_v1/hyperops-tunning-quantile-75.py)**: Optuna study script optimizing pinball loss for the upper ($Q_{75}$) confidence bound.
* **[model_v1/euler-r1/](file:///Users/etienne/optiboard/model_v1/euler-r1/)**: Directory containing remote Optuna run logs and script tarballs.

### 💾 Data Directory
Located in the `data-files/` directory.

* **`occupancy.parquet`**: Combined raw pool turnstile records.
* **`historical_weather.parquet`**: Historic weather log files.
* **`occupancy-weather.parquet`**: Merged dataset of historical weather and pool occupancy.
* **`weather_forecast.parquet`**: Incoming weather predictions used for model inputs.
* **`inference.parquet`**: The outputs containing historical actuals aligned next to ML forecast outputs and confidence intervals.

---

## 🚀 Running the Project

Optiboard uses the **Pixi** package manager for package and environment lock management.

### 1. Launch the Streamlit Dashboard
To run the predictions and explorer dashboard locally:
```bash
pixi run sl
```
This launches the app (normally on `http://localhost:8502`).

### 2. Update Data and Run Predictions
To run the Marimo orchestrator notebook and update all parquet databases:
```bash
pixi run marimo edit smartUpdate.py
```
Or execute the pipeline directly via python.

---

## ⚠️ AI Disclaimer & Model Limitations

Optiboard's forecasts are generated by machine learning models trained on historical occupancy data and meteorological forecasts. 
* **Predictions are Estimates**: Occupancy predictions are statistical expectations and can deviate from actual visitor counts due to unmodeled real-world factors (e.g., special events, school holidays, local pool renovations, or sudden turnstile sensor outages).
* **Weather Dependency**: The accuracy of the occupancy model relies on the precision of the third-party weather forecasts. Incorrect meteorological forecasts will lead to less accurate crowd predictions.
* **Uncertainty Bounds**: The shaded confidence interval ($Q_{25} - Q_{75}$) represents a 50% probability range under normal conditions. It does not account for anomalies or extreme weather deviations.
* **Frontend Coding Assist**: Gemini was utilized as an AI coding assistant to refine, layout, and polish the Streamlit frontend. Almost all other components—including data collection scripts, BigQuery integrations, training pipelines, and Optuna hyperparameter tuning—were coded by hand.

