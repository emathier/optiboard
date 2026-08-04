# 🏊 Optiboard: Public Pool Occupancy Forecasting & Data Infrastructure

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://optiboard.streamlit.app)

**Author**: Etienne Mathier  
**Repository**: [emathier/optiboard](https://github.com/emathier/optiboard)  
**Live Application**: [optiboard.streamlit.app](https://optiboard.streamlit.app)  

---

## 1. Executive Summary

**Optiboard** is an end-to-end time-series forecasting pipeline and interactive web dashboard designed to predict public pool occupancy in Zurich and Bern up to 48 hours into the future at a 5-minute sampling frequency. 

By combining real-time turnstile IoT data with high-resolution weather observations and forecast models, Optiboard trains an autoregressive multi-output Gradient Boosting framework (**LightGBM** via **MLForecast**) with Quantile Regression to provide point estimates alongside Uncertainty Quantification (UQ) bounds ($Q_{25}$–$Q_{75}$).

```
                             [DATA FLOW ARCHITECTURE]

  +-----------------------+     +------------------------+
  |  BigQuery Turnstiles  |     | Open-Meteo Weather API |
  | (Real-time Occupancy) |     |  (Historic + Forecast) |
  +-----------+-----------+     +-----------+------------+
              |                             |
              +--------------+--------------+
                             |
                             v
               +---------------------------+
               | Preprocessing & Alignment |
               | (5m Resampling/Interpol.) |
               +-------------+-------------+
                             |
                             v
               +---------------------------+
               |    Feature Engineering    |
               | (Lags [1,12,288], Dec.Hr) |
               +-------------+-------------+
                             |
                             v
               +---------------------------+
               |  LightGBM + MLForecast    |
               | (Quantiles Q25, Q50, Q75) |
               +-------------+-------------+
                             |
                             v
               +---------------------------+
               |     Batch Inference       |
               |   (48h Horizon / 576 steps|
               +-------------+-------------+
                             |
                             v
               +---------------------------+
               |    Streamlit Dashboard    |
               | (Interactive Visualizer)  |
               +---------------------------+
```

---

## 2. End-to-End Data Pipeline Architecture

```mermaid
flowchart TD
    subgraph Data_Sources ["Data Sources & Ingestion"]
        A1["IoT Turnstiles / Web Scrapers"] -->|Continuous Writes| B1[("Google BigQuery: badi_data.currentfill")]
        A2["Open-Meteo Historical API"] -->|Hourly Weather Logs| B2["Historical Weather Stream"]
        A3["Open-Meteo Forecast API"] -->|4-Day Hourly Forecast| B3["Weather Forecast Stream"]
    end

    subgraph Data_Prep ["Data Preprocessing & Grid Alignment"]
        B1 -->|"fetchOccupancy.py / smartUpdate.py"| C1["Occupancy Parquet"]
        B2 -->|"fetchHistoricalWeather.py"| C2["Historical Weather Parquet"]
        B3 -->|"fetchForecast.py"| C3["Forecast Parquet"]
        
        C1 & C2 --> D1["5-min Temporal Upsampling & Linear Interpolation"]
        D1 --> D2["Outer Join & Boundary Trimming: Europe/Zurich"]
        C3 --> D3["Forward-Fill Dynamic Covariates"]
        D2 & D3 --> E1["Merged Dataset: occupancy-weather-forecast.parquet"]
    end

    subgraph Feature_Eng ["Feature Engineering & Time-Series Framing"]
        E1 --> F1["Unpivot Multi-Series Wide Format to Long Format"]
        F1 --> F2["Autoregressive Lag Vector: 1, 12, 288"]
        F1 --> F3["Temporal Covariates: Decimal Hour, Weekday, Day of Year"]
        F1 --> F4["Exogenous Weather Covariates: Temp, Cloud, Wind, Rain"]
    end

    subgraph Training_Opt ["Training & Optuna Optimization"]
        F2 & F3 & F4 --> G1["Optuna TPE Sampler: 5000 Trials / HPC Euler R1"]
        G1 --> G2["Optimal Hyperparameters"]
        G2 --> H1["MLForecast Engine: 3 LightGBM Regressors"]
        H1 -->|Objective: Quantile alpha=0.25| I1["Q25 Model"]
        H1 -->|Objective: Quantile alpha=0.50| I2["Q50 Model - Median"]
        H1 -->|Objective: Quantile alpha=0.75| I3["Q75 Model"]
    end

    subgraph Inference_Viz ["Inference & Visualization"]
        I1 & I2 & I3 --> J1["Recursive Multi-step Prediction: h=576 steps"]
        J1 --> J2[("inference.parquet")]
        J2 --> K1["Streamlit App: Predictions.py"]
        J2 --> K2["Data Explorer: Data_Explorer.py"]
        J2 --> K3["Data Availability Heatmap: Data_Availability.py"]
    end
end
```

---

## 3. Data Flow Stages & Implementation Details

### Stage 1: Data Ingestion & Inflow Streams

The system consumes two primary asynchronous streams:

1. **Turnstile Sensor Streams**: Real-time pool visitor counts are continuously scraped into Google BigQuery (`optiswim-scraper.badi_data.currentfill`). Ingestion queries filter records newer than the current local dataset timestamp ([smartUpdate.py](file:///Users/etienne/optiboard/smartUpdate.py#L58-L92)).
2. **Meteorological Data Streams**: 
   - **Historical Weather**: Downloaded via Open-Meteo Archive API ([smartUpdate.py](file:///Users/etienne/optiboard/smartUpdate.py#L104-L128)).
   - **Weather Forecast**: Fetched via Open-Meteo 4-Day Forecast API ([smartUpdate.py](file:///Users/etienne/optiboard/smartUpdate.py#L160-L207)).

| Stream | Source | Attributes Retrieved | Update Freq | Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **Occupancy** | Google BigQuery | `timestamp`, `hallenbad_city`, `hallenbad_oerlikon`, etc. | 5 mins | Irregular (1–5 min ticks) |
| **Weather (Hist)** | Open-Meteo API | `cloud_cover`, `apparent_temperature`, `wind_speed_10m`, `precipitation` | Daily backfill | 1 hour |
| **Weather (Fcst)**| Open-Meteo API | `cloud_cover`, `apparent_temperature`, `wind_speed_10m`, `precipitation` | Execution time | 1 hour |

---

### Stage 2: Data Preprocessing & Grid Alignment

Turnstile records arrive at uneven 1-to-5 minute intervals, while weather reports arrive hourly. The pipeline unifies these disparate frequencies into a synchronous 5-minute time series ([smartUpdate.py](file:///Users/etienne/optiboard/smartUpdate.py#L140-L207)):

1. **Timezone Standardization**: All timestamps are parsed and explicitly bound to `Europe/Zurich` (`Europe/Berlin`).
2. **Temporal Upsampling & Interpolation**: Hourly weather features are upsampled to 5-minute ticks using Polars linear interpolation (`upsample(every='5m').interpolate()`).
3. **Data Fusion & Fallbacks**: Turnstile actuals and upsampled historical weather are merged. Where trailing real-time weather logs are absent, values are backfilled from the live 4-day meteorological forecast.

```
Raw Hourly Weather:     [10:00: 18.0°C] -------------------------> [11:00: 20.0°C]
                                 \                                  /
5-min Upsampled Grid:   [10:00: 18.0°C] -> [10:05: 18.17°C] ... -> [11:00: 20.0°C]
Raw Turnstile Signals:  [10:01: 42 poolers] -> [10:06: 45 poolers] -> ...
```

---

### Stage 3: Feature Engineering & Time-Series Framing

The unified time series is converted into an autoregressive tabular matrix managed by `MLForecast` ([smartUpdate.py](file:///Users/etienne/optiboard/smartUpdate.py#L249-L284)):

```python
# Feature extraction pipeline (Polars & MLForecast)
df_ml = (
    df_final
    .unpivot(index=["timestamp"] + COVARIATES)
    .rename({"timestamp": "ds", "variable": "unique_id", "value": "y"})
    .drop_nulls()
    .with_columns(pl.col("ds").dt.replace_time_zone(None))
)
```

#### Engineered Feature Matrix

1. **Autoregressive Lags**:
   - `lag_1`: Immediate prior step ($t - 5\text{ min}$)
   - `lag_12`: 1 hour prior ($t - 60\text{ min}$)
   - `lag_288`: 24 hours prior ($t - 1440\text{ min}$)
2. **Calendar & Cyclical Temporal Variables**:
   - `decimal_hour`: Continuous float metric capturing intra-day position ($\text{hour} + \frac{\text{minute}}{60} + \frac{\text{second}}{3600}$).
   - `weekday`: Integer day index ($0 = \text{Monday}, 6 = \text{Sunday}$).
   - `day_of_year`: Day integer ($1 \text{ to } 366$).
3. **Dynamic Exogenous Covariates**: `cloud_cover` (%), `apparent_temperature` (°C), `wind_speed_10m` (km/h), `precipitation` (mm).

---

### Stage 4: Model Architecture & Optuna Hyperparameter Tuning

#### 1. Quantile Regression Engine
To quantify occupancy uncertainty, the system trains three separate LightGBM models per facility targeting pinball (quantile) loss ([smartUpdate.py](file:///Users/etienne/optiboard/smartUpdate.py#L220-L232)):

$$\mathcal{L}_{\alpha}(y, \hat{y}) = \max\left(\alpha (y - \hat{y}), (\alpha - 1)(y - \hat{y})\right)$$

- $Q_{50}$ ($\alpha = 0.50$): Median point prediction.
- $Q_{25}$ ($\alpha = 0.25$): Lower confidence bound.
- $Q_{75}$ ($\alpha = 0.75$): Upper confidence bound.

#### 2. Distributed Hyperparameter Optimization (Optuna on HPC)
Hyperparameters were optimized across 5,000 trials on the ETH Euler HPC cluster using Optuna's Tree-structured Parzen Estimator (**TPE Sampler**) with isolated worker processes bounded by 70-second execution timeouts ([model_v1/hyperops-tunning-rmse.py](file:///Users/etienne/optiboard/model_v1/hyperops-tunning-rmse.py#L86-L140)).

```
                       +---------------------------------------+
                       |   Optuna TPE Sampler (5,000 Trials)   |
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       | Isolated Worker Process (70s Timeout) |
                       |    MLForecast Cross-Validation (k=7)  |
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       | Evaluated Metric: Pinball Loss / RMSE |
                       +---------------------------------------+
```

| Hyperparameter | Search Space Range | Optimal Value Selected |
| :--- | :--- | :--- |
| `max_depth` | $[1, 10]$ | **4** |
| `num_leaves` | $[20, 8000]$ (constrained to $\le 2^{\text{max\_depth}}$) | **16** |
| `min_child_samples` | $[5, 100]$ | **22** |
| `learning_rate` | $[0.001, 0.3]$ (log scale) | **0.08246** |
| `n_estimators` | $[10, 1000]$ | **53** |
| `subsample` | $[0.4, 1.0]$ | **0.47816** |
| `colsample_bytree` | $[0.4, 1.0]$ | **0.71640** |
| `reg_alpha` | $[10^{-16}, 100.0]$ (log scale) | **6.78993** |
| `reg_lambda` | $[10^{-8}, 10.0]$ (log scale) | **0.00530** |

---

### Stage 5: Batch Inference & Uncertainty Quantification

During inference, `MLForecast` recursively predicts $h = 576$ discrete 5-minute steps into the future (48 hours) using incoming weather forecasts as exogenous features $X_{df}$ ([smartUpdate.py](file:///Users/etienne/optiboard/smartUpdate.py#L291-L321)).

- **Input**: Recent historical actuals + 48-hour forward weather forecast.
- **Output**: `data-files/inference.parquet` containing timestamped predictions per pool for $Q_{25}$, $Q_{50}$, and $Q_{75}$.

---

### Stage 6: Frontend Dashboard & Data Delivery Layer

The visual dashboard is built using Streamlit and Plotly, delivering real-time predictions and historical exploration ([streamlit/Predictions.py](file:///Users/etienne/optiboard/streamlit/Predictions.py)).

```
+-----------------------------------------------------------------------------------+
|  🔮 POOL OCCUPANCY PREDICTIONS (ZÜRICH & BERN)                                    |
+-----------------------------------------------------------------------------------+
| [ Latest Occupancy: 142 ] | [ Peak Tomorrow: 380 ] | [ Weather: ☀️ 24°C / No Rain ] |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|   Occupancy                                                                       |
|      ^                                                                            |
|  400 |                     /---\   <-- Upper Bound (Q75)                          |
|  300 |         Actual     /  *  \  <-- Forecast (Q50)                             |
|  200 |        *---*      /   |   \                                                |
|  100 |       /     \____/____|____\__ <-- Lower Bound (Q25)                       |
|    0 +-------------------------------------> Time                                 |
|         Historical              48-Hour Forecast                                  |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

#### Dashboard Architecture & Page Structure

1. **Predictions Landing Page** ([streamlit/Predictions.py](file:///Users/etienne/optiboard/streamlit/Predictions.py)):
   - **Data Fetching Strategy**: Pulls raw `inference.parquet` directly from GitHub CDN with fallback to local filesystem, cached with `@st.cache_data(ttl="15m")`.
   - **Plotly Visualizations**: Renders actuals alongside dashed median forecasts ($Q_{50}$) and translucent filled confidence bands ($Q_{25}$–$Q_{75}$).
   - **Metric Cards**: Dynamic KPI summaries for *Current Occupancy*, *Predicted Peak*, and *Rain Alerts*.
2. **Data Explorer** ([streamlit/pages/Data_Explorer.py](file:///Users/etienne/optiboard/streamlit/pages/Data_Explorer.py)): Multi-pool historical comparison interface with custom date pickers, aggregated weather metrics, and CSV/Parquet data export.
3. **Data Availability Diagnostics** ([streamlit/pages/Data_Availability.py](file:///Users/etienne/optiboard/streamlit/pages/Data_Availability.py)): Heatmap monitoring system tracking sensor data completeness across historical calendar days.
4. **Analytics & Privacy**: Injected Google Analytics script with safe cross-origin iframe security error handling.

---

## 4. Pipeline Orchestration & Environment Setup

### Orchestration
The primary data and ML workflow is orchestrated reactively via **Marimo** ([smartUpdate.py](file:///Users/etienne/optiboard/smartUpdate.py)), enabling seamless execution either as an interactive notebook or headlessly via standard Python CLI.

### Package & Dependency Management
Project dependencies and build tools are locked using **Pixi** ([pixi.toml](file:///Users/etienne/optiboard/pixi.toml)):

```bash
# Launch interactive Streamlit dashboard
pixi run sl

# Trigger dataset update & model execution pipeline
pixi run refresh-data
```

---

## 5. System Specifications & Summary Matrix

| Subsystem | Tooling / Framework | Primary Data Format | Key Artifact / File |
| :--- | :--- | :--- | :--- |
| **Ingestion** | Google BigQuery Client, Open-Meteo REST API | JSON / Arrow | [scripts/fetchOccupancy.py](file:///Users/etienne/optiboard/scripts/fetchOccupancy.py) |
| **Storage** | Apache Parquet (Snappy Compression) | Parquet DataFrames | `data-files/*.parquet` |
| **Transformation** | Polars | Micro-second LazyFrames | [smartUpdate.py](file:///Users/etienne/optiboard/smartUpdate.py#L140-L207) |
| **ML Engine** | LightGBM, MLForecast | Serialized Pickle Dict | [model_v1/model1.pkl](file:///Users/etienne/optiboard/model_v1/model1.pkl) |
| **Hyperparam Tuning**| Optuna (TPE Sampler), SQLite | SQLite DB | [model_v1/hyperops-tunning-rmse.py](file:///Users/etienne/optiboard/model_v1/hyperops-tunning-rmse.py) |
| **Frontend** | Streamlit, Plotly, HTML/CSS | Interactive Web UI | [streamlit/Predictions.py](file:///Users/etienne/optiboard/streamlit/Predictions.py) |
| **Environment** | Pixi | Lockfile (`pixi.lock`) | [pixi.toml](file:///Users/etienne/optiboard/pixi.toml) |
