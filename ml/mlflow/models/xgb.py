import os
os.environ['MLFLOW_TRACKING_URI'] = "http://127.0.0.1:5000"
os.environ['MLFLOW_EXPERIMENT_NAME'] = "WeekPrediction"  
import duckdb as dd
import mlflow
from mlflow.models import infer_signature
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae


# 1. Load and clean data entirely within DuckDB
sql_query = """
SELECT cloud_cover, apparent_temperature, wind_speed_10m, precipitation,
       CAST(hallenbad_city AS DOUBLE) AS hallenbad_city,
       EXTRACT(dow FROM timestamp)::INTEGER AS weekday,
       EXTRACT(epoch FROM timestamp)::INTEGER % 86400 AS time_of_day
FROM read_parquet("data-files/{}.parquet")
"""
train_df = dd.query(sql_query.format("training")).to_df().dropna()
eval_df = dd.query(sql_query.format("validation")).to_df().dropna()


X_train, y_train = train_df.drop(columns=["hallenbad_city"]), train_df["hallenbad_city"]

# 2. Run MLflow Tracking & Automatic Evaluation
with mlflow.start_run():
    mlflow.log_input(mlflow.data.from_pandas(train_df, name="weather+timeOfDay+weekday", targets="hallenbad_city"), context="train")
    mlflow.log_input(mlflow.data.from_pandas(eval_df, name="lastWeek", targets="hallenbad_city"), context="eval")
    
    # Train & Log Model
    xgb = XGBRegressor().fit(X_train, y_train)
    signature = infer_signature(X_train, xgb.predict(X_train))
    model_info = mlflow.xgboost.log_model(xgb, "xgb-model", signature=signature)

    mlflow.log_metric("train_mae", mae(y_train, xgb.predict(X_train)))
    
    # Automatic Evaluation (Logs MAE, RMSE, R², and residual plots)
    result = mlflow.models.evaluate(
        model=model_info.model_uri,
        data=eval_df,
        targets="hallenbad_city",
        model_type="regressor",
    )