import polars as pl
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow

def getValidationScores(y_pred, target = "hallenbad_oerlikon"):
    y_true = pl.read_parquet('data-files/validation.parquet')[target]
    mlflow.log_metric("validation_week_mse", mean_squared_error(y_true, y_pred))
    mlflow.log_metric("validation_week_r2", r2_score(y_true, y_pred))
    mlflow.log_metric("validation_week_mae", mean_absolute_error(y_true, y_pred))

    