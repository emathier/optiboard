import marimo as mo
import polars as pl
from mlforecast import MLForecast
from utilsforecast.plotting import plot_series
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmse, mape
import lightgbm as lgb
from datetime import timedelta
from pyinstrument import Profiler

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


import mlforecast
import importlib.util




def decimal_hour(dates):
    """Converts a pandas datetime Index or Series to hours with decimals for mins/secs"""
    # If it's a Series, use .dt accessor; if it's an Index, use it directly
    dt = dates.dt if hasattr(dates, "dt") else dates

    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0

def cv_pipeline():
    lgb_params = {
        'verbosity': -1,
        'num_leaves': 50,
        'n_jobs' : 4
    }


    fcst = MLForecast(
        models={
            'avg': lgb.LGBMRegressor(**lgb_params, objective='rmse', alpha=0.5),
            'q75': lgb.LGBMRegressor(**lgb_params, objective='quantile', alpha=0.75),
            'q25': lgb.LGBMRegressor(**lgb_params, objective='quantile', alpha=0.25),
        },
        freq='5min',  # our series have integer timestamps, so we'll just add 1 in every timestep
        target_transforms=[],
        lags=[1,12,24*12],
        date_features=[decimal_hour, 'weekday','dayofyear'],

    )
    profiler = Profiler()
    profiler.start()


    cv_df = fcst.cross_validation(
        df=df_train,
        h=24*12,
        n_windows=3,
        static_features=[]
    )

    profiler.stop()
    profiler.open_in_browser()

    cv_results = evaluate(
        cv_df.drop(columns='cutoff'),
        metrics=[rmse,mape],
        agg_fn='mean',
    )
    return cv_results




# 2. Run your specific bottleneck
print(cv_pipeline())

