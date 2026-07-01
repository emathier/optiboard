import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    # Imports
    import marimo as mo
    import polars as pl
    import plotly.graph_objects as go
    import lightgbm as lgb
    import matplotlib.pyplot as plt

    # Data import
    df = pl.read_parquet("data-files/occupancy-weather.parquet").with_columns(
        pl.col('timestamp').dt.epoch()
    )
    TARGET = ["hallenbad_city"]
    COVARIATES = ["cloud_cover", "apparent_temperature", "wind_speed_10m", "precipitation"]

    # Data wrangling
    train,test,val = df[:int(0.8*len(df))],df[int(0.8*len(df)):int(0.9*len(df))] , df[int(0.9*len(df)):]
    X_train, y_train = train.drop(TARGET).to_pandas(), train[TARGET].to_pandas()
    X_test, y_test = test.drop(TARGET).to_pandas(), test[TARGET].to_pandas()
    X_val, y_val = val.drop(TARGET).to_pandas(), val[TARGET].to_pandas()
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
    return X_val, go, lgb, lgb_test, lgb_train, y_val


@app.cell
def _(X_val, go, lgb, lgb_test, lgb_train, y_val):
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": {"mape", "mae", "rmse"},
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": 1,
    }


    # train
    gbm = lgb.train(
        params, lgb_train, num_boost_round=100, valid_sets=lgb_test, callbacks=[lgb.early_stopping(stopping_rounds=5)]
    )

    y_pred = gbm.predict(X_val)

    def visualize(y_pred):
        _val = X_val[['timestamp']].copy()
        _val['actual'] = y_val.values
        _val['predicted'] = y_pred
        _val = _val.sort_values('timestamp')
    
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=_val['timestamp'], y=_val['actual'], mode='lines', name='Actual', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=_val['timestamp'], y=_val['predicted'], mode='lines', name='Predicted', line=dict(color='blue')))
        fig.update_layout(title='Actual vs Predicted', xaxis_title='Timestamp', yaxis_title='Value')
        return fig

    visualize(y_pred)
    return


@app.cell
def _():


    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
