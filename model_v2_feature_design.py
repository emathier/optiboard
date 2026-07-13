import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    from statsmodels.tsa.stattools import pacf, acf
    import matplotlib.pyplot as plt

    # Load dataset
    df = pl.read_parquet('data-files/occupancy-weather.parquet')

    # 1. Convert Polars Series to NumPy array
    # 2. Handle potential null/missing values (statsmodels requires clean data)
    series_data = df['hallenbad_oerlikon'].drop_nulls().to_numpy()

    # Calculate PACF (Ensure your total data length is significantly greater than nlags)
    nlags = 12 * 24

    pacf_values = pacf(series_data, nlags=nlags)

    # Plot PACF
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(pacf_values)), pacf_values, width=0.6)
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.xlabel('Lags')
    plt.ylabel('PACF')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    return acf, nlags, plt, series_data


@app.cell
def _(acf, nlags, plt, series_data):
    acf_values = acf(series_data, nlags=nlags)

    # Plot PACF
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(acf_values)), acf_values, width=0.6)
    plt.title('Autocorrelation Function (ACF)')
    plt.xlabel('Lags')
    plt.ylabel('ACF')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
