import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    return (pl,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(pl):
    # get newest date in data
    df = pl.read_parquet('data-files/occupancy-weather.parquet')
    newest_data = df.select(pl.col('timestamp').max())
    newest_data
    return (df,)


if __name__ == "__main__":
    app.run()
