import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import duckdb as dd
    import polars as pl
    import datetime
    import os
    os.chdir("/Users/etienne/optiboard/")
    return datetime, dd


@app.cell
def _(datetime, dd):
    # Get newest date in the data
    newest_date = dd.query("SELECT MAX(timestamp) FROM read_parquet('data-files/occupancy-weather.parquet')").pl()
    newest_date = newest_date[0, 0]

    # Get the date 7 days before the newest date
    validation_start_date = newest_date -  datetime.timedelta(days=7)

    # Create validation split (last 7 days of data)
    dd.query(f"""
        SELECT *
        FROM read_parquet('data-files/occupancy-weather.parquet')
        WHERE timestamp >= '{validation_start_date}' AND timestamp <= '{newest_date}'
    """).pl().write_parquet('data-files/validation.parquet')


    # Create training split (everything before the validation period)
    dd.query(f"""
        SELECT *
        FROM read_parquet('data-files/occupancy-weather.parquet')
        WHERE timestamp < '{validation_start_date}'
    """).pl().write_parquet('data-files/training.parquet')
    return


if __name__ == "__main__":
    app.run()
