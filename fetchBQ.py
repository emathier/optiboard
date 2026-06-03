import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from google.cloud import bigquery

    # 1. Explicitly state your project ID (where the query runs)
    PROJECT_ID = "optiswim-scraper" 

    # 2. Initialize the client (ADC handles the auth automatically!)
    client = bigquery.Client(project=PROJECT_ID)

    # 3. Perform a query
    QUERY = (
        'SELECT timestamp,hallenbad_city FROM `badi_data.currentfill` '
        'ORDER BY timestamp DESC'
    )


    query_job = client.query(QUERY)  # API request
    rows = query_job.result()  # Waits for query to finish

    df = pl.from_arrow(rows.to_arrow())
    df.write_parquet("city-occupancy.parquet")
    return


if __name__ == "__main__":
    app.run()
