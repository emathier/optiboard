import polars as pl
from google.cloud import bigquery
from logging_config import get_logger
log = get_logger("fetchOccupancy")
log.debug("Starting to fetch occupancy")


log.debug("Initializing BigQuery client")   
bq_client = bigquery.Client(project="optiswim-scraper")
log.debug("BigQuery client initialized successfully")



# SQL
QUERY = (
    'SELECT * FROM `badi_data.currentfill` '
    'ORDER BY timestamp DESC'
)

# Fetching data
log.debug(f"Executing query: {QUERY}")
df = pl.from_arrow(bq_client.query(QUERY).to_arrow())
log.info(f"Fetched {len(df)} rows from BigQuery")

if df.is_empty():
    log.critical("No data fetched from BigQuery")

# Write to Parquet
log.debug("Writing DataFrame to city-occupancy.parquet")
df.write_parquet("city-occupancy.parquet")
log.debug("DataFrame written to city-occupancy.parquet successfully")


