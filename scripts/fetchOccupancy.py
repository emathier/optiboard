import polars as pl
from google.cloud import bigquery
from logging_config import get_logger
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
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

if df.is_empty():
    log.critical("No data fetched from BigQuery")
    exit(1)

log.info(f"Fetched {len(df)} rows from BigQuery")

# Check timestamp freshness
ts_max = df["timestamp"].max()
if ts_max.tzinfo is None:
    ts_max = ts_max.replace(tzinfo=ZoneInfo("UTC"))
ts_max_berlin = ts_max.astimezone(ZoneInfo("Europe/Berlin"))
now_berlin = datetime.now(ZoneInfo("Europe/Berlin"))
age = now_berlin - ts_max_berlin
log.debug(
    f"Newest timestamp in data (Zürich/Berlin): {ts_max_berlin} "
    f"({age.total_seconds() / 60:.0f} min old)"
)

# Fail if data is older than 1 hour
if age > timedelta(hours=1):
    log.critical(
        f"Data is stale — newest record is {age.total_seconds() / 60:.0f} minutes old "
        f"(threshold: 60 min)"
    )
    exit(1)

# Write to Parquet
log.debug("Writing DataFrame to city-occupancy.parquet")
df.write_parquet("data-files/city-occupancy.parquet")
log.debug("DataFrame written to city-occupancy.parquet successfully")


