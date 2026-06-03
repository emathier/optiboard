import polars as pl
from google.cloud import bigquery
from logging_config import get_logger
log = get_logger(__name__)


log.info("Starting to fetch occupancy")
bq_client = bigquery.Client(project="optiswim-scraper")


"""



# SQL
QUERY = (
    'SELECT * FROM `badi_data.currentfill` '
    'ORDER BY timestamp DESC'
)

# Diagnostics
query_job = bq_client.query(QUERY)  
rows = query_job.result()  

df = pl.from_arrow(rows.to_arrow())
df.write_parquet("city-occupancy.parquet")

"""


