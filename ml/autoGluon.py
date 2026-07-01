import polars as pl
import numpy as np

df = pl.read_parquet("data-files/occupancy-weather.parquet").sort(by = "timestamp")
print(df)