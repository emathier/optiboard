import streamlit as st
import polars as pl
from logging_config import get_logger
import duckdb as dd
import plotly.graph_objects as go
from contextlib import contextmanager
import time

@contextmanager
def time_this(label):
    start = time.time()  # <-- Changed from pl.time.time()
    yield
    log.debug(f"{label} took {time.time() - start:.4f}s")


log = get_logger("streamlit-overview")
log.debug("Starting Streamlit overview dashboard")

 # Fetching data
log.debug("Reading city-occupancy.parquet into DataFrame")
df = dd.query("SELECT timestamp, hallenbad_city, hallenbad_oerlikon FROM 'city-occupancy.parquet' ORDER BY timestamp DESC LIMIT 1000").pl()
log.info(f"Read {len(df)} rows from city-occupancy.parquet")

# Streamlit 
st.title("Optiboard Overview Dashboard")
st.write("This dashboard gives an overview of the present data. Please select the pools you want to see:")


poolNames = dd.query("DESCRIBE 'city-occupancy.parquet'").to_df()["column_name"].tolist()
poolNames.remove("timestamp")

# UI for pool selection and time range
pool_selection = st.multiselect( 
    label="Select pools to display",
    default=["hallenbad_city", "hallenbad_oerlikon"],
    options=poolNames
)
log.debug(f"Selected pools: {pool_selection}")

time_select = st.selectbox(
    label="Select time range",
    options=["Last 24 hours", "Last 7 days", "Last 30 days"]
)
log.debug(f"Selected time range: {time_select}")

# Constructing SQL time filter based on selection
time_thresholds = {
    "Last 24 hours": "timestamp >= now() - interval '1 day'",
    "Last 7 days": "timestamp >= now() - interval '7 day'",
    "Last 30 days": "timestamp >= now() - interval '30 day'",
}
time_filter = time_thresholds[time_select]
log.debug(f"Time filter for SQL query: {time_filter}")

# Querying data for plot
with time_this("Querying data for plot"):
    df = dd.query(f"SELECT timestamp, {', '.join(pool_selection)} FROM 'city-occupancy.parquet' WHERE {time_filter} ORDER BY timestamp").pl()
log.debug(f"Data dimensions after filtering: {df.shape} (rows, columns)")

# Localize timestamps to Zürich/Berlin timezone
df = df.with_columns(
    pl.col("timestamp")
    .dt.replace_time_zone("UTC")
    .dt.convert_time_zone("Europe/Berlin")
)
log.debug(f"Timestamps converted to Europe/Berlin timezone")

df_with_nans=len(df)
df = df.drop_nans()     
log.debug(f"Dropped {df_with_nans - len(df)} rows with NaN values; remaining rows: {len(df)}, dropped: {(df_with_nans - len(df)) / df_with_nans:.2%}")


fig = go.Figure()

for pool in pool_selection:
    fig.add_trace(go.Scattergl( # 'Scattergl' activates WebGL rendering
        x=df["timestamp"], 
        y=df[pool], 
        name=pool,
        mode='lines',
    ))


fig.update_layout(
    height=600,
    xaxis_title="Timestamp",
    yaxis_title="Occupancy",
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.02,
        xanchor="left",
        x=0
    )
)

match time_select:
    case "Last 24 hours":
        fig.update_xaxes(
            dtick=10800000, tickformat="%H:%M",
            showgrid=True, gridwidth=1, gridcolor="lightgray",
            minor=dict(showgrid=True, dtick=3600000, gridcolor="#eee")
        )
    case "Last 7 days":
        fig.update_xaxes(dtick=86400000, tickformat="%b %d", showgrid=True, gridwidth=1, gridcolor="lightgray")
    case "Last 30 days":
        fig.update_xaxes(
            dtick=86400000 * 7, tickformat="%b %d",
            minor=dict(showgrid=True, dtick=86400000, gridcolor="#eee")
        )


st.plotly_chart(fig, width=1000)
