import streamlit as st
import polars as pl
from logging_config import get_logger
import duckdb as dd
import plotly.graph_objects as go
from contextlib import contextmanager
import time

data_source = "data-files/occupancy-weather.parquet"

@contextmanager
def time_this(label):
    start = time.time()  # <-- Changed from pl.time.time()
    yield
    log.debug(f"{label} took {time.time() - start:.4f}s")


log = get_logger("streamlit-overview")
log.debug("Starting Streamlit overview dashboard")

 # Fetching data
log.debug(f"Reading {data_source} into DataFrame")
df = dd.query(f"SELECT timestamp, hallenbad_city, hallenbad_oerlikon FROM '{data_source}' ORDER BY timestamp DESC LIMIT 1000").pl()
log.info(f"Read {len(df)} rows from {data_source}")

# Streamlit 
st.set_page_config(
    page_title="Dashboard Overview",
    page_icon="📈",
)

st.title("Optiboard Overview Dashboard")
st.write("This dashboard gives an overview of the present data. Please select the pools you want to see:")


poolNames = dd.query(f"DESCRIBE '{data_source}'").to_df()["column_name"].tolist()
exclude_cols = {"timestamp", "cloud_cover", "apparent_temperature", "wind_speed_10m", "precipitation"}
poolNames = [c for c in poolNames if c not in exclude_cols]

# --- occupancy plot
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
    df = dd.query(f"SELECT timestamp, {', '.join(pool_selection)} FROM '{data_source}' WHERE {time_filter} ORDER BY timestamp").pl()
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


# --- Weather plot
weather_cols = ["cloud_cover", "apparent_temperature", "wind_speed_10m", "precipitation"]
weather_labels = {
    "cloud_cover": "Cloud Cover (%)",
    "apparent_temperature": "Apparent Temperature (°C)",
    "wind_speed_10m": "Wind Speed (km/h)",
    "precipitation": "Precipitation (mm)",
}

with time_this("Querying weather data"):
    df_weather = dd.query(
        f"SELECT timestamp, {', '.join(weather_cols)} FROM '{data_source}' WHERE {time_filter} ORDER BY timestamp"
    ).pl()


df_weather = df_weather.with_columns(
    pl.col("timestamp")
    .dt.replace_time_zone("UTC")
    .dt.convert_time_zone("Europe/Berlin")
)

from plotly.subplots import make_subplots

fig_weather = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=[weather_labels[c] for c in weather_cols],
)

colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
for i, col in enumerate(weather_cols):
    fig_weather.add_trace(
        go.Scattergl(x=df_weather["timestamp"], y=df_weather[col], name=col, line=dict(color=colors[i]), mode="lines"),
        row=i + 1, col=1,
    )

fig_weather.update_layout(
    height=600,
    template="plotly_white",
    showlegend=False,
    margin=dict(t=30),
)

# Apply same x-axis config as occupancy plot
match time_select:
    case "Last 24 hours":
        fig_weather.update_xaxes(
            dtick=10800000, tickformat="%H:%M",
            showgrid=True, gridwidth=1, gridcolor="lightgray",
            minor=dict(showgrid=True, dtick=3600000, gridcolor="#eee")
        )
    case "Last 7 days":
        fig_weather.update_xaxes(dtick=86400000, tickformat="%b %d", showgrid=True, gridwidth=1, gridcolor="lightgray")
    case "Last 30 days":
        fig_weather.update_xaxes(
            dtick=86400000 * 7, tickformat="%b %d",
            minor=dict(showgrid=True, dtick=86400000, gridcolor="#eee")
        )

# Only show x-axis labels on bottom subplot
for i in range(1, 4):
    fig_weather.update_xaxes(visible=False, row=i, col=1)



st.plotly_chart(fig, width=1000)
st.plotly_chart(fig_weather, width=1000)