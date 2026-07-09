import streamlit as st
import polars as pl
from logging_config import get_logger
import duckdb as dd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from contextlib import contextmanager
import time
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Historical Data Explorer",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS to remove top whitespace and make layout compact
st.markdown("""
    <style>
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 0rem !important;
        }
        [data-testid="stHeader"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

log = get_logger("streamlit-explorer")
log.info("Loading Historical Data Explorer page")

data_source = "data-files/occupancy-weather.parquet"

@contextmanager
def time_this(label):
    start = time.time()
    yield
    log.debug(f"{label} took {time.time() - start:.4f}s")

# Extract available pools and time bounds
try:
    poolNames = dd.query(f"DESCRIBE '{data_source}'").to_df()["column_name"].tolist()
    exclude_cols = {"timestamp", "cloud_cover", "apparent_temperature", "wind_speed_10m", "precipitation", "bern"}
    poolNames = sorted([c for c in poolNames if c not in exclude_cols])
    
    time_bounds = dd.query(f"SELECT MIN(timestamp) as min_t, MAX(timestamp) as max_t FROM '{data_source}'").pl()
    min_t = time_bounds["min_t"][0]
    max_t = time_bounds["max_t"][0]
except Exception as e:
    st.error(f"Error reading data source: {e}")
    st.stop()

# Sidebar layout
with st.sidebar:
    st.title("⚙️ Controls")
    
    pool_selection = st.multiselect( 
        label="Select pools to display",
        default=["hallenbad_city", "hallenbad_oerlikon"] if "hallenbad_city" in poolNames and "hallenbad_oerlikon" in poolNames else poolNames[:2],
        options=poolNames
    )
    log.debug(f"Selected pools: {pool_selection}")
    
    time_select = st.selectbox(
        label="Select time range",
        options=["Last 24 hours", "Last 7 days", "Last 30 days", "Custom Range"]
    )
    log.debug(f"Selected time range: {time_select}")
    
    # Initialize dynamic delta
    range_delta = timedelta(days=1)
    
    if time_select == "Custom Range":
        default_start_dt = max_t - timedelta(days=7)
        default_end_dt = max_t
        
        st.markdown("**Start Date & Time**")
        col_start_d, col_start_t = st.columns(2)
        with col_start_d:
            start_date = st.date_input(
                "Start Date", 
                value=default_start_dt.date(), 
                min_value=min_t.date(), 
                max_value=max_t.date(),
                label_visibility="collapsed"
            )
        with col_start_t:
            start_time = st.time_input(
                "Start Time", 
                value=default_start_dt.time(),
                label_visibility="collapsed"
            )
            
        st.markdown("**End Date & Time**")
        col_end_d, col_end_t = st.columns(2)
        with col_end_d:
            end_date = st.date_input(
                "End Date", 
                value=default_end_dt.date(), 
                min_value=min_t.date(), 
                max_value=max_t.date(),
                label_visibility="collapsed"
            )
        with col_end_t:
            end_time = st.time_input(
                "End Time", 
                value=default_end_dt.time(),
                label_visibility="collapsed"
            )
            
        start_datetime = datetime.combine(start_date, start_time)
        end_datetime = datetime.combine(end_date, end_time)
        
        if start_datetime >= end_datetime:
            st.error("Error: Start time must be before End time.")
            st.stop()
            
        range_delta = end_datetime - start_datetime
        
        start_dt_str = start_datetime.strftime("%Y-%m-%d %H:%M:%S") + " Europe/Zurich"
        end_dt_str = end_datetime.strftime("%Y-%m-%d %H:%M:%S") + " Europe/Zurich"
        
        time_filter = f"timestamp >= '{start_dt_str}'::timestamptz AND timestamp <= '{end_dt_str}'::timestamptz"
    else:
        time_thresholds = {
            "Last 24 hours": "timestamp >= now() - interval '1 day'",
            "Last 7 days": "timestamp >= now() - interval '7 day'",
            "Last 30 days": "timestamp >= now() - interval '30 day'",
        }
        time_filter = time_thresholds[time_select]

# Query occupancy data
with time_this("Querying data for plot"):
    query_pools = ", ".join(pool_selection)
    df = dd.query(f"SELECT timestamp, {query_pools} FROM '{data_source}' WHERE {time_filter} ORDER BY timestamp").pl()

# Convert time zone to Europe/Zurich
df = df.with_columns(
    pl.col("timestamp")
    .dt.replace_time_zone("UTC")
    .dt.convert_time_zone("Europe/Zurich")
)
df = df.drop_nans()

# Tabs for structured viewing
tab1, tab2, tab3 = st.tabs(["📈 Occupancy Trends", "🌦️ Weather Conditions", "📊 Raw Data Explorer"])

with tab1:
    st.subheader("Occupancy Comparison")
    
    if len(df) > 0:
        fig = go.Figure()
        # Curated color palette for line charts
        colors = ["#0D9488", "#D97706", "#2563EB", "#7C3AED", "#DB2777", "#059669", "#DC2626"]
        
        for idx, pool in enumerate(pool_selection):
            color = colors[idx % len(colors)]
            fig.add_trace(go.Scattergl(
                x=df["timestamp"], 
                y=df[pool], 
                name=pool.replace("_", " ").title(),
                mode='lines',
                line=dict(color=color, width=2),
                hovertemplate="<b>%{data.name}</b><br>Time: %{x|%Y-%m-%d %H:%M}<br>Occupancy: %{y}<extra></extra>"
            ))
            
        fig.update_layout(
            height=550,
            margin=dict(l=20, r=20, t=10, b=20),
            template="plotly_white",
            hovermode="x unified",
            xaxis=dict(title="Time", showgrid=True, gridcolor="#E5E7EB"),
            yaxis=dict(title="Occupancy (People)", showgrid=True, gridcolor="#E5E7EB", rangemode="tozero"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        
        # Apply specific x-axis tick formatting
        if time_select == "Last 24 hours" or (time_select == "Custom Range" and range_delta <= timedelta(hours=36)):
            fig.update_xaxes(
                dtick=10800000, tickformat="%a %H:%M",
                minor=dict(showgrid=True, dtick=3600000, gridcolor="#F3F4F6")
            )
        elif time_select == "Last 7 days" or (time_select == "Custom Range" and range_delta <= timedelta(days=10)):
            fig.update_xaxes(dtick=86400000, tickformat="%a %d.%m")
        else:
            fig.update_xaxes(
                dtick=86400000 * 7, tickformat="%d.%m",
                minor=dict(showgrid=True, dtick=86400000, gridcolor="#F3F4F6")
            )
                
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No occupancy data available for this range.")

with tab2:
    st.subheader("Historical Weather Conditions")
    
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
        .dt.convert_time_zone("Europe/Zurich")
    )
    
    if len(df_weather) > 0:
        fig_weather = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=[weather_labels[c] for c in weather_cols],
        )
        
        w_colors = ["#3B82F6", "#EF4444", "#8B5CF6", "#0D9488"]
        for idx, col in enumerate(weather_cols):
            fig_weather.add_trace(
                go.Scattergl(
                    x=df_weather["timestamp"], 
                    y=df_weather[col], 
                    name=weather_labels[col], 
                    line=dict(color=w_colors[idx], width=2), 
                    mode="lines"
                ),
                row=idx + 1, col=1,
            )
            
        fig_weather.update_layout(
            height=650,
            template="plotly_white",
            showlegend=False,
            margin=dict(t=30, b=20, l=20, r=20),
        )
        
        # Apply x-axis formatting
        if time_select == "Last 24 hours" or (time_select == "Custom Range" and range_delta <= timedelta(hours=36)):
            fig_weather.update_xaxes(
                dtick=10800000, tickformat="%a %H:%M",
                minor=dict(showgrid=True, dtick=3600000, gridcolor="#F3F4F6")
            )
        elif time_select == "Last 7 days" or (time_select == "Custom Range" and range_delta <= timedelta(days=10)):
            fig_weather.update_xaxes(dtick=86400000, tickformat="%a %d.%m")
        else:
            fig_weather.update_xaxes(
                dtick=86400000 * 7, tickformat="%d.%m",
                minor=dict(showgrid=True, dtick=86400000, gridcolor="#F3F4F6")
            )
                
        # Hide duplicate x-axes
        for i in range(1, 4):
            fig_weather.update_xaxes(visible=False, row=i, col=1)
            
        st.plotly_chart(fig_weather, use_container_width=True)
    else:
        st.write("No weather data available for this range.")

with tab3:
    st.subheader("Raw Data Inspect & Export")
    st.write("View or export the raw records matching your sidebar settings:")
    
    if len(df) > 0:
        # Construct full data table query
        all_cols = pool_selection + ["apparent_temperature", "cloud_cover", "precipitation", "wind_speed_10m"]
        table_df = dd.query(f"SELECT timestamp, {', '.join(all_cols)} FROM '{data_source}' WHERE {time_filter} ORDER BY timestamp DESC").pl()
        
        # Convert timestamp timezone for presentation
        table_df = table_df.with_columns(
            pl.col("timestamp")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("Europe/Zurich")
            .dt.strftime("%Y-%m-%d %H:%M")
            .alias("Time")
        ).drop("timestamp")
        
        # Clean column names for display
        rename_dict = {}
        for c in pool_selection:
            rename_dict[c] = c.replace("_", " ").title()
        rename_dict.update({
            "apparent_temperature": "Apparent Temp (°C)",
            "cloud_cover": "Cloud Cover (%)",
            "precipitation": "Precipitation (mm)",
            "wind_speed_10m": "Wind Speed (km/h)"
        })
        table_df = table_df.rename(rename_dict)
        
        st.dataframe(table_df.to_pandas(), use_container_width=True, hide_index=True)
    else:
        st.write("No raw records found.")
