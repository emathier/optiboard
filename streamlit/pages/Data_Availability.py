import streamlit as st
from logging_config import get_logger
import duckdb as dd
from contextlib import contextmanager
import time
import matplotlib.pyplot as plt
import dayplot as dp

@contextmanager
def time_this(label):
    start = time.time()  # <-- Changed from pl.time.time()
    yield
    log.debug(f"{label} took {time.time() - start:.4f}s")


log = get_logger("streamlit-overview")
log.debug("Starting Streamlit overview dashboard")

 # Fetching data
log.debug("Reading occupancy-weather.parquet into DataFrame")
df = dd.query("SELECT timestamp, hallenbad_city, hallenbad_oerlikon FROM 'data-files/occupancy-weather.parquet' ORDER BY timestamp DESC LIMIT 1000").pl()
log.info(f"Read {len(df)} rows from occupancy-weather.parquet")

# Streamlit 
st.set_page_config(
    page_title="Dashboard Overview",
    page_icon="📈",
)

st.title("Data Availability Dashboard")
st.write("This plot shows on which days we have occupancy data available. The early versions of the scraper only scraped Hallenbad Oerlikon and Hallenbad City, so the plot only shows data for these two pools. ") 

query = """
SELECT DATE(timestamp) AS day,
COUNT(*) AS total_records
FROM 'data-files/occupancy-weather.parquet'
WHERE hallenbad_oerlikon IS NOT NULL
GROUP BY day
ORDER BY day
"""

df_oerlikon = dd.query(query).pl()

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(16, 12), dpi=300)

# 2026 heatmap
dp.calendar(
    dates=df_oerlikon["day"],
    values=df_oerlikon["total_records"],
    start_date="2026-01-01",
    end_date="2026-12-31",
    cmap="Reds",
    vmin=2,
    vcenter=3,
    vmax=10,
    ax=ax1,
)

# 2025 heatmap
dp.calendar(
    dates=df_oerlikon["day"],
    values=df_oerlikon["total_records"],
    start_date="2025-01-01",
    end_date="2025-12-31",
    cmap="Reds",
    vmin=2,
    vcenter=3,
    vmax=10,
    ax=ax2,
)

# 2024 heatmap
dp.calendar(
    dates=df_oerlikon["day"],
    values=df_oerlikon["total_records"],
    start_date="2024-01-01",
    end_date="2024-12-31",
    cmap="Reds",
    vmin=2,
    vcenter=3,
    vmax=10,
    ax=ax3,
)

# year labels
text_args = dict(x=-4, y=3.5, size=30, rotation=90, color="#aaa", va="center")
ax1.text(s="2026", **text_args)
ax2.text(s="2025", **text_args)
ax3.text(s="2024", **text_args)

st.pyplot(fig)

st.markdown("---")
st.subheader("All other pools")
st.write("The following plot shows the data availability for all other pools. ")
query_bern = """
SELECT DATE(timestamp) AS day,
COUNT(*) AS total_records
FROM 'data-files/occupancy-weather.parquet'
WHERE bern_wylerbad IS NOT NULL
GROUP BY day
ORDER BY day
"""

df_bern = dd.query(query_bern).pl()

fig2, (ax4, ax5, ax6) = plt.subplots(nrows=3, figsize=(16, 12), dpi=300)

dp.calendar(
    dates=df_bern["day"],
    values=df_bern["total_records"],
    start_date="2026-01-01",
    end_date="2026-12-31",
    cmap="Blues",
    vmin=0,
    vcenter=12,
    vmax=24,
    ax=ax4,
)

dp.calendar(
    dates=df_bern["day"],
    values=df_bern["total_records"],
    start_date="2025-01-01",
    end_date="2025-12-31",
    cmap="Blues",
    vmin=0,
    vcenter=12,
    vmax=24,
    ax=ax5,
)

dp.calendar(
    dates=df_bern["day"],
    values=df_bern["total_records"],
    start_date="2024-01-01",
    end_date="2024-12-31",
    cmap="Blues",
    vmin=0,
    vcenter=12,
    vmax=24,
    ax=ax6,
)

text_args2 = dict(x=-4, y=3.5, size=30, rotation=90, color="#aaa", va="center")
ax4.text(s="2026", **text_args2)
ax5.text(s="2025", **text_args2)
ax6.text(s="2024", **text_args2)

st.pyplot(fig2)