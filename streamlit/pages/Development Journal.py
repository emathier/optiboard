import streamlit as st

st.markdown("""
# Development Journal
The goal of this journal is to document my progress and justify my decisions during the development of the project. I will use AI during the writing of this journal.

---
## 2026-06-10 Initial Journal Entry
Currently the historical occupancy data and historical weather forecast data is stored and ready in parquet files. I set up scripts to split the data in a training and validation set. The last week of the data is the validation set. My current plan is to set up a first rough machine learning model to set a baseline. In doing so I also want to set up a proper experiment tracking system. Currently I think this will be mlflow. 
Current goals:
- Define a benchmark to optimize for.
- Set up mlflow for experiment tracking.
- Set up a first machine learning model to set a base line.


###### Benchmark and Loss function.
I have been thinking about which benchmarks I want to optimize for. I think the most sensible way to do this is to derive them from the potential use cases of such a tool. I want to use my tool to look up the occupancy of the pool of my interest at the same day to lookup the occupancy today and how it is going to develop or I want to use it to check which day in the next week would be best. This results in two benchamrks for me:
- Intra day benchmark: How accurate is the model predicting the occcupancy from a time in the day to the end of the day.
- Inter day benchmark: How accurate is the model predicting the occupancy for the next 7 days.


For getting started I will focus on the inter day benchmark as it is in my opinion easier to implement.
---       
""")