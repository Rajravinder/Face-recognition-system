import streamlit as st
import pandas as pd
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")


ts=time.time()
date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp=datetime.fromtimestamp(ts).strftime("%H:%M:%S")
df=pd.read_csv("Attendance/Attendance_"+ date +".csv")

st.title("Attendance Dashboard")
st.dataframe(df.style.highlight_max(axis=0))