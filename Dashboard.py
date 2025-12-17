import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("DT Dashboard")

# Sidebar
st.sidebar.header("Controls")
view = st.sidebar.selectbox("Choose a view", ["Map", "Chart"])

if view == "Map":
    MAPBOX_API_KEY = st.secrets["MAPBOX_ACCESS_KEY"]

    with open("Map.html", "r", encoding="utf-8") as f:
        map_html = f.read()

    map_html = map_html.replace("__MAPBOX_KEY__", MAPBOX_API_KEY)
    components.html(map_html, height=700, scrolling=False)

else:
    import pandas as pd
    import numpy as np
    import plotly.express as px

    df = pd.DataFrame({"x": np.arange(50), "y": np.random.randn(50).cumsum()})
    fig = px.line(df, x="x", y="y", title="Example chart")
    st.plotly_chart(fig, use_container_width=True)
