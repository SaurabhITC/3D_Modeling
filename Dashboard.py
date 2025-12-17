# Importing Libraries
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import base64
from pathlib import Path

# ---------------------------
# Page config (must be first Streamlit command)
# ---------------------------
st.set_page_config(page_title="Dashboard", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
def _b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()

# ---------------------------
# Global CSS (IMPORTANT: keep header visible so sidebar toggle works)
# ---------------------------
st.markdown(
    """
    <style>
      /* Keep Streamlit header visible (needed for sidebar toggle button) */
      header[data-testid="stHeader"]{
        background: transparent;
        box-shadow: none;
      }

      /* Optional: hide Streamlit right-side toolbar actions (Deploy/menu area) */
      header[data-testid="stHeader"] [data-testid="stToolbarActions"]{
        display: none;
      }

      /* Remove thin top decoration line */
      div[data-testid="stDecoration"]{
        display: none;
      }

      /* Reduce outer page padding (fixes big top gap) */
      .block-container{
        padding-top: 0.2rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
        padding-bottom: 0.8rem;
        max-width: 100%;
      }

      /* Make sidebar a bit narrower */
      section[data-testid="stSidebar"]{
        width: 260px !important;
      }
      section[data-testid="stSidebar"] > div{
        width: 260px !important;
      }

      /* Header row layout */
      .header-row{
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:16px;
        width:100%;
        margin: 0 0 10px 0;
        padding-top: 6px;      /* avoids logo getting cut */
        padding-right: 6px;    /* avoids right cut */
      }

      .header-title{
        font-size: 56px;
        font-weight: 800;
        line-height: 1.05;
        margin: 0;
      }

      .header-logo img{
        height: 110px;
        width: auto;
        display:block;
      }

      /* Map wrapper to add small edge spacing */
      .map-wrap{
        width: 90%;
        margin: 0 auto;   /* centers and leaves space on both sides */
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Header: Title (left) + Logo (right)
# ---------------------------
logo_path = Path("assets/itc_logo.png")
logo_html = ""

if logo_path.exists():
    logo_b64 = _b64(str(logo_path))
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" alt="ITC logo" />'
else:
    st.warning("Logo not found. Put it at: assets/itc_logo.png")

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Archivo+Narrow:wght@400;700&display=swap');

      .header-row {{
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:16px;
        width:100%;
        margin: 0 0 10px 0;
        padding-top: 6px;
        padding-right: 6px;
      }}

      .header-title {{
        font-family: "Archivo Narrow", sans-serif !important;
        font-size: 56px;
        font-weight: 700;
        letter-spacing: 0.4px;
        line-height: 1.05;
        margin: 0;
      }}

      .header-logo {{
        margin-left: auto;
      }}

      .header-logo img {{
        height: 110px;
        width: auto;
        display: block;
      }}
    </style>

    <div class="header-row">
      <div class="header-title">Urban Heat Island</div>
      <div class="header-logo">{logo_html}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Controls")
view = st.sidebar.selectbox("Choose a view", ["Map", "Chart"])

BASEMAPS = {
    "Navigation Night (default)": "mapbox://styles/mapbox/navigation-night-v1",
    "Streets": "mapbox://styles/mapbox/streets-v12",
    "Satellite": "mapbox://styles/mapbox/satellite-streets-v12",
    "Outdoors": "mapbox://styles/mapbox/outdoors-v12",
    "Light": "mapbox://styles/mapbox/light-v11",
    "Dark": "mapbox://styles/mapbox/dark-v11",
    "Navigation Day": "mapbox://styles/mapbox/navigation-day-v1",
}

# ---------------------------
# MAP VIEW
# ---------------------------
if view == "Map":
    basemap_choice = st.sidebar.selectbox("Basemap", list(BASEMAPS.keys()), index=0)
    style_url = BASEMAPS[basemap_choice]

    MAPBOX_API_KEY = st.secrets["MAPBOX_ACCESS_KEY"]

    with open("Map.html", "r", encoding="utf-8") as f:
        map_html = f.read()

    map_html = map_html.replace("__MAPBOX_KEY__", MAPBOX_API_KEY)
    map_html = map_html.replace("__STYLE_URL__", style_url)

    # ✅ add margins using spacer columns (adjust numbers to taste)
    left, mid, right = st.columns([1, 10, 1])  # try [1, 8, 1] for bigger margins
    with mid:
        components.html(map_html, height=600, scrolling=False)


# ---------------------------
# CHART VIEW
# ---------------------------
else:
    df = pd.DataFrame({"x": np.arange(50), "y": np.random.randn(50).cumsum()})
    fig = px.line(df, x="x", y="y", title="Example chart")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown(
    "<div style='opacity:0.85; font-size:13px;'>Designed by <b>Saurabh Bhagchandani</b> — <i>Faculty ITC, Master's Geo-Information Science and Earth Observation</i></div>",
    unsafe_allow_html=True,
)
