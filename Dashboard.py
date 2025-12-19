# Importing Libraries
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import json
from pathlib import Path
from io import BytesIO
from PIL import Image
import matplotlib

# ---------------------------
# Page config (must be first Streamlit command)
# ---------------------------
st.set_page_config(page_title="Dashboard", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
def _b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")

def png_to_data_url(png_path: str) -> str:
    b64 = base64.b64encode(Path(png_path).read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def png_bytes_to_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")

def boundsjson_to_coords(bounds_json_path: str):
    """
    Reads bounds from JSON written by 01_build_overlays.py
    and returns Mapbox image coordinates format:
    [[west,north],[east,north],[east,south],[west,south]]
    """
    meta = json.loads(Path(bounds_json_path).read_text(encoding="utf-8"))
    b = meta["bounds_wgs84"]
    w, s, e, n = b["west"], b["south"], b["east"], b["north"]
    return [[w, n], [e, n], [e, s], [w, s]]

def legend_bar_png(cmap_name: str, width: int = 220, height: int = 16) -> bytes:
    """
    Legend bar PNG as bytes (used inside the map as an <img>).
    width=220 fits well in the floating legend box.
    """
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    grad = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    rgba = (cmap(grad) * 255).astype(np.uint8)
    rgba = np.repeat(rgba, height, axis=0)

    img = Image.fromarray(rgba, mode="RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

# ---------------------------
# Global CSS
# ---------------------------
st.markdown(
    """
    <style>
      header[data-testid="stHeader"]{
        background: transparent;
        box-shadow: none;
      }
      header[data-testid="stHeader"] [data-testid="stToolbarActions"]{
        display: none;
      }
      div[data-testid="stDecoration"]{
        display: none;
      }
      .block-container{
        padding-top: 0.2rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
        padding-bottom: 0.8rem;
        max-width: 100%;
      }
      section[data-testid="stSidebar"]{
        width: 260px !important;
      }
      section[data-testid="stSidebar"] > div{
        width: 260px !important;
      }
      .header-row{
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:16px;
        width:100%;
        margin: 0 0 10px 0;
        padding-top: 6px;
        padding-right: 6px;
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
      .map-wrap{
        width: 90%;
        margin: 0 auto;
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

    st.sidebar.subheader("Raster overlays (dropdown)")
    selected_layers = st.sidebar.multiselect(
        "Select overlays (check/uncheck)",
        options=["LST (°C)", "SUHI (°C)"],
        default=["LST (°C)"],
    )

    show_lst = "LST (°C)" in selected_layers
    show_suhi = "SUHI (°C)" in selected_layers

    lst_opacity = st.sidebar.slider("LST opacity", 0.0, 1.0, 0.70, 0.05)
    suhi_opacity = st.sidebar.slider("SUHI opacity", 0.0, 1.0, 0.70, 0.05)

    # Load Map.html + inject key/style
    MAPBOX_API_KEY = st.secrets["MAPBOX_ACCESS_KEY"]

    with open("Map.html", "r", encoding="utf-8") as f:
        map_html = f.read()

    map_html = map_html.replace("__MAPBOX_KEY__", MAPBOX_API_KEY)
    map_html = map_html.replace("__STYLE_URL__", style_url)

    # Overlay file paths
    lst_png = Path("data/overlays/summer_lst.png")
    lst_json = Path("data/overlays/summer_lst_bounds.json")
    suhi_png = Path("data/overlays/summer_suhi.png")
    suhi_json = Path("data/overlays/summer_suhi_bounds.json")

    # ---------------------------
    # Inject overlays + legends into Map.html
    # ---------------------------
    overlays_ok = (lst_png.exists() and lst_json.exists() and suhi_png.exists() and suhi_json.exists())

    if not overlays_ok:
        st.warning(
            "Overlay files not found in data/overlays/. "
            "Generate them first (summer_lst/summer_suhi PNG + bounds.json)."
        )

        # Overlay fallbacks
        map_html = map_html.replace("__LST_URL__", "")
        map_html = map_html.replace("__SUHI_URL__", "")
        map_html = map_html.replace("__LST_COORDS__", "[]")
        map_html = map_html.replace("__SUHI_COORDS__", "[]")
        map_html = map_html.replace("__SHOW_LST__", "false")
        map_html = map_html.replace("__SHOW_SUHI__", "false")
        map_html = map_html.replace("__LST_OPACITY__", "0")
        map_html = map_html.replace("__SUHI_OPACITY__", "0")

        # Legend fallbacks
        map_html = map_html.replace("__LST_LEGEND_TITLE__", "LST (°C)")
        map_html = map_html.replace("__SUHI_LEGEND_TITLE__", "SUHI (°C)")
        map_html = map_html.replace("__LST_LEGEND_URL__", "")
        map_html = map_html.replace("__SUHI_LEGEND_URL__", "")
        map_html = map_html.replace("__LST_VMIN__", "")
        map_html = map_html.replace("__LST_VMAX__", "")
        map_html = map_html.replace("__SUHI_VMIN__", "")
        map_html = map_html.replace("__SUHI_VMAX__", "")

    else:
        # Overlay images + coords
        lst_url = png_to_data_url(str(lst_png))
        suhi_url = png_to_data_url(str(suhi_png))

        lst_coords = boundsjson_to_coords(str(lst_json))
        suhi_coords = boundsjson_to_coords(str(suhi_json))

        map_html = map_html.replace("__LST_URL__", lst_url)
        map_html = map_html.replace("__SUHI_URL__", suhi_url)
        map_html = map_html.replace("__LST_COORDS__", json.dumps(lst_coords))
        map_html = map_html.replace("__SUHI_COORDS__", json.dumps(suhi_coords))

        map_html = map_html.replace("__SHOW_LST__", "true" if show_lst else "false")
        map_html = map_html.replace("__SHOW_SUHI__", "true" if show_suhi else "false")
        map_html = map_html.replace("__LST_OPACITY__", str(lst_opacity))
        map_html = map_html.replace("__SUHI_OPACITY__", str(suhi_opacity))

        # Legend meta from JSON
        lst_meta = json.loads(lst_json.read_text(encoding="utf-8"))
        suhi_meta = json.loads(suhi_json.read_text(encoding="utf-8"))

        lst_cmap = lst_meta.get("colormap", "inferno")
        suhi_cmap = suhi_meta.get("colormap", "coolwarm")

        lst_vr = lst_meta.get("value_range_used", {})
        suhi_vr = suhi_meta.get("value_range_used", {})

        lst_vmin = lst_vr.get("vmin", "")
        lst_vmax = lst_vr.get("vmax", "")
        suhi_vmin = suhi_vr.get("vmin", "")
        suhi_vmax = suhi_vr.get("vmax", "")

        lst_legend_url = png_bytes_to_data_url(legend_bar_png(lst_cmap))
        suhi_legend_url = png_bytes_to_data_url(legend_bar_png(suhi_cmap))

        map_html = map_html.replace("__LST_LEGEND_TITLE__", "LST (°C)")
        map_html = map_html.replace("__SUHI_LEGEND_TITLE__", "SUHI (°C)")
        map_html = map_html.replace("__LST_LEGEND_URL__", lst_legend_url)
        map_html = map_html.replace("__SUHI_LEGEND_URL__", suhi_legend_url)

        def _fmt(v):
            return f"{v:.2f}" if isinstance(v, (int, float)) else str(v)

        map_html = map_html.replace("__LST_VMIN__", _fmt(lst_vmin))
        map_html = map_html.replace("__LST_VMAX__", _fmt(lst_vmax))
        map_html = map_html.replace("__SUHI_VMIN__", _fmt(suhi_vmin))
        map_html = map_html.replace("__SUHI_VMAX__", _fmt(suhi_vmax))

    left, mid, right = st.columns([1, 10, 1])
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
