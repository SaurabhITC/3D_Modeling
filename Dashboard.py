# Dashboard.py
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go
import json
import base64
from pathlib import Path
from io import BytesIO
from PIL import Image
import matplotlib

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Heat Stress", page_icon="🔥", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

@st.cache_data(show_spinner=False)
def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

@st.cache_data(show_spinner=False)
def file_to_data_url(path: str) -> str:
    b = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode("utf-8")

def boundsjson_to_coords(bounds_json_path: str):
    """
    Supports TWO formats:
    (A) your old format:
        {"bounds_wgs84":{"west":..,"south":..,"east":..,"north":..}, ...}
    (B) new format from tools/make_png_overlays.py:
        {"minLon":..,"minLat":..,"maxLon":..,"maxLat":..}
    Returns Mapbox image coordinates:
        [[minLon,maxLat],[maxLon,maxLat],[maxLon,minLat],[minLon,minLat]]
    """
    meta = read_json(bounds_json_path)

    if "bounds_wgs84" in meta:
        b = meta["bounds_wgs84"]
        w, s, e, n = b["west"], b["south"], b["east"], b["north"]
    else:
        w, s, e, n = meta["minLon"], meta["minLat"], meta["maxLon"], meta["maxLat"]

    return [[w, n], [e, n], [e, s], [w, s]]

def coords_to_fitbounds(coords):
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    w, e = min(lons), max(lons)
    s, n = min(lats), max(lats)
    return [[w, s], [e, n]]

def union_fitbounds(bounds_list):
    ws = [b[0][0] for b in bounds_list]
    ss = [b[0][1] for b in bounds_list]
    es = [b[1][0] for b in bounds_list]
    ns = [b[1][1] for b in bounds_list]
    return [[min(ws), min(ss)], [max(es), max(ns)]]

@st.cache_data(show_spinner=False)
def legend_bar_png(cmap_name: str, width: int = 220, height: int = 16) -> bytes:
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    grad = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    rgba = (cmap(grad) * 255).astype(np.uint8)
    rgba = np.repeat(rgba, height, axis=0)
    img = Image.fromarray(rgba, mode="RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def png_bytes_to_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")

def _fmt(v):
    try:
        if isinstance(v, (int, float, np.floating)) and not np.isnan(v):
            return f"{float(v):.2f}"
    except Exception:
        pass
    return str(v) if v is not None else "—"

# ---------------------------
# NEW: Overlay resolver for your overlays_png folder
# ---------------------------
OVERLAY_DIR = Path("data/overlays_png")

def get_overlay_paths(city: str, year: str) -> dict:
    """
    Uses your generated files in:
      data/overlays_png/<year>/

    File naming you have:
      2022_LST_Essen_c.png
      2022_SUHI_Essen.png
      2024_LST_Essen_C.png
      2024_LST_Wuppertal_c.png
      ...
    """
    ydir = OVERLAY_DIR / str(year)

    if str(year) == "2022":
        lst_stem = f"{year}_LST_{city}_c"
    else:
        suffix = "C" if city.lower() in ["essen", "soest"] else "c"
        lst_stem = f"{year}_LST_{city}_{suffix}"

    suhi_stem = f"{year}_SUHI_{city}"

    return {
        "lst_png":  ydir / f"{lst_stem}.png",
        "lst_json": ydir / f"{lst_stem}_bounds.json",
        "suhi_png": ydir / f"{suhi_stem}.png",
        "suhi_json":ydir / f"{suhi_stem}_bounds.json",
    }

# ---------------------------
# CSS (clean single-screen layout) + ✅ Selected-state pills
# ---------------------------
st.markdown(
    """
    <style>
      header[data-testid="stHeader"]{ background: transparent; box-shadow: none; }
      header[data-testid="stHeader"] [data-testid="stToolbarActions"]{ display: none; }
      div[data-testid="stDecoration"]{ display: none; }

      .block-container{ padding-top: 0.6rem; padding-bottom: 4.2rem; max-width: 100%; }

      .title-row{
        display:flex; align-items:center; justify-content:space-between;
        padding: 6px 2px 4px 2px;
      }
      .app-title{
        font: 900 34px system-ui, -apple-system, Segoe UI, Roboto, Arial;
        margin: 0;
        color: rgba(255,255,255,0.94);
      }

      .strip{
        background: rgba(18,18,20,0.70);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 10px 12px;
        backdrop-filter: blur(8px);
        margin-bottom: 10px;
      }
      .strip-label{
        font: 800 12px system-ui, -apple-system, Segoe UI, Roboto, Arial;
        color: rgba(255,255,255,0.70);
        margin-bottom: 6px;
      }

      .controls-box{
        background: rgba(18,18,20,0.70);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 12px 12px 10px 12px;
        backdrop-filter: blur(8px);
      }
      .controls-title{
        font: 900 14px system-ui, -apple-system, Segoe UI, Roboto, Arial;
        color: rgba(255,255,255,0.90);
        margin: 0 0 10px 0;
      }

      .chart-card{
        background: rgba(18,18,20,0.70);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 10px 10px 6px 10px;
        backdrop-filter: blur(8px);
        margin-bottom: 10px;
      }
      .chart-title{
        font: 900 13px system-ui, -apple-system, Segoe UI, Roboto, Arial;
        color: rgba(255,255,255,0.88);
        margin: 0 0 6px 0;
      }

      /* ✅ Selected-state wrapper for top pills (Cities/Year/Overview) */
      .pill-wrap button{
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        background: rgba(30,30,30,0.55) !important;
        color: rgba(255,255,255,0.90) !important;
        height: 44px !important;
      }
      .pill-wrap-selected button{
        border: 1px solid rgba(0,180,255,0.95) !important;
        box-shadow: 0 0 0 2px rgba(0,180,255,0.25), 0 10px 30px rgba(0,0,0,0.35) !important;
        background: rgba(0,180,255,0.12) !important;
        color: rgba(255,255,255,0.95) !important;
      }

      /* Fixed footer */
      .custom-footer{
        position: fixed;
        left: 0; right: 0; bottom: 0;
        z-index: 999;
        background: rgba(18,18,20,0.95);
        border-top: 1px solid rgba(255,255,255,0.10);
        padding: 10px 18px;
        font: 500 13px system-ui, -apple-system, Segoe UI, Roboto, Arial;
        color: rgba(255,255,255,0.82);
        backdrop-filter: blur(8px);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Data config
# ---------------------------
CITIES = {
    "Wuppertal": {"center": [7.1500, 51.2562], "zoom": 11.0},
    "Essen":     {"center": [7.0123, 51.4556], "zoom": 10.6},
    "Soest":     {"center": [8.1060, 51.5710], "zoom": 12.0},
}
YEARS = ["2022", "2024"]

BASEMAPS = {
    "Navigation Night (default)": "mapbox://styles/mapbox/navigation-night-v1",
    "Navigation Day": "mapbox://styles/mapbox/navigation-day-v1",
    "Streets": "mapbox://styles/mapbox/streets-v12",
    "Outdoors": "mapbox://styles/mapbox/outdoors-v12",
    "Light": "mapbox://styles/mapbox/light-v11",
    "Dark": "mapbox://styles/mapbox/dark-v11",
    "Satellite": "mapbox://styles/mapbox/satellite-streets-v12",
}
DEFAULT_BASEMAP_KEY = "Navigation Night (default)"

# ---------------------------
# Mapbox token
# ---------------------------
MAPBOX_API_KEY = st.secrets.get("MAPBOX_ACCESS_KEY") or st.secrets.get("MAPBOX_TOKEN") or ""
if not MAPBOX_API_KEY:
    st.error("Missing Mapbox token. Add MAPBOX_ACCESS_KEY (or MAPBOX_TOKEN) to Streamlit secrets.")
    st.stop()

# ---------------------------
# Session state
# ---------------------------
if "show_intro" not in st.session_state:
    st.session_state.show_intro = False
if "city_choice" not in st.session_state:
    st.session_state.city_choice = "Wuppertal"
if "year_choice" not in st.session_state:
    st.session_state.year_choice = "2022"

# ---------------------------
# Header: Title (left) + Logo (right)
# ---------------------------
logo_right = Path("assets/utwente_logo.png")

leftH, rightH = st.columns([0.82, 0.18])
with leftH:
    st.markdown("<div class='title-row'><div class='app-title'>Urban Heat Stress</div></div>", unsafe_allow_html=True)

with rightH:
    if logo_right.exists():
        st.image(str(logo_right), width=120)

st.markdown(
    "<div style='height:1px;background:rgba(255,255,255,0.12);margin:6px 0 10px 0;'></div>",
    unsafe_allow_html=True,
)

# ---------------------------
# ✅ helper to render selectable pills with highlight
# ---------------------------
def pill_button(label: str, key: str, selected: bool) -> bool:
    cls = "pill-wrap-selected" if selected else "pill-wrap"
    st.markdown(f"<div class='{cls}'>", unsafe_allow_html=True)
    clicked = st.button(label, key=key, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    return clicked

# ---------------------------
# Strip: Cities | Year | Overview
# ---------------------------
st.markdown("<div class='strip'>", unsafe_allow_html=True)
stripA, stripB, stripC = st.columns([0.60, 0.24, 0.16])

with stripA:
    st.markdown("<div class='strip-label'>Cities →</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if pill_button("Wuppertal", "city_wup", selected=(st.session_state.city_choice == "Wuppertal")):
            st.session_state.city_choice = "Wuppertal"
            st.rerun()
    with c2:
        if pill_button("Essen", "city_ess", selected=(st.session_state.city_choice == "Essen")):
            st.session_state.city_choice = "Essen"
            st.rerun()
    with c3:
        if pill_button("Soest", "city_soe", selected=(st.session_state.city_choice == "Soest")):
            st.session_state.city_choice = "Soest"
            st.rerun()

with stripB:
    st.markdown("<div class='strip-label'>Year →</div>", unsafe_allow_html=True)
    y1, y2 = st.columns(2)
    with y1:
        if pill_button("2022", "year_2022", selected=(st.session_state.year_choice == "2022")):
            st.session_state.year_choice = "2022"
            st.rerun()
    with y2:
        if pill_button("2024", "year_2024", selected=(st.session_state.year_choice == "2024")):
            st.session_state.year_choice = "2024"
            st.rerun()

with stripC:
    st.markdown("<div class='strip-label'>Note / Overview</div>", unsafe_allow_html=True)
    # Show selected state when show_intro=True (optional)
    if pill_button("Overview", "overview_btn", selected=bool(st.session_state.show_intro)):
        st.session_state.show_intro = True
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

city_choice = st.session_state.city_choice
year_choice = st.session_state.year_choice

# ---------------------------
# Main layout: Controls | Map
# ---------------------------
mainL, mainM = st.columns([0.20, 0.80], gap="small")

with mainL:
    st.markdown("<div class='controls-box'><div class='controls-title'>Controls</div>", unsafe_allow_html=True)

    layer_mode = st.radio("Show", ["LST (°C)", "SUHI (°C)", "Both"], index=2, label_visibility="collapsed")
    show_lst = layer_mode in ["LST (°C)", "Both"]
    show_suhi = layer_mode in ["SUHI (°C)", "Both"]

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    lst_opacity = st.slider("LST opacity", 0.0, 1.0, 0.72, 0.05) if show_lst else 0.0
    suhi_opacity = st.slider("SUHI opacity", 0.0, 1.0, 0.72, 0.05) if show_suhi else 0.0

    st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.10);margin:10px 0;'>", unsafe_allow_html=True)
    st.markdown("<div class='strip-label'>Planned</div>", unsafe_allow_html=True)
    st.checkbox("Hotspots / Coldspots", value=False, disabled=True)
    st.checkbox("Drivers (LCZ / NDVI / NDBI / NDWI)", value=False, disabled=True)
    st.checkbox("Morphometrics (3D buildings)", value=False, disabled=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Example charts (placeholders)
# ---------------------------
def example_chart(title: str):
    x = ["A", "B", "C", "D", "E"]
    y = [3, 7, 5, 9, 4]
    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    fig.update_layout(
        title=title,
        height=230,
        margin=dict(l=10, r=10, t=36, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.88)", size=12),
        title_font=dict(size=12),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    return fig

# ---------------------------
# Map build
# ---------------------------
with mainM:
    center = CITIES[city_choice]["center"]
    zoom = CITIES[city_choice]["zoom"]

    paths = get_overlay_paths(city_choice, year_choice)
    lst_ok = paths["lst_png"].exists() and paths["lst_json"].exists()
    suhi_ok = paths["suhi_png"].exists() and paths["suhi_json"].exists()

    fit_candidates = []
    fit_bounds = None

    lst_url = ""
    suhi_url = ""
    lst_coords = []
    suhi_coords = []

    lst_cmap = "turbo"
    suhi_cmap = "coolwarm"

    lst_vmin = ""
    lst_vmax = ""
    suhi_vmin = ""
    suhi_vmax = ""

    if lst_ok:
        try:
            lst_url = file_to_data_url(str(paths["lst_png"]))
            lst_coords = boundsjson_to_coords(str(paths["lst_json"]))
            if show_lst:
                fit_candidates.append(coords_to_fitbounds(lst_coords))
        except Exception:
            lst_ok = False

    if suhi_ok:
        try:
            suhi_url = file_to_data_url(str(paths["suhi_png"]))
            suhi_coords = boundsjson_to_coords(str(paths["suhi_json"]))
            if show_suhi:
                fit_candidates.append(coords_to_fitbounds(suhi_coords))
        except Exception:
            suhi_ok = False

    if fit_candidates:
        fit_bounds = union_fitbounds(fit_candidates)

    lst_legend_url = png_bytes_to_data_url(legend_bar_png(lst_cmap)) if (lst_ok and show_lst) else ""
    suhi_legend_url = png_bytes_to_data_url(legend_bar_png(suhi_cmap)) if (suhi_ok and show_suhi) else ""

    if show_lst and not lst_ok:
        st.warning("LST overlay not found for this City/Year (check data/overlays_png).")
    if show_suhi and not suhi_ok:
        st.warning("SUHI overlay not found for this City/Year (check data/overlays_png).")

    map_file = Path("map.html")
    if not map_file.exists():
        map_file = Path("Map.html")

    if not map_file.exists():
        st.error("map.html not found. Put map.html (your HTML file) in the same folder as Dashboard.py.")
        st.stop()

    map_html = read_text(str(map_file))

    # REQUIRED placeholders
    map_html = map_html.replace("__MAPBOX_KEY__", MAPBOX_API_KEY)
    map_html = map_html.replace("__CENTER__", json.dumps(center))
    map_html = map_html.replace("__ZOOM__", str(zoom))
    map_html = map_html.replace("__FIT_BOUNDS__", json.dumps(fit_bounds) if fit_bounds else "null")

    # basemap menu inside map
    map_html = map_html.replace("__STYLE_URL__", BASEMAPS[DEFAULT_BASEMAP_KEY])
    map_html = map_html.replace("__BASEMAPS__", json.dumps(BASEMAPS))
    map_html = map_html.replace("__DEFAULT_BASEMAP_NAME__", json.dumps(DEFAULT_BASEMAP_KEY))

    # rasters
    map_html = map_html.replace("__LST_URL__", lst_url if lst_ok else "")
    map_html = map_html.replace("__SUHI_URL__", suhi_url if suhi_ok else "")
    map_html = map_html.replace("__LST_COORDS__", json.dumps(lst_coords if lst_ok else []))
    map_html = map_html.replace("__SUHI_COORDS__", json.dumps(suhi_coords if suhi_ok else []))
    map_html = map_html.replace("__SHOW_LST__", "true" if (lst_ok and show_lst) else "false")
    map_html = map_html.replace("__SHOW_SUHI__", "true" if (suhi_ok and show_suhi) else "false")
    map_html = map_html.replace("__LST_OPACITY__", str(lst_opacity))
    map_html = map_html.replace("__SUHI_OPACITY__", str(suhi_opacity))

    # legend (inside map)
    map_html = map_html.replace("__LST_LEGEND_URL__", lst_legend_url)
    map_html = map_html.replace("__SUHI_LEGEND_URL__", suhi_legend_url)
    map_html = map_html.replace("__LST_VMIN__", json.dumps(_fmt(lst_vmin)))
    map_html = map_html.replace("__LST_VMAX__", json.dumps(_fmt(lst_vmax)))
    map_html = map_html.replace("__SUHI_VMIN__", json.dumps(_fmt(suhi_vmin)))
    map_html = map_html.replace("__SUHI_VMAX__", json.dumps(_fmt(suhi_vmax)))

    components.html(map_html, height=500, scrolling=False)

# ---------------------------
# Bottom charts row (3 boxes under map area)
# ---------------------------
b0, b1, b2, b3 = st.columns([0.18, 0.27, 0.27, 0.28], gap="small")

with b1:
    st.markdown("<div class='chart-card'><div class='chart-title'>Graph</div></div>", unsafe_allow_html=True)
    st.plotly_chart(example_chart("Example chart"), use_container_width=True)

with b2:
    st.markdown("<div class='chart-card'><div class='chart-title'>Graph</div></div>", unsafe_allow_html=True)
    st.plotly_chart(example_chart("Example chart"), use_container_width=True)

with b3:
    st.markdown("<div class='chart-card'><div class='chart-title'>Graph</div></div>", unsafe_allow_html=True)
    st.plotly_chart(example_chart("Example chart"), use_container_width=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown(
    "<div class='custom-footer'>Designed by <b>Saurabh Bhagchandani</b> — <i>Faculty ITC, Master's Geo-Information Science and Earth Observation</i></div>",
    unsafe_allow_html=True,
)
