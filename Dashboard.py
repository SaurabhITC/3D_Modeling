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
import pandas as pd
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Heat Stress", layout="wide")


# ---------------------------
# Helpers
# ---------------------------
# ✅ UPDATED: cache invalidates automatically when file changes (mtime passed)
@st.cache_data(show_spinner=False)
def read_text(path: str, mtime: float) -> str:
    # mtime is only here to bust cache when the file is modified
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
    (A) old:
        {"bounds_wgs84":{"west":..,"south":..,"east":..,"north":..}, ...}
    (B) new:
        {"minLon":..,"minLat":..,"maxLon":..,"maxLat":..}
    Returns MapLibre image coordinates:
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
# Overlays (PNG + bounds.json for map)
# ---------------------------
OVERLAY_DIR = Path("data/overlays_png")

def get_overlay_paths(city: str, year: str) -> dict:
    """
    Robust resolver:
    - Handles filename CASE differences (Windows vs Linux/Streamlit Cloud)
    - Handles _c vs _C inconsistencies
    - Returns first matching (png + bounds.json) pair
    """
    ydir = OVERLAY_DIR / str(year)

    city_variants = [city, city.upper(), city.lower(), city.title()]

    lst_candidates = []
    for cv in city_variants:
        lst_candidates.extend([
            f"{year}_LST_{cv}_c",
            f"{year}_LST_{cv}_C",
        ])

    suhi_candidates = []
    for cv in city_variants:
        suhi_candidates.extend([
            f"{year}_SUHI_{cv}",
            f"{year}_suhi_{cv}",
        ])

    def first_existing(stems):
        for stem in stems:
            png = ydir / f"{stem}.png"
            js  = ydir / f"{stem}_bounds.json"
            if png.exists() and js.exists():
                return png, js
        return Path(""), Path("")

    lst_png, lst_json = first_existing(lst_candidates)
    suhi_png, suhi_json = first_existing(suhi_candidates)

    return {
        "lst_png":  lst_png,
        "lst_json": lst_json,
        "suhi_png": suhi_png,
        "suhi_json": suhi_json,
    }


# ---------------------------
# Buildings (GeoJSON served from /static/buildings)
# ---------------------------
BUILDINGS_DIR = Path("static/buildings")

def get_buildings_url(city: str) -> str:
    base = f"{city.upper()}_buildings_4326"
    for ext in (".geojson", ".geojson"):
        fname = base + ext
        fpath = BUILDINGS_DIR / fname
        if fpath.exists():
            return f"/static/buildings/{fname}"
    return ""


# ---------------------------
# Boundaries (dual mode)
# ---------------------------
# 1) File-system boundaries for Dashboard injection (your pushed folder)
BOUNDARIES_FS_DIR = Path("Git_Data/WGS84 geojason")

# 2) Optional URL-served boundaries (only if you also copy boundaries into /static/boundarygeojson)
BOUNDARIES_DIR = Path("static/boundarygeojson")

@st.cache_data(show_spinner=False)
def _read_geojson(path_str: str) -> dict:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))

def _find_boundary_file(city: str) -> Path | None:
    if not BOUNDARIES_FS_DIR.exists():
        return None

    candidates = [
        BOUNDARIES_FS_DIR / f"{city}.geojson",
        BOUNDARIES_FS_DIR / f"{city.lower()}.geojson",
        BOUNDARIES_FS_DIR / f"{city.upper()}.geojson",
        BOUNDARIES_FS_DIR / f"{city}_boundary.geojson",
        BOUNDARIES_FS_DIR / f"{city}_boundary_4326.geojson",
        BOUNDARIES_FS_DIR / f"{city.upper()}_boundary_4326.geojson",
    ]
    for fp in candidates:
        if fp.exists():
            return fp

    city_u = city.upper()
    for fp in BOUNDARIES_FS_DIR.glob("*.geojson"):
        if fp.stem.upper() == city_u:
            return fp
    for fp in BOUNDARIES_FS_DIR.glob("*.geojson"):
        if city_u in fp.name.upper():
            return fp

    return None

def load_boundary_fc(cities: list[str]) -> dict | None:
    feats = []
    for c in cities:
        fp = _find_boundary_file(c)
        if not fp:
            continue
        try:
            gj = _read_geojson(str(fp))
            for f in gj.get("features", []):
                f.setdefault("properties", {})
                f["properties"]["name"] = c
            feats.extend(gj.get("features", []))
        except Exception:
            pass

    if not feats:
        return None
    return {"type": "FeatureCollection", "features": feats}

def get_boundary_url(city: str) -> str:
    """
    Optional static URL resolver (case-safe for Streamlit Cloud)
    """
    if not BOUNDARIES_DIR.exists():
        return ""

    candidates = [
        f"{city.upper()}_boundary_4326.geojson",
        f"{city.upper()}_boundary.geojson",
        f"{city}_boundary_4326.geojson",
        f"{city}_boundary.geojson",
        f"{city}.geojson",
        f"{city.upper()}.geojson",
        f"{city.upper()}_BOUNDARY_4326.geojson",
        f"{city.upper()}_BOUNDARY.geojson",
    ]
    for fname in candidates:
        fp = BOUNDARIES_DIR / fname
        if fp.exists():
            return f"/static/boundarygeojson/{fp.name}"

    city_u = city.upper()
    for fp in BOUNDARIES_DIR.glob("*.geojson"):
        if city_u in fp.name.upper():
            return f"/static/boundarygeojson/{fp.name}"

    return ""


# ---------------------------
# GeoTIFF graphs (read from your local final_outputs folders)
# ---------------------------
RASTER_DIRS = {
    "2022": Path(r"C:\University of Twente (Master's)\2nd year\thesis mateerial\Data\ECOSTRESS_LST_DATA\2022\final_outputs"),
    "2024": Path(r"C:\University of Twente (Master's)\2nd year\thesis mateerial\Data\ECOSTRESS_LST_DATA\2024\final_outputs"),
}

def _try_paths(paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None

def get_layer_tif(year: str, layer: str, city: str) -> Path | None:
    """
    layer: "LST" or "SUHI"
    Uses your naming:
      2024_LST_Essen_C.tif / 2022_LST_Essen_c.tif
      2024_SUHI_Essen.tif  / 2022_SUHI_Essen.tif
    """
    d = RASTER_DIRS.get(str(year))
    if not d or not d.exists():
        return None

    if layer.upper() == "LST":
        candidates = [
            d / f"{year}_LST_{city}_c.tif",
            d / f"{year}_LST_{city}_C.tif",
            d / f"{year}_LST_{city}_c.tiff",
            d / f"{year}_LST_{city}_C.tiff",
        ]
    else:  # SUHI
        candidates = [
            d / f"{year}_SUHI_{city}.tif",
            d / f"{year}_SUHI_{city}.tiff",
        ]
    return _try_paths(candidates)

def get_lcz_tif() -> Path | None:
    """
    LCZ seems to be a single raster: LCZ.tif
    We search in both year folders and return the first found.
    """
    candidates = []
    for y in ["2024", "2022"]:
        d = RASTER_DIRS.get(y)
        if d and d.exists():
            candidates.extend([
                d / "LCZ.tif",
                d / "LCZ.tiff",
                d / "LCZ_25832.tif",
                d / "LCZ_25832.tiff",
            ])
    return _try_paths(candidates)

@st.cache_data(show_spinner=False)
def read_raster_values(path_str: str, max_samples: int = 250000) -> np.ndarray:
    p = Path(path_str)
    if not p.exists():
        return np.array([], dtype=np.float32)

    with rasterio.open(p) as src:
        arr = src.read(1, masked=True)

    vals = np.asarray(arr.compressed(), dtype=np.float32)
    if vals.size == 0:
        return vals

    if vals.size > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(vals.size, size=max_samples, replace=False)
        vals = vals[idx]
    return vals

@st.cache_data(show_spinner=False)
def load_suhi_lcz_samples(suhi_path: str, lcz_path: str, max_samples: int = 250000):
    """
    Reproject LCZ onto SUHI grid (nearest), then return aligned samples.
    """
    suhi_p = Path(suhi_path)
    lcz_p = Path(lcz_path)
    if (not suhi_p.exists()) or (not lcz_p.exists()):
        return np.array([], dtype=np.float32), np.array([], dtype=np.int16)

    with rasterio.open(suhi_p) as suhi_src:
        suhi = suhi_src.read(1, masked=True)
        ref_transform = suhi_src.transform
        ref_crs = suhi_src.crs
        ref_shape = suhi.shape

    with rasterio.open(lcz_p) as lcz_src:
        lcz_in = lcz_src.read(1, masked=True)
        lcz_reproj = np.full(ref_shape, fill_value=-9999, dtype=np.int32)

        reproject(
            source=lcz_in.filled(lcz_src.nodata if lcz_src.nodata is not None else -9999),
            destination=lcz_reproj,
            src_transform=lcz_src.transform,
            src_crs=lcz_src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest,
        )

    suhi_flat = np.asarray(suhi.filled(np.nan), dtype=np.float32).reshape(-1)
    mask = np.isfinite(suhi_flat)
    suhi_vals = suhi_flat[mask]

    lcz_flat = lcz_reproj.reshape(-1)[mask].astype(np.int16)

    good = (lcz_flat > 0) & (lcz_flat != -9999)
    suhi_vals = suhi_vals[good].astype(np.float32)
    lcz_vals = lcz_flat[good].astype(np.int16)

    if suhi_vals.size > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(suhi_vals.size, size=max_samples, replace=False)
        suhi_vals = suhi_vals[idx]
        lcz_vals = lcz_vals[idx]

    return suhi_vals, lcz_vals

def make_histogram(vals: np.ndarray, title: str, x_label: str):
    if vals.size == 0:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            height=230,
            margin=dict(l=10, r=10, t=36, b=35),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.88)", size=12),
            title_font=dict(size=12),
        )
        return fig

    med = float(np.nanmedian(vals))
    p90 = float(np.nanpercentile(vals, 90))

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=vals, nbinsx=50, name="values"))
    fig.update_layout(
        title=title,
        height=230,
        margin=dict(l=10, r=10, t=36, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.88)", size=12),
        title_font=dict(size=12),
        xaxis_title=x_label,
        yaxis_title="Count",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False)

    fig.add_vline(x=med, line_width=2)
    fig.add_vline(x=p90, line_width=2, line_dash="dot")

    return fig

def make_suhi_by_lcz_box(
    suhi_vals: np.ndarray,
    lcz_vals: np.ndarray,
    title: str,
    lcz_min: int = 1,
    lcz_max: int = 17,
    min_n: int = 0
):
    df = pd.DataFrame({"LCZ": lcz_vals, "SUHI": suhi_vals}).dropna()

    fig = go.Figure()

    for cls in range(lcz_min, lcz_max + 1):
        sub = df[df["LCZ"] == cls]
        if sub.empty:
            continue
        if min_n > 0 and len(sub) < min_n:
            continue

        fig.add_trace(
            go.Box(
                y=sub["SUHI"],
                name=str(cls),
                boxpoints=False
            )
        )

    if len(fig.data) == 0:
        fig.update_layout(
            title=title,
            height=230,
            margin=dict(l=10, r=10, t=36, b=35),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.88)", size=12),
            title_font=dict(size=12),
        )
        return fig

    fig.update_layout(
        title=title,
        height=230,
        margin=dict(l=10, r=10, t=36, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.88)", size=12),
        title_font=dict(size=12),
        xaxis_title="LCZ class (1–17)",
        yaxis_title="SUHI (°C)",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    return fig


# ---------------------------
# CSS
# ---------------------------
st.markdown(
    """
    <style>
      header[data-testid="stHeader"]{ background: transparent; box-shadow: none; }
      header[data-testid="stHeader"] [data-testid="stToolbarActions"]{ display: none; }
      div[data-testid="stDecoration"]{ display: none; }

      .block-container{ padding-top: 0.6rem; padding-bottom: 4.2rem; max-width: 100%; }

      .app-title{
        font: 900 34px system-ui, -apple-system, Segoe UI, Roboto, Arial;
        margin: 0;
        color: rgba(255,255,255,0.94);
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
    "Wuppertal": {"center": [7.1500, 51.2562], "zoom": 10.5},
    "Essen":     {"center": [7.0123, 51.4556], "zoom": 10.0},
    "Soest":     {"center": [8.1060, 51.5710], "zoom": 11.0},
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
# Session state
# ---------------------------
if "city_choice" not in st.session_state:
    st.session_state.city_choice = "All cities"
if "year_choice" not in st.session_state:
    st.session_state.year_choice = "2022"

for k, default in {
    "show_lst": False,
    "show_suhi": False,
    "show_ndvi": False,
    "show_ndwi": False,
    "show_ndbi": False,
    "show_albedo": False,
    "show_lcz":  False,
    "show_buildings": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = default


# ---------------------------
# Header (Title + Logo)
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
logo_path = BASE_DIR / "assets" / "itc_logo.png"
logo_data_url = file_to_data_url(str(logo_path)) if logo_path.exists() else ""

st.markdown(
    f"""
    <div style="
        display:flex;
        align-items:center;
        justify-content:space-between;
        width:100%;
        padding: 6px 2px 4px 2px;
    ">
      <div class="app-title" style="margin:0;">Urban Heat Stress</div>
      {"<img src='" + logo_data_url + "' style='height:60px; width:auto; display:block; margin:0;'/>" if logo_data_url else ""}
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<div style='height:1px;background:rgba(255,255,255,0.12);margin:6px 0 10px 0;'></div>",
    unsafe_allow_html=True,
)

def pill_button(label: str, key: str, selected: bool) -> bool:
    cls = "pill-wrap-selected" if selected else "pill-wrap"
    st.markdown(f"<div class='{cls}'>", unsafe_allow_html=True)
    clicked = st.button(label, key=key, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    return clicked


# ---------------------------
# Strip (Cities | Year)
# ---------------------------
stripC, stripY = st.columns([0.72, 0.28])

with stripC:
    st.markdown("<div class='strip-label'>Cities →</div>", unsafe_allow_html=True)
    c0, c1, c2, c3 = st.columns(4)

    with c0:
        if pill_button("All cities", "city_all", selected=(st.session_state.city_choice == "All cities")):
            st.session_state.city_choice = "All cities"
            st.rerun()

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

with stripY:
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

city_choice = st.session_state.city_choice
year_choice = st.session_state.year_choice


# ---------------------------
# Main layout
# ---------------------------
mainL, mainM = st.columns([0.20, 0.80], gap="small")


# ---------------------------
# Controls (LEFT)
# ---------------------------
with mainL:
    st.markdown("<div class='controls-box'><div class='controls-title'>Controls</div>", unsafe_allow_html=True)

    st.markdown("<div class='strip-label'>Heat Stressors</div>", unsafe_allow_html=True)

    show_lst = st.checkbox("LST (°C)", key="show_lst")
    lst_opacity = st.slider("LST opacity", 0.0, 1.0, 0.72, 0.05, key="lst_opacity") if show_lst else 0.0

    show_suhi = st.checkbox("SUHI (°C)", key="show_suhi")
    suhi_opacity = st.slider("SUHI opacity", 0.0, 1.0, 0.72, 0.05, key="suhi_opacity") if show_suhi else 0.0

    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.10);margin:10px 0;'>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='strip-label'>Environmental Drivers</div>", unsafe_allow_html=True)

    show_ndvi = st.checkbox("NDVI", key="show_ndvi")
    ndvi_opacity = st.slider("NDVI opacity", 0.0, 1.0, 0.72, 0.05, key="ndvi_opacity") if show_ndvi else 0.0

    show_ndbi = st.checkbox("NDBI", key="show_ndbi")
    ndbi_opacity = st.slider("NDBI opacity", 0.0, 1.0, 0.72, 0.05, key="ndbi_opacity") if show_ndbi else 0.0

    show_ndwi = st.checkbox("NDWI", key="show_ndwi")
    ndwi_opacity = st.slider("NDWI opacity", 0.0, 1.0, 0.72, 0.05, key="ndwi_opacity") if show_ndwi else 0.0

    show_albedo = st.checkbox("Surface Albedo", key="show_albedo")
    albedo_opacity = st.slider("Albedo opacity", 0.0, 1.0, 0.72, 0.05, key="albedo_opacity") if show_albedo else 0.0

    show_lcz = st.checkbox("LCZ", key="show_lcz")
    lcz_opacity = st.slider("LCZ opacity", 0.0, 1.0, 0.72, 0.05, key="lcz_opacity") if show_lcz else 0.0

    show_Elevation = st.checkbox("Elevation", key="show_Elevation")
    Elevation_opacity = st.slider("Elevation opacity", 0.0, 1.0, 0.72, 0.05, key="Elevation_opacity") if show_Elevation else 0.0

    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.10);margin:10px 0;'>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='strip-label'>Urban Morphometrics</div>", unsafe_allow_html=True)

    show_buildings = st.checkbox("3D Buildings", key="show_buildings")
    buildings_opacity = st.slider("3D Buildings opacity", 0.0, 1.0, 0.90, 0.05, key="buildings_opacity") if show_buildings else 0.0

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
# Map build (RIGHT)
# ---------------------------
with mainM:
    if city_choice == "All cities":
        selected_cities = ["Wuppertal", "Essen", "Soest"]
        cs = [CITIES[c]["center"] for c in selected_cities]
        center = [sum(x[0] for x in cs) / len(cs), sum(x[1] for x in cs) / len(cs)]
        zoom = 8.9
    else:
        selected_cities = [city_choice]
        center = CITIES[city_choice]["center"]
        zoom = CITIES[city_choice]["zoom"]

    fit_candidates = []
    fit_bounds = None

    lst_url = ""
    suhi_url = ""
    lst_coords = []
    suhi_coords = []

    lst_layers = []
    suhi_layers = []

    # ✅ Boundaries: (1) injected GeoJSON, (2) optional URL list
    boundary_geojson = load_boundary_fc(selected_cities)

    boundary_layers = []
    for c in selected_cities:
        burl = get_boundary_url(c)
        if burl:
            boundary_layers.append({"id": f"bnd_{c.lower()}", "city": c, "url": burl})

    ndvi_layers, ndbi_layers, ndwi_layers, albedo_layers, lcz_layers = [], [], [], [], []

    lst_cmap = "turbo"
    suhi_cmap = "coolwarm"

    lst_vmin = ""
    lst_vmax = ""
    suhi_vmin = ""
    suhi_vmax = ""

    for c in selected_cities:
        paths = get_overlay_paths(c, year_choice)

        lst_ok_c = paths["lst_png"].exists() and paths["lst_json"].exists()
        suhi_ok_c = paths["suhi_png"].exists() and paths["suhi_json"].exists()

        if lst_ok_c:
            try:
                _url = file_to_data_url(str(paths["lst_png"]))
                _coords = boundsjson_to_coords(str(paths["lst_json"]))
                lst_layers.append({"id": f"{c}_lst", "city": c, "url": _url, "coords": _coords})
                if show_lst:
                    fit_candidates.append(coords_to_fitbounds(_coords))
                if c == city_choice:
                    lst_url = _url
                    lst_coords = _coords
            except Exception:
                pass

        if suhi_ok_c:
            try:
                _url = file_to_data_url(str(paths["suhi_png"]))
                _coords = boundsjson_to_coords(str(paths["suhi_json"]))
                suhi_layers.append({"id": f"{c}_suhi", "city": c, "url": _url, "coords": _coords})
                if show_suhi:
                    fit_candidates.append(coords_to_fitbounds(_coords))
                if c == city_choice:
                    suhi_url = _url
                    suhi_coords = _coords
            except Exception:
                pass

    if city_choice != "All cities":
        if (not lst_url) and lst_layers:
            match = next((L for L in lst_layers if L.get("city") == city_choice), lst_layers[0])
            lst_url = match.get("url", "")
            lst_coords = match.get("coords", [])

        if (not suhi_url) and suhi_layers:
            match = next((L for L in suhi_layers if L.get("city") == city_choice), suhi_layers[0])
            suhi_url = match.get("url", "")
            suhi_coords = match.get("coords", [])

    lst_available = bool(lst_url) and isinstance(lst_coords, list) and len(lst_coords) == 4
    suhi_available = bool(suhi_url) and isinstance(suhi_coords, list) and len(suhi_coords) == 4

    if fit_candidates:
        fit_bounds = union_fitbounds(fit_candidates)

    lst_legend_url = png_bytes_to_data_url(legend_bar_png(lst_cmap)) if (lst_available and show_lst) else ""
    suhi_legend_url = png_bytes_to_data_url(legend_bar_png(suhi_cmap)) if (suhi_available and show_suhi) else ""

    if city_choice != "All cities":
        if show_lst and not lst_available:
            st.warning("LST overlay not found for this City/Year (check data/overlays_png).")
        if show_suhi and not suhi_available:
            st.warning("SUHI overlay not found for this City/Year (check data/overlays_png).")
    else:
        if show_lst and not lst_layers:
            st.warning("LST overlay not found for ALL cities for this Year (check data/overlays_png).")
        if show_suhi and not suhi_layers:
            st.warning("SUHI overlay not found for ALL cities for this Year (check data/overlays_png).")

    buildings_url = get_buildings_url(city_choice) if (show_buildings and city_choice != "All cities") else ""
    if show_buildings and city_choice != "All cities" and not buildings_url:
        st.warning("3D Buildings GeoJSON not found in static/buildings for this city.")

    overlays_payload = {
        "mode": "all" if city_choice == "All cities" else "single",
        "cities": selected_cities,
        "year": year_choice,
        "center": center,
        "zoom": zoom,
        "fit_bounds": fit_bounds,

        # ✅ boundaries: injected + optional URL list + selection highlight
        "selected_city": city_choice,
        "boundary_geojson": boundary_geojson,
        "boundary_layers": boundary_layers,

        # Heat Stressors
        "show_lst": bool(show_lst),
        "show_suhi": bool(show_suhi),
        "lst_opacity": float(lst_opacity),
        "suhi_opacity": float(suhi_opacity),
        "lst_layers": lst_layers,
        "suhi_layers": suhi_layers,
        "lst_legend_url": lst_legend_url,
        "suhi_legend_url": suhi_legend_url,
        "lst_vmin": _fmt(lst_vmin),
        "lst_vmax": _fmt(lst_vmax),
        "suhi_vmin": _fmt(suhi_vmin),
        "suhi_vmax": _fmt(suhi_vmax),

        # Environmental Drivers (active toggles; layers added later)
        "show_ndvi": bool(show_ndvi),
        "show_ndbi": bool(show_ndbi),
        "show_ndwi": bool(show_ndwi),
        "show_albedo": bool(show_albedo),
        "show_lcz":  bool(show_lcz),

        "ndvi_opacity": float(ndvi_opacity),
        "ndbi_opacity": float(ndbi_opacity),
        "ndwi_opacity": float(ndwi_opacity),
        "albedo_opacity": float(albedo_opacity),
        "lcz_opacity":  float(lcz_opacity),

        "ndvi_layers": ndvi_layers,
        "ndbi_layers": ndbi_layers,
        "ndwi_layers": ndwi_layers,
        "albedo_layers": albedo_layers,
        "lcz_layers":  lcz_layers,

        # Urban Morphometrics
        "show_buildings": bool(show_buildings),
        "buildings_opacity": float(buildings_opacity),
        "buildings_url": buildings_url,
    }

    map_file = Path("map.html")
    if not map_file.exists():
        map_file = Path("Map.html")

    if not map_file.exists():
        st.error("map.html not found. Put map.html (your HTML file) in the same folder as Dashboard.py.")
        st.stop()

    map_html = read_text(str(map_file), map_file.stat().st_mtime)

    map_html = map_html.replace("__MAPBOX_KEY__", "")

    map_html = map_html.replace("__CENTER__", json.dumps(center))
    map_html = map_html.replace("__ZOOM__", str(zoom))
    map_html = map_html.replace("__FIT_BOUNDS__", json.dumps(fit_bounds) if fit_bounds else "null")

    map_html = map_html.replace("__LST_URL__", lst_url if lst_available else "")
    map_html = map_html.replace("__SUHI_URL__", suhi_url if suhi_available else "")
    map_html = map_html.replace("__LST_COORDS__", json.dumps(lst_coords if lst_available else []))
    map_html = map_html.replace("__SUHI_COORDS__", json.dumps(suhi_coords if suhi_available else []))
    map_html = map_html.replace("__SHOW_LST__", "true" if (lst_available and show_lst) else "false")
    map_html = map_html.replace("__SHOW_SUHI__", "true" if (suhi_available and show_suhi) else "false")
    map_html = map_html.replace("__LST_OPACITY__", str(lst_opacity))
    map_html = map_html.replace("__SUHI_OPACITY__", str(suhi_opacity))

    map_html = map_html.replace("__SHOW_BUILDINGS__", "true" if (show_buildings and bool(buildings_url)) else "false")
    map_html = map_html.replace("__BUILDINGS_URL__", buildings_url if buildings_url else "")

    map_html = map_html.replace("__LST_LEGEND_URL__", lst_legend_url)
    map_html = map_html.replace("__SUHI_LEGEND_URL__", suhi_legend_url)
    map_html = map_html.replace("__LST_VMIN__", json.dumps(_fmt(lst_vmin)))
    map_html = map_html.replace("__LST_VMAX__", json.dumps(_fmt(lst_vmax)))
    map_html = map_html.replace("__SUHI_VMIN__", json.dumps(_fmt(suhi_vmin)))
    map_html = map_html.replace("__SUHI_VMAX__", json.dumps(_fmt(suhi_vmax)))

    map_html = map_html.replace("__OVERLAYS_PAYLOAD__", json.dumps(overlays_payload))

    layer_state = (
        f"lst{int(show_lst)}_suhi{int(show_suhi)}_"
        f"ndvi{int(show_ndvi)}_ndbi{int(show_ndbi)}_ndwi{int(show_ndwi)}_alb{int(show_albedo)}_lcz{int(show_lcz)}_"
        f"bld{int(show_buildings)}"
    )
    map_key = f"{city_choice}|{year_choice}|{layer_state}"
    map_html = map_html.replace("__MAP_KEY__", json.dumps(map_key))

    components.html(map_html, height=650, scrolling=False)


# ---------------------------
# Bottom charts row (LST dist | SUHI dist | SUHI by LCZ)
# ---------------------------
b0, b1, b2, b3 = st.columns([0.18, 0.27, 0.27, 0.28], gap="small")

chart_cities = ["Wuppertal", "Essen", "Soest"] if city_choice == "All cities" else [city_choice]
lcz_path = get_lcz_tif()

lst_vals_all = []
suhi_vals_all = []
suhi_paths_for_lcz = []

for c in chart_cities:
    lst_p = get_layer_tif(year_choice, "LST", c)
    suhi_p = get_layer_tif(year_choice, "SUHI", c)

    if lst_p:
        lst_vals_all.append(read_raster_values(str(lst_p)))
    if suhi_p:
        suhi_vals_all.append(read_raster_values(str(suhi_p)))
        suhi_paths_for_lcz.append(suhi_p)

lst_vals = np.concatenate(lst_vals_all) if lst_vals_all else np.array([], dtype=np.float32)
suhi_vals = np.concatenate(suhi_vals_all) if suhi_vals_all else np.array([], dtype=np.float32)

with b1:
    st.markdown("<div class='chart-card'><div class='chart-title'>LST distribution</div></div>", unsafe_allow_html=True)
    if lst_vals.size:
        title = f"LST distribution — {city_choice} ({year_choice})"
        st.plotly_chart(make_histogram(lst_vals, title, "LST (°C)"), use_container_width=True)
    else:
        st.info("LST GeoTIFF not found for this selection (check your final_outputs filenames).")

with b2:
    st.markdown("<div class='chart-card'><div class='chart-title'>SUHI distribution</div></div>", unsafe_allow_html=True)
    if suhi_vals.size:
        title = f"SUHI distribution — {city_choice} ({year_choice})"
        st.plotly_chart(make_histogram(suhi_vals, title, "SUHI (°C)"), use_container_width=True)
    else:
        st.info("SUHI GeoTIFF not found for this selection (check your final_outputs filenames).")

with b3:
    st.markdown("<div class='chart-card'><div class='chart-title'>SUHI by LCZ</div></div>", unsafe_allow_html=True)

    if (not lcz_path) or (not suhi_paths_for_lcz):
        st.info("Need LCZ.tif + SUHI GeoTIFF for this selection.")
    else:
        suhi_samps = []
        lcz_samps = []
        for sp in suhi_paths_for_lcz:
            sv, lv = load_suhi_lcz_samples(str(sp), str(lcz_path))
            if sv.size and lv.size:
                suhi_samps.append(sv)
                lcz_samps.append(lv)

        if suhi_samps and lcz_samps:
            sv = np.concatenate(suhi_samps)
            lv = np.concatenate(lcz_samps)
            title = f"SUHI by LCZ — {city_choice} ({year_choice})"
            st.plotly_chart(make_suhi_by_lcz_box(sv, lv, title), use_container_width=True)
        else:
            st.info("Could not align SUHI with LCZ (check CRS/extent of LCZ.tif).")


# ---------------------------
# Footer
# ---------------------------
st.markdown(
    "<div class='custom-footer'>Designed by <b>Saurabh Bhagchandani</b> — <i>Faculty ITC, Master's Geo-Information Science and Earth Observation</i></div>",
    unsafe_allow_html=True,
)
