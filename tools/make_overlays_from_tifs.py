import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
import matplotlib
from PIL import Image


def make_overlay(tif_path: Path, out_png: Path, out_json: Path, cmap: str, qmin=2, qmax=98):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata

        mask = ~np.isfinite(arr)
        if nodata is not None:
            mask |= (arr == nodata)

        valid = arr[~mask]
        if valid.size == 0:
            raise ValueError(f"No valid pixels in {tif_path}")

        vmin = float(np.percentile(valid, qmin))
        vmax = float(np.percentile(valid, qmax))
        if np.isclose(vmin, vmax):
            vmin = float(valid.min())
            vmax = float(valid.max())

        norm = (arr - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0, 1)
        norm[mask] = 0

        cmap_obj = matplotlib.colormaps.get_cmap(cmap)
        rgba = (cmap_obj(norm) * 255).astype(np.uint8)
        rgba[..., 3] = np.where(mask, 0, 255).astype(np.uint8)

        Image.fromarray(rgba, mode="RGBA").save(out_png, optimize=True)

        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        meta = {
            "bounds_wgs84": {"west": west, "south": south, "east": east, "north": north},
            "colormap": cmap,
            "value_range_used": {"vmin": vmin, "vmax": vmax},
            "source_tif": str(tif_path),
        }
        out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"✅ {out_png}")
    print(f"✅ {out_json}")


def build_city_year(city: str, year: str, lst_tif: Path, suhi_tif: Path, out_root: Path):
    out_dir = out_root / year / city
    make_overlay(lst_tif, out_dir / "summer_lst.png", out_dir / "summer_lst_bounds.json", cmap="inferno")
    make_overlay(suhi_tif, out_dir / "summer_suhi.png", out_dir / "summer_suhi_bounds.json", cmap="coolwarm")


if __name__ == "__main__":
    # EDIT ONLY THESE PATHS (your TIFFs)
    year = "2024"

    essen_dir = Path(r"C:\University of Twente (Master's)\2nd year\thesis mateerial\Data\ECOSTRESS_LST_DATA\2024\ESSEN\outputs\final_outputs\rasters")
    wupp_dir  = Path(r"C:\University of Twente (Master's)\2nd year\thesis mateerial\Data\ECOSTRESS_LST_DATA\2024\WUPPERTAL\outputs\final_outputs\rasters")

    essen_lst  = essen_dir / "summer_LST_median_QC_25832_C.tif"
    essen_suhi = essen_dir / "summer_SUHI_median_QC_masked_n3_ETRS25832_C.tif"

    wupp_lst   = wupp_dir / "summer_LST_median_QC_25832_C.tif"
    wupp_suhi  = wupp_dir / "summer_SUHI_median_QC_masked_n3_ETRS25832_C.tif"

    # This is your dashboard data folder
    out_root = Path("data")

    build_city_year("Essen", year, essen_lst, essen_suhi, out_root)
    build_city_year("Wuppertal", year, wupp_lst, wupp_suhi, out_root)
