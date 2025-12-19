import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
import matplotlib.cm as cm
from PIL import Image


def _read_band1(tif_path: Path, max_size: int | None):
    """
    Read band 1. If max_size is set, downsample so the longest side <= max_size
    (helps performance on Streamlit Cloud).
    """
    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width

        if max_size and max(h, w) > max_size:
            scale = max_size / float(max(h, w))
            out_h = max(1, int(h * scale))
            out_w = max(1, int(w * scale))

            arr = src.read(
                1,
                out_shape=(out_h, out_w),
                resampling=Resampling.bilinear,
            ).astype("float32")
        else:
            scale = 1.0
            arr = src.read(1).astype("float32")

        nodata = src.nodata
        bounds = src.bounds
        crs = src.crs

    return arr, nodata, bounds, crs, (h, w), scale


def tif_to_png_and_bounds(
    tif_path: Path,
    out_png: Path,
    out_json: Path,
    cmap_name: str = "inferno",
    p_low: float = 2.0,
    p_high: float = 98.0,
    max_size: int | None = 2048,
):
    arr, nodata, bounds, crs, (orig_h, orig_w), scale = _read_band1(tif_path, max_size)

    # Mask nodata/invalid
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    arr = np.where(np.isfinite(arr), arr, np.nan)

    # Robust stretch
    vmin = float(np.nanpercentile(arr, p_low))
    vmax = float(np.nanpercentile(arr, p_high))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    # Normalize 0..1
    norm = (arr - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)
    norm = np.where(np.isfinite(arr), norm, 0)

    # Colormap -> RGBA
    cmap = cm.get_cmap(cmap_name)
    rgba = (cmap(norm) * 255).astype(np.uint8)

    # Transparent where NaN
    alpha = np.where(np.isfinite(arr), 255, 0).astype(np.uint8)
    rgba[..., 3] = alpha

    # Save PNG
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(out_png, format="PNG", optimize=True)

    # Bounds to EPSG:4326 (lon/lat)
    if crs and crs.to_string() != "EPSG:4326":
        west, south, east, north = transform_bounds(
            crs, "EPSG:4326",
            bounds.left, bounds.bottom, bounds.right, bounds.top,
            densify_pts=21,
        )
    else:
        west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top

    payload = {
        "tif": str(tif_path.as_posix()),
        "png": str(out_png.as_posix()),
        "bounds_wgs84": {"west": west, "south": south, "east": east, "north": north},
        "stretch_percentiles": {"low": p_low, "high": p_high},
        "colormap": cmap_name,
        "value_range_used": {"vmin": vmin, "vmax": vmax},
        "downsample": {
            "max_size": max_size,
            "orig_size": [orig_w, orig_h],
            "scale_applied": scale,
            "output_size": [int(rgba.shape[1]), int(rgba.shape[0])],
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    print(f"✅ PNG   : {out_png}")
    print(f"✅ BOUNDS: {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--p_low", type=float, default=2.0)
    ap.add_argument("--p_high", type=float, default=98.0)
    ap.add_argument("--max_size", type=int, default=2048)  # set 0 to disable downsampling
    args = ap.parse_args()

    max_size = None if args.max_size == 0 else args.max_size

    tif_to_png_and_bounds(
        Path(args.tif),
        Path(args.out_png),
        Path(args.out_json),
        cmap_name=args.cmap,
        p_low=args.p_low,
        p_high=args.p_high,
        max_size=max_size,
    )


if __name__ == "__main__":
    main()
