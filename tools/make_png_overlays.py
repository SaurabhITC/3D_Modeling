import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_bounds_json(out_json: Path, bounds_4326):
    minx, miny, maxx, maxy = bounds_4326
    out_json.write_text(json.dumps({
        "minLon": float(minx),
        "minLat": float(miny),
        "maxLon": float(maxx),
        "maxLat": float(maxy),
    }, indent=2))


def save_png_rgba(out_png: Path, arr, nodata_mask, cmap_name, vmin=None, vmax=None):
    valid = arr[~nodata_mask]
    if valid.size == 0:
        raise RuntimeError(f"No valid pixels for {out_png.name}")

    # robust auto stretch if not provided
    if vmin is None or vmax is None:
        vmin = np.nanpercentile(valid, 2)
        vmax = np.nanpercentile(valid, 98)
        if np.isclose(vmin, vmax):
            vmin, vmax = float(valid.min()), float(valid.max())

    x = np.clip(arr, vmin, vmax)
    x = (x - vmin) / (vmax - vmin + 1e-12)

    cmap = plt.get_cmap(cmap_name)
    rgba = (cmap(x) * 255).astype(np.uint8)  # HxWx4
    rgba[..., 3] = np.where(nodata_mask, 0, 255).astype(np.uint8)

    import imageio.v2 as imageio
    imageio.imwrite(out_png.as_posix(), rgba)


def process_tif(tif_path: Path, out_dir: Path):
    name = tif_path.stem.upper()
    layer_type = "SUHI" if "SUHI" in name else ("LST" if "LST" in name else None)
    if layer_type is None:
        print(f"↪️ Skipping: {tif_path.name}")
        return

    with rasterio.open(tif_path) as ds:
        arr = ds.read(1).astype(np.float32)
        nodata = ds.nodata
        if nodata is None:
            nodata_mask = ~np.isfinite(arr)
        else:
            nodata_mask = (~np.isfinite(arr)) | (arr == nodata)

        b4326 = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)

    ensure_dir(out_dir)

    out_png = out_dir / f"{tif_path.stem}.png"
    out_json = out_dir / f"{tif_path.stem}_bounds.json"

    if layer_type == "SUHI":
        # diverging, symmetric around 0
        valid = arr[~nodata_mask]
        lim = np.nanpercentile(np.abs(valid), 98) if valid.size else 1.0
        save_png_rgba(out_png, arr, nodata_mask, cmap_name="coolwarm", vmin=-lim, vmax=lim)
    else:
        save_png_rgba(out_png, arr, nodata_mask, cmap_name="turbo")

    write_bounds_json(out_json, b4326)
    print(f"✅ {tif_path.name} -> {out_png.name} + {out_json.name}")


def main():
    base = Path("data")
    inputs = {
        "2022": base / "2022_final_outputs",
        "2024": base / "2024_final_outputs",
    }
    out_base = base / "overlays_png"

    for year, in_dir in inputs.items():
        if not in_dir.exists():
            print(f"⚠️ Missing folder: {in_dir}")
            continue

        out_dir = out_base / year
        for tif in sorted(in_dir.glob("*.tif")):
            process_tif(tif, out_dir)


if __name__ == "__main__":
    main()
