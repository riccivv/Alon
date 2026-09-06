"""Philippine waters overview map: chlorophyll-a with coastline + EEZ boundary.

Produces ``docs/diagrams/eez_chl_map.png`` -- a clean, single-frame
presentation map. By default it plots the **long-term annual mean** of the
gap-filled MODIS-Aqua chlorophyll cube over the whole study region, so the 
slide shows the persistent productivity pattern across the Philippine EEZ.

The Philippine EEZ polygon is pulled once from the Marine Regions *World EEZ
v12 (low resolution)* archive (CC-BY-4.0, VLIZ/Flanders Marine Institute),
subset to Philippine polygons and cached under
``data/raw/boundaries/ph_eez.geojson``. If the download fails, the script
falls back to a coastline-only map (bathymetry contour) so it never breaks.

Usage:
    python scripts/map_chl.py                 # annual-mean overview
    python scripts/map_chl.py --year 2022     # single-year mean
    python scripts/map_chl.py --time 2023-01-01  # one 8-day composite window
    python scripts/map_chl.py --no-eez        # coastline only (offline)
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PHILIPPINES_BBOX  # noqa: E402
from src.processing.pipeline import CUBE_PATH, load_cube  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("mapchl")

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOUNDARY_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "boundaries"
BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)
EEZ_CACHE = BOUNDARY_DIR / "ph_eez.geojson"

DEFAULT_EEZ_URL = (
    "https://zenodo.org/records/16314546/files/World_EEZ_v12_20231025_LR.zip?download=1"
)


def _download_eez(url: str, dest: Path) -> Path:
    """Download the World EEZ archive to ``dest`` (zip) with a range check."""
    import requests

    logger.info("Downloading World EEZ (low res) archive (~27 MB)...")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    size_mb = dest.stat().st_size / (1 << 20)
    if size_mb < 1:
        raise RuntimeError(f"Suspiciously small EEZ download ({size_mb:.1f} MB).")
    logger.info("Downloaded %s (%.1f MB)", dest.name, size_mb)
    return dest


def _first_vector_member(zip_path: Path, tmp: Path) -> Path | None:
    """Extract shapefile/geojson members so geopandas can read them."""
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        candidates = [m for m in members if m.lower().endswith((".shp", ".geojson", ".json", ".gml", ".gpkg"))]
        if not candidates:
            logger.warning("No vector member inside %s (%s)", zip_path.name, members[:10])
            return None
        name = candidates[0]
        out = tmp / Path(name).name
        with zf.open(name) as src, out.open("wb") as dst:
            dst.write(src.read())
        return out


def _philippine_eez(url: str, cache: Path) -> object:
    """Return a geopandas GeoDataFrame of Philippine EEZ polygons."""
    if cache.exists():
        logger.info("Using cached EEZ polygons from %s", cache)
        return _read_geodataframe(cache)

    try:
        import geopandas as gpd
    except ImportError:
        logger.warning("geopandas not installed; falling back to coastline-only map.")
        return None

    tmp = Path(tempfile.mkdtemp(prefix="eez_"))
    zip_path = tmp / "world_eez.zip"
    try:
        _download_eez(url, zip_path)
        member = _first_vector_member(zip_path, tmp)
        if member is None:
            raise RuntimeError("No readable vector file inside EEZ archive.")
        world = gpd.read_file(member)
        name_col = next((c for c in world.columns if c.lower() in ("sovereign", "name", "geoname")), None)
        if name_col is None:
            raise RuntimeError(f"Unknown EEZ attribute schema: {list(world.columns)}")
        ph = world[world[name_col].str.upper().str.contains("PHILIPPINE", na=False)]
        if ph.empty:
            raise RuntimeError(f"No Philippine polygons found in {name_col}: {set(world[name_col])}")
        ph = ph.set_crs("EPSG:4326", allow_override=True).dissolve()
        ph.to_file(cache, driver="GeoJSON")
        logger.info("Cached %d Philippine EEZ polygons to %s", len(ph), cache)
        return ph
    except Exception as exc:  # noqa: BLE001
        logger.warning("EEZ fetch failed (%s); coastline-only map.", exc)
        return None
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


def _read_geodataframe(path: Path):
    import geopandas as gpd

    return gpd.read_file(path)


def _pick_chl(cube: xr.Dataset, year: int | None, time: str | None) -> xr.DataArray:
    da = cube["chl"]
    if time is not None:
        picked = da.sel(time=str(time), method="nearest")
        logger.info("Using single 8-day window: %s", str(picked["time"].values[()])[:10])
        return picked
    if year is not None:
        picked = da.sel(time=str(year))
        logger.info("Averaging %d windows of %s", picked.sizes["time"], year)
        return picked.mean("time")
    logger.info("Averaging %d windows into an annual mean (2015-2025).", da.sizes["time"])
    return da.mean("time")


def _depth_grid(cube: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lat, lon, depth) on the study grid, from the cube or cached bathymetry."""
    if "depth" in cube.data_vars:
        d = cube["depth"]
        if "time" in d.dims:
            d = d.isel(time=0)
        return d["lat"].values, d["lon"].values, np.asarray(d.values, dtype="float64")
    from src.data.bathymetry import fetch_bathymetry
    from src.processing.grid import create_grid
    from src.processing.spatial import regrid_to_grid

    depth = regrid_to_grid(
        fetch_bathymetry(PHILIPPINES_BBOX, use_cached=True)["depth"],
        create_grid(),
        method="bilinear",
    )
    return depth["lat"].values, depth["lon"].values, np.asarray(depth.values, dtype="float64")


def _land_mask(depth: np.ndarray) -> np.ndarray:
    return np.isnan(depth) | ~(depth < 0.0)


def _plot_eez(ax, eez, in_bbox) -> None:
    if eez is None:
        return
    for poly in eez.geometry.iloc[0].geoms if hasattr(eez.geometry.iloc[0], "geoms") else [eez.geometry.iloc[0]]:
        if poly is None or poly.is_empty:
            continue
        xs, ys = poly.exterior.xy
        ax.plot(xs, ys, color="#222222", lw=1.0, ls="--", alpha=0.7)


def main() -> None:
    parser = argparse.ArgumentParser(description="PH EEZ + chlorophyll overview map")
    parser.add_argument("--year", type=int, default=None, help="Single year mean (default: full-period mean)")
    parser.add_argument("--time", default=None, help="One 8-day window, e.g. 2023-01-01")
    parser.add_argument("--no-eez", action="store_true", help="Skip EEZ polygon download")
    parser.add_argument("--eez-url", default=DEFAULT_EEZ_URL)
    parser.add_argument("--out", default=OUT_DIR / "eez_chl_map.png")
    args = parser.parse_args()

    cube = load_cube(CUBE_PATH)
    da = _pick_chl(cube, args.year, args.time)
    data = np.asarray(da.values, dtype="float64")

    depth_lat, depth_lon, depth = _depth_grid(cube)
    mask_land = _land_mask(depth)
    data = np.where(mask_land, np.nan, data)

    fig, ax = plt.subplots(figsize=(9.5, 8), dpi=150)
    lon = cube["lon"].values
    lat = cube["lat"].values
    lr = lon[1] - lon[0]
    tr = lat[1] - lat[0]
    extent = [lon[0] - lr / 2, lon[-1] + lr / 2, lat[0] - tr / 2, lat[-1] + tr / 2]

    if data.shape != mask_land.shape:
        raise RuntimeError(
            f"Grid mismatch: chl {data.shape} vs land mask {mask_land.shape}. "
            "Rebuild the cube (scripts/run_pipeline.py)."
        )

    finite = data[np.isfinite(data)]
    vmin, vmax = np.nanpercentile(finite, 2), np.nanpercentile(finite, 98)
    vmin = max(vmin, np.nextafter(0.0, 1.0))

    im = ax.imshow(
        data,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
        interpolation="nearest",
    )

    # Grey land silhouette (first), then coastline contour.
    land_plot = np.ma.masked_where(~mask_land, mask_land.astype(float))
    ax.imshow(land_plot, extent=extent, origin="lower", aspect="auto",
              cmap="Greys", vmin=0, vmax=1, alpha=0.85, zorder=2)
    ax.contour(depth_lon, depth_lat, land_plot, levels=[0.5],
               colors="0.25", linewidths=0.7, zorder=3)

    # Philippine EEZ boundary overlay.
    eez = None
    if not args.no_eez:
        eez = _philippine_eez(args.eez_url, EEZ_CACHE)
    _plot_eez(ax, eez, PHILIPPINES_BBOX)

    pad = 0.6
    ax.set_xlim(extent[0] - pad + lr, extent[1] + pad - lr)
    ax.set_ylim(extent[2] - pad + tr, extent[3] + pad - tr)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if args.time:
        import pandas as pd

        time_label = f"8-day window {str(pd.Timestamp(da['time'].values[()]))[:10]}"
    elif args.year:
        time_label = f"Annual mean {args.year}"
    else:
        time_label = "Annual mean 2015-2025"
    ax.set_title(
        f"Chlorophyll-a in Philippine waters\n{time_label}",
        fontsize=13,
        fontweight="bold",
    )

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Chlorophyll-a (mg/m$^3$)", fontsize=11)

    ax.text(
        0.02, 0.02,
        "Data: NASA MODIS-Aqua L3m (8-day, 9 km)\nEEZ: Marine Regions v12 (VLIZ)",
        va="bottom", ha="left", fontsize=8, color="0.3",
        transform=ax.transAxes,
    )

    out = Path(args.out)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


if __name__ == "__main__":
    main()