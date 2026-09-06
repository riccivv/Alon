"""Generate before/after diagrams of the preprocessing pipeline.

Produces figures saved under docs/diagrams/ showing a raw satellite granule
(with cloud gaps visible as NaN) at each processing stage:

    stage 0: raw granule subset (cloud gaps / coastlines still raw)
    stage 1: after temporal gap-fill
    stage 2: after regridding onto the study grid
    stage 3: derived indices (frontal, productivity) -- when SST/CHL available

Usage:
    python scripts/visualize_process.py                          # all datasets
    python scripts/visualize_process.py --dataset chlorophyll --granule <filepath>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PHILIPPINES_BBOX, RAW_DATA_DIR  # noqa: E402
from src.data.bathymetry import fetch_bathymetry  # noqa: E402
from src.processing.grid import create_grid  # noqa: E402
from src.processing.indices import add_indices, frontal_index, productivity_index  # noqa: E402
from src.processing.pipeline import VARIABLE_MAP  # noqa: E402
from src.processing.spatial import regrid_to_grid, standardize_axes  # noqa: E402
from src.processing.temporal import gap_fill_timeseries  # noqa: E402
from src.utils.geo import select_bbox  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("viz")

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def pick_granule(dataset_key: str) -> Path:
    """Find a suitable single granule for visualization."""
    if dataset_key not in VARIABLE_MAP:
        raise KeyError(f"Unknown dataset {dataset_key!r}")
    files = sorted(RAW_DATA_DIR.glob(f"{dataset_key}/**/*.nc"))
    if not files:
        raise FileNotFoundError(
            f"No granules for {dataset_key} in {RAW_DATA_DIR}. Fetch data first."
        )
    return files[len(files) // 2]  # middle of the record


def load_granule_series(dataset_key: str, n: int = 5) -> tuple[xr.DataArray, Path]:
    """Load ``n`` consecutive granules as a time series (time from filename).

    Returns (series DataArray with dims (time, lat, lon), the middle granule).
    """
    from src.processing.pipeline import granule_time

    src_var = VARIABLE_MAP[dataset_key]["src"]
    files = sorted(RAW_DATA_DIR.glob(f"{dataset_key}/**/*.nc"))
    if not files:
        raise FileNotFoundError(
            f"No granules for {dataset_key} in {RAW_DATA_DIR}. Fetch data first."
        )
    window = files[max(0, len(files) // 2 - n // 2): len(files) // 2 + n // 2 + 1]

    das = []
    for f in window:
        with xr.open_dataset(f, decode_times=True) as ds:
            sub = _subset_bbox(ds, src_var)
            da = standardize_axes(sub[src_var])
        da = da.assign_coords(time=granule_time(f))
        das.append(da.expand_dims("time"))
    series = xr.concat(das, dim="time").sortby("time")
    middle = window[len(window) // 2]
    return series, middle


def load_single_granule(path: Path, dataset_key: str) -> xr.DataArray:
    """Load one granule, subset to the study bbox, return the variable."""
    src_var = VARIABLE_MAP[dataset_key]["src"]
    with xr.open_dataset(path, decode_times=True) as ds:
        sub = _subset_bbox(ds, src_var)
        return standardize_axes(sub[src_var])


def _subset_bbox(ds: xr.Dataset, var: str) -> xr.Dataset:
    return select_bbox(ds, PHILIPPINES_BBOX, pad=0.6)[var].to_dataset(name=var)


def plot_stage(ax, da: xr.DataArray, title: str, cmap: str, vmin=None, vmax=None) -> None:
    da = da.transpose("lat", "lon")
    img = ax.imshow(
        np.flipud(da.values),
        extent=[da["lon"].min(), da["lon"].max(), da["lat"].min(), da["lat"].max()],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)


def visualize_dataset(dataset_key: str, granule: Path | None = None) -> Path:
    """Produce the before/after figure for one dataset."""
    grad_guide = {
        "chlorophyll": ("chl (mg/m^3)", "YlGn", "log"),
        "sst": ("SST (degC)", "RdYlBu_r", "linear"),
        "par": ("PAR (E/m^2/day)", "viridis", "linear"),
        "kd490": ("Kd490 (1/m)", "magma", "linear"),
    }
    raw_label, cmap, scale = grad_guide[dataset_key]
    grid = create_grid()

    if granule is None:
        series, granule = load_granule_series(dataset_key, n=5)
        raw = series.isel(time=len(series) // 2)  # the middle time slice
    else:
        series = load_single_granule(granule, dataset_key).expand_dims("time")
        raw = load_single_granule(granule, dataset_key)
    logger.info("Visualizing %s using %s", dataset_key, granule.name)

    # Stage 1: gap fill along time
    filled_series = gap_fill_timeseries(series, method="linear", max_gap_steps=4)
    idx = (len(series) - 1) // 2 if granule is None else 0
    gapfilled = filled_series.isel(time=idx)

    # Stage 2: regrid onto the study grid
    regridded = regrid_to_grid(gapfilled, grid, method="bilinear")

    # A stable v-range across the raw/filled/regridded stages.
    finite = raw.values[np.isfinite(raw.values)]
    if scale == "log":
        finite = finite[finite > 0]
    vmin, vmax = np.nanpercentile(finite, 5), np.nanpercentile(finite, 95)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Data Through the Pipeline - {raw_label}\n{granule.name}\n"
        f"(time window {pd.to_datetime(series.time.values[0]).date()} -> "
        f"{pd.to_datetime(series.time.values[-1]).date()})",
        fontsize=12,
        fontweight="bold",
    )

    plot_stage(axes[0, 0], raw, "Stage 0: Raw granule (cloud gaps = blank)", cmap, vmin, vmax)

    fill_finite = gapfilled.values[np.isfinite(gapfilled.values)]
    v2min, v2max = np.nanpercentile(fill_finite, 2), np.nanpercentile(fill_finite, 98)
    nfilled = int(np.isfinite(gapfilled.values).sum() - np.isfinite(raw.values).sum())
    plot_stage(
        axes[0, 1], gapfilled,
        f"Stage 1: After cloud gap-fill\n(+{nfilled} cells recovered in this slice)",
        cmap, v2min, v2max,
    )

    reg_finite = regridded.values[np.isfinite(regridded.values)]
    v3min, v3max = np.nanpercentile(reg_finite, 2), np.nanpercentile(reg_finite, 98)
    plot_stage(axes[1, 0], regridded, "Stage 2: Regridded to 0.1 deg study grid", cmap, v3min, v3max)

    # Stage 3: derived index (if applicable)
    if dataset_key == "sst":
        index = frontal_index(gapfilled).squeeze()
        plot_stage(axes[1, 1], index, "Stage 3: Frontal index (SST gradient)", "cividis")
    elif dataset_key == "chlorophyll":
        index = productivity_index(gapfilled).squeeze()
        plot_stage(axes[1, 1], index, "Stage 3: Productivity index (log chl)", "viridis")
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5, 0.5,
            "Derived indices use SST/CHL\n(diagram generated for those datasets)",
            ha="center", va="center", fontsize=11, wrap=True,
        )

    # Overlay coastline from bathymetry for context.
    try:
        depth = fetch_bathymetry(PHILIPPINES_BBOX, use_cached=True)["depth"]
        depth = regrid_to_grid(depth, grid, method="bilinear")
        land = ~(depth < 0)
        for ax in axes.flat:
            ax.contour(depth["lon"], depth["lat"], np.flipud(land.values),
                       levels=[0.5], colors="0.35", linewidths=0.6)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not overlay coastline: %s", exc)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / f"stage_before_after_{dataset_key}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate before/after pipeline diagrams")
    parser.add_argument("--dataset", choices=list(VARIABLE_MAP), default=None)
    parser.add_argument("--granule", type=Path, default=None)
    args = parser.parse_args()

    keys = [args.dataset] if args.dataset else ["sst", "chlorophyll", "par", "kd490"]
    for key in keys:
        try:
            visualize_dataset(key, args.granule)
        except Exception as exc:  # noqa: BLE001
            logger.error("Skipping {%s}: %s", key, exc)


if __name__ == "__main__":
    main()