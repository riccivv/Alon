"""End-to-end preprocessing pipeline: raw NetCDF granules -> feature cube.

Stages (each stage is diagrammed in docs/diagrams/):
    RAW granules
        -> subset to Philippine bbox & regrid onto the study grid (spatial.py)
        -> gap fill cloud-covered pixels along the time axis (temporal.py)
        -> rolling means, climatology anomalies, seasonal encoding (temporal.py)
        -> derived indices: frontal, productivity, upwelling (indices.py)
        -> spatial neighborhood lags (spatial.py)
        -> long-format feature matrix (rows = cell x time)
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.config import (
    FEATURE_CONFIG,
    MODELS_DIR,
    PHILIPPINES_BBOX,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SATELLITE_DATASETS,
)
from src.data.bathymetry import fetch_bathymetry
from src.utils.geo import select_bbox
from src.processing.grid import Grid, create_grid, land_mask_from_depth
from src.processing.indices import add_indices
from src.processing.spatial import compute_spatial_lags, distance_to_coast, regrid_to_grid
from src.processing.temporal import apply_temporal_features, gap_fill_timeseries

logger = logging.getLogger(__name__)

CUBE_PATH = PROCESSED_DATA_DIR / "ph_cube.zarr"
FEATURES_PATH = PROCESSED_DATA_DIR / "feature_matrix.pkl"

# Map dataset key -> canonical feature name and the source NetCDF variable.
VARIABLE_MAP = {
    "chlorophyll": {"src": "chlor_a", "feature": "chl"},
    "sst": {"src": "sst", "feature": "sst"},
    "par": {"src": "par", "feature": "par"},
    "kd490": {"src": "Kd_490", "feature": "kd490"},
}


# --------------------------------------------------------------------------- #
# Per-file preprocessing (called inside open_mfdataset)
# --------------------------------------------------------------------------- #
def _subset_preprocess(ds: xr.Dataset, src_var: str, var: str) -> xr.Dataset:
    """Subset a single L3 granule to the study region and keep only the target."""
    rename = {}
    for name in ("latitude", "lat", "y"):
        if name in ds.dims and name not in rename:
            rename[name] = "lat"
    for name in ("longitude", "lon", "x"):
        if name in ds.dims and name not in rename:
            rename[name] = "lon"
    if rename:
        ds = ds.rename(rename)

    ds = select_bbox(ds, PHILIPPINES_BBOX, pad=0.8)
    if src_var in ds.variables:
        ds = ds.get(src_var).to_dataset(name=var)
    else:
        raise ValueError(f"Variable {src_var!r} not found in granule: {list(ds.variables)}")
    if "time" in ds.dims and "lat" in ds.dims:
        ds = ds.transpose("time", "lat", "lon")
    return ds


def _granule_year(path: Path) -> int | None:
    """Guess the year from the filename or file modification time."""
    import re

    m = re.search(r"\.(\d{4})\d{2}", path.name)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{4})", path.name)
    if m:
        return int(m.group(1))
    return None


def granule_time(path: Path) -> pd.Timestamp:
    """Derive a timestamp for a granule from its filename.

    OB.DAAC L3 mapped granules store time in the file name, e.g.
    ``AQUA_MODIS.20180525_20180601.L3m.8D.CHL.chlor_a.9km.nc``. We use the
    midpoint of the composite window so that 8-day bins align centrally.
    """
    import re

    m = re.search(r"\d{4}\.\d{8}_\d{8}\.", path.name) or re.search(r"_MODIS.(\d{8})_(\d{8}).", path.name)
    if m and m.lastindex == 2:
        start = pd.Timestamp(m.group(1))
        end = pd.Timestamp(m.group(2))
        return start + (end - start) / 2
    m2 = re.search(r"(\d{8})", path.name)
    if m2:
        return pd.Timestamp(m2.group(1))
    return pd.Timestamp(path.stat().st_mtime, unit="s")


# --------------------------------------------------------------------------- #
# Per-variable cube building
# --------------------------------------------------------------------------- #
def build_variable_cube(
    dataset_key: str,
    grid: Grid,
    years: list[int] | None = None,
) -> xr.DataArray:
    """Combine, subset, regrid, and gap-fill one satellite variable."""
    if dataset_key not in VARIABLE_MAP:
        raise KeyError(f"Unknown variable key {dataset_key!r}")
    cfg = VARIABLE_MAP[dataset_key]
    src_var = cfg["src"]
    feature_name = cfg["feature"]

    files = sorted(RAW_DATA_DIR.glob(f"{dataset_key}/**/*.nc"))
    if not files:
        raise FileNotFoundError(
            f"No granules found under data/raw/{dataset_key}/. Run the fetch "
            "script first (scripts/fetch_satellite.py)."
        )
    if years:
        files = [f for f in files if _granule_year(f) in years]

    logger.info("Combining %s granules for %s", len(files), dataset_key)

    das = []
    for f in files:
        try:
            with xr.open_dataset(f, decode_times=True) as ds_open:
                sub = _subset_preprocess(ds_open, src_var, feature_name)
                da = sub[feature_name]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s: %s", f.name, exc)
            continue
        da = da.assign_coords(time=granule_time(f))
        das.append(da.expand_dims("time"))

    if not das:
        raise RuntimeError(
            f"No usable granules processed for {dataset_key}. Check raw files in "
            f"{RAW_DATA_DIR / dataset_key}."
        )

    combined = xr.concat(das, dim="time", join="outer").sortby("time")
    da = regrid_to_grid(combined, grid, method="bilinear")
    da = gap_fill_timeseries(da, method="linear", max_gap_steps=4)
    da = da.rename(feature_name)
    da.attrs["units"] = SATELLITE_DATASETS[dataset_key]["units"]
    da.attrs["feature"] = feature_name
    return da


# --------------------------------------------------------------------------- #
# Full feature cube
# --------------------------------------------------------------------------- #
def build_cube(
    grid: Grid,
    variables: list[str] | None = None,
    years: list[int] | None = None,
    include_bathymetry: bool = True,
    include_spatial_lags: bool = True,
    cache: bool = True,
) -> xr.Dataset:
    """Assemble all variables into one feature cube on the study grid."""
    variables = variables or list(VARIABLE_MAP.keys())

    parts: dict[str, xr.DataArray] = {}
    for key in variables:
        feature = VARIABLE_MAP[key]["feature"]
        parts[feature] = build_variable_cube(key, grid, years=years)

    cube = xr.Dataset(parts)

    # Static spatial fields.
    if include_bathymetry:
        depth = fetch_bathymetry(PHILIPPINES_BBOX)["depth"]
        depth = regrid_to_grid(depth, grid, method="bilinear").rename("depth")
        cube["depth"] = depth
        cube["slope"] = regrid_to_grid(
            fetch_bathymetry(PHILIPPINES_BBOX)["slope"], grid, method="bilinear"
        ).rename("slope")
        cube["dist_coast"] = distance_to_coast(grid, depth)

    # Mask land everywhere.
    land = land_mask_from_depth(cube["depth"])
    for name in list(cube.data_vars):
        if name in ("depth", "slope", "dist_coast"):
            continue
        cube[name] = cube[name].where(~land)

    # Temporal features (rolling means, anomalies, seasonal encoding).
    cube = apply_temporal_features(cube, FEATURE_CONFIG)

    # Derived indices.
    cube = add_indices(cube)

    # Spatial lags for the raw environmental fields.
    if include_spatial_lags:
        lag_vars = [v for v in ("sst", "chl", "frontal_index") if v in cube.data_vars]
        lags = compute_spatial_lags(
            cube[lag_vars], grid, FEATURE_CONFIG.get("spatial_lag_radii_km")
        )
        for name in lags.data_vars:
            cube[name] = lags[name]

    cube.attrs.update(
        {
            "description": "OceanForecast PH feature cube",
            "grid_res": grid.res,
            "bbox": str(PHILIPPINES_BBOX),
            "created": dt.datetime.now(dt.timezone.utc).isoformat(),
            "years": str(years or "all"),
        }
    )
    logger.info("Feature cube assembled: %s", cube)
    if cache:
        save_cube(cube)
    return cube


# --------------------------------------------------------------------------- #
# Long-format feature matrix
# --------------------------------------------------------------------------- #
def to_long_dataframe(cube: xr.Dataset) -> pd.DataFrame:
    """Flatten the cube into (lat, lon, time, feature...) rows."""
    df = cube.to_dataframe().reset_index()
    df["lat"] = df["lat"].round(6)
    df["lon"] = df["lon"].round(6)
    df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")
    return df


def save_features(df: pd.DataFrame, path: Path | str = FEATURES_PATH) -> None:
    """Persist the long-format matrix (parquet if pyarrow available, else pickle)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(str(path.with_suffix(".parquet")))
        logger.info("Saved feature matrix to %s", path.with_suffix(".parquet"))
    except Exception:  # noqa: BLE001
        df.to_pickle(path)
        logger.info("Saved feature matrix to %s (pickle)", path)


# --------------------------------------------------------------------------- #
# Cube persistence
# --------------------------------------------------------------------------- #
def save_cube(cube: xr.Dataset, path: Path | str = CUBE_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cube.to_zarr(path, mode="w")
        logger.info("Saved feature cube to %s", path)
    except Exception:  # noqa: BLE001
        cube.to_netcdf(path.with_suffix(".nc"))
        logger.info("Saved feature cube (netcdf) to %s", path.with_suffix(".nc"))


def load_cube(path: Path | str = CUBE_PATH) -> xr.Dataset:
    path = Path(path)
    if path.exists() and (path / ".zgroup").exists():
        return xr.open_zarr(path)
    nc = path.with_suffix(".nc")
    if nc.exists():
        return xr.open_dataset(nc)
    raise FileNotFoundError(f"No cached cube found at {path} (or .nc). Build it first.")


# --------------------------------------------------------------------------- #
# Convenience runner
# --------------------------------------------------------------------------- #
def run_pipeline(
    years: list[int] | None = None,
    variables: list[str] | None = None,
    grid: Grid | None = None,
    cache: bool = True,
    include_spatial_lags: bool = True,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """Run the full preprocessing chain: raw files -> feature DataFrame."""
    grid = grid or create_grid()
    cube = build_cube(
        grid,
        variables=variables,
        years=years,
        cache=cache,
        include_spatial_lags=include_spatial_lags,
    )
    df = to_long_dataframe(cube)
    save_features(df)
    return cube, df