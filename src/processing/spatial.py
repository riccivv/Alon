"""Spatial operations: regridding and neighborhood (lag) features."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import ndimage
from scipy.spatial import cKDTree

from src.utils.geo import select_bbox, ensure_ascending
from .grid import Grid, land_mask_from_depth


# --------------------------------------------------------------------------- #
# Regridding
# --------------------------------------------------------------------------- #
def standardize_axes(da: xr.DataArray) -> xr.DataArray:
    """Rename lat/lon axes to a canonical (lat, lon) ordering."""
    rename = {}
    for name in ("latitude", "lat", "y"):
        if name in da.dims and name not in rename:
            rename[name] = "lat"
    for name in ("longitude", "lon", "x"):
        if name in da.dims and name not in rename:
            rename[name] = "lon"
    if rename:
        da = da.rename(rename)
    if "lat" in da.dims and "lon" in da.dims:
        da = da.transpose(..., "lat", "lon")
    return da


def regrid_to_grid(
    da: xr.DataArray,
    grid: Grid,
    method: str = "bilinear",
    pad: float = 0.6,
) -> xr.DataArray:
    """Interpolate a data array onto the uniform study grid.

    Optionally pads the selection window so edge cells do not go NaN.
    Requires ``lat``/``lon`` dimensions (see :func:`standardize_axes`).
    """
    da = standardize_axes(da)
    da = ensure_ascending(da, "lat").sortby("lat")
    da = ensure_ascending(da, "lon").sortby("lon")

    sel = {
        "lat": slice(grid.lat_min - pad, grid.lat_max + pad),
        "lon": slice(grid.lon_min - pad, grid.lon_max + pad),
    }
    try:
        da = da.sel(sel)
    except KeyError:
        da = da

    kw = {"fill_value": None} if method == "nearest" else {"fill_value": np.nan}
    xr_method = "linear" if method == "bilinear" else method
    return da.interp(lat=grid.lats, lon=grid.lons, method=xr_method, kwargs=kw)


# --------------------------------------------------------------------------- #
# Distance to coast
# --------------------------------------------------------------------------- #
def distance_to_coast(grid: Grid, depth: xr.DataArray) -> xr.DataArray:
    """Distance (km) from each ocean cell to the nearest land cell."""
    land = land_mask_from_depth(depth)
    lon2d, lat2d = grid.mesh()
    land_lon = lon2d[land.values]
    land_lat = lat2d[land.values]

    if land_lon.size == 0:
        raise ValueError("No land cells found in the grid to measure distance to.")

    # Equirectangular projection near the study latitude for fast geodesic-ish
    # distances. Errors are < 0.5% at Philippine latitudes for 100 km scales.
    scale_lat = np.cos(np.radians(np.clip(lat2d.mean(), -80, 80)))
    x_land = land_lon * scale_lat
    y_land = land_lat
    tree = cKDTree(np.column_stack([x_land, y_land]))

    x_grid = lon2d.ravel() * scale_lat
    y_grid = lat2d.ravel()
    _, dist_d = tree.query(np.column_stack([x_grid, y_grid]), k=1)

    km_per_deg = 111.32
    dist_km = dist_d.reshape(lon2d.shape) * km_per_deg
    return xr.DataArray(
        dist_km,
        coords={"lat": grid.lats, "lon": grid.lons},
        dims=("lat", "lon"),
        attrs={"units": "km", "long_name": "distance to nearest land"},
    )


# --------------------------------------------------------------------------- #
# Neighborhood (spatial lag) means
# --------------------------------------------------------------------------- #
def _disk_kernel(radius_cells: float, ndim: int = 2) -> np.ndarray:
    """Boolean disk kernel of the given radius (in cells).

    Returns a kernel with the same dimensionality as the data being blurred
    (2D for a (lat, lon) map, 3D with a singleton time axis otherwise). The
    disk lives in the last two axes.
    """
    r = int(np.ceil(radius_cells))
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    disk = (x**2 + y**2) <= radius_cells**2
    if ndim == 3:
        disk = disk[np.newaxis, ...]
    elif ndim != 2:
        raise ValueError(f"Unsupported ndim {ndim}; expected 2 or 3")
    return disk


def neighborhood_mean(
    data: np.ndarray, radius_cells: float, fill_nan_mask: bool = True
) -> np.ndarray:
    """NaN-aware mean of each cell's neighbors within a disk radius.

    The center cell is included. Cells are treated as symmetric so coastline
    cells use only their valid neighbors. Returns an array with the same shape
    as ``data`` (assumes last two axes are lat/lon).
    """
    kernel = _disk_kernel(radius_cells, data.ndim)

    mask = np.isfinite(data)
    data_f = np.nan_to_num(data, nan=0.0)

    norm = ndimage.convolve(mask.astype("float64"), kernel, mode="nearest")
    sm = ndimage.convolve(data_f, kernel, mode="nearest")

    out = np.where(norm > 0, sm / np.maximum(norm, 1e-12), np.nan)
    if fill_nan_mask:
        out = np.where(mask, data, out)  # keep original values w/ neighbor fill
    return out


def compute_spatial_lags(
    ds: xr.Dataset,
    grid: Grid,
    radii_km: list[float] | None = None,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Add neighborhood-mean features for given variables.

    Each variable ``v`` at radius ``r`` km produces feature ``v_lag_{r}km``.
    """
    from src.config import FEATURE_CONFIG

    radii_km = radii_km or FEATURE_CONFIG["spatial_lag_radii_km"]
    variables = variables or [v for v in ds.data_vars if ds[v].ndim >= 2]

    meters_per_deg = 111_320.0
    result = xr.Dataset(coords=ds.coords)
    for name in ds.data_vars:
        if name not in (variables or [name]):
            continue
        da = ds[name]
        if da.ndim < 2:
            result[name] = da
            continue

        # Radius in cells at this latitude band.
        # Use the grid resolution scaled by cos(lat) for lon-direction.
        cell_lat = abs(float(np.diff(grid.lats).mean() or grid.res))
        cell_lon = abs(float(np.diff(grid.lons).mean() or grid.res))

        for r_km in radii_km:
            radius_lat_cells = r_km * 1000 / (meters_per_deg * cell_lat)
            # Reduce lon cell count near high latitudes to keep circularity.
            cos_lat = np.cos(np.radians(np.clip(grid.lats.mean(), -75, 75)))
            radius_lon_cells = r_km * 1000 / (meters_per_deg * cell_lon * cos_lat)
            radius_cells = min(radius_lat_cells, radius_lon_cells)

            vals = da.values
            if vals.ndim == 3:
                # (time, lat, lon) -> iterate over the leading dimensions
                lag = np.stack(
                    [
                        neighborhood_mean(vals[i], radius_cells)
                        for i in range(vals.shape[0])
                    ],
                    axis=0,
                )
            else:
                lag = neighborhood_mean(vals, radius_cells)
            feature_name = f"{name}_lag{int(round(r_km))}km"
            result[feature_name] = (da.dims, lag)

    return result