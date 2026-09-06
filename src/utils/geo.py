"""Small geospatial helpers shared across the pipeline and visualization."""

from __future__ import annotations

import numpy as np
import xarray as xr


def ensure_ascending(da: xr.DataArray, dim: str = "lat") -> xr.DataArray:
    """Sort a data array so its spatial axis is ascending, propagating coords."""
    if dim in da.dims:
        coords = da[dim].values
        if len(coords) > 1 and coords[0] > coords[-1]:
            da = da.sortby(dim)
    return da


def select_bbox(
    da: xr.DataArray | xr.Dataset,
    bbox: tuple[float, float, float, float],
    pad: float = 0.8,
) -> xr.DataArray | xr.Dataset:
    """Subset by (lon_min, lat_min, lon_max, lat_max) regardless of axis order.

    Handles both ascending and descending latitude/longitude coordinate axes
    (MODIS L3 products store latitude descending from +90).
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    sel = {}
    if "lat" in da.dims:
        lat = da["lat"].values
        lo, hi = (lat_min - pad, lat_max + pad)
        if len(lat) > 1 and lat[0] > lat[-1]:  # descending
            lo, hi = hi, lo
        sel["lat"] = slice(lo, hi)
    if "lon" in da.dims:
        lon = da["lon"].values
        lo, hi = (lon_min - pad, lon_max + pad)
        if len(lon) > 1 and lon[0] > lon[-1]:  # descending
            lo, hi = hi, lo
        sel["lon"] = slice(lo, hi)
    return da.sel(sel)