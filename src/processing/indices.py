"""Derived oceanographic indices that proxy fish habitat quality.

These composite the primary satellite fields (SST, chlorophyll-a, Kd490) into
features that correlate with forage fish aggregation:
    * frontal_index   -- magnitude of the SST spatial gradient
                          (thermal fronts attract fish aggregations).
    * productivity_index -- log-scale chlorophyll biomass (food availability).
    * upwelling_proxy -- positive when surface waters are colder than their
                          climatology (i.e., freshly upwelled, nutrient rich).
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def frontal_index(sst: xr.DataArray) -> xr.DataArray:
    """SST horizontal gradient magnitude (degC / deg).

    Computed per time slice using index-space gradients scaled by lat/lon
    spacing. Expects dims (time, lat, lon) last two axes being lat/lon.
    """
    if sst.ndim < 2:
        return sst * 0.0
    lat = sst["lat"].values.astype("float64")
    lon = sst["lon"].values.astype("float64")
    dlat = abs(float(np.mean(np.diff(lat)))) or 1.0
    dlon = abs(float(np.mean(np.diff(lon)))) or 1.0
    cos_lat = np.cos(np.radians(np.clip(lat.mean(), -75, 75)))

    def _grad(values_2d: np.ndarray) -> np.ndarray:
        gy, gx = np.gradient(np.nan_to_num(values_2d, nan=np.nan), dlat, dlon)
        gx = gx / cos_lat
        mag = np.sqrt(gx**2 + gy**2)
        return np.where(np.isnan(values_2d), np.nan, mag)

    vals = sst.values
    if vals.ndim == 3:
        mags = np.stack([_grad(vals[i]) for i in range(vals.shape[0])], axis=0)
    else:
        mags = _grad(vals)
    return xr.DataArray(
        mags,
        coords=sst.coords,
        dims=sst.dims,
        attrs={"units": "degC/deg", "long_name": "SST frontal index"},
    )


def productivity_index(chl: xr.DataArray) -> xr.DataArray:
    """Log-scaled chlorophyll biomass (a proxy for prey availability)."""
    eps = 1e-4
    return np.log10(chl.where(chl > 0).clip(min=eps) + eps)


def upwelling_proxy(sst_anom: xr.DataArray) -> xr.DataArray:
    """Positive when surface water is cooler than climatology.

    Cooling relative to the monthly climatology indicates recently upwelled
    (nutrient-rich) water. Larger positive = stronger upwelling signal.
    """
    return (-sst_anom).clip(min=0.0)


def add_indices(ds: xr.Dataset) -> xr.Dataset:
    """Add derived indices to the cube if their inputs are present."""
    out = ds
    if "sst" in ds.data_vars:
        out["frontal_index"] = frontal_index(ds["sst"])
        if "sst_anom" in ds.data_vars:
            out["upwelling_proxy"] = upwelling_proxy(ds["sst_anom"])
    if "chl" in ds.data_vars:
        out["productivity_index"] = productivity_index(ds["chl"])
    return out