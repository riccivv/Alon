"""Temporal processing: gap filling, rolling means, anomalies, encodings."""

from __future__ import annotations

import numpy as np
import xarray as xr

from src.config import FEATURE_CONFIG


# --------------------------------------------------------------------------- #
# Gap filling (cloud cover)
# --------------------------------------------------------------------------- #
def gap_fill_timeseries(
    da: xr.DataArray,
    method: str = "linear",
    max_gap_steps: int | None = 8,
) -> xr.DataArray:
    """Interpolate NaNs (cloud gaps) along the time axis.

    ``max_gap_steps`` limits interpolation to gaps of at most N time steps;
    longer gaps stay NaN to avoid over-smoothing seasonal transitions.
    """
    if "time" not in da.dims:
        return da

    out = da.sortby("time").interpolate_na(dim="time", method=method, fill_value="extrapolate")

    # Re-introduce NaN where the gap is too long.
    if max_gap_steps is not None:
        isnan = da.sortby("time").isnull()
        gap_id = isnan.astype("int").cumsum("time") - isnan.astype("int").cumsum("time").where(isnan)
        # Simple gap-length mask: run-length encode along time.
        gap_mask = _gap_mask(da.sortby("time"), max_gap_steps)
        out = out.where(~gap_mask)

    return out


def _gap_mask(da: xr.DataArray, max_gap_steps: int) -> xr.DataArray:
    """Boolean array True where gaps exceed max_gap_steps (per cell)."""
    isnan = da.isnull().values
    # Transpose so time is last for run-length coding per cell.
    if isnan.ndim == 3:
        arr = np.moveaxis(isnan, 0, -1)  # (lat, lon, time)
        mask = np.zeros_like(arr, dtype=bool)
        for idx in np.ndindex(arr.shape[:-1]):
            mask[idx] = _long_gaps(arr[idx], max_gap_steps)
        mask = np.moveaxis(mask, -1, 0)
    else:
        mask = _long_gaps(isnan, max_gap_steps)
    return xr.DataArray(mask, coords=da.coords, dims=da.dims)


def _long_gaps(series: np.ndarray, max_len: int) -> np.ndarray:
    """Mark positions inside runs of NaN longer than max_len."""
    out = np.zeros(series.shape, dtype=bool)
    i = 0
    n = len(series)
    while i < n:
        if series[i]:
            j = i
            while j < n and series[j]:
                j += 1
            if j - i > max_len:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def fill_with_climatology(
    da: xr.DataArray,
    max_gap_steps: int | None = None,
) -> xr.DataArray:
    """Fill any residual NaNs with the monthly climatological mean."""
    if da.isnull().sum() == 0:
        return da
    clim = da.groupby("time.month").mean("time")
    months = da["time"].dt.month
    filled = da.where(~da.isnull(), clim.sel(month=months))
    return filled


# --------------------------------------------------------------------------- #
# Rolling means
# --------------------------------------------------------------------------- #
def rolling_mean(da: xr.DataArray, window_days: int) -> xr.DataArray:
    """Rolling mean along time measured in days, centered, NaN-tolerant.

    Windows are mapped to the nearest number of 8-day composite steps.
    """
    if "time" not in da.dims:
        return da
    # Estimate average timestep from the actual axis.
    diffs = np.median(np.diff(da["time"].values).astype("timedelta64[D]").astype(int))
    steps = max(1, int(round(window_days / max(diffs, 1))))
    return da.rolling(time=steps, center=True, min_periods=1).mean()


# --------------------------------------------------------------------------- #
# Anomalies vs climatology
# --------------------------------------------------------------------------- #
def monthly_climatology(da: xr.DataArray) -> xr.DataArray:
    """Mean per calendar month across all years (ignores NaNs)."""
    return da.groupby("time.month").mean("time")


def anomaly_from_climatology(da: xr.DataArray) -> xr.DataArray:
    """Difference of each observation from its month's climatology."""
    if "time" not in da.dims:
        return da - da
    clim = monthly_climatology(da)
    months = da["time"].dt.month
    return da - clim.sel(month=months)


# --------------------------------------------------------------------------- #
# Cyclical time encoding
# --------------------------------------------------------------------------- #
def month_sin_cos(time: xr.DataArray) -> dict[str, xr.DataArray]:
    """Encode the calendar month as sine/cosine coordinates."""
    months = time.dt.month.values.astype("float64")
    period = 12.0
    sin = np.sin(2 * np.pi * months / period)
    cos = np.cos(2 * np.pi * months / period)
    out = {}
    out["month_sin"] = xr.DataArray(sin, coords={"time": time}, dims="time")
    out["month_cos"] = xr.DataArray(cos, coords={"time": time}, dims="time")
    return out


def apply_temporal_features(
    ds: xr.Dataset,
    config: dict | None = None,
) -> xr.Dataset:
    """Add rolling means, anomalies, and seasonal encoding to the cube.

    Operating variables: sst, chl, par, kd490 (present if in the cube).
    """
    config = config or FEATURE_CONFIG
    rolling_windows = config.get("rolling_windows_days", [7, 14, 30, 90])
    result = xr.Dataset(coords=ds.coords)

    # Preserve static (non-time-varying) fields, e.g. bathymetry features.
    for name in ds.data_vars:
        if "time" not in ds[name].dims:
            result[name] = ds[name]

    for var in ("sst", "chl", "par", "kd490"):
        if var not in ds.data_vars:
            continue
        da = ds[var]
        result[var] = da

        # Rolling means
        for w in rolling_windows:
            result[f"{var}_roll{w}d"] = rolling_mean(da, w)

        # Anomaly vs monthly climatology
        if ds["time"].size > 12:
            result[f"{var}_anom"] = anomaly_from_climatology(da)

    if "time" in ds.coords and config.get("use_sin_cos_time", True):
        enc = month_sin_cos(ds["time"])
        result["month_sin"] = enc["month_sin"]
        result["month_cos"] = enc["month_cos"]

    return result