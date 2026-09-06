"""Bathymetry acquisition for the Philippine study region.

Provides a single ``fetch_bathymetry()`` entry point. It subsets the NOAA
ETOPO 2022 global relief grid (1 arc-minute) over the Philippine bounding box
using OPeNDAP, so only the region's values are transferred. The subset is
cached to a local NetCDF along with a derived seabed slope field.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# Provider list, tried in order. OPeNDAP (dodsC) endpoints allow us to pull
# only the study-region subset from the global grid (491 MB) without
# downloading the whole file.
PROVIDERS = [
    {
        "id": "etopo2022-60s",
        "url": (
            "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/"
            "60s/60s_bed_elev_netcdf/ETOPO_2022_v1_60s_N90W180_bed.nc"
        ),
        "file_url": (
            "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/"
            "60s/60s_bed_elev_netcdf/ETOPO_2022_v1_60s_N90W180_bed.nc"
        ),
        "var": "z",
        "lat_dim": "lat",
        "lon_dim": "lon",
    },
]


def _open_provider_subset(
    bbox: tuple[float, float, float, float], provider: dict
) -> xr.Dataset | None:
    """Open a provider grid via OPeNDAP and subset to the bbox."""
    lon_min, lat_min, lon_max, lat_max = bbox
    lat_dim = provider["lat_dim"]
    lon_dim = provider["lon_dim"]

    lat_lo, lat_hi = min(lat_min, lat_max), max(lat_min, lat_max)
    lon_lo, lon_hi = min(lon_min, lon_max), max(lon_min, lon_max)

    ds = None
    for engine in ("netcdf4", "pydap"):
        try:
            ds = xr.open_dataset(provider["url"], engine=engine, decode_times=False)
            logger.info("Provider %s opened with engine=%s", provider["id"], engine)
            break
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Provider %s engine=%s failed: %s", provider["id"], engine, exc
            )
            ds = None
    if ds is None:
        return None

    try:
        if lat_dim not in ds.dims or lon_dim not in ds.dims:
            raise ValueError(f"Unexpected dims for {provider['id']}: {dict(ds.sizes)}")

        sub = ds.sel(
            {lat_dim: slice(lat_lo, lat_hi), lon_dim: slice(lon_lo, lon_hi)}
        )
        da = sub[provider["var"]]
        if da.ndim > 2:  # squeeze trailing single dims (e.g. a depth/z axis)
            da = da.squeeze(drop=True)

        return xr.Dataset(
            {"depth": (["lat", "lon"], np.asarray(da.values, dtype="float32"))},
            coords={
                "lat": np.asarray(da[lat_dim].values, dtype="float32"),
                "lon": np.asarray(da[lon_dim].values, dtype="float32"),
            },
            attrs={"units": "metres", "long_name": "elevation above sea level"},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Provider %s subset failed: %s", provider["id"], exc)
        return None


def _download_full(provider: dict, dest: Path, bbox: tuple[float, float, float, float]) -> xr.Dataset:
    """Fallback: download the full global grid once, then subset locally."""
    import requests

    if not dest.exists():
        logger.info("Full download fallback for %s (%.0f MB)...", provider["id"], 491)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(provider["file_url"], stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
    ds = xr.open_dataset(dest, decode_times=False)
    lat_lo, lat_hi = sorted((bbox[1], bbox[3]))
    lon_lo, lon_hi = sorted((bbox[0], bbox[2]))
    sub = ds.sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi))
    da = sub[provider["var"]].squeeze(drop=True)
    return xr.Dataset(
        {"depth": (["lat", "lon"], np.asarray(da.values, dtype="float32"))},
        coords={"lat": da[provider["lat_dim"]].values,
                "lon": da[provider["lon_dim"]].values},
    )


def slope_from_depth(depth: xr.DataArray) -> xr.DataArray:
    """Compute seabed slope magnitude (degrees) from an elevation grid.

    Slope is the maximum rate of change between a cell and its neighbors,
    expressed in degrees from horizontal.
    """
    d = depth.values.astype("float64")
    grad_y, grad_x = np.gradient(d)

    lat = depth["lat"].values
    lon = depth["lon"].values
    dlat = abs(float(np.mean(np.diff(lat)))) * 111_320.0  # metres per degree
    dlon = abs(float(np.mean(np.diff(lon)))) * 111_320.0 * np.cos(
        np.radians(np.clip(lat.mean(), -75, 75))
    )
    dlon = max(dlon, 1.0)
    gx = grad_x / dlon
    gy = grad_y / dlat
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    return xr.DataArray(
        np.degrees(slope_rad),
        coords=depth.coords,
        dims=depth.dims,
        attrs={"units": "degrees", "long_name": "seabed slope"},
    )


def fetch_bathymetry(
    bbox: tuple[float, float, float, float],
    out_path: Path | str | None = None,
    use_cached: bool = True,
    allow_full_download: bool = True,
) -> xr.Dataset:
    """Return (and optionally cache) the Philippine bathymetry subset.

    Parameters
    ----------
    bbox: (lon_min, lat_min, lon_max, lat_max)
    out_path: where to cache the NetCDF subset (optional).
    use_cached: return the cached file if present.
    allow_full_download: if OPeNDAP fails, fetch the global grid once.
    """
    from src.config import RAW_DATA_DIR

    if out_path is None:
        out_path = RAW_DATA_DIR / "bathymetry" / "ph_bathymetry.nc"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if use_cached and out_path.exists():
        logger.info("Loading cached bathymetry from %s", out_path)
        return xr.open_dataset(out_path)

    ds = None
    for provider in PROVIDERS:
        ds = _open_provider_subset(bbox, provider)
        if ds is None and allow_full_download:
            full_path = RAW_DATA_DIR / "bathymetry" / "ETOPO_2022_v1_60s_global.nc"
            try:
                ds = _download_full(provider, full_path, bbox)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Full download fallback failed: %s", exc)
        if ds is not None:
            logger.info("Bathymetry sourced from %s", provider["id"])
            break
    if ds is None:
        raise RuntimeError("All bathymetry providers failed. Check network access.")

    ds.attrs.update({"source": "NOAA ETOPO 2022 (OPeNDAP subset)"})
    ds["slope"] = slope_from_depth(ds["depth"])
    ds.to_netcdf(out_path)
    logger.info("Wrote bathymetry subset to %s", out_path)
    return ds


def load_bathymetry(path: Path | str | None = None) -> xr.Dataset:
    """Load a previously saved Philippine bathymetry NetCDF."""
    from src.config import RAW_DATA_DIR

    if path is None:
        path = RAW_DATA_DIR / "bathymetry" / "ph_bathymetry.nc"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Bathymetry file not found at {path}. Call fetch_bathymetry() first."
        )
    return xr.open_dataset(path)