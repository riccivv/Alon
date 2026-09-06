"""Common spatial grid over the Philippine study region.

All satellite variables are regridded onto this uniform lat/lon grid so that
features can be joined across sensors. Land cells are masked out using
bathymetry (elevation >= 0 is land).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from src.config import PHILIPPINES_BBOX

DEFAULT_RES = 0.1  # degrees -> ~11 km cell size at the equator


@dataclass
class Grid:
    """Uniform lon/lat grid definition."""

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    res: float = DEFAULT_RES

    def __post_init__(self) -> None:
        self.lons: np.ndarray = np.round(
            np.arange(self.lon_min, self.lon_max + self.res, self.res), 6
        )
        self.lats: np.ndarray = np.round(
            np.arange(self.lat_min, self.lat_max + self.res, self.res), 6
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.lats), len(self.lons))

    @property
    def n_cells(self) -> int:
        return len(self.lats) * len(self.lons)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lon2d, lat2d) broadcast arrays for the grid."""
        lon2d, lat2d = np.meshgrid(self.lons, self.lats)
        return lon2d, lat2d

    def as_dataset(self) -> xr.Dataset:
        return xr.Dataset(coords={"lon": self.lons, "lat": self.lats})

    def cell_area_km2(self) -> xr.DataArray:
        """Approximate cell area in km^2 (spherical)."""
        lat = np.radians(self.lats)
        dy = self.res * 111.32
        dx = self.res * 111.32 * np.cos(lat[:, None])
        return xr.DataArray(dx * dy, coords={"lat": self.lats, "lon": self.lons})


def create_grid(bbox: tuple[float, float, float, float] = PHILIPPINES_BBOX) -> Grid:
    """Build the default Philippine study grid."""
    lon_min, lat_min, lon_max, lat_max = bbox
    return Grid(lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max)


def ocean_mask_from_depth(depth: xr.DataArray) -> xr.DataArray:
    """Mask grid cells that are ocean (depth < 0 is below sea level)."""
    ocean = depth < 0.0
    ocean = ocean.fillna(False)
    return ocean.astype("bool")


def land_mask_from_depth(depth: xr.DataArray) -> xr.DataArray:
    """Inverse of the ocean mask; True where the cell is land."""
    return ~ocean_mask_from_depth(depth)


def mask_land(ds: xr.Dataset | xr.DataArray, depth: xr.DataArray) -> None:
    """In-place: set NaN on land cells for every data variable."""
    land = land_mask_from_depth(depth)
    if isinstance(ds, xr.DataArray):
        ds = ds.where(~land)  # noqa: PLW2901 (local)
        return ds
    out = ds
    for var in out.data_vars:
        out[var] = out[var].where(~land)
    return out