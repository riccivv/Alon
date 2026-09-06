"""Shared cached loaders for the Streamlit app.

Everything heavy (cube, feature matrix, trained models) is cached so the UI
stays responsive. All functions here degrade gracefully when artifacts are
not built yet (e.g. before the full pipeline has run).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

from src.config import MODELS_DIR, PHILIPPINE_REGIONS, PHILIPPINES_BBOX, TARGET_SPECIES
from src.processing.pipeline import CUBE_PATH, FEATURES_PATH, load_cube

logger = logging.getLogger(__name__)

CUBE_HINT = (
    "Ocean data cube not built yet. Run:\n"
    "  python scripts/fetch_satellite.py   # after downloads\n"
    "  python scripts/run_pipeline.py\n"
)


# --------------------------------------------------------------------------- #
# Loaders (cached)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading ocean cube ...")
def get_cube() -> xr.Dataset:
    try:
        return load_cube(CUBE_PATH)
    except FileNotFoundError as exc:
        st.error(CUBE_HINT)
        raise st.stop from exc


@st.cache_resource(show_spinner="Loading feature matrix ...")
def get_feature_matrix() -> pd.DataFrame:
    parquet = Path(str(FEATURES_PATH).replace(".pkl", ".parquet"))
    if parquet.exists():
        return pd.read_parquet(parquet)
    if FEATURES_PATH.exists():
        return pd.read_pickle(FEATURES_PATH)
    st.error(CUBE_HINT)
    raise st.stop


@st.cache_resource(show_spinner="Loading trained models ...")
def get_models(species_codes: tuple[str, ...]) -> dict[str, dict]:
    from src.models.train import load_bundle

    bundles = {}
    for code in species_codes:
        dest = Path(MODELS_DIR) / code
        if (dest / "bundle.pkl").exists() and (dest / "model.pkl").exists():
            bundles[code] = load_bundle(MODELS_DIR, code)
        else:
            logger.info("No saved model for %s; using habitat-suitability fallback.", code)
    return bundles


# --------------------------------------------------------------------------- #
# Domain helpers
# --------------------------------------------------------------------------- #
def species_options() -> list[dict]:
    """All target species except the composite 'all' entry."""
    return [s for s in TARGET_SPECIES if s["code"] != "all"]


def region_of(lon: float, lat: float) -> str | None:
    """Return the FMA region name containing a point, else None."""
    for region in PHILIPPINE_REGIONS:
        lon_min, lat_min, lon_max, lat_max = region["bbox"]
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return region["name"]
    return None


def time_options(cube: xr.Dataset) -> list[pd.Timestamp]:
    times = pd.to_datetime(cube["time"].values)
    return sorted(times)


def land_points(cube: xr.Dataset) -> pd.DataFrame:
    """Coarse land overlay points (for a grey silhouette backdrop)."""
    depth = cube["depth"].load() if hasattr(cube["depth"], "load") else cube["depth"]
    land = (~(depth < 0.0)).fillna(True)
    if land.ndim > 2:
        land = land.isel(time=0) if "time" in land.dims else land[0]
    lat, lon = np.meshgrid(land["lat"].values, land["lon"].values, indexing="ij")
    mask = land.values
    if mask.shape != lat.shape:
        # transpose guard: xarray dim order (lat, lon)
        mask = np.asarray(mask)
    return pd.DataFrame({"lon": lon[mask], "lat": lat[mask]})


def in_bbox(point_lon: float, point_lat: float, bbox: tuple[float, float, float, float]) -> bool:
    lon_min, lat_min, lon_max, lat_max = bbox
    return lon_min <= point_lon <= lon_max and lat_min <= point_lat <= lat_max


def snapshot_frame(cube: xr.Dataset, time: pd.Timestamp, features: list[str]) -> pd.DataFrame:
    """Long-format feature frame for one composite window (for inference).

    Every requested feature column is produced; any not present in the cube
    is filled with NaN so the model's median imputer can substitute later.
    """
    da = cube.sel(time=str(time), method="nearest")
    df = da.to_dataframe().reset_index()

    present = [f for f in features if f in df.columns]
    out = df[["lat", "lon"]].copy()
    for col in features:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[col] = np.nan
    return out.dropna(subset=["lat", "lon"])