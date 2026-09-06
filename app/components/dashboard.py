"""FMA dashboard: region rankings and condition trends.

* Region ranking -- mean predicted yield (or suitability) per FMA for the
  selected composite, derived from the same forecast used on the map.
* Condition trend -- region-averaged habitat suitability over time for the
  selected species (cheap to compute straight from the cube).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

from app.components.state import (
    get_models, species_options, time_options,
)
from app.components.forecast_map import _forecast_series
from src.config import PHILIPPINE_REGIONS as REGIONS_CFG


def region_ranking(
    cube: xr.Dataset,
    species_code: str,
    time: pd.Timestamp,
    bundles: dict[str, dict],
) -> pd.DataFrame:
    series = _forecast_series(cube, species_code, time, bundles)
    rows = series.reset_index()
    rows["region"] = [in_region(lon, lat) for lon, lat in zip(rows["lon"], rows["lat"])]
    rows = rows.dropna(subset=["region"])
    grouped = (
        rows.groupby("region")["value"]
        .agg(cells="count", mean="mean", p90=lambda v: np.percentile(v, 90), max="max")
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    grouped.columns = ["FMA region", "Cells", "Mean yield", "P90 hot-spot", "Peak cell"]
    return grouped


def in_region(lon: float, lat: float) -> str | None:
    for region in REGIONS_CFG:
        lon_min, lat_min, lon_max, lat_max = region["bbox"]
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return region["name"]
    return None


@st.cache_resource(show_spinner="Computing condition trend ...")
def species_trend(
    _cube: xr.Dataset, species_code: str, region_name: str | None
) -> pd.DataFrame:
    """Region-averaged habitat suitability over time for a species."""
    import warnings

    from src.data.fisheries import habitat_suitability_from_cube

    suitability = habitat_suitability_from_cube(_cube, species_code)
    if region_name is not None:
        bbox = next(r["bbox"] for r in REGIONS_CFG if r["name"] == region_name)
        lon_min, lat_min, lon_max, lat_max = bbox
        suitability = suitability.sel(
            lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
        )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        mean = suitability.mean(dim=["lat", "lon"], skipna=True)
    return pd.DataFrame(
        {"time": pd.to_datetime(mean["time"].values), "suitability": mean.values}
    )


def dashboard_tab(cube: xr.Dataset) -> None:
    species = species_options()
    times = time_options(cube)

    c1, c2 = st.columns(2)
    with c1:
        code = st.selectbox("Species", [s["code"] for s in species],
                            format_func=lambda c: next(s["name"] for s in species if s["code"] == c),
                            key="dash_species")
    with c2:
        idx = st.slider("Base composite", 0, len(times) - 1, len(times) - 1, key="dash_time")
        time = times[idx]

    bundles = get_models((code,))

    st.subheader("FMA rankings (current window)")
    ranking = region_ranking(cube, code, time, bundles)
    st.dataframe(ranking.round(3), width="stretch")

    st.subheader("Regional condition trend vs time")
    region_names = [r["name"] for r in REGIONS_CFG]
    region = st.selectbox("Region", ["All Philippine waters"] + region_names)
    trend = species_trend(cube, code, None if region.startswith("All") else region)
    st.line_chart(trend.set_index("time"))