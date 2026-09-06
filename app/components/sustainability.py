"""Sustainability flags: current vs historical (MSY-proxy) per FMA.

A crude MSY baseline is the historical PEAK of the region-averaged habitat
suitability over the full record (mirroring ``msy_proxy``'s peak-catch
convention). The current composite is compared to that baseline and flagged
against the thresholds in ``SUSTAINABILITY`` config:

    green  ratio <= 0.60  -> yields within sustainable bounds
    yellow 0.60 < ratio <= 0.80 -> caution
    red    ratio > 0.80   -> overfishing risk (exceeding baseline)

Swap in authenticated BFAR/NFRDI MSY numbers later without changing the UI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

from app.components.dashboard import in_region
from app.components.state import get_cube, species_options, time_options
from src.config import SUSTAINABILITY, PHILIPPINE_REGIONS
from src.data.fisheries import habitat_suitability_from_cube


def sustainability_table(cube: xr.Dataset, species_code: str) -> pd.DataFrame:
    """Per-region current flag vs a historical MSY proxy."""
    ratios = []
    for region in PHILIPPINE_REGIONS:
        series = _region_series(cube, species_code, region["name"])
        if series is None or series["mean"].isna().all():
            continue
        historical = float(series["mean"].max())
        current = float(series["mean"].iloc[-1]) if len(series) else np.nan
        if historical <= 0 or np.isnan(current):
            ratios.append({"FMA region": region["name"], "Current": current,
                           "MSY proxy": float(historical), "Ratio": np.nan, "Flag": "n/a"})
            continue
        ratio = current / historical
        ratios.append({
            "FMA region": region["name"],
            "Current": current,
            "MSY proxy": float(historical),
            "Ratio": ratio,
            "Flag": flag_for(ratio),
        })
    return pd.DataFrame(ratios)


def _region_series(cube: xr.Dataset, species_code: str, region_name: str):
    import numpy as np
    import pandas as pd
    import warnings

    suitability = habitat_suitability_from_cube(cube, species_code)
    lon_min, lat_min, lon_max, lat_max = next(
        r["bbox"] for r in PHILIPPINE_REGIONS if r["name"] == region_name
    )
    block = suitability.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        mean = block.mean(dim=["lat", "lon"])
    return pd.DataFrame({
        "time": pd.to_datetime(mean["time"].values),
        "mean": mean.values,
    })


def flag_for(ratio: float) -> str:
    r = float(ratio)
    if r <= SUSTAINABILITY["msy_ratio_sustainable"]:
        return "sustainable"
    if r <= SUSTAINABILITY["msy_ratio_caution"]:
        return "caution"
    return "overfishing risk"


FLAG_COLORS = {
    "sustainable": "background-color: #1d6f42; color: white;",
    "caution": "background-color: #b5a642; color: black;",
    "overfishing risk": "background-color: #b0352f; color: white;",
    "n/a": "background-color: #555555; color: white;",
}


def sustainability_tab(cube: xr.Dataset) -> None:
    species = [s for s in species_options() if s["code"] != "all"]
    code = st.selectbox(
        "Species", [s["code"] for s in species],
        format_func=lambda c: next(s["name"] for s in species if s["code"] == c),
        key="sus_species",
    )

    table = sustainability_table(cube, code)
    if table.empty:
        st.info("No data available to compute sustainability flags yet.")
        return

    styled = table.style.map(lambda v: FLAG_COLORS.get(v, ""), subset=["Flag"])
    st.dataframe(styled, width="stretch")

    counts = table["Flag"].value_counts()
    st.caption(
        f"{counts.get('sustainable', 0)} sustainable  |  "
        f"{counts.get('caution', 0)} caution  |  "
        f"{counts.get('overfishing risk', 0)} overfishing risk"
    )
    st.info(
        "Baseline (MSY proxy) = historical peak regional suitability. "
        "Ratio = current / baseline. Swap in BFAR/NFRDI MSY assessments when available."
    )