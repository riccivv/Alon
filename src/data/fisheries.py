"""Fisheries ground-truth pipeline.

Supports the hybrid target design:

1. **Real catch data** (when available): a CSV of the form
   ``[fma, species, year, quarter, volume_mt, lat, lon]`` from PSA OpenStat /
   BFAR is joined to grid cells so satellite features can be supervised by
   actual catch volumes (annual/quarterly, province/grid resolution).

2. **Habitat-suitability pseudo-labels** (always): a 0-1 suitability score for
   each species derived from its SST and chlorophyll-a preference ranges
   (from ``configs/species_regions.yaml``). This is what drives the fine-grid
   "high-probability fishing zones" in the MVP map before real catch labels
   are aggregated.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.config import RAW_DATA_DIR, TARGET_SPECIES

logger = logging.getLogger(__name__)

DEFAULT_CATCH_CSV = RAW_DATA_DIR / "fisheries" / "catch_data.csv"

CATCH_COLUMNS = [
    "fma",
    "species",
    "year",
    "quarter",
    "volume_mt",
    "lat",
    "lon",
]


# --------------------------------------------------------------------------- #
# Habitat suitability pseudo-labels
# --------------------------------------------------------------------------- #
def _gaussian(x: np.ndarray, lo: float, hi: float, peak: float | None = None) -> np.ndarray:
    """Trapezoid-ish membership in [lo, hi] with soft shoulders.

    Values inside [lo, hi] score 1.0; outside it decays with a Gaussian tail
    whose width is 20% of the interval.
    """
    x = np.asarray(x, dtype="float64")
    width = max((hi - lo) * 0.2, 0.5)
    score = np.where(
        (x >= lo) & (x <= hi),
        1.0,
        np.where(
            x < lo,
            np.exp(-((x - lo) ** 2) / (2 * width**2)),
            np.exp(-((x - hi) ** 2) / (2 * width**2)),
        ),
    )
    return score


def habitat_suitability_from_cube(
    cube: xr.Dataset, species_code: str
) -> xr.DataArray:
    """Compute a 0-1 habitat-suitability field for a species.

    Uses the geometric mean of the SST and chlorophyll memberships so that a
    cell must be *reasonable in both* to score highly.
    """
    matches = [s for s in TARGET_SPECIES if s["code"] == species_code]
    if not matches:
        raise KeyError(f"Unknown species code {species_code!r}")
    species = matches[0]

    sst_lo, sst_hi = species["optimal_sst_range"]
    chl_lo, chl_hi = species["optimal_chla_range"]

    if "sst" not in cube.data_vars or "chl" not in cube.data_vars:
        raise ValueError(
            "cube must contain 'sst' and 'chl' to compute habitat suitability"
        )

    sst_score = _gaussian(cube["sst"].values, sst_lo, sst_hi)
    chl_values = cube["chl"].values
    with np.errstate(invalid="ignore", divide="ignore"):
        chl_score = _gaussian(
            np.where(chl_values > 0, np.log10(chl_values), 0), chl_lo, chl_hi
        )

    suitability = np.sqrt(sst_score * chl_score)
    return xr.DataArray(
        suitability,
        coords=cube["sst"].coords,
        dims=cube["sst"].dims,
        name=f"suitability_{species_code}",
        attrs={"units": "0-1", "long_name": f"habitat suitability for {species['name']}"},
    )


def habitat_suitability_from_frame(
    df: pd.DataFrame,
    species_code: str,
    sst_col: str = "sst",
    chl_col: str = "chl",
) -> pd.Series:
    """Vectorized 0-1 habitat-suitability from a long feature DataFrame.

    Mirrors ``habitat_suitability_from_cube`` (geometric mean of SST and
    log-chlorophyll memberships) so training can compute labels directly
    from the feature matrix without materializing the full cube.
    """
    matches = [s for s in TARGET_SPECIES if s["code"] == species_code]
    if not matches:
        raise KeyError(f"Unknown species code {species_code!r}")
    species = matches[0]

    if sst_col not in df.columns or chl_col not in df.columns:
        raise ValueError(
            f"Feature frame must contain {sst_col!r} and {chl_col!r} "
            f"to compute habitat suitability"
        )

    sst_lo, sst_hi = species["optimal_sst_range"]
    chl_lo, chl_hi = species["optimal_chla_range"]

    sst_score = _gaussian(df[sst_col].to_numpy(), sst_lo, sst_hi)
    chl_values = df[chl_col].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        chl_log = np.log10(np.where(chl_values > 0, chl_values, 1.0))
    chl_score = _gaussian(np.where(chl_values > 0, chl_log, 0.0), chl_lo, chl_hi)
    suitability = np.sqrt(sst_score * chl_score)
    return pd.Series(suitability, index=df.index, name=f"suitability_{species_code}")


def add_suitability_labels(cube: xr.Dataset, species_code: str = "all") -> xr.Dataset:
    """Add the suitability pseudo-label field(s) onto the cube."""
    out = cube
    codes = [species_code] if species_code != "all" else [s["code"] for s in TARGET_SPECIES]
    for code in codes:
        if code == "all":
            continue
        try:
            out[f"suitability_{code}"] = habitat_suitability_from_cube(cube, code)
        except (ValueError, KeyError) as exc:
            logger.warning("Skipping suitability for %s: %s", code, exc)
    return out


# --------------------------------------------------------------------------- #
# Real catch data
# --------------------------------------------------------------------------- #
def load_catch_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load BFAR/PSA catch statistics from a CSV file.

    Expected schema (``CATCH_COLUMNS``): fma, species, year, quarter,
    volume_mt, lat, lon. Empty DataFrame if the file is absent.
    """
    path = Path(path or DEFAULT_CATCH_CSV)
    if not path.exists():
        logger.warning("No catch data found at %s. Using suitability labels only.", path)
        return pd.DataFrame(columns=CATCH_COLUMNS)
    df = pd.read_csv(path)
    missing = [c for c in CATCH_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Catch CSV missing required columns: {missing}")
    df["year"] = df["year"].astype(int)
    df["quarter"] = df["quarter"].astype(int).clip(1, 4)
    return df


def map_catch_to_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Attach grid cell coordinates to catch rows.

    Each catch row must carry an anchor point (lat/lon = fishing-ground or
    province centroid). This expands the point to the nearest 0.1 deg cell.
    """
    if df.empty:
        return df
    cell = np.round(np.column_stack([df["lon"], df["lat"]]), 1)
    out = df.copy()
    out["cell_lon"] = cell[:, 0]
    out["cell_lat"] = cell[:, 1]
    return out


# --------------------------------------------------------------------------- #
# MSY estimate / overfishing baseline (placeholder, refined with BFAR data)
# --------------------------------------------------------------------------- #
def msy_proxy(accumulated_ts: pd.DataFrame, species: str) -> float:
    """Crude MSY baseline from the historical catch peak (tonnes).

    Replace with BFAR/NFRDI stock-assessment MSY values once integrated.
    """
    if accumulated_ts.empty:
        return 0.0
    series = accumulated_ts[accumulated_ts["species"] == species]["volume_mt"]
    if series.empty:
        return 0.0
    return float(series.max())