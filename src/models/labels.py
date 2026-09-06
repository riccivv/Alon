"""Training-target (label) providers.

Two providers implement a single :class:`LabelProvider` interface so the
training pipeline does not care where supervision comes from:

1. ``HabitatSuitabilityProvider`` -- self-supervised 0-1 score from the
   species' SST / chlorophyll preference ranges (works today, no external
   data). This drives the MVP pollution-free "high-probability zones".
2. ``CatchVolumeProvider`` -- joins real BFAR/PSA quarterly catch volumes
   (CSV schema in :mod:`src.data.fisheries`) onto the feature matrix. It
   returns ``None`` while no catch file is present, so training falls back
   to suitability labels without code changes.

Switching providers is a config choice, not a code change: a provider just
attaches a continuous ``label`` column to a feature DataFrame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import TARGET_SPECIES
from src.data.fisheries import (
    DEFAULT_CATCH_CSV,
    CATCH_COLUMNS,
    habitat_suitability_from_frame,
    load_catch_data,
    map_catch_to_cells,
)

logger = logging.getLogger(__name__)

PROVIDER_HABITAT = "habitat"
PROVIDER_CATCH = "catch"


@dataclass
class LabelResult:
    """A label column plus metadata for reporting."""

    series: pd.Series
    name: str
    kind: str


class LabelProvider:
    """Base class: attach a continuous target column to a feature frame."""

    kind = "base"

    def attach(self, df: pd.DataFrame, species_code: str) -> LabelResult | None:
        """Return LabelResult or None when no labels are available."""
        raise NotImplementedError


class HabitatSuitabilityProvider(LabelProvider):
    """0-1 habitat suitability from SST/chlorophyll within species ranges."""

    kind = PROVIDER_HABITAT

    def attach(self, df: pd.DataFrame, species_code: str) -> LabelResult:
        series = habitat_suitability_from_frame(df, species_code)
        return LabelResult(
            series=series,
            name=f"suitability_{species_code}",
            kind=self.kind,
        )


class CatchVolumeProvider(LabelProvider):
    """Quarterly BFAR/PSA catch volumes (tonnes) merged per cell.

    Requires ``data/raw/fisheries/catch_data.csv`` (see ``CATCH_COLUMNS``).
    Returns ``None`` (skipped) when the file is absent.
    """

    kind = PROVIDER_CATCH

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or DEFAULT_CATCH_CSV)

    def attach(self, df: pd.DataFrame, species_code: str) -> LabelResult | None:
        catches = load_catch_data(self.path)
        if catches.empty:
            logger.warning(
                "No catch data at %s; skipping real-catch labels for %s. "
                "Falling back to habitat suitability.",
                self.path,
                species_code,
            )
            return None

        if species_code not in set(catches["species"]):
            logger.warning(
                "No rows for species %r in catch data; skipping catch labels.",
                species_code,
            )
            return None

        catches = map_catch_to_cells(catches)
        catches = catches[catches["species"] == species_code].copy()

        labels = _merge_catches_to_features(df, catches, species_code)
        if labels is None or labels.dropna().empty:
            return None
        return LabelResult(series=labels, name="catch_volume_mt", kind=self.kind)


def _merge_catches_to_features(
    df: pd.DataFrame, catches: pd.DataFrame, species_code: str
) -> pd.Series | None:
    """Join quarterly cell-level catches onto the (lat, lon, date) feature frame.

    Quarterly values are applied to every 8-day composite row within that
    quarter (forward-fill within quarter at each cell).
    """
    required = {"lat", "lon", "year", "quarter", "volume_mt"}
    missing = required - set(catches.columns)
    catches = catches.dropna(subset=["lat", "lon", "volume_mt"])
    if missing or catches.empty:
        return None

    features = df.copy()
    features["lat_cell"] = features["lat"].round(1)
    features["lon_cell"] = features["lon"].round(1)
    features["year"] = pd.to_datetime(features["time"]).dt.year
    features["quarter"] = pd.to_datetime(features["time"]).dt.quarter

    catch_key = catches[["lat", "lon", "year", "quarter", "volume_mt"]].rename(
        columns={"lat": "lat_cell", "lon": "lon_cell"}
    )
    merged = features[["lat_cell", "lon_cell", "year", "quarter"]].merge(
        catch_key,
        on=["lat_cell", "lon_cell", "year", "quarter"],
        how="left",
        validate="many_to_one",
    )
    series = merged["volume_mt"].astype("float64")
    series.index = df.index
    series.name = f"catch_volume_mt_{species_code}"
    return series


LABEL_PROVIDERS = {
    PROVIDER_HABITAT: HabitatSuitabilityProvider,
    PROVIDER_CATCH: CatchVolumeProvider,
}


def provider_order(kind: str = "prefer_catch") -> list[LabelProvider]:
    """Return providers in the order they should be tried.

    * ``prefer_catch`` -- use real catches when available, suitability otherwise.
    * ``habitat_only``  -- always the self-supervised score.
    * ``both``          -- both labels (used for diagnostics/comparison).
    """
    if kind == "habitat_only":
        return [HabitatSuitabilityProvider()]
    if kind == "both":
        return [HabitatSuitabilityProvider(), CatchVolumeProvider()]
    return [CatchVolumeProvider(), HabitatSuitabilityProvider()]


def attach_labels(
    df: pd.DataFrame,
    species_code: str,
    kind: str = "prefer_catch",
) -> tuple[pd.Series, str]:
    """Attach the winning label for ``species_code``.

    Returns ``(label_series, provider_kind)``. Raises ``ValueError`` if no
    provider produced labels.
    """
    for provider in provider_order(kind):
        result = provider.attach(df, species_code)
        if result is not None and not result.series.isna().all():
            logger.info(
                "Labels for %s from provider %r (%s) -- %d rows",
                species_code,
                provider.kind,
                result.name,
                result.series.notna().sum(),
            )
            return result.series, provider.kind
    raise ValueError(
        f"No labels were produced for {species_code!r} with kind={kind!r}"
    )


def available_species() -> list[dict]:
    """Species metadata (excluding the 'all' composite) for training."""
    return [s for s in TARGET_SPECIES if s["code"] != "all"]