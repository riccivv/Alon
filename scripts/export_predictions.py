"""Export per-cell yield predictions for a species and composite to CSV.

Usage:
    python scripts/export_predictions.py --species sardinella
    python scripts/export_predictions.py --species mackerel --time 2025-12-15
    python scripts/export_predictions.py --species scad --label-kind habitat_only

Default behaviour: latest composite in the cube, trained model when present,
else the habitat-suitability score. Writes data/predictions/<species>.csv.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.components.forecast_map import _forecast_series  # noqa: E402
from app.components.state import get_cube, get_models, time_options  # noqa: E402
from src.config import DATA_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export yield predictions to CSV")
    parser.add_argument("--species", required=True, help="Species code (e.g. sardinella)")
    parser.add_argument("--time", default=None, help="Composite date ('YYYY-MM-DD'); "
                        "defaults to the latest available")
    parser.add_argument("--label-kind", choices=["prefer_catch", "habitat_only"],
                        default="prefer_catch")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "predictions",
                        help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("export")

    cube = get_cube()
    times = time_options(cube)
    time_val = pd.Timestamp(args.time) if args.time else times[-1]

    bundles = get_models((args.species,))
    series = _forecast_series(cube, args.species, time_val,
                              {k: v for k, v in bundles.items() if k == args.species})
    if series.empty:
        logger.error("No predictions produced for %s at %s", args.species, time_val)
        raise SystemExit(1)

    out = series.reset_index()
    out["time"] = pd.Timestamp(time_val).strftime("%Y-%m-%d")
    out["species"] = args.species
    out.columns = ["lat", "lon", "yield", "time", "species"]

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.species}.csv"
    out.to_csv(path, index=False)
    logger.info("Exported %d cells to %s", len(out), path)


if __name__ == "__main__":
    main()