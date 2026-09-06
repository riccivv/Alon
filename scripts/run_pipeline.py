"""Run the full preprocessing pipeline and save the feature matrix.

Usage:
    python scripts/run_pipeline.py                       # all years, all vars
    python scripts/run_pipeline.py --years 2018 2019     # specific years
    python scripts/run_pipeline.py --variables sst chl   # subset of variables
    python scripts/run_pipeline.py --no-spatial-lags     # skip lag features
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DATA_DIR  # noqa: E402
from src.processing.pipeline import run_pipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Philippine feature cube")
    parser.add_argument("--years", type=int, nargs="*", default=None,
                        help="Only process specific years (default: all)")
    parser.add_argument("--variables", nargs="*", default=None,
                        help="Satellite variables to process (default: all)")
    parser.add_argument("--no-spatial-lags", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    cube, df = run_pipeline(
        years=args.years,
        variables=args.variables,
        include_spatial_lags=not args.no_spatial_lags,
        cache=not args.no_cache,
    )

    logger.info("Feature cube: %s", cube)
    logger.info("Feature matrix: %s rows x %s cols", *df.shape)
    logger.info("Saved under %s", PROCESSED_DATA_DIR)


if __name__ == "__main__":
    main()