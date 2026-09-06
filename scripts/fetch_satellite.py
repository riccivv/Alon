"""Fetch NASA satellite granules for the Philippine study region.

Usage:
    python scripts/fetch_satellite.py                  # all datasets, default range
    python scripts/fetch_satellite.py --dataset sst --start 2018-01-01 --end 2018-12-31
    python scripts/fetch_satellite.py --dataset chlorophyll --max-files 10   # smoke test

Downloads land in data/raw/<dataset>/, resumable (skips existing granules).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SATELLITE_DATASETS  # noqa: E402
from src.data.satellite import login, download_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fetch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NASA ocean-color granules")
    parser.add_argument("--dataset", choices=list(SATELLITE_DATASETS),
                        default=None, help="Which dataset to fetch (default: all)")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit granule count (smoke testing)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable the live progress bar (e.g. for CI/headless)")
    parser.add_argument("--cloud", action="store_true",
                        help="Use the AWS cloud-hosted collection (via "
                             "obdaac-tea.earthdatacloud.nasa.gov) instead of the "
                             "on-prem oceandata host. Works for chlorophyll, par, "
                             "kd490; NOT available for sst.")
    args = parser.parse_args()

    login()

    keys = [args.dataset] if args.dataset else list(SATELLITE_DATASETS)
    for key in keys:
        if args.cloud and key == "sst":
            logger.warning("SST has no cloud-hosted collection; "
                           "skipping it (use on-prem fetch).")
            continue
        logger.info("=== Fetching {%s}: %s -> %s ===", key, args.start, args.end)
        new = download_dataset(
            key,
            start_date=args.start,
            end_date=args.end,
            max_files=args.max_files,
            threads=args.threads,
            show_progress=not args.no_progress,
            cloud_hosted=args.cloud,
        )
        logger.info("Downloaded %s new granules for {%s}.", len(new), key)


if __name__ == "__main__":
    main()