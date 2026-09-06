"""Create the BFAR/PSA catch-data CSV template.

Generates a header-only CSV at data/raw/fisheries/catch_data.csv following the
schema consumed by src.data.fisheries.load_catch_data:

    fma,species,year,quarter,volume_mt,lat,lon

Fill one row per FMA x species x quarter with the reported catch volume (tonnes)
and the fishing-ground / province-centroid coordinates. Fishing Management Area
names are provided as a row example per species.

Run: python scripts/generate_catch_template.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR, TARGET_SPECIES
from src.data.fisheries import CATCH_COLUMNS

FMAS = [
    "FMA1 - North Philippine Sea", "FMA2 - West Philippine Sea",
    "FMA3 - Sulu-Celebes Sea", "FMA4 - Visayan Sea",
    "FMA5 - West Sulu Sea", "FMA6 - East Philippine Sea",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=RAW_DATA_DIR / "fisheries" / "catch_data.csv")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and args.out.stat().st_size > 0:
        raise SystemExit(f"{args.out} already exists; refusing to overwrite.")

    rows = []
    for species in TARGET_SPECIES:
        if species["code"] == "all":
            continue
        for fma in FMAS:
            rows.append({
                "fma": fma,
                "species": species["code"],
                "year": 2020, "quarter": 1, "volume_mt": None,
                "lat": 13.0, "lon": 122.0,
            })
    pd.DataFrame(rows, columns=CATCH_COLUMNS).to_csv(args.out, index=False)
    print(f"Template written to {args.out}")
    print("Fill `volume_mt`, `year`, `quarter`, and real lat/lon per FMA.")


if __name__ == "__main__":
    main()