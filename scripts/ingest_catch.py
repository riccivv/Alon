"""Ingest PSA fisheries volume exports into the catch-data convention.

Reads every ``*_Volume of Production by Geolocation*`` xlsx in ``data/raw/``
(the Commercial + Marine Municipal matrices; the "(2)" full-species exports
are preferred over the 7-species originals) and aggregates them into
``data/raw/fisheries/catch_data.csv`` with the schema used by the training
pipeline:

    fma, species, year, quarter, volume_mt, lat, lon

Processing notes
----------------
- Only province-level geolocations are kept (``....`` rows). ``PHILIPPINES``
  and regional totals are ignored. ``National Capital Region (NCR)`` is kept
  as its own anchor (it reports landing-centre volumes).
- PSA species are bucketed to the 5 target species; two PSA rows map onto
  ``sardinella`` (Tamban + Tunsoy) and ``mackerel`` (Alumahan + Hasa-hasa).
  All other PSA taxa (tuna, squid, big-eyed scad, ...) are dropped.
- ``.`` (category not applicable) and ``..`` (data not available) both become
  missing values (NaN), per decision docs. Offshore/inland provinces end up
  with no labels and are naturally excluded at training time.
- Commercial and Municipal are summed (skipna) so a sector reporting alone is
  kept.
- Province centroids come from ``configs/province_centroids.csv`` (PSA/OCHA
  adm2 centre points plus manual anchors for NCR and the HUC "City of" rows).
  Anchor coordinates are rounded to the 0.1 deg grid cell, then volumes are
  summed per (species, year, quarter, cell) so the downstream ``many_to_one``
  join in ``labels.py`` stays collision-free.
- ``Sulu`` is reported three ways in the export (Region IX, BARMM, plus an
  annotated footnote row); variants are unified by taking the maximum value
  per (species, year, quarter) to avoid double counting.
- FMA comes from the curated ``configs/fma_province_map.yaml`` lookup.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from openpyxl import load_workbook

REPO = Path(__file__).resolve().parent.parent
RAW_DIR = REPO / "data" / "raw"
DEFAULT_RAW = RAW_DIR / "fisheries"
CENTROIDS_CSV = REPO / "configs" / "province_centroids.csv"
FMA_YAML = REPO / "configs" / "fma_province_map.yaml"
OUT_CSV = RAW_DIR / "fisheries" / "catch_data.csv"

RESULTS_COLUMNS = ["fma", "species", "year", "quarter", "volume_mt", "lat", "lon"]

# PSA species row label -> target species code. Rows with several PSA labels
# per code are summed. Anything not listed is ignored.
SPECIES_ALIASES = {
    "..Anchovies (Dilis)": "anchovy",
    "..Bali sardinella (Tamban)": "sardinella",
    "..Fimbriated sardines (Tunsoy)": "sardinella",
    "..Indian mackerel (Alumahan)": "mackerel",
    "..Indo-pacific mackerel (Hasa-hasa)": "mackerel",
    "..Roundscad (Galunggong)": "scad",
    "..Slipmouth (Sapsap)": "slipmouth",
}

# Unify the three PSA spellings of Sulu.
PROVINCE_ALIASES = {
    "Sulu a/": "Sulu",
    "Sulu a/\nStill part of BARMM prior to 2026\n": "Sulu",
}

logger = logging.getLogger("ingest_catch")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def find_matrix_files(raw_dir: Path) -> list[Path]:
    hits = []
    for path in sorted(raw_dir.glob("*.xlsx")):
        try:
            wb = load_workbook(path, read_only=True, data_only=True)
            ws = wb.active
            head = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
            title = (head[0] or "") if head else ""
            if "Volume of Production by Geolocation" in str(title):
                hits.append(path)
            wb.close()
        except Exception:
            logger.warning("Cannot open %s as spreadsheet; skipping.", path.name)
    return hits


def parse_matrix(path: Path) -> pd.DataFrame:
    """Return a long frame: geo, species, year, quarter, volume_mt."""
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))

    if not rows:
        return pd.DataFrame(columns=["geo", "species", "year", "quarter", "volume_mt"])

    quarters_idx = next(
        (i for i, r in enumerate(rows) if any(v == "Quarter 1" for v in r)),
        None,
    )
    if quarters_idx is None:
        raise ValueError(f"{path.name}: could not locate the quarter header row")

    def is_year(val) -> bool:
        if isinstance(val, (int, float)):
            return 2000 <= val <= 2100
        if isinstance(val, str) and val.strip().isdigit():
            return 2000 <= int(val.strip()) <= 2100
        return False

    years_idx = next(
        (i for i in range(quarters_idx - 1, max(0, quarters_idx - 4) - 1, -1)
         if any(is_year(v) for v in rows[i])),
        quarters_idx - 1,
    )
    years_row = rows[years_idx]
    quarters_row = rows[quarters_idx]

    col_year: dict[int, int] = {}
    col_quarter: dict[int, int] = {}
    for ci, val in enumerate(years_row):
        if is_year(val):
            col_year[ci] = int(str(val).strip())
    for ci, val in enumerate(quarters_row):
        if isinstance(val, str) and val.startswith("Quarter"):
            col_quarter[ci] = int(val.split()[-1])

    # Years are printed once per 4-quarter block, so every quarter column needs
    # its year resolved from the nearest preceding year label.
    year_starts = sorted(col_year)
    col_yq: dict[int, tuple[int, int]] = {}
    for ci, q in col_quarter.items():
        year = next(
            (col_year[s] for s in reversed(year_starts) if s <= ci), None
        )
        if year is not None:
            col_yq[ci] = (year, q)

    records: list[dict] = []
    geo = None
    for r in rows[quarters_idx + 1 :]:
        if not r:
            continue
        a = r[0]
        b = r[1] if len(r) > 1 else None
        if isinstance(a, str):
            geo = a  # forward-fill geolocation (col A present only on block head)
        if not isinstance(b, str):
            continue
        if b not in SPECIES_ALIASES:
            continue
        if not (geo and geo.startswith("....")):
            if geo != "..National Capital Region (NCR)":
                continue
        species = SPECIES_ALIASES[b]
        for ci, (year, quarter) in col_yq.items():
            cell = r[ci] if ci < len(r) else None
            if cell in (None, ".", "..", ""):
                continue
            try:
                vol = float(cell)
            except (TypeError, ValueError):
                logger.debug("%s: non-numeric %r at row %s", path.name, cell, ci)
                continue
            records.append(
                {
                    "geo_raw": geo,
                    "species": species,
                    "year": year,
                    "quarter": quarter,
                    "volume_mt": vol,
                }
            )
    wb.close()
    return pd.DataFrame(records)


def normalise_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # PSA reports Sulu in three equivalent ways (Region IX "Sulu", BARMM
    # "Sulu a/", plus the footnoted duplicate). They are the same province, so
    # keep the maximum volume per (species, year, quarter) instead of letting a
    # later groupby sum them together.
    is_sulu = df["geo_raw"].str.lstrip(".").str.startswith("Sulu")
    if is_sulu.any():
        sulu_merged = (
            df[is_sulu]
            .groupby(["species", "year", "quarter"], as_index=False)
            .agg({"geo_raw": "first", "volume_mt": "max"})
        )
        df = pd.concat([df[~is_sulu], sulu_merged], ignore_index=True)

    df["geo"] = df["geo_raw"].str.lstrip(".").map(lambda s: PROVINCE_ALIASES.get(s, s))
    df = df.drop(columns="geo_raw")

    # Collapse PSA rows that map onto the same target code (sardinella, mackerel).
    df = (
        df.groupby(["geo", "species", "year", "quarter"], as_index=False)
        .agg({"volume_mt": lambda s: s.sum(min_count=1)})
        .dropna(subset=["volume_mt"])
    )
    return df


def build_lookups() -> tuple[pd.DataFrame, dict[str, int]]:
    centroids = pd.read_csv(CENTROIDS_CSV)

    with open(FMA_YAML) as f:
        fma_cfg = yaml.safe_load(f)
    fma_of: dict[str, int] = {}
    for fma_num, block in fma_cfg["fma_province_map"].items():
        for prov in block["provinces"]:
            fma_of[prov["name"]] = int(fma_num)

    return centroids, fma_of


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="data/raw directory")
    ap.add_argument("--out", type=Path, default=OUT_CSV, help="output CSV path")
    args = ap.parse_args(argv)

    files = find_matrix_files(args.raw_dir)
    if not files:
        logger.error("No PSA 'Volume of Production by Geolocation' xlsx found in %s", args.raw_dir)
        return 1
    logger.info("Found %d PSA matrix file(s): %s", len(files), ", ".join(p.name for p in files))

    frames: list[pd.DataFrame] = []
    for path in files:
        frame = parse_matrix(path)
        frame = normalise_frame(frame)
        # Tag the table type from the file itself is unreliable (same matrix
        # layout for both); deduce it from the sheet title via the workbook?
        # PSA internal codes keep this unambiguous:
        #   Commercial = '2E4GVCP0', Marine Municipal = '2E4GVMP0'.
        code_hits = [c for c in ("2E4GVCP0", "2E4GVMP0") if c.upper() in path.name.upper()]
        if len(code_hits) != 1:
            logger.warning("%s: cannot infer table type (%s); treating as extra source", path.name, code_hits)
            frame["_type"] = "extra"
        else:
            frame["_type"] = "commercial" if code_hits[0] == "2E4GVCP0" else "municipal"
        frames.append(frame)
        logger.info(
            "%s: %d rows parsed (species %s)",
            path.name,
            len(frame),
            ", ".join(sorted(frames[-1]["species"].unique())),
        )

    all_rows = pd.concat(frames, ignore_index=True)

    # Step 1: within each table type, dedupe superset-vs-subset files (the
    # 7-species originals vs the "(2)" full-species exports) by taking max.
    deduped = (
        all_rows.groupby(
            ["_type", "geo", "species", "year", "quarter"], as_index=False
        )["volume_mt"]
        .agg("max")
    )
    # Step 2: sum commercial + municipal.
    totals = (
        deduped.groupby(["geo", "species", "year", "quarter"], as_index=False)
        .agg({"volume_mt": lambda s: s.sum(min_count=1)})
        .dropna(subset=["volume_mt"])
    )

    centroids, fma_of = build_lookups()
    cent = centroids.set_index("psa_name")

    out_rows: list[dict] = []
    for row in totals.itertuples(index=False):
        geo = row.geo
        if geo not in fma_of:
            logger.warning("No FMA mapping for %r; dropping row", geo)
            continue
        if geo not in cent.index:
            logger.warning("No centroid anchor for %r; dropping row", geo)
            continue
        lat, lon = cent.loc[geo, "lat"], cent.loc[geo, "lon"]

        out_rows.append(
            {
                "fma": fma_of[geo],
                "species": row.species,
                "year": int(row.year),
                "quarter": int(row.quarter),
                "volume_mt": float(row.volume_mt),
                "lat": lat,
                "lon": lon,
            }
        )

    catches = pd.DataFrame(out_rows, columns=RESULTS_COLUMNS)

    # Snap to the 0.1 deg grid and sum per (species, year, quarter, cell) to
    # keep the downstream many_to_one join unique.
    catches["lat"] = np.round(catches["lat"].to_numpy(), 1)
    catches["lon"] = np.round(catches["lon"].to_numpy(), 1)
    cells = (
        catches.groupby(
            ["species", "year", "quarter", "lat", "lon"], as_index=False
        )["volume_mt"]
        .agg("sum")
        # keep the FMA of the largest contributor to the cell
    )
    fma_of_cell = catches.assign(
        rank=catches.groupby(["species", "year", "quarter", "lat", "lon"])["volume_mt"].transform("max")
    )
    fma_of_cell = (
        fma_of_cell[fma_of_cell["volume_mt"] == fma_of_cell["rank"]]
        .drop_duplicates(["species", "year", "quarter", "lat", "lon"])[["species", "year", "quarter", "lat", "lon", "fma"]]
    )
    catches = cells.merge(fma_of_cell, on=["species", "year", "quarter", "lat", "lon"], how="left")
    catches = catches[RESULTS_COLUMNS].sort_values(
        ["species", "fma", "year", "quarter", "lat", "lon"]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    catches.to_csv(args.out, index=False)

    # Sanity report
    logger.info("Wrote %s (%d rows)", args.out, len(catches))
    per_species = (
        catches.groupby("species")["volume_mt"].agg(["count", "sum"])
    )

    for sp, cnt, s in per_species.itertuples():
        logger.info("  %-10s rows=%6d cumulative tonnes=%12.1f", sp, cnt, s)
    years = (catches["year"].min(), catches["year"].max())
    logger.info("Year range: %d-%d", *years)
    return 0


if __name__ == "__main__":
    sys.exit(main())