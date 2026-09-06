"""Dataset overview diagrams: raw (pre-processed) and processed distributions.

Produces two figures under ``docs/diagrams/``:

* ``data_overview_pre.png``  -- what the raw downloaded granules look like:
  temporal coverage per year, cloud/fill coverage, spatial coverage, and
  value distributions per satellite variable.
* ``data_overview_post.png`` -- what the data looks like AFTER preprocessing:
  remaining NaN fraction per feature in the cube, rows per year in the
  feature matrix, feature value distributions, and the per-species target
  (label) distributions used to train the models.

Usage (run these yourself after building the pipeline):
    python scripts/visualize_data.py                 # pre + post (post skipped
                                                     # until the cube is built)
    python scripts/visualize_data.py --stage pre     # raw granules only
    python scripts/visualize_data.py --stage post    # processed artifacts only
    python scripts/visualize_data.py --samples 24    # more raw samples/dataset
    python scripts/visualize_data.py --map-var sst   # which var for the map

Post-processing still has spatial before/after maps per dataset in
``scripts/visualize_process.py`` (stage_before_after_*.png).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PHILIPPINES_BBOX, RAW_DATA_DIR, TARGET_SPECIES  # noqa: E402
from src.data.fisheries import habitat_suitability_from_frame  # noqa: E402
from src.processing.grid import create_grid  # noqa: E402
from src.processing.pipeline import (  # noqa: E402
    CUBE_PATH,
    FEATURES_PATH,
    VARIABLE_MAP,
    load_cube,
)
from src.processing.spatial import standardize_axes  # noqa: E402
from src.utils.geo import select_bbox  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("vizdata")

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAR_LABELS = {
    "chlorophyll": "Chlorophyll-a (mg/m^3)",
    "sst": "Sea surface temp (degC)",
    "par": "PAR (E/m^2/day)",
    "kd490": "Kd490 (1/m)",
}

# Model-relevant features for the post-preprocessing NaN overview.
MODEL_FEATURES = [
    "sst",
    "sst_anom",
    "chl",
    "chl_roll30d",
    "chl_anom",
    "kd490",
    "par",
    "frontal_index",
    "productivity_index",
    "upwelling_proxy",
]


# --------------------------------------------------------------------------- #
# Raw (pre-processed) statistics
# --------------------------------------------------------------------------- #
def _year_of(path: Path) -> int | None:
    m = re.search(r"\.(\d{4})\d{2}\.\d{4}\d{2}\.", path.name)
    return int(m.group(1)) if m else None


def _read_raw_sample(dataset_key: str, path: Path) -> xr.DataArray:
    """Open one granule and subset to the study region (returns expected)."""
    src_var = VARIABLE_MAP[dataset_key]["src"]
    with xr.open_dataset(path, decode_times=True) as ds:
        if src_var not in ds.variables:
            raise ValueError(f"{src_var!r} missing from {path.name}")
        da = select_bbox(ds[src_var], PHILIPPINES_BBOX, pad=0.6)
    return standardize_axes(da)


def raw_stats(
    dataset_key: str, n_samples: int, grid, rng: np.random.Generator
) -> dict:
    """Aggregate raw-granule statistics for one dataset (sampled)."""
    files = sorted(RAW_DATA_DIR.glob(f"{dataset_key}/**/*.nc"))
    if not files:
        logger.warning("No raw granules for %s", dataset_key)
        return {}

    indices = np.unique(np.linspace(0, len(files) - 1, n_samples).astype(int))
    nan_fracs = []
    values: list[np.ndarray] = []
    coverage = np.zeros(grid.shape)
    coverage_cnt = 0
    lon_edges = np.linspace(
        grid.lons[0] - grid.res / 2, grid.lons[-1] + grid.res / 2, len(grid.lons) + 1
    )
    lat_edges = np.linspace(
        grid.lats[0] - grid.res / 2, grid.lats[-1] + grid.res / 2, len(grid.lats) + 1
    )

    for i in indices:
        try:
            da = _read_raw_sample(dataset_key, files[i])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s: %s", files[i].name, exc)
            continue
        vals = da.values.astype("float64")
        nan_fracs.append(float(np.isnan(vals).mean()))
        valid = ~np.isnan(vals)
        if valid.any():
            v = vals[valid].ravel()
            if len(v) > 150_000:
                v = rng.choice(v, 150_000, replace=False)
            values.append(v)
            lat2d, lon2d = np.meshgrid(da["lat"].values, da["lon"].values, indexing="ij")
            hist, _, _ = np.histogram2d(
                lon2d[valid].ravel(),
                lat2d[valid].ravel(),
                bins=[lon_edges, lat_edges],
            )
            coverage += hist.T
            coverage_cnt += 1

    year_counts = {y: 0 for y in range(2015, 2026)}
    for f in files:
        y = _year_of(f)
        if y in year_counts:
            year_counts[y] += 1

    coverage = coverage / max(coverage_cnt, 1)
    return {
        "dataset_key": dataset_key,
        "nan_fracs": np.asarray(nan_fracs),
        "values": np.concatenate(values) if values else np.asarray([]),
        "coverage": coverage,
        "year_counts": year_counts,
    }


def _plot_value_hist(ax, key: str, values: np.ndarray, color: str) -> None:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        ax.set_visible(False)
        return
    hist_kw = dict(bins=50, density=True, color=color, alpha=0.75)
    if key == "chlorophyll":
        v = v[v > 0]
        ax.hist(np.log10(v), **hist_kw)
        ax.set_xlabel("log10(chlorophyll-a)")
    else:
        ax.hist(v, **hist_kw)
        ax.set_xlabel(VAR_LABELS[key])
    ax.set_ylabel("density")


def figure_pre(stats: dict[str, dict]) -> plt.Figure:
    n = len(stats)
    if n == 0:
        raise RuntimeError("No raw data to plot. Fetch granules first.")
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.1, 1])

    # -- Temporal coverage per dataset ---------------------------------------- #
    ax_t = fig.add_subplot(gs[0, 0])
    years = list(range(2015, 2026))
    xs = np.arange(len(years))
    width = 0.8 / max(n, 1)
    for i, (key, st) in enumerate(stats.items()):
        counts = [st["year_counts"].get(y, 0) for y in years]
        ax_t.bar(xs + (i - (n - 1) / 2) * width, counts, width, label=VAR_LABELS.get(key, key))
    ax_t.set_xticks(xs, years, rotation=45)
    ax_t.set_title("Granules on disk per year (expected ~46/yr)")
    ax_t.legend(fontsize=9)

    # -- Cloud / invalid coverage per dataset ----------------------------------- #
    ax_n = fig.add_subplot(gs[0, 1])
    keys = list(stats)
    means = [float(st["nan_fracs"].mean()) * 100 for st in stats.values()]
    stds = [float(st["nan_fracs"].std()) * 100 for st in stats.values()]
    bars = ax_n.bar(range(n), means, yerr=stds, capsize=4, color=plt.cm.Set2.colors[:n])
    ax_n.set_xticks(range(n), [VAR_LABELS.get(k, k) for k in keys], rotation=15, ha="right")
    ax_n.set_ylabel("% invalid (cloud/fill) cells")
    ax_n.set_title("Raw cloud coverage per 8-day composite (mean +/- std)")
    for b, m in zip(bars, means):
        ax_n.text(b.get_x() + b.get_width() / 2, m + 1, f"{m:.1f}%", ha="center", fontsize=9)

    # -- Spatial coverage map ---------------------------------------------------- #
    ax_m = fig.add_subplot(gs[1, 0])
    map_key = next((k for k in ("chlorophyll", "sst", "par", "kd490") if k in stats), keys[0])
    cov = stats[map_key]["coverage"]
    grid = create_grid()
    lon_edges = np.linspace(grid.lons[0] - grid.res / 2, grid.lons[-1] + grid.res / 2, len(grid.lons) + 1)
    lat_edges = np.linspace(grid.lats[0] - grid.res / 2, grid.lats[-1] + grid.res / 2, len(grid.lats) + 1)
    im = ax_m.imshow(cov, origin="lower", cmap="YlGnBu", aspect="auto",
                     extent=[lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]])
    ax_m.set_title(f"Spatial coverage: fraction of sampled windows valid ({VAR_LABELS.get(map_key, map_key)})")
    ax_m.set_xlabel("Longitude")
    ax_m.set_ylabel("Latitude")
    fig.colorbar(im, ax=ax_m, fraction=0.046, pad=0.04, label="valid fraction")

    # -- Value distributions (one sub-panel per dataset) -------------------------- #
    outer = gs[1, 1]
    inner = outer.subgridspec(1, n)
    colors = plt.cm.Set2.colors[:n]
    for i, key in enumerate(keys):
        ax = fig.add_subplot(inner[0, i])
        _plot_value_hist(ax, key, stats[key]["values"], colors[i])
        ax.set_title(VAR_LABELS.get(key, key).split(" (")[0])

    fig.suptitle("Raw granules -- before preprocessing (2015-2025)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# --------------------------------------------------------------------------- #
# Processed (post-preprocessing) statistics
# --------------------------------------------------------------------------- #
def _load_matrix() -> pd.DataFrame | None:
    parquet = Path(str(FEATURES_PATH).replace(".pkl", ".parquet"))
    if parquet.exists():
        return pd.read_parquet(parquet)
    if FEATURES_PATH.exists():
        return pd.read_pickle(FEATURES_PATH)
    return None


def figure_post(cube, df: pd.DataFrame | None) -> plt.Figure:
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.05, 1.15])

    # -- Residual NaN fraction per feature (after gap-fill) ----------------------- #
    ax_n = fig.add_subplot(gs[0, 0])
    features = [v for v in MODEL_FEATURES if v in cube.data_vars]
    fracs = [float(cube[v].isnull().mean().item()) * 100 for v in features]
    order = np.argsort(fracs)
    ax_n.barh([features[i] for i in order], [fracs[i] for i in order], color="steelblue")
    ax_n.set_xlabel("% cells still NaN after preprocessing")
    ax_n.set_title("Cloud gaps remaining per feature (gap-fill removes most)")

    # -- Feature matrix rows per year --------------------------------------------- #
    ax_y = fig.add_subplot(gs[0, 1])
    if df is not None and not df.empty and "time" in df.columns:
        per_year = pd.to_datetime(df["time"]).dt.year.value_counts().sort_index()
        ax_y.bar(per_year.index.astype(str), per_year.values, color="seagreen")
        ax_y.set_ylabel("rows (cells x 8-day window)")
        ax_y.set_title(f"Feature matrix rows per year (total {len(df):,})")
    else:
        ax_y.text(0.5, 0.5, "Feature matrix not found -- build pipeline first",
                  ha="center", va="center")
        ax_y.axis("off")

    # -- Feature value distributions ---------------------------------------------- #
    outer = gs[1, 0]
    inner = outer.subgridspec(2, 2)
    plot_vars = [("sst", "SST (degC)", None), ("chl", "Chl-a (mg/m^3)", "log"),
                 ("kd490", "Kd490 (1/m)", None), ("par", "PAR (E/m^2/day)", None)]
    for i, (var, label, scale) in enumerate(plot_vars):
        ax = fig.add_subplot(inner[i // 2, i % 2])
        if var not in cube.data_vars:
            ax.set_visible(False)
            continue
        v = np.asarray(cube[var].values, dtype="float64")
        v = v[np.isfinite(v)]
        if scale == "log":
            v = v[v > 0]
            v = np.log10(v)
            label = f"log10 {label}"
        ax.hist(v, bins=60, density=True, color="#4477AA", alpha=0.8)
        ax.set_title(label, fontsize=9)
        ax.set_ylabel("density", fontsize=8)

    # -- Target (label) distributions per species --------------------------------- #
    outer2 = gs[1, 1]
    ax_l = fig.add_subplot(outer2)
    if df is not None and not df.empty and {"sst", "chl"} <= set(df.columns):
        cap = min(len(df), 200_000)
        sub = df[["sst", "chl"]].dropna().sample(cap, random_state=42)
        codes = [s["code"] for s in TARGET_SPECIES if s["code"] != "all"]
        colors = plt.cm.tab10(np.linspace(0, 1, len(codes) + 1)[1:])
        drawn = False
        for code, color in zip(codes, colors):
            try:
                series = habitat_suitability_from_frame(sub, code)
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping label for %s: %s", code, exc)
                continue
            scored = series.notna().sum()
            ax_l.hist(series.dropna().values, bins=40, density=True,
                      alpha=0.55, color=color, label=f"{code} (n={scored:,})")
            drawn = True
        if drawn:
            ax_l.legend(fontsize=8)
        ax_l.set_xlabel("habitat suitability (0-1 target)")
        ax_l.set_ylabel("density")
        ax_l.set_title("Training targets: habitat-suitability 'class' distribution per species")
    else:
        ax_l.text(0.5, 0.5, "Feature matrix not found", ha="center", va="center")
        ax_l.axis("off")

    fig.suptitle("After preprocessing -- features and training targets", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset overview diagrams")
    parser.add_argument("--stage", choices=["pre", "post", "both"], default="both")
    parser.add_argument("--samples", type=int, default=12,
                        help="Raw granule samples per dataset (default 12)")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    if args.stage in ("pre", "both"):
        stats = {
            key: raw_stats(key, args.samples, create_grid(), rng)
            for key in VARIABLE_MAP
        }
        stats = {k: v for k, v in stats.items() if v}
        if stats:
            fig = figure_pre(stats)
            out = OUT_DIR / "data_overview_pre.png"
            fig.savefig(out, dpi=140)
            plt.close(fig)
            logger.info("Saved %s", out)

    if args.stage in ("post", "both"):
        try:
            cube = load_cube(CUBE_PATH)
        except FileNotFoundError as exc:
            logger.warning("No cube built yet -- skipping post diagram: %s", exc)
            return
        df = _load_matrix()
        fig = figure_post(cube, df)
        out = OUT_DIR / "data_overview_post.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        logger.info("Saved %s", out)


if __name__ == "__main__":
    main()