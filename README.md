# OceanForecast Philippines

Dynamic oceanographic fisheries yield forecasting for Philippine waters.
Predict seasonal catch density / yield for target fish species using remote-
sensing ocean data (SST, chlorophyll-`a`, PAR, Kd490, bathymetry) rather than
basic historical catch totals.

```
┌─────────────────────────────────────────────────────────────┐
│                  Streamlit UI (map + controls)              │
├─────────────────────────────────────────────────────────────┤
│                  Prediction engine (XGBoost)                │
├───────────────┬───────────────┬─────────────────────────────┤
│  Feature cube │  Feature Eng  │  Spatial / temporal agg     │
├───────────────┴───────────────┴─────────────────────────────┤
│  Data layer: xarray feature cube (Zarr) + manifest          │
├───────────────────────┬─────────────────────────────────────┤
│  NASA Earthdata (OB.DAAC)  │  ETOPO 2022 bathymetry         │
└───────────────────────┴─────────────────────────────────────┘
```

## Repository layout

```
app/                  Streamlit application (app.py + components/)
configs/              YAML configs: species/regions, satellite datasets, model
data/
  raw/                Downloaded satellite granules, bathymetry, catch CSVs
  processed/          Feature cube (Zarr) + long feature matrix
  predictions/        Exported per-cell yield forecasts (CSV)
docs/diagrams/        Architecture & pipeline diagrams (Mermaid + real-data PNGs)
models/saved/         Trained model artifacts (<species>/model.pkl + bundle.pkl)
scripts/              fetch_satellite · run_pipeline · train_model · export · diagrams
src/
  config.py           Central Python configuration
  data/               satellite.py · bathymetry.py · fisheries.py
  processing/         grid · spatial · temporal · indices · pipeline
  models/             labels.py (providers) · train.py (XGBoost) · predict helpers
  utils/geo.py        Shared geospatial helpers
```

## Quickstart

```bash
# 1. Environment
python -m venv .venv && .venv\Scripts\activate   # Windows (use source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

# 2. NASA Earthdata credentials (free)
cp .env.example .env          # then fill in EARTHDATA_USERNAME / PASSWORD
python -c "import earthaccess; earthaccess.login()"

# 3. Fetch satellite data (resumable, shows a live progress bar)
python scripts/fetch_satellite.py                # all datasets 2015-2025
python scripts/fetch_satellite.py --no-progress  # headless/CI

# 4. Run the preprocessing pipeline (feature cube + matrix)
python scripts/run_pipeline.py

# 5. Train the per-species forecast models
python scripts/train_model.py                     # all species, chronological split
python scripts/train_model.py --species scad mackerel --force

# 6. Launch the interactive app
streamlit run app.py

# 7. Extras
python scripts/export_predictions.py --species sardinella       # CSV forecast
python scripts/generate_catch_template.py                       # BFAR/PSA schema template
python scripts/visualize_process.py                             # before/after pipeline PNGs
python scripts/benchmark_models.py                              # algorithm selection study (run before step 5)
python scripts/visualize_data.py                                # dataset overview diagrams (pre+post)
python scripts/visualize_models.py --benchmark                  # model comparison + algorithm evidence
```

## Which satellite products

| Feature | Dataset | Resolution | Temporal |
|---------|---------|------------|----------|
| Chlorophyll-`a` | `MODISA_L3m_CHL` | 9 km | 8-day |
| Sea surface temp | `MODISA_L3m_SST` | 9 km | 8-day |
| PAR | `MODISA_L3m_PAR` | 9 km | 8-day |
| Optical depth | `MODISA_L3m_Kd490` | 9 km | 8-day |
| Bathymetry | NOAA ETOPO 2022 | 1 arc-min | static |

Study region bbox: lon 116..127, lat 4.5..21.5 (Philippine EEZ).

## Preprocessing stages

1. **Raw granules** (L3 mapped, no time variable — timestamp from filename)
2. **Subset + regrid** to a 0.1 deg (~11 km) study grid (handles descending lat axes)
3. **Cloud gap-fill** via temporal interpolation (gaps > 4 steps stay NaN)
4. **Temporal features**: rolling means (7/14/30/90 d), monthly-climatology
   anomalies, month sin/cos seasonal encoding
5. **Derived indices**: frontal (SST gradient), productivity (log chl),
   upwelling proxy
6. **Spatial lags**: neighborhood means at 20 / 50 / 100 km

Output: `data/processed/ph_cube.zarr` + `feature_matrix.parquet` (fallback
`.pkl`; long format, rows = [lat, lon, time, features]).

## Target species (configurable in `configs/species_regions.yaml`)

Sardines (tamban) · Anchovies (dilis) · Indian mackerel (alumahan) ·
Round scad (galunggong) · Slipmouth (sap-sap) · All-species composite

## Model training

- **Labels**: two interchangeable providers (`src/models/labels.py`).
  - `HabitatSuitabilityProvider` — self-supervised 0-1 score from each species'
    SST/chlorophyll preference ranges (works with no external data).
  - `CatchVolumeProvider` — real BFAR/PSA quarterly catches joined per grid cell
    (`--label-kind prefer_catch` uses catches when present, else suitability).
- **Split** is chronological (no random CV leakage): train <= 2022, val 2023-2024,
  holdout 2025. Configured via flags (`--train-until`, `--val-start/end`, `--holdout`).
- **Features**: the 15 in `configs/model_config.yaml`; alias names (`chlor_a`,
  `bathymetry_depth`, ...) are resolved to pipeline columns (`chl`, `depth`, ...).
- **Artifacts** per species in `models/saved/<code>/`: `model.pkl`, `bundle.pkl`
  (features, median imputer, target transform, metrics), `metrics.json`.

## Streamlit app

Tabs: **Ocean map** (animated SST/chl/PAR/Kd490 + indices, with land silhouette) ·
**Yield forecast** (per-species predicted yield for the next 8-day window) ·
**FMA dashboard** (region rankings + condition trends) ·
**Sustainability** (current vs historical-peak flag per FMA, MSY-proxy).

The forecast tab uses the trained model when available and transparently falls
back to the habitat-suitability score otherwise.

## Roadmap

- [x] Stage 1 — data acquisition (satellite + bathymetry, resumable, progress bar)
- [x] Stage 2 — preprocessing pipeline + before/after diagrams
- [x] Stage 3 — hybrid labels (habitat suitability + BFAR/PSA catch provider)
- [x] Stage 4 — XGBoost training with chronological split + model artifacts
- [x] Streamlit app: ocean map, yield forecast, dashboard, sustainability flags
- [x] Prediction export (`scripts/export_predictions.py`)
- [ ] Refinements — hyperparameter search, per-species stacking, SHAP explainers,
      authenticated BFAR/NFRDI stock assessments, deployment

## Design decisions

Why these datasets, this preprocessing, these training targets, and this model
family is explained in `docs/decisions.md` (with the evidence diagrams
generated by `scripts/visualize_data.py` and `scripts/visualize_models.py`).

See `docs/diagrams/` for architecture, pipeline, and data before/after diagrams.