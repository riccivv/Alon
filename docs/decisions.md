# Design Decisions

This document explains *why* the OceanForecast Philippines pipeline looks the
way it does: the data sources, the preprocessing choices, the training
targets, and the model family. It is the readable, evidence-based companion to
the code and to the diagnostic diagrams produced by
`scripts/visualize_data.py` and `scripts/visualize_models.py`.

See also: `configs/satellite_datasets.yaml`, `configs/species_regions.yaml`,
`configs/model_config.yaml`, and the architecture diagrams in
`docs/diagrams/`.

---

## 1. Why these datasets

The forecasting target is **pelagic fish yield density** in Philippine waters.
Pelagic forage fish aggregate where the physical and biological environment is
favourable, so the predictors are *oceanographic satellite fields* rather than
raw catch time series. The four moving variables + one static field were chosen
as a minimal, established set of drivers.

| Feature | Dataset | Why it matters for fisheries |
|---|---|---|
| Chlorophyll-a | MODIS-Aqua L3m CHL | Proxy for primary production / food availability; log-transformed as a productivity index |
| Sea surface temp | MODIS-Aqua L3m SST | Thermal niche; each target species has an optimal SST band; SST gradients define fronts (aggregation sites); cold anomalies signal upwelling |
| PAR | MODIS-Aqua L3m PAR | Sunlight available for phytoplankton growth; drives the productivity signal |
| Kd490 | MODIS-Aqua L3m KD | Water turbidity / optical depth; near-shore habitat quality for small pelagics |
| Depth, slope, dist-to-coast | NOAA ETOPO 2022 bathymetry | Static benthic habitat: shelf vs slope vs open ocean; inshore/offshore separation |

**Why MODIS-Aqua specifically:**
- A single, consistent, cross-calibrated sensor (Aqua, 2002–present) avoids the
  inter-sensor calibration jumps of multi-sensor SST/CHL analyses.
- 9 km spatial resolution matches the scale of Philippine municipal & coastal
  fisheries (thin shelves, small islands) without the cost of 4 km daily data.
- **8-day composites** average out most cloud coverage (monsoon clouds are the
  main noise source in the tropics) while still resolving intra-seasonal
  change (see the raw cloud-coverage panel in `data_overview_pre.png`).
- Free and public via NASA Earthdata; L3 mapped granules have a time-stamped
  filename even though the NetCDF itself has no time dimension.

**Why the study region / grid:**
- Bounding box `116..127 lon × 4.5..21.5 lat` = Philippine EEZ (`src/config.py`).
- Regridded to a uniform **0.1° (~11 km)** grid: finer than the 9 km sensors
  would not add information; coarser would smear the patchy coastal signal.
- Land is masked using bathymetry (depth < 0 = ocean), so models never see
  lakes/land cells.

**Alternatives considered:** GEBCO is referenced in configs for bathymetry;
NOAA ETOPO 2022 was preferred for its 1 arc-min resolution and OPeNDAP
subsetting. Higher-frequency products (4 km daily) were deferred for storage/
cloud reasons. In situ buoy data is too sparse over PH waters to be a predictor.
Fishing a different SST source changed later — see "Known limitations".

---

## 2. Why this preprocessing

- **Cloud gap-fill along time** (`src/processing/temporal.py`): L3 8-day
  composites still have cloud gaps near the coasts. Gaps are linearly
  interpolated in time; gaps longer than 4 windows stay NaN (avoid smoothing
  real seasonal transitions). The effect of the fill is visible as the drop in
  % invalid cells between `data_overview_pre.png` and the "residual NaN per
  feature" panel in `data_overview_post.png`.
- **Temporal features**: rolling means (7/14/30/90 d) capture recent regime;
  deviations from the **monthly climatology** encode anomalies (used for the
  upwelling proxy: waters colder than their climatology = freshly upwelled);
  month sin/cos give the model the calendar seasonality explicitly.
- **Derived indices** (`src/processing/indices.py`): frontal index (SST
  gradient magnitude), productivity index (log chlorophyll), upwelling proxy.
  These compress raw fields into the quantities fishers and oceanographers
  actually reason about.
- **Spatial lags** at 20/50/100 km: neighbourhood context (a fish school sits
  at a front, not just at a pixel). Coastal cells average only their valid
  neighbours.

The result is the long feature matrix `data/processed/feature_matrix.parquet`
(rows = lat × lon × time), which is what every model is trained on.

---

## 3. Why these training targets (labels)

Forecasting requires a continuous target for every grid cell at every time.
Two providers are implemented behind one interface (`src/models/labels.py`):

1. **Habitat-suitability pseudo-label (used now).** A 0–1 score = geometric
   mean of the species' SST and log-chlorophyll memberships inside its
   preference bands (`configs/species_regions.yaml`). It is self-supervised
   (no external data), biologically plausible, and lets the pipeline train and
   the app forecast today. *It is a relative habitat-quality score, NOT an
   absolute tonnage prediction.*
2. **Real catch volume (BFAR/PSA, active once catch_data.csv exists).** When
   `data/raw/fisheries/catch_data.csv` exists, quarterly catch tonnage is joined
   to grid cells and used instead (`--label-kind prefer_catch` falls back to
   suitability when catches are absent). The target distribution is then
   heavy-tailed, so catch targets are modelled on `log1p`.

This hybrid lets the system ship now and improve automatically the moment
catch statistics are ingested — `scripts/ingest_catch.py` turns the raw PSA
exports into that file.

---

## 3a. Real catch data: where it comes from and how it is normalised

The CSV schema (`fma, species, year, quarter, volume_mt, lat, lon`) is fed by
`scripts/ingest_catch.py`, which reads the PSA OpenStat exports the user pulled:

- **Commercial Fisheries: Volume of Production** (`2E4GVCP0 ...xlsx`) and
  **Marine Municipal Fisheries: Volume of Production** (`2E4GVMP0 ...xlsx`),
  both in PX-Web matrix layout: rows = Geolocation × Species, columns =
  2015–2025 × 4 quarters. The "(2)" exports list all 31 PSA taxa; the
  7-species originals are kept but ignored when the superset is present.
- Ingest keeps **province-level** geolocations only (national/regional totals
  are dropped), plus `NCR` as its own anchor. Unknown geolocations are logged
  and dropped.

Decisions built into the ingest:

| Decision | Rationale |
|---|---|
| Species bucketing | PSA rows are mapped to the 5 target codes; two PSA rows feed `sardinella` (Tamban + Tunsoy) and `mackerel` (Alumahan + Hasa-hasa); `Roundscad`→scad, `Anchovies`→anchovy, `Slipmouth (Sapsap)`→slipmouth. Non-target taxa (tuna, squid, big-eyed scad, ...) are excluded. |
| `.` and `..` both → missing | Per user choice: a "category not applicable" cell (inland province) is treated the same as "data not available" — both are absent labels, so landlocked CAR/Luzon provinces simply have no training rows. |
| Commercial + Municipal summed | Total marine-capture volume per province; `sum(min_count=1)` keeps a sector that reports alone. |
| Province → lat/lon anchor | PSA/OCHA `phl_admin2` **centroid** (configs/province_centroids.csv), with manual anchors for NCR and the HUC "City of" rows. Anchors are rounded to the 0.1° grid cell; volumes are then summed *per cell* so the label join's `many_to_one` invariant holds. |
| Province → FMA | Curated static lookup (`configs/fma_province_map.yaml`, aligned with the BFAR FMA-263 zones). FMA is *not* consumed by the training join (src/models/labels.py merges on lat/lon only), so border-province assignments are cosmetic for the app's FMA breakdown/sustainability panels — refined if an official FMA polygon set appears. |
| Sulu dedup | PSA files Sulu in three ways (Region IX, BARMM, and a footnoted duplicate); variants are unified by taking the **max** per (species, year, quarter) to avoid double counting. |
| Sanity check | Province sums reconcile exactly with the exports' `PHILIPPINES` national totals (verified during development). |

Notes on expectations after ingestion:

- **R² will drop** from ≈0.999 (habitat proxy) to a realistic level on real
  tonnage — that is the point; the benchmark and per-species metrics are only
  meaningful once `--label-kind` actually uses catch labels (`metrics.json`
  records which label policy was used).
- The **model-selection benchmark** should be re-run *after* ingestion so the
  six candidate algorithms are compared on real (log1p) catch targets rather
  than on the near-degenerate habitat labels (interrupted run before ingest).
  *Done — see §3b.*

---

## 3b. Model selection on real catch labels (post-ingestion benchmark)

`scripts/benchmark_classification.py` is the post-ingestion re-run promised in
§3a. Each algorithm is fit twice on the **same rows and the same chronological
split** (match the §§2/4 scheme): once as a **regressor** on the continuous
log1p catch target (the primary measure) and once as a **classifier** on a
binarised high-/low-yield version (threshold = the species' **train-window
median**, so no future leakage). This makes explicit that the two metric
families answer different questions:

- **R² / RMSE / MAE** — how well predictions track *absolute* tonnage level.
  This is the primary measure, and the sort key for the ranking.
- **Accuracy / Precision / Recall / F1 / ROC-AUC / Spearman / Hit@k** — how
  well predictions *rank* cells into hotspots at the high-yield threshold.

Results: median across the 5 species, validation window 2023–2024 (saved in
`docs/model_selection_results_unified.csv` and `docs/diagrams/model_classification.png`):

| Algorithm | R² | RMSE | Acc | Prec | Rec | F1 | ROC-AUC | Spear. | Hit@k |
|---|---|---|---|---|---|---|---|---|---|
| XGBoost | **0.585** | 1.210 | 0.789 | 0.715 | 0.839 | 0.760 | **0.876** | 0.754 | 0.767 |
| LightGBM | 0.584 | **1.205** | **0.792** | 0.713 | **0.849** | **0.765** | 0.874 | 0.751 | 0.742 |
| CatBoost | 0.571 | 1.209 | 0.772 | **0.723** | 0.837 | 0.755 | 0.875 | 0.750 | 0.760 |
| RandomForest | 0.554 | 1.220 | 0.767 | 0.707 | 0.837 | 0.751 | 0.859 | 0.705 | 0.760 |
| LogisticRegression* | 0.012 | 1.804 | 0.568 | 0.530 | 0.620 | 0.565 | 0.601 | 0.211 | 0.530 |
| NoSkill | -0.027 | 1.833 | 0.590 | 0.000 | 0.000 | 0.000 | 0.500 | NaN | 0.445 |

\* `LogisticRegression` name is historical — the regressor stand-in is a scaled
linear model (see the algorithm table in §4), so its R²≈0.01 is the honest
"linear baseline" result.

Reading the table:

- **The three GBMs are statistically tied** (ΔR² ≤ 0.01, ΔROC-AUC ≤ 0.002).
  XGBoost tops the primary axis (R²) and ROC-AUC, LightGBM edges F1/RMSE/recall.
  The decision therefore stays **XGBoost**: deterministic seeds, first-class
  feature importance, and fastest of the group on this row count.
- **RandomForest is a clear step behind** on both metric families (nonlinear
  structure exists, but bagged averaging is weaker than boosting).
- **Logistic regression collapses** (R² ≈ 0, ROC-AUC ≈ 0.60 ≈ rank guessing):
  the interactions/tree structure the GBMs exploit are essential.
- **NoSkill sits at the theoretical floor** (ROC-AUC 0.5, R² ≤ 0, zero precision
  because it predicts the all-majority class), confirming the models add skill.
- ROC-AUC ≈ 0.87-0.88 and Spearman ≈ 0.75 (GBMs) mean the forecast *ranks*
  cells credibly: the top-k (Hit@k ≈ 0.76) regularly land on genuinely high-yield
  cells, even though R² says tonnage *level* is only partially explained.

Per-species detail is in the CSV; the conclusions hold on the individual
sard/anchovy/mack/scad/slipspecies rows as well (no single-species flip of the
ordering).

---

## 4. Why this model (XGBoost)

### Choosing the algorithm -- model-selection study (run BEFORE final training)

Several candidate algorithms are benchmarked on identical inputs (same 15
features, chronological split) by
`scripts/benchmark_models.py` **before** the per-species models are trained.
It saves `docs/model_selection_results.csv` and `docs/diagrams/model_selection.png`.
*Since real catch labels exist, the authoritative re-run is
`scripts/benchmark_classification.py` on log1p catch targets — see §3b; the
conclusion is unchanged.*
The winner is set as `model.type` in `configs/model_config.yaml`.

| Algorithm | Why it is in the study |
|---|---|
| **XGBoost** | Incumbent: gradient-boosted trees, handles interactions + missing values, CPU-friendly at our row count |
| **LightGBM** | Strongest XGBoost alternative; histogram-based (fast), near-identical accuracy class |
| **CatBoost** | Third major GBM; robust to label noise, built-in regularization (needs `pip install catboost`) |
| **Random Forest** | Bagged-tree baseline that GBMs are expected to beat; guards against overfitting claims |
| **Ridge (linear)** | Honest linear sanity check (logistic regression is a classifier; suitability is continuous, so a scaled linear regressor stands in) |
| **No-skill (mean)** | Floor: predict the training mean; proves the other models add real predictive skill (Isolation Forest is an anomaly detector, not a regressor, and cannot be benchmarked for this target) |

Winning logic: primary metric = validation R2/RMSE on the chronological
holdout; training time is secondary and shown on the same figure.

### The manual-default justification (fallback when the study is skipped)

### Problem traits
- **Tabular** features (~15 real-valued columns), no image/sequence structure yet.
- **Large but not massive** row count (millions of cell-window rows) — fits in RAM, no distributed training needed.
- **Nonlinear** relationships and **feature interactions** (SST × chl × depth near coasts).
- **Missing values** persist at edges despite gap-filling.
- **Requirement**: explainable-ish for stakeholders (feature importance), fast CPU training, reproducible seeds.

### Options considered

| Approach | Verdict | Why |
|---|---|---|
| Linear / ridge | Rejected | Cannot capture interactions without hand-engineering every pair |
| Random forest | Viable | Good baseline, but ensembles of trees predict by *averaging*; GBMs are usually stronger for heterogeneous tabular regression |
| LightGBM | Strong close second | Faster histograms; near-identical accuracy to XGB on this kind of data |
| **XGBoost** | **Selected** | Mature, deterministic, handles missing values + interactions well, first-class feature importance, CPU-friendly at our row count |
| Deep learning (ConvLSTM) | Deferred | Needs spatiotemporal serialization + GPU; typically buys little over GBMs at this data volume; revisit for sequence forecasts post-MVP |

`requirements.txt` already pins `xgboost` (primary) and `lightgbm`
(benchmark/alternative). `scripts/visualize_models.py --benchmark` retrains
XGBoost vs LightGBM vs RandomForest on a down-sampled train slice and writes
`model_algorithm_benchmark.png` as direct evidence for the choice.

### Validation strategy
- **Chronological split** (not random): `train ≤ 2022`, `val = 2023–2024`,
  `holdout = 2025`. Random K-fold would leak the future into the past and
  overstate accuracy for a forecasting system.
- **Metrics on the transformed target space** (habitat: raw 0–1; catch:
  log1p): RMSE, MAE, R2. Reported per species in `models/saved/<code>/metrics.json`
  and visualised in `model_comparison.png` / `model_fit.png`.
- Possible limitation: pre-catch training used habitat labels, so val/holdout
  R2 measured fit to the *environmental suitability proxy*, not "true" fish
  tonnage (see the "Reading the near-perfect R2" note below, which applies to
  that habitat phase only). With real catch labels ingested (§3a/3b and
  `models/saved/<code>/metrics.json` reporting `label_kind=catch`), the
  per-species R2 is now a genuine tonnage-skill figure and sits at the realistic
  0.39–0.65 range described in §3a.
- **Reading the near-perfect R2 (≈0.999).** This is *expected and by design*,
  not a warning sign in itself: suitability labels are a smooth function of
  `sst` and `chl`, and those same fields are model inputs — XGBoost essentially
  recovers the label-generating formula. The metric therefore confirms the
  model learns the mapping, but says nothing about real catch. What matters for
  credibility is (a) the chronological (leakage-free) split, (b) the
  error is tiny in 0-1 label units, and (c) real-catch labels will lower R2 to
  meaningful, realistic values.

### Hyperparameters
A single manual configuration (`configs/model_config.yaml`): 300 trees, depth
6, lr 0.05, subsampling 0.8 — chosen as a sane default. Systematic search
(`Optuna`) is roadmap work; the benchmark flag gives a first comparison cheaply.

---

## 5. Known limitations / next steps

- Predictions are **relative yield hotspots** until catch-based labels and
  ground-truth validation land. Treat the map as "where to look", not "tonnes expected".
- SST was originally the on-prem MODIS L3 product; if that host becomes
  unreachable the acquisition is switched to a NASA cloud-hosted L4 SST and
  resampled to the same 8-day cadence (same `sst` feature, so no downstream change).
- Next: hyperparameter search, per-species stacking, SHAP explainers,
  BFAR/NFRDI stock assessments, probabilistic forecasts, deployment.