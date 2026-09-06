# Diagrams

These diagrams document the OceanForecast Philippines system at three levels:
**(1) architecture**, **(2) data flow**, and **(3) data **before/after** evidence
generated from real satellite granules.

## How to view

| File | Type | What it shows |
|------|------|---------------|
| `architecture.mmd` | Mermaid | Full system: data sources -> acquisition -> preprocessing -> ML -> Streamlit UI |
| `data_pipeline.mmd` | Mermaid | Stage 1 & 2 as a flow with storage artifacts at each step |
| `processing_stages.mmd` | Mermaid | Concept-level before/after at every preprocessing stage |
| `model_flow.mmd` | Mermaid | Offline training vs. online prediction, and the UI outputs |
| `stage_before_after_*.png` | PNG | **Real data** at each stage (raw / gap-filled / regridded / derived) |

Mermaid files render in: VS Code (Mermaid extension), GitHub, GitLab, or
[mermaid.live](https://mermaid.live).

The PNG figures are generated from actual MODIS-Aqua granules by:

```bash
python scripts/visualize_process.py            # all datasets
python scripts/visualize_process.py --dataset sst
```

## Regenerating the before/after figures

```bash
python scripts/fetch_satellite.py --dataset sst --start 2018-06-01 --end 2018-06-30
python scripts/visualize_process.py --dataset sst
```

Each figure is a 2x2 panel:

```
┌─────────────────────────────┬─────────────────────────────┐
│ Stage 0: RAW granule        │ Stage 1: After cloud fill   │
│ (blank cells = cloud gaps)  │ (gap-fill along time axis)  │
├─────────────────────────────┼─────────────────────────────┤
│ Stage 2: Regridded to study │ Stage 3: Derived index      │
│ grid (0.1 deg, land masked) │ (frontal / productivity)    │
└─────────────────────────────┴─────────────────────────────┘
```

Land contours come from the cached ETOPO 2022 bathymetry subset.