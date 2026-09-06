# ==============================================================================
# OceanForecast Philippines - Application Configuration
# ==============================================================================
# Central configuration for regions, species, data sources, and models.
# ==============================================================================

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
load_dotenv(BASE_DIR / ".env")

# ==============================================================================
# Paths
# ==============================================================================
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = BASE_DIR / "models" / "saved"
CONFIGS_DIR = BASE_DIR / "configs"
APP_DIR = BASE_DIR / "app"

for _dir in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# NASA Earthdata Credentials
# ==============================================================================
EARTHDATA_USERNAME = os.getenv("EARTHDATA_USERNAME", "")
EARTHDATA_PASSWORD = os.getenv("EARTHDATA_PASSWORD", "")

# ==============================================================================
# Philippine Study Region (Geographic bounds)
# ==============================================================================
# Full Philippine exclusive economic zone (EEZ) bounding box
# lon_min, lat_min, lon_max, lat_max
PHILIPPINES_BBOX = (116.0, 4.5, 127.0, 21.5)

# Sub-regions used for forecasting aggregation
PHILIPPINE_REGIONS = [
    {
        "name": "FMA1 - North Philippine Sea",
        "bbox": (120.0, 15.0, 125.0, 21.5),
        "description": "Northern Luzon, Batanes, Babuyan",
    },
    {
        "name": "FMA2 - West Philippine Sea",
        "bbox": (116.0, 8.0, 120.5, 21.5),
        "description": "South China Sea side, Palawan",
    },
    {
        "name": "FMA3 - Sulu-Celebes Sea",
        "bbox": (117.5, 3.0, 122.5, 9.5),
        "description": "Tawi-Tawi, Basilan, Sulu",
    },
    {
        "name": "FMA4 - Visayan Sea",
        "bbox": (122.5, 9.0, 125.0, 12.5),
        "description": "Central Visayas, Panay, Negros",
    },
    {
        "name": "FMA5 - West Sulu Sea",
        "bbox": (117.0, 7.0, 121.0, 12.0),
        "description": "Palawan east coast, eastern Sulu",
    },
    {
        "name": "FMA6 - East Philippine Sea",
        "bbox": (124.0, 6.0, 127.0, 18.0),
        "description": "Pacific coast, Eastern Visayas to Bicol",
    },
]

# ==============================================================================
# Target Fish Species (Philippine)
# ==============================================================================
TARGET_SPECIES = [
    {
        "code": "sardinella",
        "name": "Sardines (Sardinella)",
        "species": ["Sardinella tawilis", "Sardinella lemuru", "Sardinella longiceps"],
        "local_names": ["Tamban", "Tunsoy"],
        "optimal_sst_range": (26.0, 31.0),
        "optimal_chla_range": (0.2, 3.0),
    },
    {
        "code": "anchovy",
        "name": "Anchovies (Stolephorus/Encrasicholina)",
        "species": ["Encrasicholina punctifer", "Stolephorus indicus"],
        "local_names": ["Dilis", "Bolinao"],
        "optimal_sst_range": (26.5, 30.0),
        "optimal_chla_range": (0.3, 4.0),
    },
    {
        "code": "mackerel",
        "name": "Indian Mackerel (Rastrelliger)",
        "species": ["Rastrelliger kanagurta", "Rastrelliger brachysoma"],
        "local_names": ["Alumahan", "Guro"],
        "optimal_sst_range": (27.0, 31.0),
        "optimal_chla_range": (0.1, 2.0),
    },
    {
        "code": "scad",
        "name": "Round Scad (Decapterus)",
        "species": ["Decapterus macrosoma", "Decapterus maruadsi"],
        "local_names": ["Galunggong", "Mamale"],
        "optimal_sst_range": (26.0, 30.0),
        "optimal_chla_range": (0.2, 3.5),
    },
    {
        "code": "slipmouth",
        "name": "Slipmouth (Leiognathidae)",
        "species": ["Leiognathus spp.", "Secutor spp."],
        "local_names": ["Sap-sap", "Tambong"],
        "optimal_sst_range": (25.0, 30.0),
        "optimal_chla_range": (0.3, 5.0),
    },
    {
        "code": "all",
        "name": "All Species (Total Pelagic Yield)",
        "species": ["All pelagic species"],
        "local_names": ["Composite"],
        "optimal_sst_range": (25.0, 31.0),
        "optimal_chla_range": (0.1, 5.0),
    },
]

# ==============================================================================
# NASA Satellite Dataset Configuration
# ==============================================================================
SATELLITE_DATASETS = {
    "chlorophyll": {
        "short_name": "MODISA_L3m_CHL",
        "granule_pattern": "*.8D*.9km*",  # 8-day, 9km resolution monthly lookups
        "variable": "chlor_a",
        "units": "mg/m^3",
        "spatial_res": "9km",
        "temporal_res": "8-day",
    },
    "sst": {
        "short_name": "MODISA_L3m_SST",
        "granule_pattern": "*.8D*.9km*",
        "variable": "sst",
        "units": "degC",
        "spatial_res": "9km",
        "temporal_res": "8-day",
    },
    "par": {
        "short_name": "MODISA_L3m_PAR",
        "granule_pattern": "*.8D*.9km*",
        "variable": "par",
        "units": "E/m^2/day",
        "spatial_res": "9km",
        "temporal_res": "8-day",
    },
    "kd490": {
        "short_name": "MODISA_L3m_Kd490",
        "granule_pattern": "*.8D*.9km*",
        "variable": "kd_490",
        "units": "1/m",
        "spatial_res": "9km",
        "temporal_res": "8-day",
    },
}

# Default temporal range for data acquisition (MODIS Aqua starts 2002)
SATELLITE_DEFAULT_TEMPORAL = ("2015-01-01", "2025-12-31")

# Cloud cover filtering (percent)
CLOUD_COVER_MAX = 50

# ==============================================================================
# Bathymetry (GEBCO)
# ==============================================================================
GEBCO_URL = "https://www.gebco.net/data_and_products/gridded_bathymetry_data/gebco_2024/GEBCO_2024.nc"
GEBCO_DEPTH_VARIABLE = "elevation"  # negative = below sea level

# ==============================================================================
# Feature Engineering Configuration
# ==============================================================================
FEATURE_CONFIG = {
    "rolling_windows_days": [7, 14, 30, 90],  # Rolling means for temporal features
    "lag_days": [7, 14, 30],                  # Lag features
    "spatial_lag_radii_km": [20, 50, 100],    # Spatial lag neighbors
    "use_sin_cos_time": True,                 # Cyclical time encoding
    "cloud_fill_method": "temporal_interp",   # Gap filling: None | temporal_interp | climatology
}

# ==============================================================================
# Model Training Configuration
# ==============================================================================
MODEL_CONFIG = {
    "model_type": "xgboost",
    "random_state": 42,
    "cv_folds": 5,
    "test_size": 0.2,
    "features": [
        "sst",
        "sst_anomaly",
        "chlor_a",
        "chlor_a_rolling_30",
        "chlor_a_anomaly",
        "kd490",
        "par",
        "bathymetry_depth",
        "bathymetry_slope",
        "distance_to_coast",
        "frontal_index",
        "productivity_index",
        "upwelling_proxy",
        "month_sin",
        "month_cos",
    ],
    "xgboost_params": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
    },
}

# ==============================================================================
# Overfishing / Sustainability Thresholds
# ==============================================================================
# Ratio of predicted yield to estimated MSY (Maximum Sustainable Yield)
SUSTAINABILITY = {
    "msy_ratio_sustainable": 0.6,   # Green zone: yield < 60% of MSY
    "msy_ratio_caution": 0.8,       # Yellow zone: 60-80% of MSY
    # Red zone: > 80% of MSY (overfishing risk)
}

# ==============================================================================
# App Settings
# ==============================================================================
APP_TITLE = "OceanForecast Philippines"
APP_SUBTITLE = "Dynamic Oceanographic Fisheries Yield Forecasting"
DEFAULT_MAP_CENTER = [12.0, 122.0]
DEFAULT_MAP_ZOOM = 6
