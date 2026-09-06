"""Per-species XGBoost training on the feature matrix.

Workflow (``scripts/train_model.py`` wrapper):
    feature matrix (data/processed/feature_matrix.parquet|pkl)
        -> label provider attach (habitat suitability | BFAR/PSA catches)
        -> drop/clean rows, impute feature NaNs (median)
        -> time-based split (train / val / holdout)  [no random shuffling]
        -> fit XGBoost, evaluate on val, save model + artifacts

Artifacts per species (models/saved/<code>/):
    model.pkl           -- fitted XGBoostRegressor
    bundle.pkl          -- feature columns, median imputer, target transform,
                           evaluation metrics, provider kind
    metrics.json        -- human-readable evaluation summary
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODEL_CONFIG, MODELS_DIR, TARGET_SPECIES
from src.models.labels import attach_labels

logger = logging.getLogger(__name__)

# Config feature names -> feature-matrix column names (see pipeline.py).
FEATURE_ALIASES = {
    "sst": "sst",
    "sst_anomaly": "sst_anom",
    "chlor_a": "chl",
    "chlor_a_rolling_30": "chl_roll30d",
    "chlor_a_anomaly": "chl_anom",
    "kd490": "kd490",
    "par": "par",
    "bathymetry_depth": "depth",
    "bathymetry_slope": "slope",
    "distance_to_coast": "dist_coast",
    "frontal_index": "frontal_index",
    "productivity_index": "productivity_index",
    "upwelling_proxy": "upwelling_proxy",
    "month_sin": "month_sin",
    "month_cos": "month_cos",
}

# Default chronological holdout window (years).
DEFAULT_TRAIN_UNTIL = 2022
DEFAULT_VAL = (2023, 2024)
DEFAULT_HOLDOUT = 2025

# Catch volumes are right-skewed (tonnes); model on log1p and invert at use.
TARGET_TRANSFORMS = {"habitat": None, "catch": "log1p"}


def resolve_features(feature_columns: pd.Index, requested: list[str] | None = None) -> list[str]:
    """Map configured feature names to columns actually present in the frame."""
    requested = requested or MODEL_CONFIG["features"]
    resolved = []
    for cfg_name in requested:
        col = FEATURE_ALIASES.get(cfg_name, cfg_name)
        if col in feature_columns:
            resolved.append(col)
        else:
            logger.warning("Requested feature %r (%r) not in matrix; dropping.", cfg_name, col)
    if not resolved:
        raise ValueError("No requested features are present in the feature matrix.")
    return resolved


def build_dataset(
    df: pd.DataFrame,
    species_code: str,
    features: list[str],
    label_kind: str = "prefer_catch",
    drop_unlabeled: bool = True,
) -> tuple[pd.DataFrame, pd.Series, str, dict[str, float], pd.Index]:
    """Assemble X/y plus a median-imputer dict for the species.

    Returns ``(X, y, used_label, imputer, positions)`` where ``positions`` are
    the original ``df.index`` values of the surviving rows (so training code
    can align chronological split masks computed over the full frame).
    """
    labels, used_label = attach_labels(df, species_code, kind=label_kind)
    y = pd.to_numeric(labels, errors="coerce")

    X = df[features].astype("float32")
    imputer: dict[str, float] = {
        col: float(X[col].median()) if not np.isnan(X[col].median()) else 0.0
        for col in features
    }

    if drop_unlabeled:
        keep = y.notna()
        X = X[keep]
        y = y[keep]
        positions = df.index[keep.to_numpy()]
    else:
        positions = df.index

    # Impute remaining feature NaNs (edge cells, occasionally missing indices).
    for col in features:
        X[col] = X[col].fillna(imputer[col])

    if X.empty:
        raise ValueError(
            f"No usable training rows for {species_code!r} after cleaning "
            f"(label={used_label}). Is the feature matrix empty?"
        )

    transform = TARGET_TRANSFORMS.get(used_label)
    if transform == "log1p":
        y = np.log1p(y.clip(lower=0.0))

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    logger.info(
        "Dataset %s: %d rows, %d features, label=%s transform=%s",
        species_code, len(X), len(features), used_label, transform,
    )
    return X, y, used_label, imputer, positions


def build_classification_dataset(
    df: pd.DataFrame,
    species_code: str,
    features: list[str],
    label_kind: str = "prefer_catch",
    pos_ratio: float = 0.5,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, str, dict[str, float], pd.Index, float]:
    """Binarize catches into a high/low-yield hotspot problem.

    Returns ``(X, y_bin, y_cont, used_label, imputer, positions, threshold)``.
    ``y_bin`` is 1 where the transformed catch exceeds the *train-window*
    quantile (default the median, ``pos_ratio=0.5``) and 0 otherwise; ``y_cont``
    is the continuous transformed label (for rank/Spearman diagnostics). The
    threshold is fitted on training rows only so validation/holdout stay clean.
    """
    labels, used_label = attach_labels(df, species_code, kind=label_kind)
    y = pd.to_numeric(labels, errors="coerce")

    X = df[features].astype("float32")
    imputer: dict[str, float] = {
        col: float(X[col].median()) if not np.isnan(X[col].median()) else 0.0
        for col in features
    }

    keep = y.notna()
    X = X[keep]
    positions = df.index[keep.to_numpy()]
    for col in features:
        X[col] = X[col].fillna(imputer[col])

    transform = TARGET_TRANSFORMS.get(used_label)
    y_cont = pd.Series(
        np.log1p(y[keep].clip(lower=0.0)) if transform == "log1p" else y[keep],
        index=positions,
    )

    year = pd.to_datetime(df["time"], errors="coerce").dt.year
    train_mask = (year <= DEFAULT_TRAIN_UNTIL).reindex(positions).fillna(False).to_numpy()
    threshold = float(y_cont[train_mask].quantile(1.0 - pos_ratio))
    y_bin = (y_cont > threshold).astype("int64")

    X = X.reset_index(drop=True)
    y_bin = y_bin.reset_index(drop=True)
    y_cont = y_cont.reset_index(drop=True)
    if y_bin.nunique() < 2:
        logger.warning(
            "%s: threshold %.3f did not separate classes (%d pos / %d neg); "
            "check the catch distribution.",
            species_code, threshold, int((y_bin == 1).sum()), int((y_bin == 0).sum()),
        )
    logger.info(
        "Classification dataset %s: %d rows, label=%s threshold=%.3f "
        "(pos=%d, neg=%d)",
        species_code, len(X), used_label, threshold,
        int((y_bin == 1).sum()), int((y_bin == 0).sum()),
    )
    return X, y_bin, y_cont, used_label, imputer, positions, threshold


def chronological_split(
    df: pd.DataFrame,
    train_until: int = DEFAULT_TRAIN_UNTIL,
    val: tuple[int, int] = DEFAULT_VAL,
    holdout: int = DEFAULT_HOLDOUT,
) -> dict[str, pd.Series]:
    """Boolean masks per split based on the year of each 'time' string."""
    year = pd.to_datetime(df["time"], errors="coerce").dt.year

    train_mask = year <= train_until
    val_mask = (year >= val[0]) & (year <= val[1])
    test_mask = year >= holdout

    for name, mask in [
        ("train", train_mask), ("val", val_mask), ("holdout", test_mask)
    ]:
        if not mask.any():
            logger.warning("Split %s is empty for this data window.", name)
    return {"train": train_mask, "val": val_mask, "test": test_mask}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    """Classification metrics for the binary high-yield problem.

    ``y_score`` is the predicted probability of the positive (high-yield) class.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype="int64")
    y_pred = np.asarray(y_pred, dtype="int64")
    y_score = np.asarray(y_score, dtype="float64")
    res = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        res["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        res["roc_auc"] = float("nan")
    return res


def fit_xgboost(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    params: dict | None = None,
    random_state: int | None = None,
) -> tuple[object, dict[str, float]]:
    """Fit an XGBoost regressor; return (model, val_metrics)."""
    from xgboost import XGBRegressor

    cfg = MODEL_CONFIG["xgboost_params"].copy()
    if params:
        cfg.update(params)
    cfg.setdefault("random_state", random_state or MODEL_CONFIG["random_state"])
    cfg.setdefault("verbosity", 0)

    model = XGBRegressor(**cfg)
    model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr)], verbose=False)

    metrics: dict[str, float] = {}
    if X_val is not None and y_val is not None and len(X_val) > 0:
        metrics = _metrics(y_val, model.predict(X_val))
    else:
        metrics = _metrics(y_tr, model.predict(X_tr))
        metrics["on"] = "train"
    return model, metrics


def feature_importance(model: object, features: list[str]) -> list[dict[str, str | float]]:
    try:
        imp = np.asarray(model.feature_importances_, dtype="float64")
    except AttributeError:
        return []
    order = np.argsort(imp)[::-1]
    return [
        {"feature": features[i], "importance": float(imp[i])}
        for i in order
        if imp[i] > 0
    ]


def train_species(
    df: pd.DataFrame,
    species_code: str,
    features: list[str] | None = None,
    label_kind: str = "prefer_catch",
    train_until: int = DEFAULT_TRAIN_UNTIL,
    val: tuple[int, int] = DEFAULT_VAL,
    holdout: int = DEFAULT_HOLDOUT,
    out_dir: Path | str | None = None,
) -> dict:
    """Train one species end-to-end and persist artifacts."""
    features = features or resolve_features(df.columns)
    X, y, used_label, imputer, positions = build_dataset(df, species_code, features, label_kind)

    try:
        from src.models.labels import available_species
        species = next(s for s in available_species() if s["code"] == species_code)
    except StopIteration:
        species = {"code": species_code, "name": species_code}

    masks = chronological_split(df, train_until=train_until, val=val, holdout=holdout)
    # Align chronological masks to the cleaned rows via their original positions.
    mask_pos = {
        name: mask.reindex(positions).to_numpy()
        for name, mask in masks.items()
    }

    X_train, y_train = X[mask_pos["train"]], y[mask_pos["train"]]
    X_val, y_val = X[mask_pos["val"]], y[mask_pos["val"]]
    X_test, y_test = X[mask_pos["test"]], y[mask_pos["test"]]

    model, val_metrics = fit_xgboost(X_train, y_train, X_val, y_val)

    # Evaluate also on the chronological holdout when it is non-empty.
    report = {
        "species_code": species_code,
        "species_name": species["name"],
        "label_kind": used_label,
        "target_transform": TARGET_TRANSFORMS.get(used_label),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_holdout": int(len(X_test)),
        "val_metrics": val_metrics,
        "feature_importance": feature_importance(model, features),
        "features": features,
    }
    if len(X_test) > 0:
        report["holdout_metrics"] = _metrics(y_test, model.predict(X_test))

    out_dir = Path(out_dir or MODELS_DIR)
    save_artifacts(out_dir, species_code, model, report, imputer, features)
    return report


def save_artifacts(
    out_dir: Path,
    species_code: str,
    model: object,
    report: dict,
    imputer: dict[str, float],
    features: list[str],
) -> Path:
    import pickle

    dest = Path(out_dir) / species_code
    dest.mkdir(parents=True, exist_ok=True)

    with (dest / "model.pkl").open("wb") as fh:
        pickle.dump(model, fh)
    with (dest / "bundle.pkl").open("wb") as fh:
        pickle.dump(
            {
                "features": features,
                "imputer": imputer,
                "report": report,
            },
            fh,
        )
    with (dest / "metrics.json").open("w") as fh:
        json.dump(report, fh, indent=2, default=str)

    logger.info("Saved %s model (+bundle) to %s", species_code, dest)
    return dest


def load_bundle(out_dir: Path | str, species_code: str) -> dict:
    """Load a saved training bundle (features, imputer, report, model)."""
    import pickle

    dest = Path(out_dir) / species_code
    with (dest / "bundle.pkl").open("rb") as fh:
        bundle = pickle.load(fh)
    with (dest / "model.pkl").open("rb") as fh:
        bundle["model"] = pickle.load(fh)
    bundle["code"] = species_code
    return bundle


def predict_frame(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """Run a trained bundle on a feature frame (already using resolved cols)."""
    X = frame[bundle["features"]].astype("float32")
    for col in bundle["features"]:
        X[col] = X[col].fillna(bundle["imputer"].get(col, 0.0))
    pred = bundle["model"].predict(X)
    transform = bundle["report"].get("target_transform")
    if transform == "log1p":
        pred = np.expm1(pred)
    return pred