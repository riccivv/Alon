"""Unified algorithm benchmark: accuracy view + the primary regression measure.

The production task is regression (tonnes per cell), so the **primary** measures
(VAL/HOLDOUT RMSE, MAE, R2 on the continuous log1p catch) are always reported.
On top of that, each species is reframed as a binary high-yield vs low-yield
problem (threshold = the species' *train-window* median, so validation/holdout
stay leak-free) to make the classification metrics meaningful:

    accuracy, precision, recall, f1, roc_auc          -- classification core
    spearman                                         -- rank agreement with the
                                                       continuous (log) catch
    hit_at_k                                         -- of the top-k predicted
                                                       cells (k = n positive),
                                                       how many are truly high-yield
    rmse, mae, r2                                     -- primary regression measures
                                                       on the continuous target

Every algorithm is run BOTH as a classifier (hotspot detector) and as a
regressor (tonnage forecaster) on the same rows / same chronological split.
This is the single-source comparison behind the training decision
(``configs/model_config.yaml``) and the forecasting map.

Usage:
    python scripts/benchmark_classification.py
    python scripts/benchmark_classification.py --algos xgboost logisticregression
    python scripts/benchmark_classification.py --species scad
    python scripts/benchmark_classification.py --quick
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.labels import available_species  # noqa: E402
from src.models.train import (  # noqa: E402
    DEFAULT_HOLDOUT,
    DEFAULT_TRAIN_UNTIL,
    DEFAULT_VAL,
    _class_metrics,
    _metrics,
    build_classification_dataset,
    chronological_split,
    resolve_features,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("bench-unified")

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_CSV = BASE_DIR / "docs" / "model_selection_results_unified.csv"
OUT_DIR = BASE_DIR / "docs" / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
RANDOM_STATE = SEED

CLASS_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "spearman", "hit_at_k"]
REGR_METRICS = ["rmse", "mae", "r2"]


# --------------------------------------------------------------------------- #
# Algorithm registry (classifier + regressor per candidate)
# --------------------------------------------------------------------------- #
def make_classifier(name: str, quick: bool = False):
    """Instantiate the classification variant of ``name``."""
    if name == "XGBoost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=1000, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=50, eval_metric="auc",
            random_state=RANDOM_STATE, verbosity=0,
        ), "gbm"

    if name == "LightGBM":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=1000, num_leaves=31, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE, verbose=-1,
        ), "lgbm"

    if name == "CatBoost":
        try:
            from catboost import CatBoostClassifier

            return CatBoostClassifier(
                iterations=1000, depth=6, learning_rate=0.05,
                early_stopping_rounds=50, random_seed=RANDOM_STATE,
                loss_function="Logloss", eval_metric="AUC",
                verbose=False, allow_writing_files=False,
            ), "gbm"
        except ImportError:
            logger.warning("catboost is not installed (pip install catboost); skipping.")
            return None, "gbm"

    if name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier

        n = 100 if quick else 200
        return RandomForestClassifier(
            n_estimators=n, max_depth=22, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE,
        ), "rf"

    if name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]
        ), "linear"

    if name == "NoSkill":
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE), "baseline"

    raise ValueError(f"Unknown algorithm {name!r}")


def make_regressor(name: str, quick: bool = False):
    """Instantiate the regression variant of ``name`` (the primary measure)."""
    if name == "XGBoost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=1000, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=50, random_state=RANDOM_STATE, verbosity=0,
        ), "gbm"

    if name == "LightGBM":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=1000, num_leaves=31, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE, verbose=-1,
        ), "lgbm"

    if name == "CatBoost":
        try:
            from catboost import CatBoostRegressor

            return CatBoostRegressor(
                iterations=1000, depth=6, learning_rate=0.05,
                early_stopping_rounds=50, random_seed=RANDOM_STATE,
                loss_function="RMSE", verbose=False, allow_writing_files=False,
            ), "gbm"
        except ImportError:
            return None, "gbm"

    if name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor

        n = 100 if quick else 200
        return RandomForestRegressor(
            n_estimators=n, max_depth=22, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE,
        ), "rf"

    if name == "LogisticRegression":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]
        ), "linear"

    if name == "NoSkill":
        return None, "baseline"

    raise ValueError(f"Unknown algorithm {name!r}")


def _fit_predict(
    name: str, model, kind: str, Xtr, ytr, Xva, yva, predict_score: bool
) -> tuple[object, np.ndarray, np.ndarray, float]:
    """Fit ``model``; return (model, hard/min-class pred, score, train_s).

    For regressors ``yva`` is continuous and ``score`` is NaN unless
    ``predict_score`` is True (only classifiers produce probabilities).
    """
    t0 = time.perf_counter()
    if kind == "gbm":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    elif kind == "lgbm":
        try:
            from lightgbm import early_stopping

            callbacks = [early_stopping(50, verbose=False)]
            try:
                model.fit(Xtr, ytr, eval_X=Xva, eval_y=yva, callbacks=callbacks)
            except TypeError:  # older lightgbm: only eval_set exists
                model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=callbacks)
        except (ImportError, TypeError):
            model.fit(Xtr, ytr)
    else:
        model.fit(Xtr, ytr)
    train_s = time.perf_counter() - t0

    y_pred = np.asarray(model.predict(Xva))
    y_score = np.full(len(y_pred), float("nan"))
    if predict_score:
        proba = model.predict_proba(Xva)
        y_score = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    return model, y_pred, y_score, train_s


def _rank_metrics(y_cont: np.ndarray, y_score: np.ndarray, y_bin: np.ndarray) -> dict[str, float]:
    """Spearman correlation with the continuous label + precision@k."""
    y_cont = np.asarray(y_cont, dtype="float64")
    y_score = np.asarray(y_score, dtype="float64")
    y_bin = np.asarray(y_bin, dtype="int64")

    res: dict[str, float] = {"spearman": float("nan"), "hit_at_k": float("nan")}
    try:
        from scipy.stats import spearmanr

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # constant input (NoSkill) -> NaN
            rho, _ = spearmanr(y_cont, y_score)
        if np.isfinite(rho):
            res["spearman"] = float(rho)
    except Exception:  # noqa: BLE001
        pass

    k = int(y_bin.sum())
    if k > 0 and len(y_score) and not np.isnan(y_score).any():
        topk = np.sort(np.argsort(y_score)[::-1][:k])
        res["hit_at_k"] = float(y_bin[topk].mean())
    return res


# --------------------------------------------------------------------------- #
# Per-species benchmark
# --------------------------------------------------------------------------- #
def load_matrix() -> pd.DataFrame:
    parquet = BASE_DIR / "data" / "processed" / "feature_matrix.parquet"
    if parquet.exists():
        logger.info("Loading feature matrix from %s", parquet)
        return pd.read_parquet(parquet)
    raise FileNotFoundError("No feature matrix. Run scripts/run_pipeline.py first.")


def benchmark_species(
    df: pd.DataFrame,
    code: str,
    features: list[str],
    algos: list[str],
    quick: bool,
    pos_ratio: float,
    rng: np.random.Generator,
) -> list[dict]:
    X, y_bin, y_cont, used_label, _imputer, positions, threshold = (
        build_classification_dataset(df, code, features, pos_ratio=pos_ratio)
    )

    masks = chronological_split(
        df, train_until=DEFAULT_TRAIN_UNTIL, val=DEFAULT_VAL, holdout=DEFAULT_HOLDOUT,
    )
    mask_pos = {s: m.reindex(positions).to_numpy() for s, m in masks.items()}

    Xtr, ytr = X[mask_pos["train"]], y_bin[mask_pos["train"]]
    Xva, yva = X[mask_pos["val"]], y_bin[mask_pos["val"]]
    Xte, yte = X[mask_pos["test"]], y_bin[mask_pos["test"]]
    y_cont_tr = y_cont[mask_pos["train"]]
    y_cont_va = y_cont[mask_pos["val"]]
    y_cont_te = y_cont[mask_pos["test"]]

    if quick:
        keep = rng.choice(len(Xtr), min(150_000, len(Xtr)), replace=False)
        Xtr, ytr = Xtr.iloc[keep], ytr.iloc[keep]
        y_cont_tr = y_cont_tr.iloc[keep]

    logger.info(
        "Benchmarking %s: %d train / %d val / %d hold rows, thresh=%.3f (%s)",
        code, len(Xtr), len(Xva), len(Xte), threshold, used_label,
    )

    rows = []
    for name in algos:
        clf, ckind = make_classifier(name, quick=quick)
        reg, rkind = make_regressor(name, quick=quick)
        if clf is None and reg is None:
            logger.warning("Skipping %s for %s (unavailable).", name, code)
            continue

        row = {
            "species": code,
            "algorithm": name,
            "label_kind": used_label,
            "threshold": threshold,
            "n_train": len(Xtr),
            "n_val": len(Xva),
            "n_holdout": len(Xte),
        }
        c_s = float("nan")
        r_s = float("nan")

        # Classification view -------------------------------------------------
        if clf is not None:
            try:
                _, _, y_score, c_s = _fit_predict(
                    name, clf, ckind, Xtr, ytr, Xva, yva, predict_score=True
                )
                y_pred = np.asarray(clf.predict(Xva))
                row.update({f"{m}_val": v for m, v in _class_metrics(yva, y_pred, y_score).items()})
                row.update(
                    {f"{m}_val": v for m, v in _rank_metrics(y_cont_va.to_numpy(), y_score, yva).items()}
                )
                if len(Xte) > 0:
                    y_pred_te = np.asarray(clf.predict(Xte))
                    proba_te = clf.predict_proba(Xte)
                    score_te = proba_te[:, 1] if proba_te.shape[1] > 1 else proba_te[:, 0]
                    row.update({f"{m}_hold": v for m, v in _class_metrics(yte, y_pred_te, score_te).items()})
                    row.update(
                        {f"{m}_hold": v for m, v in _rank_metrics(y_cont_te.to_numpy(), score_te, yte).items()}
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("%s classification failed for %s: %s", name, code, exc)

        # Regression view (the primary measure) --------------------------------
        if reg is not None:
            try:
                _, _, _, r_s = _fit_predict(
                    name, reg, rkind, Xtr, y_cont_tr, Xva, y_cont_va, predict_score=False
                )
                row.update(
                    {f"{m}_val": v for m, v in _metrics(y_cont_va, reg.predict(Xva)).items()}
                )
                if len(Xte) > 0:
                    row.update(
                        {f"{m}_hold": v for m, v in _metrics(y_cont_te, reg.predict(Xte)).items()}
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("%s regression failed for %s: %s", name, code, exc)
        else:
            r_s = 0.0
            y_pred = np.full(len(y_cont_va), float(y_cont_tr.mean()))
            row.update({f"{m}_val": v for m, v in _metrics(y_cont_va, y_pred).items()})
            if len(Xte) > 0:
                y_pred_te = np.full(len(y_cont_te), float(y_cont_tr.mean()))
                row.update({f"{m}_hold": v for m, v in _metrics(y_cont_te, y_pred_te).items()})

        row["train_s"] = round(c_s + r_s, 3)
        rows.append(row)

        shown = " ".join(
            f"{m}={row.get(m+'_val', float('nan')):.3f}" if not np.isnan(row.get(m+"_val", float("nan"))) else f"{m}=nan"
            for m in CLASS_METRICS[:2] + ["roc_auc", "r2"]
        )
        logger.info("  %-20s %s (%.1f s)", name, shown, row["train_s"])
    return rows


# --------------------------------------------------------------------------- #
# Figures + CLI
# --------------------------------------------------------------------------- #
def summary_figure(results: pd.DataFrame) -> plt.Figure:
    order = [
        a for a in ("XGBoost", "LightGBM", "CatBoost", "RandomForest", "LogisticRegression", "NoSkill")
        if a in results["algorithm"].unique()
    ]
    panels = [
        ("roc_auc", "ROC AUC (classifier, high-yield)", 0.0, 1.05),
        ("f1", "F1 (classifier, high-yield)", 0.0, 1.05),
        ("hit_at_k", "Hit@k (top-k cells hit)", 0.0, 1.05),
        ("spearman", "Spearman (rank vs log catch)", -0.1, 1.05),
        ("r2", "R2 (regressor, primary)", -0.1, 1.05),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(26, 5.5))
    for ax, (m, title, lo, hi) in zip(axes, panels):
        pivot = results.pivot(index="species", columns="algorithm", values=f"{m}_val")[order]
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel(title)
        ax.set_ylim(lo, hi)
        ax.tick_params(axis="x", rotation=0)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle(
        "Unified benchmark: primary regression measure (R2) + hotspot-detection view",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified algorithm benchmark (regression + classification)")
    parser.add_argument("--species", nargs="*", default=None)
    parser.add_argument("--algos", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true",
                        help="Subsample 150k rows (fast draft)")
    parser.add_argument("--pos-ratio", type=float, default=0.5,
                        help="Train share below the threshold that is the positive class")
    args = parser.parse_args()

    all_algos = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "LogisticRegression", "NoSkill"]
    algos = args.algos or all_algos
    unknown = [a for a in algos if a not in all_algos]
    if unknown:
        parser.error(f"Unknown algorithms: {unknown}. Choose from {all_algos}")

    codes = args.species or [s["code"] for s in available_species()]
    df = load_matrix()
    features = resolve_features(df.columns)

    logger.info(
        "Testing %s on %d species (unified: regression + classification, chronological split).",
        ", ".join(algos), len(codes),
    )

    rng = np.random.default_rng(SEED)
    all_rows: list[dict] = []
    for code in codes:
        all_rows.extend(benchmark_species(df, code, features, algos, args.quick, args.pos_ratio, rng))

    results = pd.DataFrame(all_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    logger.info("Saved results table to %s", OUT_CSV)

    print("\n=== Unified ranking (median across species) ===")
    show_cols = ["r2", "rmse", "accuracy", "precision", "recall", "f1", "roc_auc", "spearman", "hit_at_k"]
    agg = (
        results.groupby("algorithm").agg(
            **{m: (f"{m}_val", "median") for m in show_cols},
            train_s=("train_s", "median"),
        )
        .sort_values("r2", ascending=False)
        .reset_index()
    )
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.3f}" if not np.isnan(v) else "nan"))
    print()

    fig = summary_figure(results)
    out = OUT_DIR / "model_classification.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    logger.info("Saved %s", out)


if __name__ == "__main__":
    main()