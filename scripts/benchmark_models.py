"""Model-selection study: which algorithm predicts fish suitability best?

Run this ONCE after ``scripts/run_pipeline.py`` and BEFORE training the final
per-species models. It trains every candidate on the SAME rows, evaluates on
the SAME chronological windows, and produces a ranked comparison table
(``docs/model_selection_results.csv``) plus a summary figure
(``docs/diagrams/model_selection.png``).

Algorithms tested and why they are in the study (rationale summary):

1. **XGBoost**            - the incumbent: mature gradient-boosted trees,
                            handles missing values + feature interactions,
                            CPU-friendly at millions of rows.
2. **LightGBM**           - the strongest competitor to XGBoost in the same
                            family; histogram-based, typically much faster.
3. **CatBoost**           - third major GBM; robust to noisy inputs, decent
                            regularization. Needs ``pip install catboost``
                            (absent -> the script skips it with a warning).
4. **Random Forest**      - bagged-tree baseline: averages many trees instead
                            of gradient descent; usually the "safe" benchmark
                            that GBMs should beat.
5. **Ridge (linear)**     - sanity baseline for "is a plain linear model
                            enough?" (scaled). Logistic regression is a
                            *classifier*; suitability is a continuous target,
                            so Ridge is the honest linear stand-in.
6. **No-skill (mean)**    - the floor: predict the training mean. Every
                            algorithm must beat this to add real value
                            (Isolation Forest is an anomaly detector, not a
                            regressor, so it cannot be benchmarked here).

Method: same 15 features, habitat-suitability labels, and chronological split
as real training (train <= 2022, val 2023-2024, holdout 2025). Boosters get
1000 trees with early stopping on the validation window; RF gets 200; all use
a fixed seed. Metrics are computed on the transformed label space exactly as
``train_model.py`` reports them. The winner should be set as
``model.type`` in ``configs/model_config.yaml``.

Usage:
    python scripts/benchmark_models.py                     # all species, full data
    python scripts/benchmark_models.py --algos xgboost ridge   # subset of algos
    python scripts/benchmark_models.py --species scad       # one species
    python scripts/benchmark_models.py --quick              # subsample (fast draft)
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

from src.config import MODELS_DIR  # noqa: E402
from src.models.labels import available_species  # noqa: E402
from src.models.train import (  # noqa: E402
    DEFAULT_HOLDOUT,
    DEFAULT_TRAIN_UNTIL,
    DEFAULT_VAL,
    _metrics,
    build_dataset,
    chronological_split,
    resolve_features,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("bench")

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_CSV = BASE_DIR / "docs" / "model_selection_results.csv"
OUT_DIR = BASE_DIR / "docs" / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
RANDOM_STATE = SEED


# --------------------------------------------------------------------------- #
# Algorithm registry
# --------------------------------------------------------------------------- #
def make_model(name: str, quick: bool = False):
    """Instantiate one candidate; return ``(sklearn_like_model, tree_flag)``.

    Boosting models get an ``eval_set``-aware fit via ``fit_and_eval`` below;
    the returned object is ``None`` for unavailable/unsupported algorithms
    (e.g. catboost not installed) and for the no-skill baseline.
    """
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
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=50, random_state=RANDOM_STATE, verbose=-1,
        ), "gbm"

    if name == "CatBoost":
        try:
            from catboost import CatBoostRegressor

            return CatBoostRegressor(
                iterations=1000, depth=6, learning_rate=0.05,
                early_stopping_rounds=50, random_seed=RANDOM_STATE,
                loss_function="RMSE", verbose=False, allow_writing_files=False,
            ), "gbm"
        except ImportError:
            logger.warning("catboost is not installed (pip install catboost); skipping.")
            return None, "gbm"

    if name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor

        n = 100 if quick else 200
        return RandomForestRegressor(
            n_estimators=n, max_depth=22, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE,
        ), "rf"

    if name == "Ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]
        ), "linear"

    if name == "NoSkill":
        return None, "baseline"

    raise ValueError(f"Unknown algorithm {name!r}")


def _fit_eval(name: str, model, kind: str, Xtr, ytr, Xva, yva) -> tuple:
    """Fit ``model`` timing it; returns (model_or_pred, metrics_dict, train_s)."""
    t0 = time.perf_counter()
    if kind == "baseline":
        pred = np.full(len(yva), float(ytr.mean()))
        train_s = time.perf_counter() - t0
        return pred, _metrics(yva, pred), train_s

    if kind == "gbm":
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            verbose=False,
        )
        pred = model.predict(Xva)
    else:
        model.fit(Xtr, ytr)
        pred = model.predict(Xva)
    train_s = time.perf_counter() - t0
    return model, _metrics(yva, pred), train_s


# --------------------------------------------------------------------------- #
# Per-species benchmark
# --------------------------------------------------------------------------- #
def load_matrix() -> pd.DataFrame:
    parquet = Path(str(BASE_DIR / "data" / "processed" / "feature_matrix.parquet"))
    if parquet.exists():
        logger.info("Loading feature matrix from %s", parquet)
        return pd.read_parquet(parquet, columns=None)
    raise FileNotFoundError("No feature matrix. Run scripts/run_pipeline.py first.")


def benchmark_species(
    df: pd.DataFrame,
    code: str,
    features: list[str],
    algos: list[str],
    quick: bool,
    rng: np.random.Generator,
) -> list[dict]:
    """Train all algorithms for one species and return result rows."""
    X, y, used_label, imputer, positions = build_dataset(
        df, code, features, label_kind="prefer_catch",
    )

    masks = chronological_split(
        df, train_until=DEFAULT_TRAIN_UNTIL, val=DEFAULT_VAL, holdout=DEFAULT_HOLDOUT,
    )
    mask_pos = {s: m.reindex(positions).to_numpy() for s, m in masks.items()}

    Xtr, ytr = X[mask_pos["train"]], y[mask_pos["train"]]
    Xva, yva = X[mask_pos["val"]], y[mask_pos["val"]]
    Xte, yte = X[mask_pos["test"]], y[mask_pos["test"]]

    if quick:
        keep_tr = rng.choice(len(Xtr), min(150_000, len(Xtr)), replace=False)
        keep_va = rng.choice(len(Xva), min(40_000, len(Xva)), replace=False)
        Xtr, ytr = Xtr.iloc[keep_tr], ytr.iloc[keep_tr]
        Xva, yva = Xva.iloc[keep_va], yva.iloc[keep_va]

    logger.info("Benchmarking %s on %d train / %d val / %d hold rows (%s label)",
                code, len(Xtr), len(Xva), len(Xte), used_label)

    rows = []
    for name in algos:
        model, kind = make_model(name, quick=quick)
        if model is None and kind != "baseline":
            logger.warning("Skipping %s for %s (unavailable).", name, code)
            continue
        try:
            _, val_m, train_s = _fit_eval(name, model, kind, Xtr, ytr, Xva, yva)
        except Exception as exc:  # noqa: BLE001
            logger.error("%s failed for %s: %s", name, code, exc)
            continue

        row = {
            "species": code,
            "algorithm": name,
            "label_kind": used_label,
            "n_train": len(Xtr),
            "n_val": len(Xva),
            "n_holdout": len(Xte),
            "train_s": round(train_s, 3),
            "rmse_val": val_m.get("rmse"),
            "mae_val": val_m.get("mae"),
            "r2_val": val_m.get("r2"),
        }
        if len(Xte) > 0:
            if kind == "baseline":
                pred_te = np.full(len(yte), float(ytr.mean()))
            else:
                pred_te = model.predict(Xte)
            hold = _metrics(yte, pred_te)
            row.update({"rmse_hold": hold["rmse"], "mae_hold": hold["mae"], "r2_hold": hold["r2"]})
        rows.append(row)
        logger.info("  %-12s val RMSE=%.4f R2=%.3f (%.1f s)", name,
                    row["rmse_val"], row["r2_val"], row["train_s"])
    return rows


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def summary_figure(results: pd.DataFrame) -> plt.Figure:
    # Order as in the study list.
    order = [a for a in ("XGBoost", "LightGBM", "CatBoost", "RandomForest", "Ridge", "NoSkill")
             if a in results["algorithm"].unique()]
    pivot_r2 = results.pivot(index="species", columns="algorithm", values="r2_val")[order]
    pivot_rmse = results.pivot(index="species", columns="algorithm", values="rmse_val")[order]
    times = results.groupby("algorithm")["train_s"].median().reindex(order)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    pivot_r2.plot(kind="bar", ax=axes[0], width=0.8)
    axes[0].axhline(0.0, color="0.3", lw=1)
    axes[0].set_ylabel("R2 on validation (higher better)")
    axes[0].set_title("Validation R2 per algorithm")
    axes[0].set_ylim(-0.15, 1.05)
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].tick_params(axis="x", rotation=0)

    pivot_rmse.plot(kind="bar", ax=axes[1], width=0.8)
    axes[1].set_ylabel("RMSE (lower better)")
    axes[1].set_title("Validation RMSE per algorithm")
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].tick_params(axis="x", rotation=0)

    med_r2 = pivot_r2.median(axis=0)
    x = np.arange(len(order))
    ax = axes[2]
    ax.bar(x - 0.22, times.values, width=0.44, color="#CC6677", label="median train time (s)")
    ax.bar(x + 0.22, med_r2.values, width=0.44, color="#4477AA", label="median R2")
    ax.set_xticks(x, order, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("time (s) / R2")
    ax.set_title("Accuracy vs. training cost (median across species)")
    ax.legend(fontsize=8)

    fig.suptitle("Model-selection study: algorithm vs. environment-suitability prediction",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark candidate algorithms")
    parser.add_argument("--species", nargs="*", default=None,
                        help="Species codes (default: all five)")
    parser.add_argument("--algos", nargs="*", default=None,
                        help="Algorithm subset (default: all six)")
    parser.add_argument("--quick", action="store_true",
                        help="Subsample 150k/40k rows (fast draft)")
    args = parser.parse_args()

    all_algos = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "Ridge", "NoSkill"]
    algos = args.algos or all_algos
    unknown = [a for a in algos if a not in all_algos]
    if unknown:
        parser.error(f"Unknown algorithms: {unknown}. Choose from {all_algos}")

    codes = args.species or [s["code"] for s in available_species()]
    df = load_matrix()
    features = resolve_features(df.columns)

    logger.info("Testing %s on %d species (15 features, chronological split).",
                ", ".join(algos), len(codes))

    rng = np.random.default_rng(SEED)
    all_rows: list[dict] = []
    for code in codes:
        all_rows.extend(benchmark_species(df, code, features, algos, args.quick, rng))

    results = pd.DataFrame(all_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    logger.info("Saved results table to %s", OUT_CSV)

    # Console ranking: median R2 per algorithm across species.
    print("\n=== Model-selection ranking (median across species) ===")
    agg = results.groupby("algorithm").agg(
        r2_val=("r2_val", "median"),
        rmse_val=("rmse_val", "median"),
        train_s=("train_s", "median"),
    ).sort_values("r2_val", ascending=False).reset_index()
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()

    fig = summary_figure(results)
    out = OUT_DIR / "model_selection.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    logger.info("Saved %s", out)


if __name__ == "__main__":
    main()