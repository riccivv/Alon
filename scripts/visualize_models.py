"""Trained-model diagnostics and comparison diagrams.

Produces (under ``docs/diagrams/``) for every trained species in
``models/saved/``:

* ``model_comparison.png``  -- RMSE / MAE / R2 per species, chronological
  validation vs holdout (the numbers stored in each ``metrics.json``).
* ``model_fit.png``         -- actual vs predicted scatter for the validation
  and holdout windows per species (identity line + R2).
* ``model_importance.png``  -- top feature importances per species.
* ``model_algorithm_benchmark.png`` -- optional: XGBoost vs LightGBM vs
  RandomForest on a subsample, as evidence for the model choice.

Usage (run after ``python scripts/train_model.py``):
    python scripts/visualize_models.py                     # all trained species
    python scripts/visualize_models.py --species scad mackerel
    python scripts/visualize_models.py --benchmark         # + algorithm compare

Note: model_fit / model_importance reuse the saved artifacts; they do NOT
retrain anything. The benchmark trains three small models on a subsample and
reports validation error as evidence for "why XGBoost" (see docs/decisions.md).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS_DIR  # noqa: E402
from src.models.train import (  # noqa: E402
    DEFAULT_HOLDOUT,
    DEFAULT_TRAIN_UNTIL,
    DEFAULT_VAL,
    _metrics,
    chronological_split,
    load_bundle,
    predict_frame,
)
from src.models.labels import attach_labels  # noqa: E402
from src.processing.pipeline import FEATURES_PATH  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("vizmodel")

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _label_policy(stored_kind: str | None) -> str:
    """Stored report label_kind is a provider ('habitat'/'catch'); the label
    functions expect a provider-order policy ('habitat_only'/'prefer_catch')."""
    return "habitat_only" if stored_kind == "habitat" else "prefer_catch"


def load_feature_matrix() -> pd.DataFrame:
    parquet = Path(str(FEATURES_PATH).replace(".pkl", ".parquet"))
    if parquet.exists():
        logger.info("Loading feature matrix from %s", parquet)
        return pd.read_parquet(parquet)
    if FEATURES_PATH.exists():
        logger.info("Loading feature matrix from %s", FEATURES_PATH)
        return pd.read_pickle(FEATURES_PATH)
    raise FileNotFoundError(
        "No feature matrix found. Run scripts/run_pipeline.py first."
    )


def trained_species() -> list[str]:
    return sorted(p.parent.name for p in MODELS_DIR.glob("*/metrics.json"))


def _scores_for(bundle: dict, df: pd.DataFrame, split: str) -> pd.DataFrame:
    """(y_true, y_pred, year) rows for one chronological split."""
    from src.models.train import TARGET_TRANSFORMS

    pred = predict_frame(bundle, df)
    y, used = attach_labels(df, bundle["code"], kind=_label_policy(bundle["report"].get("label_kind")))
    transform = TARGET_TRANSFORMS.get(used)

    masks = chronological_split(
        df,
        train_until=DEFAULT_TRAIN_UNTIL,
        val=DEFAULT_VAL,
        holdout=DEFAULT_HOLDOUT,
    )
    out = pd.DataFrame({"y": y.to_numpy(), "pred": pred}, index=df.index)
    out["year"] = pd.to_datetime(df["time"], errors="coerce").dt.year
    out["split"] = split
    keep = masks[split].to_numpy() & out["y"].notna().to_numpy()
    return out.loc[keep]


def figure_metrics(bundles: dict[str, dict]) -> plt.Figure:
    rows = []
    for code, b in bundles.items():
        rep = b["report"]
        vm, hm = rep.get("val_metrics", {}), rep.get("holdout_metrics", {})
        rows.append({
            "species": code,
            "label": rep.get("label_kind", ""),
            "rmse_val": vm.get("rmse"), "rmse_hold": hm.get("rmse"),
            "mae_val": vm.get("mae"), "mae_hold": hm.get("mae"),
            "r2_val": vm.get("r2"), "r2_hold": hm.get("r2"),
        })
    res = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    species = res["species"].tolist()
    x = np.arange(len(species))
    width = 0.18

    for i, (metric, label) in enumerate([("rmse", "RMSE"), ("mae", "MAE")]):
        for j, split, color in [(0, "val", "#4477AA"), (1, "hold", "#CC6677")]:
            axes[0].bar(
                x + (i * 2 + j - 1.5) * width,
                res[f"{metric}_{split}"].fillna(0),
                width, color=color, label=f"{label} {split}",
            )
    axes[0].set_xticks(x, species)
    axes[0].set_ylabel("error (lower better)")
    axes[0].set_title("Prediction error -- validation vs holdout")
    axes[0].legend(fontsize=9)

    for j, (split, color) in enumerate([("val", "#4477AA"), ("hold", "#CC6677")]):
        axes[1].bar(x + (j - 0.5) * width, res[f"r2_{split}"].fillna(0), width,
                    color=color, label=f"R2 {split}")
    axes[1].axhline(0, color="0.4", lw=0.8)
    axes[1].set_xticks(x, species)
    axes[1].set_ylabel("R2 (higher better)")
    axes[1].set_title("R2 -- validation vs holdout")
    axes[1].legend(fontsize=9)

    fig.suptitle("Trained XGBoost models per species   |   labels: " +
                 "/".join(sorted({r["label"] for r in rows}) or ["?"]), fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def figure_fit(bundles: dict[str, dict], df: pd.DataFrame) -> plt.Figure:
    codes = list(bundles)
    rows = (len(codes) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(13, 4.5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, code in zip(axes, codes):
        b = bundles[code]
        try:
            val = _scores_for(b, df, "val")
            hold = _scores_for(b, df, "test")
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping fit figure for %s: %s", code, exc)
            ax.set_visible(False)
            continue
        data = {"val": val, "holdout": hold}
        for name, sub in data.items():
            if sub.empty:
                continue
            ax.scatter(sub["y"], sub["pred"], s=6, alpha=0.35,
                       label=f"{name} (R2 {_metrics(sub['y'], sub['pred'])['r2']:.3f})")
        lim = ax.get_xlim()
        lo, hi = min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
        ax.set_title(code)
        ax.set_xlabel("actual (target)")
        ax.set_ylabel("predicted")
        ax.legend(fontsize=8)

    for ax in axes[len(codes):]:
        ax.axis("off")
    fig.suptitle("Actual vs predicted -- per species (identity = perfect)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def figure_importance(bundles: dict[str, dict], top: int = 12) -> plt.Figure:
    codes = [c for c in bundles if bundles[c]["report"].get("feature_importance")]
    if not codes:
        raise RuntimeError("No feature importances stored in any model bundle.")
    rows = (len(codes) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(13, 4.5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, code in zip(axes, codes):
        imp = pd.DataFrame(bundles[code]["report"]["feature_importance"]).head(top)
        y_pos = np.arange(len(imp))[::-1]
        ax.barh(y_pos, imp["importance"], color="teal")
        ax.set_yticks(y_pos, imp["feature"], fontsize=8)
        ax.set_xlabel("gain importance")
        ax.set_title(code)

    for ax in axes[len(codes):]:
        ax.axis("off")
    fig.suptitle("Top feature importances per trained species", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# --------------------------------------------------------------------------- #
# Optional algorithm benchmark (evidence for the "why XGBoost" decision)
# --------------------------------------------------------------------------- #
def _benchmark(df: pd.DataFrame, bundles: dict[str, dict], estimators: int) -> plt.Figure:
    from src.models.train import build_dataset, resolve_features

    rng = np.random.default_rng(7)
    results = []
    for code, b in bundles.items():
        try:
            features = resolve_features(df.columns, b["report"].get("features"))
            X, y, used, imputer, positions = build_dataset(
                df, code, features,
                label_kind=_label_policy(b["report"].get("label_kind")),
            )
        except (ValueError, KeyError) as exc:
            logger.warning("Benchmark skip %s: %s", code, exc)
            continue
        masks = chronological_split(df, train_until=DEFAULT_TRAIN_UNTIL,
                                    val=DEFAULT_VAL, holdout=DEFAULT_HOLDOUT)
        mask_tr = masks["train"].reindex(positions).to_numpy()
        mask_va = masks["val"].reindex(positions).to_numpy()

        Xtr, ytr = X[mask_tr].to_numpy(), y[mask_tr].to_numpy()
        Xva, yva = X[mask_va].to_numpy(), y[mask_va].to_numpy()
        if len(Xtr) > 150_000:
            keep = rng.choice(len(Xtr), 150_000, replace=False)
            Xtr, ytr = Xtr[keep], ytr[keep]
        if len(Xva) > 40_000:
            keep = rng.choice(len(Xva), 40_000, replace=False)
            Xva, yva = Xva[keep], yva[keep]

        algos = {
            "XGBoost": _xgb(estimators),
            "RandomForest": _rf(estimators),
        }
        try:
            algos["LightGBM"] = _lgb(estimators)
        except ImportError:
            logger.warning("lightgbm not installed; benchmarking 2 algorithms.")

        logger.info("Benchmark %s: %s train / %s val rows", code, len(Xtr), len(Xva))
        for name, mdl in algos.items():
            mdl.fit(Xtr, ytr)
            pred = mdl.predict(Xva)
            m = _metrics(yva, pred)
            results.append({"species": code, "algorithm": name, "rmse": m["rmse"], "r2": m["r2"]})

    res = pd.DataFrame(results)
    if res.empty:
        raise RuntimeError("No benchmark results produced.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, label in [(axes[0], "rmse", "RMSE (lower better)"),
                              (axes[1], "r2", "R2 (higher better)")]:
        pivot = res.pivot(index="species", columns="algorithm", values=metric)
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel(label)
        ax.set_title(f"Algorithm comparison on validation window -- {label}")
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle("Model choice evidence: XGBoost vs LightGBM vs RandomForest",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def _xgb(n: int):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=n, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
    )


def _lgb(n: int):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=n, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=-1,
    )


def _rf(n: int):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(n_estimators=n, max_depth=12, n_jobs=-1, random_state=42)


def _print_summary(bundles: dict[str, dict]) -> None:
    print("\n=== Model training summary ===")
    for code, b in bundles.items():
        rep = b["report"]
        vm, hm = rep.get("val_metrics", {}), rep.get("holdout_metrics", {})
        print(
            f"{code:<12} label={rep.get('label_kind',''):<8} "
            f"train={rep.get('n_train',0):>9,} val={rep.get('n_val',0):>7,} "
            f"holdout={rep.get('n_holdout',0):>7,} | val RMSE={vm.get('rmse',float('nan')):.4f} "
            f"R2={vm.get('r2',float('nan')):.3f}  hold R2={hm.get('r2',float('nan')):.3f}"
        )
    print()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Model diagnostics diagrams")
    parser.add_argument("--species", nargs="*", default=None,
                        help="Species codes (default: all trained)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Also run the XGB/LightGBM/RF validation benchmark")
    parser.add_argument("--estimators", type=int, default=200,
                        help="Trees for the benchmark (default 200)")
    args = parser.parse_args()

    df = load_feature_matrix()
    codes = args.species or trained_species()
    unknown = [c for c in codes if not (MODELS_DIR / c / "bundle.pkl").exists()]
    if unknown:
        parser.error(f"No saved model bundle for species: {unknown}")

    bundles = {c: load_bundle(MODELS_DIR, c) for c in codes}
    _print_summary(bundles)

    for fig, name in [
        (figure_metrics(bundles), "model_comparison.png"),
        (figure_fit(bundles, df), "model_fit.png"),
        (figure_importance(bundles), "model_importance.png"),
    ]:
        out = OUT_DIR / name
        fig.savefig(out, dpi=140)
        plt.close(fig)
        logger.info("Saved %s", out)

    if args.benchmark:
        fig = _benchmark(df, bundles, args.estimators)
        out = OUT_DIR / "model_algorithm_benchmark.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        logger.info("Saved %s", out)


if __name__ == "__main__":
    main()