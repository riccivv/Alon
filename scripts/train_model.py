"""Train per-species yield-forecast models on the processed feature matrix.

Usage:
    python scripts/train_model.py                       # all species
    python scripts/train_model.py --species scad mackerel
    python scripts/train_model.py --label-kind habitat_only
    python scripts/train_model.py --train-until 2022 --holdout 2025

Artifacts land in models/saved/<species_code>/ (see src/models/train.py).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS_DIR, TARGET_SPECIES  # noqa: E402
from src.models.labels import available_species  # noqa: E402
from src.models.train import (  # noqa: E402
    DEFAULT_HOLDOUT,
    DEFAULT_TRAIN_UNTIL,
    DEFAULT_VAL,
    resolve_features,
    train_species,
)
from src.processing.pipeline import FEATURES_PATH  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train")


def load_feature_matrix() -> "object":
    """Load the feature matrix (parquet preferred, pickle fallback)."""
    import pandas as pd

    parquet = Path(str(FEATURES_PATH).replace(".pkl", ".parquet"))
    if parquet.exists():
        logger.info("Loading feature matrix from %s", parquet)
        return pd.read_parquet(parquet)
    if FEATURES_PATH.exists():
        logger.info("Loading feature matrix from %s", FEATURES_PATH)
        return pd.read_pickle(FEATURES_PATH)
    raise FileNotFoundError(
        "No feature matrix found. Run scripts/run_pipeline.py first, "
        "or check scripts/run_pipeline.py --help."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train per-species XGBoost models")
    parser.add_argument(
        "--species", nargs="*", default=None,
        help="Species codes to train (default: all non-composite species)",
    )
    parser.add_argument(
        "--label-kind", choices=["prefer_catch", "habitat_only", "both"],
        default="prefer_catch",
        help="Label provider strategy (default: real catches if available, else habitat)",
    )
    parser.add_argument("--features", nargs="*", default=None,
                        help="Override the configured feature list")
    parser.add_argument("--train-until", type=int, default=DEFAULT_TRAIN_UNTIL)
    parser.add_argument("--val-start", type=int, default=DEFAULT_VAL[0])
    parser.add_argument("--val-end", type=int, default=DEFAULT_VAL[1])
    parser.add_argument("--holdout", type=int, default=DEFAULT_HOLDOUT)
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if a saved model already exists")
    args = parser.parse_args()

    df = load_feature_matrix()
    species = args.species or [s["code"] for s in available_species()]
    unknown = [s for s in species if s not in {t["code"] for t in TARGET_SPECIES}]
    if unknown:
        parser.error(f"Unknown species codes: {unknown}")

    features = resolve_features(df.columns, args.features)
    logger.info("Training %d species on %d features: %s", len(species), len(features), features)

    results = []
    for code in species:
        dest = args.out_dir / code / "metrics.json"
        if dest.exists() and not args.force:
            logger.info("Model exists for %s (--force to retrain). Skipping.", code)
            continue
        report = train_species(
            df,
            code,
            features=features,
            label_kind=args.label_kind,
            train_until=args.train_until,
            val=(args.val_start, args.val_end),
            holdout=args.holdout,
            out_dir=args.out_dir,
        )
        results.append(report)

    if results:
        print("\n=== Training summary ===")
        for r in results:
            vm = r["val_metrics"]
            print(
                f"{r['species_code']:<12} label={r['label_kind']:<8} "
                f"train={r['n_train']:>9,} val={r['n_val']:>7,} "
                f"holdout={r['n_holdout']:>7,} | val RMSE={vm.get('rmse', float('nan')):.4f} "
                f"R2={vm.get('r2', float('nan')):.3f}"
            )
        print(f"\nArtifacts in {args.out_dir}")


if __name__ == "__main__":
    main()