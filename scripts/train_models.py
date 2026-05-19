"""Entraînement et sauvegarde des 3 pipelines ML.

Usage :
    python scripts/train_models.py

Chaque pipeline enchaîne :
    SimpleImputer(median) → StandardScaler → régresseur

Les modèles sont sauvegardés dans models/ au format .joblib.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permet les imports absolus depuis src/ même sans variable d'env PYTHONPATH
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import MODELS_DIR
from data import FEATURES, load_dataset_split


def _build_pipelines() -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", HistGradientBoostingRegressor(
                max_iter=500,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
            )),
        ]),
    }


def main() -> None:
    print("Chargement du dataset...")
    X_train, X_test, y_train, y_test = load_dataset_split()
    print(f"  Train : {X_train.shape[0]} lignes | Test : {X_test.shape[0]} lignes")
    print(f"  Features ({len(FEATURES)}) : {FEATURES}\n")

    pipelines = _build_pipelines()

    for key, pipeline in pipelines.items():
        print(f"Entraînement : {key}...", end=" ", flush=True)
        pipeline.fit(X_train, y_train)

        out_path = MODELS_DIR / f"{key}.joblib"
        joblib.dump(pipeline, out_path)
        print(f"sauvegarde : {out_path}")

    print("\nTous les modeles sont prets. Lancez maintenant : python scripts/main.py")


if __name__ == "__main__":
    main()
