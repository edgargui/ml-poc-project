"""Chargement et préparation du dataset Mutual Funds.

Target : R_3_to_5 — rendement annualisé pur entre la fin de l'an 3 et la fin de l'an 5,
calculé par intérêts composés pour isoler la performance post-an3.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import MUTUAL_FUNDS_CSV

# Colonnes disponibles à l'issue de l'an 3 uniquement (anti-leakage)
FEATURES = [
    "fund_return_1year",
    "fund_return_3years",
    "fund_yield",
    "fund_sector_technology",
    "fund_sector_financial_services",
    "fund_sector_healthcare",
    "fund_sector_energy",
    "fund_sector_industrials",
    "fund_annual_report_net_expense_ratio",
]


def _compute_target(df: pd.DataFrame) -> pd.Series:
    """R_3_to_5 = ((1+R5)^5 / (1+R3)^3)^(1/2) - 1, exprimé en pourcentage.

    Les colonnes fund_return_* sont déjà en décimal (0.138 = 13.8%),
    donc pas de division par 100 — on multiplie uniquement la sortie par 100.
    """
    r3 = df["fund_return_3years"]
    r5 = df["fund_return_5years"]

    ratio = (1.0 + r5) ** 5 / (1.0 + r3) ** 3

    # Les ratios négatifs produiraient un NaN par ** 0.5 — on les masque
    target = np.where(ratio > 0, ratio ** 0.5 - 1.0, np.nan) * 100.0
    return pd.Series(target, index=df.index, name="R_3_to_5")


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Charge le CSV, calcule la target et retourne (X_train, X_test, y_train, y_test)."""
    df = pd.read_csv(MUTUAL_FUNDS_CSV, low_memory=False)

    df["R_3_to_5"] = _compute_target(df)

    # Supprime les lignes sans target valide
    df = df.dropna(subset=["R_3_to_5", "fund_return_3years", "fund_return_5years"])
    df = df[~np.isinf(df["R_3_to_5"])]

    X = df[FEATURES].copy()
    y = df["R_3_to_5"].copy()

    return tuple(train_test_split(X, y, test_size=0.20, random_state=42))
