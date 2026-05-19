"""Métriques d'évaluation pour la régression du rendement R_3_to_5."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Retourne MAE, RMSE, R² et Win Rate pour une paire (y_true, y_pred).

    Win Rate = proportion de fonds pour lesquels le signe du rendement prédit
    correspond au signe du rendement réel (précision directionnelle).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    # Accord de signe : les deux positifs ou les deux négatifs
    win_rate = float(np.mean(np.sign(y_true) == np.sign(y_pred)))

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Win Rate": round(win_rate, 4),
    }
