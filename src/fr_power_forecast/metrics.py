"""Evaluation metrics for day-ahead price forecasts.

Point forecasts follow Lago et al. (2021); probabilistic forecasts are represented as a set of
quantiles, with the quantile levels on the last axis of the prediction array.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def _as_same_shape(*arrays: ArrayLike) -> list[np.ndarray]:
    out = [np.asarray(a, dtype=float) for a in arrays]
    if any(a.shape != out[0].shape for a in out[1:]):
        raise ValueError(f"shape mismatch: {[a.shape for a in out]}")
    return out


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error."""
    y, p = _as_same_shape(y_true, y_pred)
    return float(np.mean(np.abs(y - p)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error."""
    y, p = _as_same_shape(y_true, y_pred)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def smape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Symmetric MAPE in percent, as used in the EPF literature.

    Pairs where both the price and the forecast are zero contribute no error.
    """
    y, p = _as_same_shape(y_true, y_pred)
    denom = np.abs(y) + np.abs(p)
    ratio = np.divide(2 * np.abs(y - p), denom, out=np.zeros_like(denom), where=denom > 0)
    return float(100 * np.mean(ratio))


def rmae(y_true: ArrayLike, y_pred: ArrayLike, y_naive: ArrayLike) -> float:
    """Relative MAE: MAE of the forecast divided by the MAE of a naive forecast.

    Scale-free, so it stays comparable across periods with very different price levels
    (e.g. before and during the 2022 energy crisis). Below 1 means better than naive.
    """
    y, p, n = _as_same_shape(y_true, y_pred, y_naive)
    naive_mae = np.mean(np.abs(y - n))
    if naive_mae == 0:
        raise ValueError("naive forecast is perfect, rMAE is undefined")
    return float(np.mean(np.abs(y - p)) / naive_mae)


def _check_quantiles(y_true: ArrayLike, q_pred: ArrayLike, quantiles: ArrayLike):
    y = np.asarray(y_true, dtype=float)
    q = np.asarray(q_pred, dtype=float)
    levels = np.asarray(quantiles, dtype=float)
    if levels.ndim != 1 or np.any((levels <= 0) | (levels >= 1)):
        raise ValueError("quantile levels must be a 1-D array in (0, 1)")
    if q.shape != (*y.shape, levels.size):
        raise ValueError(f"q_pred shape {q.shape} != y_true shape {y.shape} + ({levels.size},)")
    return y, q, levels


def pinball_loss(y_true: ArrayLike, q_pred: ArrayLike, quantiles: ArrayLike) -> float:
    """Pinball (quantile) loss averaged over observations and quantile levels.

    ``q_pred`` has shape ``y_true.shape + (len(quantiles),)``.
    """
    y, q, levels = _check_quantiles(y_true, q_pred, quantiles)
    diff = y[..., None] - q
    return float(np.mean(np.maximum(levels * diff, (levels - 1) * diff)))


def crps_from_quantiles(y_true: ArrayLike, q_pred: ArrayLike, quantiles: ArrayLike) -> float:
    """CRPS approximated as twice the mean pinball loss over the quantile levels.

    The approximation improves with the number of evenly spaced levels.
    """
    return 2 * pinball_loss(y_true, q_pred, quantiles)


def interval_coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Share of observations falling inside [lower, upper]."""
    y, lo, hi = _as_same_shape(y_true, lower, upper)
    if np.any(lo > hi):
        raise ValueError("lower bound above upper bound")
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_interval_width(lower: ArrayLike, upper: ArrayLike) -> float:
    """Average width of the prediction intervals (sharpness)."""
    lo, hi = _as_same_shape(lower, upper)
    return float(np.mean(hi - lo))


@dataclass(frozen=True)
class DMResult:
    statistic: float
    p_value: float


def diebold_mariano(
    y_true: ArrayLike, pred_1: ArrayLike, pred_2: ArrayLike, norm: int = 1
) -> DMResult:
    """Multivariate one-sided Diebold-Mariano test (Lago et al., 2021).

    Inputs have shape ``(n_days, 24)``. The 24 hourly errors of each day are aggregated into one
    daily loss (mean absolute error for ``norm=1``, mean squared error for ``norm=2``), so the
    loss differential is a single series with no intra-day correlation to correct for.

    H0: ``pred_2`` is not more accurate than ``pred_1``. A small p-value means ``pred_2`` is
    significantly better.
    """
    y, p1, p2 = _as_same_shape(y_true, pred_1, pred_2)
    if y.ndim != 2:
        raise ValueError("expected arrays of shape (n_days, n_hours)")
    if norm not in (1, 2):
        raise ValueError("norm must be 1 or 2")

    loss_1 = np.mean(np.abs(y - p1) ** norm, axis=1)
    loss_2 = np.mean(np.abs(y - p2) ** norm, axis=1)
    delta = loss_1 - loss_2

    n_days = delta.size
    if n_days < 2:
        raise ValueError("need at least two days")
    std = np.std(delta, ddof=1)
    if std == 0:
        raise ValueError("loss differential is constant, DM statistic is undefined")

    statistic = float(np.mean(delta) / (std / np.sqrt(n_days)))
    return DMResult(statistic=statistic, p_value=float(1 - stats.norm.cdf(statistic)))
