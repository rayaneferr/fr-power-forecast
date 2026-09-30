"""LightGBM baseline: one gradient-boosted model shared by the 24 delivery hours.

Each (day, hour) pair is a training row. Features are the hour and calendar, the same-hour prices
of d-1, d-2 and d-7, the level and range of the day d-1 prices, and for each exogenous forecast
its same-hour value on days d, d-1 and d-7 plus its daily mean on day d. Missing features are
left to LightGBM's native NaN handling. The point model minimises the absolute error; quantile
models are trained on request for probabilistic forecasts.

LightGBM is imported lazily: on macOS its OpenMP runtime (Homebrew libomp) and PyTorch's crash
when both are loaded in one process, which would break the foundation models.
"""

import numpy as np
import pandas as pd

from fr_power_forecast.models.base import TARGET, Forecaster, as_days, lagged
from fr_power_forecast.models.lear import DEFAULT_EXOG

DEFAULT_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 0,
    "verbose": -1,
}


def _per_hour(matrix: np.ndarray) -> np.ndarray:
    return matrix.ravel()


def _per_day(values: np.ndarray) -> np.ndarray:
    return np.repeat(values, 24)


class LightGBM(Forecaster):
    """LightGBM trained on the last ``window_days`` target days passed to :meth:`fit`."""

    def __init__(
        self,
        window_days: int = 1456,
        exog: tuple[str, ...] = DEFAULT_EXOG,
        quantiles: tuple[float, ...] = (),
        params: dict | None = None,
    ):
        self.window_days = window_days
        self.exog = tuple(exog)
        self.quantiles = tuple(sorted(quantiles))
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.name = f"lightgbm_{window_days}"

    def _features(self, panel: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
        yesterday = lagged(panel, TARGET, 1, days)
        columns = {
            "hour": np.tile(np.arange(24), len(days)),
            "dayofweek": _per_day(days.dayofweek.to_numpy()),
            "month": _per_day(days.month.to_numpy()),
            "price_lag1": _per_hour(yesterday),
            "price_lag2": _per_hour(lagged(panel, TARGET, 2, days)),
            "price_lag7": _per_hour(lagged(panel, TARGET, 7, days)),
            "price_lag1_mean": _per_day(np.nanmean(yesterday, axis=1)),
            "price_lag1_min": _per_day(np.nanmin(yesterday, axis=1)),
            "price_lag1_max": _per_day(np.nanmax(yesterday, axis=1)),
        }
        for var in self.exog:
            today = lagged(panel, var, 0, days)
            columns[var] = _per_hour(today)
            columns[f"{var}_mean"] = _per_day(np.nanmean(today, axis=1))
            columns[f"{var}_lag1"] = _per_hour(lagged(panel, var, 1, days))
            columns[f"{var}_lag7"] = _per_hour(lagged(panel, var, 7, days))
        return pd.DataFrame(columns)

    def _model(self, objective: str, **extra):
        import lightgbm

        return lightgbm.LGBMRegressor(objective=objective, **extra, **self.params)

    def fit(self, panel: pd.DataFrame, days) -> "LightGBM":
        days = as_days(days)[-self.window_days :]
        x = self._features(panel, days)
        y = _per_hour(lagged(panel, TARGET, 0, days))
        known = np.isfinite(y)
        x, y = x[known], y[known]

        self._point = self._model("l1").fit(x, y)
        self._quantile_models = [self._model("quantile", alpha=q).fit(x, y) for q in self.quantiles]
        return self

    def predict(self, panel: pd.DataFrame, days) -> pd.DataFrame:
        days = as_days(days)
        values = self._point.predict(self._features(panel, days)).reshape(len(days), 24)
        return self._frame(values, days)

    def predict_quantiles(self, panel: pd.DataFrame, days) -> np.ndarray:
        """(len(days), 24, n_quantiles) forecasts, sorted along the last axis to avoid crossing."""
        if not self.quantiles:
            raise ValueError("model was fitted without quantiles")
        days = as_days(days)
        x = self._features(panel, days)
        values = np.stack([m.predict(x) for m in self._quantile_models], axis=-1)
        return np.sort(values.reshape(len(days), 24, -1), axis=-1)
