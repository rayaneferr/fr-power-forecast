"""LEAR: LASSO-estimated autoregressive model of Lago et al. (2021).

One linear model per delivery hour, all sharing the same regressors: the 24 prices of days
d-1, d-2, d-3 and d-7, each exogenous forecast on days d, d-1 and d-7, and day-of-week dummies.
Prices and regressors are normalised with the median and MAD of the calibration window and then
passed through asinh, a variance-stabilising transform that tames price spikes (Uniejewski et
al., 2018). The LASSO penalty of each hour is selected by AIC on the LARS path.
"""

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.linear_model import Lasso, LassoLarsIC

from fr_power_forecast.models.base import HOURS, TARGET, Forecaster, as_days, lagged

PRICE_LAGS = (1, 2, 3, 7)
EXOG_LAGS = (0, 1, 7)
DEFAULT_EXOG = ("load_forecast", "forecast_wind_speed_100m", "forecast_shortwave_radiation")


class AsinhScaler:
    """Column-wise ``asinh((x - median) / MAD)``; constant columns get a unit scale."""

    def fit(self, x: np.ndarray) -> "AsinhScaler":
        self.median = np.median(x, axis=0)
        mad = median_abs_deviation(x, axis=0)
        self.mad = np.where(mad > 0, mad, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.arcsinh((x - self.median) / self.mad)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.sinh(z) * self.mad + self.median


class LEAR(Forecaster):
    """LEAR calibrated on the last ``window_days`` target days passed to :meth:`fit`."""

    def __init__(self, window_days: int = 1456, exog: tuple[str, ...] = DEFAULT_EXOG):
        self.window_days = window_days
        self.exog = tuple(exog)
        self.name = f"lear_{window_days}"

    def _regressors(self, panel: pd.DataFrame, days: pd.DatetimeIndex):
        blocks = [lagged(panel, TARGET, lag, days) for lag in PRICE_LAGS]
        blocks += [lagged(panel, var, lag, days) for var in self.exog for lag in EXOG_LAGS]
        return np.hstack(blocks), np.eye(7)[days.dayofweek]

    def fit(self, panel: pd.DataFrame, days) -> "LEAR":
        days = as_days(days)[-self.window_days :]
        x, dummies = self._regressors(panel, days)
        y = lagged(panel, TARGET, 0, days)
        complete = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        if complete.sum() <= x.shape[1] + dummies.shape[1]:
            raise ValueError(
                f"{complete.sum()} complete days for {x.shape[1] + dummies.shape[1]} regressors"
            )
        x, dummies, y = x[complete], dummies[complete], y[complete]

        # Constant regressors (e.g. night-time solar radiation) only destabilise the LARS path.
        self._varying = np.ptp(x, axis=0) > 0
        x = x[:, self._varying]
        self._x_scaler = AsinhScaler().fit(x)
        self._y_scaler = AsinhScaler().fit(y)
        design = np.hstack([self._x_scaler.transform(x), dummies])
        target = self._y_scaler.transform(y)

        self._models = []
        for hour in HOURS:
            alpha = LassoLarsIC(criterion="aic", max_iter=2500).fit(design, target[:, hour]).alpha_
            self._models.append(Lasso(alpha=alpha, max_iter=10_000).fit(design, target[:, hour]))
        return self

    def predict(self, panel: pd.DataFrame, days) -> pd.DataFrame:
        days = as_days(days)
        x, dummies = self._regressors(panel, days)
        complete = np.isfinite(x).all(axis=1)
        out = np.full((len(days), 24), np.nan)
        if complete.any():
            x = x[complete][:, self._varying]
            design = np.hstack([self._x_scaler.transform(x), dummies[complete]])
            z = np.column_stack([model.predict(design) for model in self._models])
            out[complete] = self._y_scaler.inverse(z)
        return self._frame(out, days)
