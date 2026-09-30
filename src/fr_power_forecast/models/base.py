"""Daily panel of the modelling dataset and the interface shared by all forecasters.

A panel has one row per delivery day (Paris local date) and one column per (variable, hour).
Every model forecasts the 24 prices of a delivery day ``d`` under the day-ahead information set:
prices up to day ``d - 1`` and exogenous forecasts up to day ``d``.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from fr_power_forecast.data.timegrid import to_daily_matrix

TARGET = "price"
HOURS = range(24)


def daily_panel(hourly: pd.DataFrame) -> pd.DataFrame:
    """Reshape the hourly UTC dataset into a (day x (variable, hour)) panel."""
    matrices = {column: to_daily_matrix(hourly[column]) for column in hourly.columns}
    return pd.concat(matrices, axis=1, names=["variable", "hour"])


def lagged(panel: pd.DataFrame, variable: str, lag: int, days: pd.DatetimeIndex) -> np.ndarray:
    """(len(days), 24) values of ``variable`` on each day minus ``lag`` days. NaN if absent."""
    return panel[variable].reindex(days - pd.Timedelta(days=lag)).to_numpy(dtype=float)


def as_days(days) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(days, name="date")


class Forecaster(ABC):
    """Forecaster of the 24 hourly prices of each delivery day.

    ``fit`` uses the prices of the target ``days`` it is given (and lags of them); ``predict``
    must only read prices strictly before each forecast day and exogenous columns up to it.
    """

    name: str
    zero_shot = False  # True for models that are never fitted on this dataset

    def fit(self, panel: pd.DataFrame, days) -> "Forecaster":
        return self

    @abstractmethod
    def predict(self, panel: pd.DataFrame, days) -> pd.DataFrame:
        """(len(days), 24) point forecasts indexed by day."""

    def predict_with_quantiles(self, panel: pd.DataFrame, days):
        """Point forecasts and, for models with ``quantiles``, the (days, 24, n) quantiles."""
        point = self.predict(panel, days)
        has_quantiles = bool(getattr(self, "quantiles", ()))
        return point, self.predict_quantiles(panel, days) if has_quantiles else None

    @staticmethod
    def _frame(values: np.ndarray, days: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame(values, index=as_days(days), columns=pd.Index(HOURS, name="hour"))
