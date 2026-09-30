"""Seasonal naive benchmark of Lago et al. (2021), the reference model of rMAE."""

import numpy as np
import pandas as pd

from fr_power_forecast.models.base import TARGET, Forecaster, as_days, lagged

# Monday, Saturday and Sunday are forecast by the same day one week earlier.
WEEKLY_DAYS = (0, 5, 6)


class SeasonalNaive(Forecaster):
    """Prices of the same day last week for Monday and weekends, of the previous day otherwise."""

    name = "naive"

    def predict(self, panel: pd.DataFrame, days) -> pd.DataFrame:
        days = as_days(days)
        weekly = np.isin(days.dayofweek, WEEKLY_DAYS)[:, None]
        values = np.where(weekly, lagged(panel, TARGET, 7, days), lagged(panel, TARGET, 1, days))
        return self._frame(values, days)
