# Baseline models

All models live in `fr_power_forecast.models` and share one interface. The hourly dataset is
first reshaped into a daily panel, one row per delivery day and one column per (variable, hour):

```python
from fr_power_forecast.models import LEAR, daily_panel

panel = daily_panel(pd.read_parquet("data/processed/epex_fr_hourly.parquet"))
model = LEAR().fit(panel, train_days)
forecast = model.predict(panel, test_days)  # (days x 24) prices
```

Information set: the forecast for day `d` reads prices up to `d - 1` and exogenous forecasts up
to `d`. A test tampers with every later price and exogenous value and checks that each model's
forecast does not change.

| Model | Description |
|---|---|
| `SeasonalNaive` | Lago et al. (2021) naive: day `d - 7` for Monday, Saturday and Sunday, day `d - 1` otherwise. Reference model of rMAE. |
| `LEAR` | One LASSO per hour on the 24 prices of `d-1`, `d-2`, `d-3`, `d-7`, each exogenous series on `d`, `d-1`, `d-7`, and day-of-week dummies. Median/MAD + asinh scaling, penalty chosen by AIC. |
| `LightGBM` | One model for all hours on (day, hour) rows: hour, calendar, same-hour price lags, day `d-1` price level and range, exogenous values and daily means. L1 objective; optional quantile models. |

Default exogenous columns: `load_forecast`, `forecast_wind_speed_100m`,
`forecast_shortwave_radiation`. Both models calibrate on the last `window_days` (1456 by
default) target days they are given.

## Sanity check

Not a benchmark result: a single calibration on 2021-2024, no recalibration, tested on the
365 days of 2025. The rolling-origin backtest comes in v0.4.

| Model | MAE (EUR/MWh) | rMAE | Fit time (M4 Pro) |
|---|---:|---:|---:|
| Seasonal naive | 23.96 | 1.000 | - |
| LEAR | 15.65 | 0.653 | 3 s |
| LightGBM (point + 3 quantiles) | 16.03 | 0.669 | 40 s |

The LightGBM 10-90 % interval covers 62 % of the 2025 prices against 80 % nominal, which is
what the conformal calibration of v0.6 is meant to fix.
