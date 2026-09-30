# Backtest

## Protocol

Rolling-origin backtest following Lago et al. (2021). Each test day `d` is forecast with a model
calibrated on the days before it, using prices up to `d - 1` and exogenous forecasts for `d` (the
information available at gate closure). Models are recalibrated every `--recalibrate-every` days
(daily by default) on their last `--window-days` days (1456 by default).

```bash
uv run fr-power-forecast backtest --model naive --start 2022-01-01
uv run fr-power-forecast backtest --model lear --start 2022-01-01 --n-jobs 12
uv run fr-power-forecast evaluate                       # all results in results/
uv run fr-power-forecast evaluate --start 2023-01-01    # one period only
```

Forecasts are written to `results/<model>/forecasts.parquet` (one row per day and hour, with
`calibrated_until`, the last day of calibration data) and fit and predict times to
`results/<model>/timings.parquet`. `evaluate` scores every model on the days they all cover and
prints, per model, the last recalibration date. When the dataset has price data beyond a
backtest, it says so: results are only up to date after a recalibration, i.e. rebuilding the
dataset and running `backtest --resume`, which forecasts only the new days.

## Results (January 2022 to September 2026)

Last recalibration of both models: on data up to 2026-09-29. Daily recalibration, 1,734 test
days. The DM p-value tests whether the model is more accurate than the naive benchmark.

| Period | Days | Model | MAE (EUR/MWh) | rMAE | RMSE | DM p-value |
|---|---:|---|---:|---:|---:|---:|
| 2022-2026 | 1,734 | Seasonal naive | 30.37 | 1.000 | 55.14 | |
| | | LEAR | **20.09** | **0.662** | **33.00** | < 1e-15 |
| 2022 (energy crisis) | 365 | Seasonal naive | 52.97 | 1.000 | 98.57 | |
| | | LEAR | **35.53** | **0.671** | **57.82** | < 1e-15 |
| 2023-2026 | 1,369 | Seasonal naive | 24.34 | 1.000 | 35.51 | |
| | | LEAR | **15.97** | **0.656** | **22.08** | < 1e-15 |

- LEAR cuts the error by about a third in both regimes. The absolute MAE more than halves after
  2022 because price levels dropped, while the rMAE barely moves: this is why rMAE is the
  headline metric across regimes.
- sMAPE is not reported in the table: with prices close to zero or negative (frequent since 2023)
  it explodes and ranks models on a handful of hours.
- Cost: one LEAR calibration takes 5.2 s on average (single thread, 12 in parallel on an Apple
  M4 Pro) and one forecast 4 ms; the whole daily backtest takes 13 minutes.

LightGBM is not backtested yet: with its quantile models one calibration takes about 40 s on all
cores, so it will run with a coarser recalibration in a later release.
