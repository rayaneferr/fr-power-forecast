"""Rolling-origin backtest and evaluation (Lago et al., 2021).

Test days are walked forward in blocks of ``recalibrate_every`` days. Before each block the model
is recalibrated on every day before the block, then it forecasts the days of the block. Each
forecast records the last day of its calibration data (``calibrated_until``), so every result
table can say how old the model behind it is. Blocks are independent and can run in parallel.
"""

import copy
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fr_power_forecast import metrics
from fr_power_forecast.models.base import TARGET, Forecaster, as_days


@dataclass
class Backtest:
    """Long-format forecasts indexed by (date, hour) and one timing row per recalibration."""

    forecasts: pd.DataFrame
    timings: pd.DataFrame


def _quantile_column(q: float) -> str:
    return f"q{q:g}"


def _run_block(model: Forecaster, panel: pd.DataFrame, block: pd.DatetimeIndex):
    model = copy.deepcopy(model)
    train = panel.index[panel.index < block[0]]
    start = time.perf_counter()
    model.fit(panel, train)
    fitted = time.perf_counter()
    point = model.predict(panel, block)
    quantiles = model.predict_quantiles(panel, block) if getattr(model, "quantiles", ()) else None
    predicted = time.perf_counter()

    frame = point.stack(future_stack=True).rename("forecast").to_frame()
    if quantiles is not None:
        for i, q in enumerate(model.quantiles):
            frame[_quantile_column(q)] = quantiles[:, :, i].ravel()
    frame["calibrated_until"] = train[-1]
    timing = {
        "calibrated_until": train[-1],
        "first_day": block[0],
        "n_days": len(block),
        "fit_seconds": fitted - start,
        "predict_seconds": predicted - fitted,
    }
    return frame, timing


def rolling_backtest(
    model: Forecaster,
    panel: pd.DataFrame,
    test_days,
    recalibrate_every: int = 1,
    n_jobs: int = 1,
) -> Backtest:
    """Forecast every day of ``test_days``, recalibrating ``model`` every ``recalibrate_every``."""
    test_days = as_days(test_days)
    if len(test_days) == 0:
        raise ValueError("no test days")
    if test_days[0] <= panel.index[0]:
        raise ValueError("the first test day needs at least one day of history")
    blocks = [
        test_days[i : i + recalibrate_every] for i in range(0, len(test_days), recalibrate_every)
    ]

    if n_jobs == 1:
        results = [_run_block(model, panel, block) for block in blocks]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            n = len(blocks)
            results = list(pool.map(_run_block, [model] * n, [panel] * n, blocks))

    forecasts = pd.concat([frame for frame, _ in results]).sort_index()
    timings = pd.DataFrame([timing for _, timing in results])
    return Backtest(forecasts=forecasts, timings=timings)


def extend(previous: Backtest, new: Backtest) -> Backtest:
    """Append the forecasts of later days to a stored backtest."""
    overlap = previous.forecasts.index.intersection(new.forecasts.index)
    if len(overlap):
        raise ValueError(f"{len(overlap)} (date, hour) pairs are already in the backtest")
    return Backtest(
        forecasts=pd.concat([previous.forecasts, new.forecasts]).sort_index(),
        timings=pd.concat([previous.timings, new.timings], ignore_index=True),
    )


def save(backtest: Backtest, directory) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    backtest.forecasts.to_parquet(directory / "forecasts.parquet")
    backtest.timings.to_parquet(directory / "timings.parquet")


def load(directory) -> Backtest:
    directory = Path(directory)
    return Backtest(
        forecasts=pd.read_parquet(directory / "forecasts.parquet"),
        timings=pd.read_parquet(directory / "timings.parquet"),
    )


def _daily(forecasts: pd.DataFrame, column: str) -> pd.DataFrame:
    return forecasts[column].unstack("hour")


def quantile_levels(forecasts: pd.DataFrame) -> list[float]:
    return sorted(float(c[1:]) for c in forecasts.columns if c.startswith("q"))


def evaluate(
    results: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    reference: str = "naive",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Accuracy of each model on the days where every model and the true price are available.

    ``start`` and ``end`` (exclusive) restrict the scored days, e.g. to one market regime.

    ``reference`` is the model rMAE is relative to and that the Diebold-Mariano test compares
    against (a small ``dm_p_value`` means the model is significantly more accurate).
    """
    if reference not in results:
        raise ValueError(f"reference model {reference!r} is missing from the results")
    points = {name: _daily(frame, "forecast") for name, frame in results.items()}
    truth = panel[TARGET]
    days = truth.index[truth.notna().all(axis=1)]
    if start is not None:
        days = days[days >= start]
    if end is not None:
        days = days[days < end]
    for point in points.values():
        days = days.intersection(point.index[point.notna().all(axis=1)])
    if len(days) < 2:
        raise ValueError("fewer than two days are common to all models")

    y = truth.loc[days].to_numpy()
    ref = points[reference].loc[days].to_numpy()
    rows = []
    for name, point in points.items():
        pred = point.loc[days].to_numpy()
        row = {
            "model": name,
            "mae": metrics.mae(y, pred),
            "rmae": metrics.rmae(y, pred, ref),
            "rmse": metrics.rmse(y, pred),
            "smape": metrics.smape(y, pred),
            "dm_p_value": np.nan
            if name == reference
            else metrics.diebold_mariano(y, ref, pred).p_value,
        }
        levels = quantile_levels(results[name])
        if levels:
            q = np.stack(
                [_daily(results[name], _quantile_column(lv)).loc[days] for lv in levels], axis=-1
            )
            row["pinball"] = metrics.pinball_loss(y, q, levels)
            row[f"coverage_{levels[0]:g}_{levels[-1]:g}"] = metrics.interval_coverage(
                y, q[..., 0], q[..., -1]
            )
        rows.append(row)
    table = pd.DataFrame(rows).set_index("model")
    table.attrs["days"] = days
    return table


def freshness(results: dict[str, pd.DataFrame], panel: pd.DataFrame) -> pd.DataFrame:
    """Per model: last forecast day, last recalibration, and the days the data has beyond it."""
    known = panel.index[panel[TARGET].notna().all(axis=1)]
    rows = []
    for name, frame in results.items():
        dates = frame.index.get_level_values("date")
        rows.append(
            {
                "model": name,
                "last_forecast_day": dates.max(),
                "last_recalibration": frame["calibrated_until"].max(),
                "days_not_backtested": int((known > dates.max()).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("model")
