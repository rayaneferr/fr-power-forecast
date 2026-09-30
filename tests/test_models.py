import numpy as np
import pandas as pd
import pytest

from fr_power_forecast.metrics import mae
from fr_power_forecast.models import LEAR, LightGBM, SeasonalNaive, daily_panel
from fr_power_forecast.models.base import lagged

FAST_GBM = {"n_estimators": 50, "num_leaves": 15}


def synthetic_panel(n_days: int = 400, seed: int = 0) -> pd.DataFrame:
    """Prices driven by the same-day load forecast plus a daily shape and AR(1) day level."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2024-01-01", periods=n_days, freq="D", name="date")
    shape = 10 * np.sin(np.arange(24) / 24 * 2 * np.pi)
    load = 50_000 + 8_000 * rng.standard_normal((n_days, 1)) + 2_000 * shape / 10
    load += 500 * rng.standard_normal((n_days, 24))  # hourly regressors must not be collinear
    level = np.zeros(n_days)
    for i in range(1, n_days):
        level[i] = 0.7 * level[i - 1] + 5 * rng.standard_normal()
    price = 20 + (load - 50_000) / 200 + shape + level[:, None] + rng.standard_normal((n_days, 24))
    frames = {"price": price, "load_forecast": load}
    return pd.concat(
        {k: pd.DataFrame(v, index=days, columns=range(24)) for k, v in frames.items()},
        axis=1,
        names=["variable", "hour"],
    )


def test_daily_panel_has_one_column_per_variable_and_hour():
    index = pd.date_range(
        "2025-03-29", "2025-04-01", freq="1h", inclusive="left", tz="Europe/Paris"
    )
    hourly = pd.DataFrame(
        {"price": np.arange(len(index), dtype=float), "load_forecast": 1.0},
        index=index.tz_convert("UTC"),
    )
    panel = daily_panel(hourly)
    assert panel.shape == (3, 48)
    assert panel["price"].notna().all().all()
    assert lagged(panel, "price", 1, pd.DatetimeIndex(["2025-03-30"]))[0, 0] == 0.0


def test_naive_uses_last_week_on_monday_and_weekends_and_yesterday_otherwise():
    panel = synthetic_panel(30)
    days = pd.date_range("2024-01-15", periods=7, freq="D")  # Monday to Sunday
    forecast = SeasonalNaive().predict(panel, days)
    for day in days:
        lag = 7 if day.dayofweek in (0, 5, 6) else 1
        expected = panel["price"].loc[day - pd.Timedelta(days=lag)].to_numpy()
        np.testing.assert_array_equal(forecast.loc[day].to_numpy(), expected)


def test_naive_is_nan_without_history():
    panel = synthetic_panel(10)
    assert SeasonalNaive().predict(panel, panel.index[:1]).isna().all().all()


MODELS = [
    lambda: SeasonalNaive(),
    lambda: LEAR(window_days=300, exog=("load_forecast",)),
    lambda: LightGBM(window_days=300, exog=("load_forecast",), params=FAST_GBM),
]


@pytest.mark.parametrize("make_model", MODELS)
def test_forecast_does_not_read_future_prices_or_exogenous_data(make_model):
    panel = synthetic_panel()
    day = panel.index[350]
    model = make_model().fit(panel, panel.index[:350])
    before = model.predict(panel, [day])

    tampered = panel.copy()
    tampered.loc[tampered.index >= day, "price"] = 1e6
    tampered.loc[tampered.index > day, "load_forecast"] = -1e6
    pd.testing.assert_frame_equal(model.predict(tampered, [day]), before)


@pytest.mark.parametrize("make_model", MODELS[1:])
def test_models_beat_the_naive_benchmark(make_model):
    panel = synthetic_panel()
    train, test = panel.index[:340], panel.index[340:]
    model = make_model().fit(panel, train)
    truth = panel["price"].loc[test].to_numpy()
    naive = SeasonalNaive().predict(panel, test).to_numpy()
    assert mae(truth, model.predict(panel, test).to_numpy()) < 0.5 * mae(truth, naive)


def test_lear_uses_only_the_calibration_window():
    panel = synthetic_panel()
    short = LEAR(window_days=250, exog=("load_forecast",)).fit(panel, panel.index[:300])
    same = LEAR(window_days=250, exog=("load_forecast",)).fit(panel, panel.index[50:300])
    test = panel.index[300:310]
    pd.testing.assert_frame_equal(short.predict(panel, test), same.predict(panel, test))


def test_lear_rejects_a_window_shorter_than_its_regressors():
    panel = synthetic_panel()
    with pytest.raises(ValueError, match="regressors"):
        LEAR(window_days=100, exog=("load_forecast",)).fit(panel, panel.index[:300])


def test_lear_skips_days_with_missing_regressors():
    panel = synthetic_panel()
    panel.iloc[100:110, 0] = np.nan
    model = LEAR(window_days=300, exog=("load_forecast",)).fit(panel, panel.index[:340])
    forecast = model.predict(panel, panel.index[[105, 108, 340]])
    assert forecast.iloc[0].isna().all()  # d-3 price missing
    assert forecast.iloc[2].notna().all()


def test_lightgbm_quantiles_are_ordered_and_cover_the_price():
    panel = synthetic_panel()
    train, test = panel.index[:340], panel.index[340:]
    model = LightGBM(
        window_days=300, exog=("load_forecast",), quantiles=(0.9, 0.1, 0.5), params=FAST_GBM
    ).fit(panel, train)
    q = model.predict_quantiles(panel, test)
    assert q.shape == (len(test), 24, 3)
    assert (np.diff(q, axis=-1) >= 0).all()
    truth = panel["price"].loc[test].to_numpy()
    coverage = np.mean((truth >= q[..., 0]) & (truth <= q[..., 2]))
    assert 0.6 < coverage < 0.97


def test_lightgbm_without_quantiles_refuses_quantile_forecasts():
    panel = synthetic_panel(60)
    model = LightGBM(window_days=50, exog=("load_forecast",), params=FAST_GBM)
    model.fit(panel, panel.index[10:50])
    with pytest.raises(ValueError, match="quantiles"):
        model.predict_quantiles(panel, panel.index[50:])
