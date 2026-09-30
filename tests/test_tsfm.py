import os

import numpy as np
import pandas as pd
import pytest
from test_models import synthetic_panel

from fr_power_forecast import backtest as bt
from fr_power_forecast.models import SeasonalNaive
from fr_power_forecast.models.tsfm import Chronos2, TimesFM3, ZeroShot


class FakeZeroShot(ZeroShot):
    """Last context day plus the future load, spread by a fixed band for the quantiles."""

    label = "fake"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def _forecast(self, inputs):
        self.calls += 1
        point = np.stack([x["target"][-24:] for x in inputs])
        if self.exog:
            point = point + np.stack([x["future"]["load_forecast"] for x in inputs]) / 1e4
        offsets = (np.array(self.quantiles) - 0.5) * 10
        return point[..., None] + offsets


def test_contexts_end_the_day_before_and_future_covariates_cover_the_day():
    panel = synthetic_panel(60)
    day = panel.index[50]
    (x,) = FakeZeroShot(context_days=14, exog=("load_forecast",)).contexts(panel, [day])
    assert x["target"].shape == (14 * 24,)
    np.testing.assert_array_equal(x["target"][-24:], panel["price"].loc[panel.index[49]])
    np.testing.assert_array_equal(x["past"]["load_forecast"][:24], panel["load_forecast"].iloc[36])
    np.testing.assert_array_equal(x["future"]["load_forecast"], panel["load_forecast"].loc[day])


def test_contexts_fill_gaps_and_pad_before_the_first_day():
    panel = synthetic_panel(30)
    panel.iloc[20, 3] = np.nan
    (x,) = FakeZeroShot(context_days=28).contexts(panel, [panel.index[25]])
    assert np.isfinite(x["target"]).all()
    assert x["target"].shape == (28 * 24,)


def test_zero_shot_forecast_does_not_read_future_data():
    panel = synthetic_panel(80)
    day = panel.index[60]
    model = FakeZeroShot(context_days=28, exog=("load_forecast",))
    before = model.predict(panel, [day])
    tampered = panel.copy()
    tampered.loc[tampered.index >= day, "price"] = 1e6
    tampered.loc[tampered.index > day, "load_forecast"] = -1e6
    pd.testing.assert_frame_equal(model.predict(tampered, [day]), before)


def test_point_forecast_is_the_median_and_inference_runs_once():
    panel = synthetic_panel(80)
    model = FakeZeroShot(context_days=28, batch_size=4)
    days = panel.index[40:50]
    point, quantiles = model.predict_with_quantiles(panel, days)
    assert model.calls == 3  # 10 days in batches of 4
    assert quantiles.shape == (10, 24, 9)
    np.testing.assert_array_equal(point.to_numpy(), quantiles[..., 4])


def test_zero_shot_models_require_the_median_and_timesfm_the_deciles():
    with pytest.raises(ValueError, match="median"):
        FakeZeroShot(quantiles=(0.1, 0.9))
    with pytest.raises(ValueError, match="deciles"):
        TimesFM3(quantiles=(0.1, 0.5, 0.9))
    assert Chronos2(exog=("load_forecast",)).name == "chronos2_exog"


def test_backtest_runs_zero_shot_models_in_one_batch_without_calibration():
    panel = synthetic_panel(80)
    days = panel.index[40:60]
    result = bt.rolling_backtest(FakeZeroShot(context_days=28), panel, days, recalibrate_every=1)
    assert len(result.timings) == 1
    assert result.forecasts["calibrated_until"].isna().all()
    assert {"forecast", "q0.1", "q0.9"} <= set(result.forecasts.columns)

    results = {
        "naive": bt.rolling_backtest(SeasonalNaive(), panel, days).forecasts,
        "fake": result.forecasts,
    }
    table = bt.evaluate(results, panel)
    assert table.loc["fake", "coverage_0.1_0.9"] >= 0
    assert pd.isna(bt.freshness(results, panel).loc["fake", "last_recalibration"])


@pytest.mark.skipif(not os.environ.get("RUN_TSFM"), reason="set RUN_TSFM=1 (downloads weights)")
@pytest.mark.parametrize("model_class", [Chronos2, TimesFM3])
def test_real_foundation_model_forecasts_negative_prices(model_class):
    pytest.importorskip("torch")
    panel = synthetic_panel(80)
    panel["price"] -= 60  # mostly negative prices
    model = model_class(context_days=28, exog=("load_forecast",), device="cpu")
    point, quantiles = model.predict_with_quantiles(panel, panel.index[70:72])
    assert quantiles.shape == (2, 24, 9)
    assert (point.to_numpy() < 0).any()
