import numpy as np
import pandas as pd
import pytest
from test_models import FAST_GBM, synthetic_panel

from fr_power_forecast import backtest as bt
from fr_power_forecast.cli import main
from fr_power_forecast.models import LEAR, LightGBM, SeasonalNaive


def test_each_block_is_calibrated_on_the_days_before_it():
    panel = synthetic_panel(60)
    days = panel.index[40:50]
    result = bt.rolling_backtest(SeasonalNaive(), panel, days, recalibrate_every=4)
    until = result.forecasts["calibrated_until"].unstack("hour")[0]
    expected = [days[0]] * 4 + [days[4]] * 4 + [days[8]] * 2
    assert (until.to_numpy() == np.array(expected) - pd.Timedelta(days=1)).all()
    assert result.timings["n_days"].tolist() == [4, 4, 2]


def test_backtest_forecasts_match_a_direct_fit():
    panel = synthetic_panel()
    days = panel.index[350:353]
    model = LEAR(window_days=300, exog=("load_forecast",))
    result = bt.rolling_backtest(model, panel, days, recalibrate_every=3)
    direct = model.fit(panel, panel.index[:350]).predict(panel, days)
    np.testing.assert_allclose(result.forecasts["forecast"].unstack("hour"), direct)


def test_parallel_backtest_equals_serial():
    panel = synthetic_panel(80)
    days = panel.index[60:70]
    serial = bt.rolling_backtest(SeasonalNaive(), panel, days, 3)
    parallel = bt.rolling_backtest(SeasonalNaive(), panel, days, 3, n_jobs=2)
    pd.testing.assert_frame_equal(serial.forecasts, parallel.forecasts)


def test_backtest_needs_history():
    panel = synthetic_panel(20)
    with pytest.raises(ValueError, match="history"):
        bt.rolling_backtest(SeasonalNaive(), panel, panel.index[:5])


def test_extend_refuses_overlapping_days():
    panel = synthetic_panel(40)
    first = bt.rolling_backtest(SeasonalNaive(), panel, panel.index[20:30])
    second = bt.rolling_backtest(SeasonalNaive(), panel, panel.index[30:35])
    assert len(bt.extend(first, second).forecasts) == 15 * 24
    with pytest.raises(ValueError, match="already"):
        bt.extend(first, first)


def test_evaluate_scores_models_on_common_days_against_the_reference():
    panel = synthetic_panel()
    days = panel.index[340:370]
    results = {
        "naive": bt.rolling_backtest(SeasonalNaive(), panel, days).forecasts,
        "lightgbm": bt.rolling_backtest(
            LightGBM(
                window_days=300, exog=("load_forecast",), quantiles=(0.1, 0.9), params=FAST_GBM
            ),
            panel,
            days[5:],
            recalibrate_every=25,
        ).forecasts,
    }
    table = bt.evaluate(results, panel)
    assert len(table.attrs["days"]) == 25
    assert table.loc["naive", "rmae"] == pytest.approx(1.0)
    assert table.loc["lightgbm", "rmae"] < 0.6
    assert table.loc["lightgbm", "dm_p_value"] < 0.05
    assert 0 < table.loc["lightgbm", "coverage_0.1_0.9"] <= 1
    assert np.isnan(table.loc["naive", "pinball"])

    window = bt.evaluate(results, panel, start=days[10], end=days[20])
    assert window.attrs["days"].equals(days[10:20])


def test_freshness_reports_days_beyond_the_backtest():
    panel = synthetic_panel(40)
    forecasts = bt.rolling_backtest(SeasonalNaive(), panel, panel.index[20:30], 5).forecasts
    row = bt.freshness({"naive": forecasts}, panel).loc["naive"]
    assert row["last_forecast_day"] == panel.index[29]
    assert row["last_recalibration"] == panel.index[24]
    assert row["days_not_backtested"] == 10


def test_cli_backtest_resume_and_evaluate(tmp_path, capsys):
    panel = synthetic_panel(60)
    hourly_index = pd.date_range(
        pd.Timestamp(panel.index[0]).tz_localize("Europe/Paris"),
        periods=len(panel) * 24,
        freq="1h",
    )
    # January and February 2024: no DST change, so the panel maps one-to-one to hours.
    hourly = pd.DataFrame(
        {var: panel[var].to_numpy().ravel() for var in ["price", "load_forecast"]},
        index=hourly_index.tz_convert("UTC"),
    )
    dataset = tmp_path / "hourly.parquet"
    hourly.to_parquet(dataset)
    common = ["--dataset", str(dataset), "--results-dir", str(tmp_path / "results")]

    main(["backtest", "--model", "naive", "--start", "2024-02-01", "--end", "2024-02-10", *common])
    main(["backtest", "--model", "naive", "--start", "2024-02-01", "--resume", *common])
    stored = bt.load(tmp_path / "results" / "naive").forecasts
    assert stored.index.get_level_values("date").max() == panel.index[-1]
    assert len(stored) == (len(panel) - 31) * 24

    main(["evaluate", *common])
    out = capsys.readouterr().out
    assert "last_recalibration" in out and "Recalibrate" not in out
