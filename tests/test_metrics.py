import numpy as np
import pytest

from fr_power_forecast import metrics


def test_point_metrics_on_known_values():
    y = [10.0, 20.0, 30.0]
    p = [12.0, 18.0, 33.0]
    assert metrics.mae(y, p) == pytest.approx(7 / 3)
    assert metrics.rmse(y, p) == pytest.approx(np.sqrt(17 / 3))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        metrics.mae([1, 2, 3], [1, 2])


def test_smape_handles_zero_prices():
    # Both zero: no error. Negative prices are legitimate on EPEX and must not break it.
    assert metrics.smape([0.0, -10.0], [0.0, -10.0]) == 0.0
    assert metrics.smape([10.0], [-10.0]) == pytest.approx(200.0)


def test_rmae_is_one_for_the_naive_forecast_itself():
    y = np.array([50.0, 60.0, 55.0])
    naive = np.array([45.0, 58.0, 70.0])
    assert metrics.rmae(y, naive, naive) == pytest.approx(1.0)
    assert metrics.rmae(y, y, naive) == 0.0


def test_rmae_rejects_perfect_naive():
    with pytest.raises(ValueError):
        metrics.rmae([1.0, 2.0], [1.5, 2.5], [1.0, 2.0])


def test_pinball_median_is_half_mae():
    y = np.array([10.0, 20.0, 30.0])
    median = np.array([12.0, 18.0, 33.0])
    loss = metrics.pinball_loss(y, median[:, None], [0.5])
    assert loss == pytest.approx(metrics.mae(y, median) / 2)


def test_pinball_is_asymmetric():
    # Under-forecasting is penalised by q, over-forecasting by 1 - q.
    assert metrics.pinball_loss([10.0], [[0.0]], [0.9]) == pytest.approx(9.0)
    assert metrics.pinball_loss([0.0], [[10.0]], [0.9]) == pytest.approx(1.0)


def test_crps_of_perfect_quantiles_is_zero():
    y = np.array([5.0, 7.0])
    q = np.repeat(y[:, None], 3, axis=1)
    assert metrics.crps_from_quantiles(y, q, [0.1, 0.5, 0.9]) == 0.0


def test_crps_approximates_closed_form_for_a_gaussian():
    # For N(mu, sigma), E[CRPS] at y = mu is sigma * (sqrt(2) - 1) / sqrt(pi).
    from scipy import stats

    sigma = 3.0
    levels = np.linspace(0.005, 0.995, 199)
    q = stats.norm.ppf(levels, scale=sigma)[None, :]
    expected = sigma * (np.sqrt(2) - 1) / np.sqrt(np.pi)
    assert metrics.crps_from_quantiles([0.0], q, levels) == pytest.approx(expected, rel=0.02)


@pytest.mark.parametrize("levels", [[0.0, 0.5], [0.5, 1.0], [[0.5]]])
def test_invalid_quantile_levels_raise(levels):
    with pytest.raises(ValueError):
        metrics.pinball_loss([1.0], [[1.0, 1.0]], levels)


def test_coverage_and_width():
    y = [1.0, 5.0, 10.0, 20.0]
    lo = [0.0, 0.0, 0.0, 0.0]
    hi = [2.0, 4.0, 10.0, 30.0]
    assert metrics.interval_coverage(y, lo, hi) == 0.75
    assert metrics.mean_interval_width(lo, hi) == 11.5


def test_diebold_mariano_detects_a_better_forecast():
    rng = np.random.default_rng(0)
    y = rng.normal(60, 20, size=(365, 24))
    good = y + rng.normal(0, 5, size=y.shape)
    bad = y + rng.normal(0, 15, size=y.shape)

    better = metrics.diebold_mariano(y, bad, good)
    assert better.statistic > 0
    assert better.p_value < 0.01

    worse = metrics.diebold_mariano(y, good, bad)
    assert worse.p_value > 0.99


def test_diebold_mariano_does_not_reject_equal_forecasts():
    rng = np.random.default_rng(1)
    y = rng.normal(60, 20, size=(365, 24))
    f1 = y + rng.normal(0, 10, size=y.shape)
    f2 = y + rng.normal(0, 10, size=y.shape)
    assert 0.01 < metrics.diebold_mariano(y, f1, f2, norm=2).p_value < 0.99


def test_diebold_mariano_requires_daily_matrix():
    with pytest.raises(ValueError):
        metrics.diebold_mariano(np.zeros(24), np.zeros(24), np.ones(24))
