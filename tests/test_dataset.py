import httpx
import numpy as np
import pandas as pd
import pytest

from fr_power_forecast.data import dataset, weather


def utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def constant_series(name: str, value: float):
    def fetch(start, end):
        return pd.Series(value, index=dataset.hourly_index(start, end), name=name)

    return fetch


def test_assemble_reindexes_on_the_full_grid():
    start, end = utc("2025-01-01"), utc("2025-01-01T04:00")
    price = pd.Series([1.0, 2.0], index=[start, start + pd.Timedelta(hours=2)], name="price")
    frame = dataset.assemble([price], start, end)
    assert len(frame) == 4
    assert frame["price"].isna().tolist() == [False, True, False, True]


def test_assemble_rejects_duplicate_columns():
    start, end = utc("2025-01-01"), utc("2025-01-01T01:00")
    s = pd.Series([1.0], index=[start], name="price")
    with pytest.raises(ValueError, match="duplicate"):
        dataset.assemble([s, s], start, end)


def test_missing_report_measures_share_and_longest_gap():
    frame = pd.DataFrame({"a": [1, np.nan, np.nan, 4, np.nan], "b": [1, 2, 3, 4, 5]})
    report = dataset.missing_report(frame)
    assert report.loc["a", "missing_share"] == pytest.approx(0.6)
    assert report.loc["a", "longest_gap_hours"] == 2
    assert report.loc["b", "longest_gap_hours"] == 0


def test_only_complete_past_years_are_cached(tmp_path):
    calls = []

    def fetch(start, end):
        calls.append((start, end))
        return constant_series("price", 1.0)(start, end)

    start, end, now = utc("2024-07-01"), utc("2026-03-01"), utc("2026-03-01")
    first = dataset.fetch_cached("price", fetch, start, end, tmp_path, now=now)
    second = dataset.fetch_cached("price", fetch, start, end, tmp_path, now=now)

    assert sorted(p.name for p in (tmp_path / "price").iterdir()) == ["2025.parquet"]
    # 2024 is partial and 2026 is ongoing: both are fetched twice, 2025 once.
    assert len(calls) == 5
    pd.testing.assert_frame_equal(first, second, check_freq=False)
    assert len(first) == len(dataset.hourly_index(start, end))


def test_build_dataset_joins_all_sources():
    class FakeEntsoe:
        day_ahead_prices = staticmethod(constant_series("price", 50.0))
        load_forecast = staticmethod(constant_series("load_forecast", 60_000.0))

        @staticmethod
        def wind_solar_forecast(start, end):
            index = dataset.hourly_index(start, end)
            return pd.DataFrame({"solar_forecast": 1.0, "wind_onshore_forecast": 2.0}, index=index)

    start, end = utc("2025-01-01"), utc("2025-01-02")
    frame = dataset.build_dataset(start, end, FakeEntsoe(), weather_source=None)
    assert list(frame.columns) == [
        "price",
        "load_forecast",
        "solar_forecast",
        "wind_onshore_forecast",
    ]
    assert frame.shape == (24, 4) and not frame.isna().any().any()


def test_weather_averages_cities_and_prefixes_the_source():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["start_date"] == "2025-01-01"
        assert request.url.params["end_date"] == "2025-01-01"
        times = ["2025-01-01T00:00", "2025-01-01T01:00"]

        def city(values):
            return {"hourly": {"time": times, **{v: values for v in weather.VARIABLES}}}

        # A city with missing values must not drag the national mean.
        return httpx.Response(200, json=[city([0.0, 2.0]), city([10.0, 12.0]), city([None, None])])

    http = httpx.Client(transport=httpx.MockTransport(handler))
    frame = weather.fetch_weather(utc("2025-01-01"), utc("2025-01-01T02:00"), "forecast", http=http)
    assert list(frame.columns) == [f"forecast_{v}" for v in weather.VARIABLES]
    assert frame["forecast_temperature_2m"].tolist() == [5.0, 7.0]


def test_weather_rejects_unknown_source():
    with pytest.raises(ValueError):
        weather.fetch_weather(utc("2025-01-01"), utc("2025-01-02"), "satellite")
