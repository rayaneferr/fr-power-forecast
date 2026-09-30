import httpx
import pandas as pd
import pytest

from fr_power_forecast.data import energy_charts, rte
from fr_power_forecast.data.http import get_with_retries


def utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def unix(value: str) -> int:
    return int(utc(value).timestamp())


def client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_retries_stop_after_max_and_return_last_response():
    delays = []
    http = client(lambda r: httpx.Response(429))
    response = get_with_retries(http, "https://x", {}, max_retries=2, sleep=delays.append)
    assert response.status_code == 429
    assert delays == [1.0, 2.0]


def test_energy_charts_pads_dates_trims_and_averages_quarter_hours():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params)
        # 22:00 is hourly; 23:00 is the first quarter-hour delivery period.
        times = ["2025-09-30T21:00Z", "2025-09-30T22:00Z"]
        times += [f"2025-09-30T23:{m:02d}Z" for m in (0, 15, 30, 45)]
        return httpx.Response(
            200,
            json={"unix_seconds": [unix(t) for t in times], "price": [1, 50, 10, 20, 30, 40]},
        )

    series = energy_charts.fetch_day_ahead_prices(
        utc("2025-09-30T22:00"), utc("2025-10-01T00:00"), http=client(handler)
    )
    assert seen[0]["bzn"] == "FR"
    assert seen[0]["start"] == "2025-09-29" and seen[0]["end"] == "2025-10-02"
    assert series.name == "price"
    assert series.tolist() == [50.0, 25.0]


def test_energy_charts_skips_periods_without_content():
    http = client(lambda r: httpx.Response(404, text="no content available"))
    series = energy_charts.fetch_day_ahead_prices(utc("2014-01-01"), utc("2014-01-02"), http=http)
    assert series.empty


# Real ODRE rows around the 2025 spring DST change: the non-existent local 02:xx rows share their
# UTC timestamp with 03:xx and must be dropped.
SPRING_RECORDS = [
    {"date": "2025-03-30", "heure": h, "date_heure": t, "prevision_j1": v}
    for h, t, v in [
        ("01:30", "2025-03-30T00:30:00+00:00", 47000),
        ("01:45", "2025-03-30T00:45:00+00:00", 47000),
        ("03:00", "2025-03-30T01:00:00+00:00", 48100),
        ("02:00", "2025-03-30T01:00:00+00:00", 47000),
        ("03:15", "2025-03-30T01:15:00+00:00", 47050),
        ("02:15", "2025-03-30T01:15:00+00:00", 47000),
        ("03:30", "2025-03-30T01:30:00+00:00", 46000),
        ("02:30", "2025-03-30T01:30:00+00:00", 47000),
        ("03:45", "2025-03-30T01:45:00+00:00", 45100),
        ("02:45", "2025-03-30T01:45:00+00:00", 47550),
    ]
]


def test_rte_drops_non_existent_local_hours_and_averages_to_hourly():
    series = rte.parse_records(SPRING_RECORDS)
    assert series.index.tolist() == [utc("2025-03-30T00:00"), utc("2025-03-30T01:00")]
    assert series.iloc[1] == pytest.approx((48100 + 47050 + 46000 + 45100) / 4)


def test_rte_prefers_consolidated_data_and_fills_with_real_time():
    def records(hour: int, value: float) -> list[dict]:
        t = utc("2025-06-01") + pd.Timedelta(hours=hour)
        local = t.tz_convert("Europe/Paris")
        return [
            {
                "date": local.strftime("%Y-%m-%d"),
                "heure": local.strftime("%H:%M"),
                "date_heure": t.isoformat(),
                "prevision_j1": value,
            }
        ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert "prevision_j1 is not null" in request.url.params["where"]
        if rte.CONSOLIDATED in request.url.path:
            return httpx.Response(200, json=records(0, 50_000))
        return httpx.Response(200, json=records(0, 1) + records(1, 51_000))

    series = rte.fetch_load_forecast(
        utc("2025-06-01"), utc("2025-06-01T02:00"), http=client(handler)
    )
    assert series.name == "load_forecast"
    assert series.tolist() == [50_000.0, 51_000.0]


def test_empty_rte_export_gives_empty_series():
    assert rte.parse_records([]).empty
