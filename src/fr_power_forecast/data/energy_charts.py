"""French day-ahead prices from the Energy-Charts API (Fraunhofer ISE), no API key required.

Prices are published under CC BY 4.0 (Bundesnetzagentur | SMARD.de). Data for the FR bidding
zone is available from 2016 at least. Since October 2025 the market clears 15-minute products;
they are averaged to hourly values like in the ENTSO-E client.
"""

import time
from collections.abc import Callable

import httpx
import pandas as pd

from fr_power_forecast.data.http import get_with_retries
from fr_power_forecast.data.timegrid import check_utc, yearly_chunks

URL = "https://api.energy-charts.info/price"
BIDDING_ZONE = "FR"
ATTRIBUTION = "Day-ahead prices: Bundesnetzagentur | SMARD.de (CC BY 4.0), via Energy-Charts"


def fetch_day_ahead_prices(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    http: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.Series:
    """Hourly day-ahead prices in EUR/MWh on ``[start, end)``, indexed in UTC."""
    check_utc(start, end)
    http = http or httpx.Client(timeout=120)

    parts = []
    for lower, upper in yearly_chunks(start, end):
        # The API takes inclusive local dates: pad by a day on each side, then trim in UTC.
        params = {
            "bzn": BIDDING_ZONE,
            "start": (lower - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "end": (upper + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        }
        # The public API rate-limits bursts for about a minute.
        response = get_with_retries(http, URL, params, max_retries=6, base_delay=10, sleep=sleep)
        if response.status_code == 404:  # "no content available" for the period
            continue
        response.raise_for_status()
        payload = response.json()
        index = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
        parts.append(pd.Series(payload["price"], index=index, dtype=float))

    if not parts:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"), name="price")
    series = pd.concat(parts).sort_index()
    series = series[~series.index.duplicated(keep="first")]
    series = series.resample("1h").mean()
    return series[(series.index >= start) & (series.index < end)].rename("price")
