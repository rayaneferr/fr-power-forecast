"""RTE day-ahead load forecast (eCO2mix), from the ODRE open data portal, no API key required.

``prevision_j1`` is RTE's forecast of the national consumption issued the day before delivery,
the same forecast RTE reports to ENTSO-E. Data is published under the Licence Ouverte v2.0
(Etalab). Two datasets are merged: the consolidated one (from 2012, lagging a few months behind)
and the real-time one, which fills the most recent months.

RTE rows carry both a local date/hour and a UTC timestamp. On the spring DST day the file also
contains the non-existent local hour 02:00, stamped with the same UTC time as 03:00; such rows
are detected by checking that both timestamps agree, and dropped.
"""

import time
from collections.abc import Callable

import httpx
import pandas as pd

from fr_power_forecast.data.http import get_with_retries
from fr_power_forecast.data.timegrid import TZ, check_utc, yearly_chunks

BASE_URL = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets"
CONSOLIDATED = "eco2mix-national-cons-def"
REAL_TIME = "eco2mix-national-tr"
ATTRIBUTION = "Load forecast: RTE eCO2mix via ODRE (Licence Ouverte v2.0)"


def parse_records(records: list[dict]) -> pd.Series:
    """Hourly load forecast in MW from ODRE export records."""
    if not records:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    frame = pd.DataFrame(records)
    utc = pd.to_datetime(frame["date_heure"], utc=True)
    declared = pd.to_datetime(frame["date"] + " " + frame["heure"])
    consistent = (utc.dt.tz_convert(TZ).dt.tz_localize(None) == declared).to_numpy()

    series = pd.Series(frame["prevision_j1"].astype(float).to_numpy(), index=pd.DatetimeIndex(utc))
    series = series[consistent].sort_index()
    series = series[~series.index.duplicated(keep="first")]
    return series.resample("1h").mean()


def _fetch(http, dataset, lower, upper, sleep) -> pd.Series:
    where = (
        f'date_heure >= "{lower.isoformat()}" and date_heure < "{upper.isoformat()}"'
        " and prevision_j1 is not null"
    )
    params = {
        "where": where,
        "select": "date,heure,date_heure,prevision_j1",
        "order_by": "date_heure",
    }
    response = get_with_retries(http, f"{BASE_URL}/{dataset}/exports/json", params, sleep=sleep)
    response.raise_for_status()
    return parse_records(response.json())


def fetch_load_forecast(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    http: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.Series:
    """Hourly day-ahead load forecast in MW on ``[start, end)``, indexed in UTC."""
    check_utc(start, end)
    http = http or httpx.Client(timeout=120)

    merged = []
    for lower, upper in yearly_chunks(start, end):
        consolidated = _fetch(http, CONSOLIDATED, lower, upper, sleep)
        real_time = _fetch(http, REAL_TIME, lower, upper, sleep)
        merged.append(consolidated.combine_first(real_time))

    series = pd.concat(merged).sort_index()
    series = series[(series.index >= start) & (series.index < end)]
    return series.rename("load_forecast")
