"""Client for the ENTSO-E Transparency Platform REST API.

Only the documents needed for day-ahead price forecasting in France are covered:

- A44 day-ahead prices,
- A65 day-ahead total load forecast,
- A69 day-ahead wind and solar generation forecast.

Every series is returned at hourly resolution in UTC. Since the switch of the European day-ahead
market to 15-minute products (October 2025), prices come as quarter-hours; they are averaged to
hourly values so that the whole history shares one grid.
"""

import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable

import httpx
import numpy as np
import pandas as pd

from fr_power_forecast.data.http import get_with_retries
from fr_power_forecast.data.timegrid import check_utc, yearly_chunks

BASE_URL = "https://web-api.tp.entsoe.eu/api"
FRANCE = "10YFR-RTE------C"
DAY_AHEAD_CONTRACT = "A01"
PSR_TYPES = {"B16": "solar", "B18": "wind_offshore", "B19": "wind_onshore"}

_RESOLUTION = re.compile(r"^PT(\d+)M$")
_NO_DATA_REASON = "999"


class EntsoeError(RuntimeError):
    """The API returned an error or a document that cannot be parsed."""


class NoDataError(EntsoeError):
    """The API has no data for the requested period."""


def _name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _name(child) == name]


def _text(element: ET.Element, name: str) -> str | None:
    found = _children(element, name)
    return found[0].text if found else None


def _required(element: ET.Element, name: str) -> str:
    value = _text(element, name)
    if value is None:
        raise EntsoeError(f"<{_name(element)}> has no <{name}>")
    return value


def _raise_for_acknowledgement(root: ET.Element) -> None:
    if _name(root) != "Acknowledgement_MarketDocument":
        return
    reasons = _children(root, "Reason")
    code = _text(reasons[0], "code") if reasons else None
    message = (_text(reasons[0], "text") if reasons else None) or "unknown reason"
    if code == _NO_DATA_REASON:
        raise NoDataError(message)
    raise EntsoeError(message)


def _parse_period(period: ET.Element, value_tag: str, curve_type: str) -> pd.Series:
    interval = _children(period, "timeInterval")[0]
    start = pd.Timestamp(_required(interval, "start"))
    end = pd.Timestamp(_required(interval, "end"))
    resolution = _required(period, "resolution")
    match = _RESOLUTION.match(resolution)
    if not match:
        raise EntsoeError(f"unsupported resolution {resolution}")

    index = pd.date_range(start, end, freq=f"{match[1]}min", inclusive="left")
    values = np.full(len(index), np.nan)
    for point in _children(period, "Point"):
        position = int(_required(point, "position"))
        if not 1 <= position <= len(index):
            raise EntsoeError(f"point position {position} outside the period")
        values[position - 1] = float(_required(point, value_tag))

    series = pd.Series(values, index=index)
    # Curve type A03 omits a point when its value equals the previous one.
    if curve_type == "A03":
        series = series.ffill()
    return series.resample("1h").mean()


def parse_timeseries(document: bytes | str, value_tag: str) -> list[tuple[dict, pd.Series]]:
    """Parse an ENTSO-E market document into ``(attributes, hourly series)`` pairs.

    Attributes hold the fields used to tell series apart: ``psr_type`` and ``contract_type``.
    """
    root = ET.fromstring(document)
    _raise_for_acknowledgement(root)

    parsed = []
    for timeseries in _children(root, "TimeSeries"):
        psr = _children(timeseries, "MktPSRType")
        attributes = {
            "psr_type": _text(psr[0], "psrType") if psr else None,
            "contract_type": _text(timeseries, "contract_MarketAgreement.type"),
        }
        curve_type = _text(timeseries, "curveType") or "A01"
        for period in _children(timeseries, "Period"):
            parsed.append((attributes, _parse_period(period, value_tag, curve_type)))
    return parsed


def _combine(parts: list[pd.Series], name: str) -> pd.Series:
    if not parts:
        return pd.Series(dtype=float, name=name, index=pd.DatetimeIndex([], tz="UTC"))
    series = pd.concat(parts).sort_index()
    series = series[~series.index.duplicated(keep="first")]
    return series.rename(name)


class EntsoeClient:
    def __init__(
        self,
        api_key: str,
        *,
        http: httpx.Client | None = None,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not api_key:
            raise ValueError("an ENTSO-E API key is required")
        self._api_key = api_key
        self._http = http or httpx.Client(timeout=60)
        self._max_retries = max_retries
        self._sleep = sleep

    def _get(self, params: dict) -> bytes:
        response = get_with_retries(
            self._http,
            BASE_URL,
            {"securityToken": self._api_key, **params},
            max_retries=self._max_retries,
            sleep=self._sleep,
        )
        # Errors such as "no data" come back as an XML acknowledgement with status 400.
        acknowledgement = b"Acknowledgement_MarketDocument" in response.content
        if response.status_code == 400 and acknowledgement:
            return response.content
        response.raise_for_status()
        return response.content

    def _query(
        self, params: dict, start: pd.Timestamp, end: pd.Timestamp, value_tag: str
    ) -> list[tuple[dict, pd.Series]]:
        check_utc(start, end)
        parsed = []
        for lower, upper in yearly_chunks(start, end):
            window = {
                "periodStart": lower.strftime("%Y%m%d%H%M"),
                "periodEnd": upper.strftime("%Y%m%d%H%M"),
            }
            try:
                parsed += parse_timeseries(self._get({**params, **window}), value_tag)
            except NoDataError:
                continue
        return [(attrs, s[(s.index >= start) & (s.index < end)]) for attrs, s in parsed]

    def day_ahead_prices(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Hourly day-ahead prices in EUR/MWh."""
        params = {"documentType": "A44", "in_Domain": FRANCE, "out_Domain": FRANCE}
        parsed = self._query(params, start, end, "price.amount")
        # Intraday auction prices share the A44 document type; keep the day-ahead contract only.
        parts = [s for attrs, s in parsed if attrs["contract_type"] in (None, DAY_AHEAD_CONTRACT)]
        return _combine(parts, "price")

    def load_forecast(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Day-ahead forecast of the total load in MW."""
        params = {"documentType": "A65", "processType": "A01", "outBiddingZone_Domain": FRANCE}
        parsed = self._query(params, start, end, "quantity")
        return _combine([s for _, s in parsed], "load_forecast")

    def wind_solar_forecast(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Day-ahead forecast of solar, onshore and offshore wind generation in MW."""
        params = {"documentType": "A69", "processType": "A01", "in_Domain": FRANCE}
        parsed = self._query(params, start, end, "quantity")
        columns = {}
        for code, label in PSR_TYPES.items():
            parts = [s for attrs, s in parsed if attrs["psr_type"] == code]
            columns[f"{label}_forecast"] = _combine(parts, f"{label}_forecast")
        return pd.DataFrame(columns)
