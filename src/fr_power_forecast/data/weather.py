"""Hourly weather for mainland France from Open-Meteo (no API key required).

Two sources are available, and the difference matters for leakage:

- ``"reanalysis"``: ERA5 observed weather, available back to 1940. It is *not* known at gate
  closure, so features built from it are an oracle upper bound, never a fair covariate.
- ``"forecast"``: archived weather forecasts, close to the information a trader actually had.
  Values exist from 2017 onwards (2016 is accepted by the API but empty). The archive is stitched
  from the first hours of successive model runs, so it is still slightly more accurate than a
  genuine D+1 forecast issued at gate closure.

National values are the plain mean over a fixed set of large cities.
"""

from collections.abc import Iterator

import httpx
import pandas as pd

URLS = {
    "reanalysis": "https://archive-api.open-meteo.com/v1/archive",
    "forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
}
VARIABLES = ("temperature_2m", "wind_speed_100m", "shortwave_radiation")
CITIES = {
    "Paris": (48.86, 2.35),
    "Lyon": (45.76, 4.84),
    "Marseille": (43.30, 5.37),
    "Toulouse": (43.60, 1.44),
    "Lille": (50.63, 3.06),
    "Bordeaux": (44.84, -0.58),
    "Nantes": (47.22, -1.55),
    "Strasbourg": (48.57, 7.75),
}


def _yearly_dates(start: pd.Timestamp, end: pd.Timestamp) -> Iterator[tuple[str, str]]:
    """Inclusive date ranges of at most one year covering ``[start, end)``."""
    current = start.normalize()
    last = (end - pd.Timedelta(hours=1)).normalize()
    while current <= last:
        upper = min(current + pd.DateOffset(years=1) - pd.Timedelta(days=1), last)
        yield current.strftime("%Y-%m-%d"), upper.strftime("%Y-%m-%d")
        current = upper + pd.Timedelta(days=1)


def fetch_weather(
    start: pd.Timestamp,
    end: pd.Timestamp,
    source: str = "reanalysis",
    *,
    http: httpx.Client | None = None,
) -> pd.DataFrame:
    """National hourly weather on ``[start, end)`` in UTC, one column per variable.

    Columns are prefixed with the source (``reanalysis_temperature_2m`` ...) so that oracle
    features cannot be mistaken for ex-ante ones downstream.
    """
    if source not in URLS:
        raise ValueError(f"source must be one of {sorted(URLS)}")
    http = http or httpx.Client(timeout=120)
    latitudes = ",".join(str(lat) for lat, _ in CITIES.values())
    longitudes = ",".join(str(lon) for _, lon in CITIES.values())

    chunks = []
    for first_day, last_day in _yearly_dates(start, end):
        response = http.get(
            URLS[source],
            params={
                "latitude": latitudes,
                "longitude": longitudes,
                "start_date": first_day,
                "end_date": last_day,
                "hourly": ",".join(VARIABLES),
                "timezone": "GMT",
            },
        )
        response.raise_for_status()
        payload = response.json()
        locations = payload if isinstance(payload, list) else [payload]
        per_city = [
            pd.DataFrame(
                {var: loc["hourly"][var] for var in VARIABLES},
                index=pd.to_datetime(loc["hourly"]["time"]).tz_localize("UTC"),
                dtype=float,
            )
            for loc in locations
        ]
        chunks.append(pd.concat(per_city).groupby(level=0).mean())

    frame = pd.concat(chunks).sort_index()
    frame = frame[(frame.index >= start) & (frame.index < end)]
    return frame.add_prefix(f"{source}_")
