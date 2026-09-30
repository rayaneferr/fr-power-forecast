"""Assembly of the hourly modelling dataset from all sources.

Raw downloads are cached per source and per calendar year, and only for years that are over, so
rebuilding the dataset re-downloads nothing but the current year.
"""

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from fr_power_forecast.data.entsoe import EntsoeClient
from fr_power_forecast.data.weather import fetch_weather

Fetcher = Callable[[pd.Timestamp, pd.Timestamp], pd.Series | pd.DataFrame]


def hourly_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="1h", inclusive="left", tz="UTC", name="timestamp")


def assemble(parts: list[pd.Series | pd.DataFrame], start, end) -> pd.DataFrame:
    """Outer-join all parts on the full hourly UTC grid. Gaps stay NaN."""
    frames = [p.to_frame() if isinstance(p, pd.Series) else p for p in parts]
    frame = pd.concat(frames, axis=1)
    if frame.columns.duplicated().any():
        raise ValueError(f"duplicate columns: {list(frame.columns[frame.columns.duplicated()])}")
    return frame.reindex(hourly_index(start, end))


def missing_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Share of missing hours and longest gap (in hours) per column."""

    def longest_gap(column: pd.Series) -> int:
        missing = column.isna()
        if not missing.any():
            return 0
        runs = missing.ne(missing.shift()).cumsum()
        return int(missing.groupby(runs).sum().max())

    return pd.DataFrame(
        {
            "missing_share": frame.isna().mean(),
            "longest_gap_hours": frame.apply(longest_gap),
        }
    )


def _year_bounds(start: pd.Timestamp, end: pd.Timestamp):
    for year in range(start.year, end.year + 1):
        lower = max(start, pd.Timestamp(f"{year}-01-01", tz="UTC"))
        upper = min(end, pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))
        if lower < upper:
            yield year, lower, upper


def fetch_cached(
    name: str,
    fetch: Fetcher,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path | None,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch ``[start, end)`` year by year, caching the complete calendar years on disk."""
    now = now or pd.Timestamp.now(tz="UTC")
    chunks = []
    for year, lower, upper in _year_bounds(start, end):
        full_year = lower.month == 1 and lower.day == 1 and lower.hour == 0
        full_year = full_year and upper == pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        cacheable = cache_dir is not None and full_year and upper <= now
        path = cache_dir / name / f"{year}.parquet" if cache_dir is not None else None

        if cacheable and path.exists():
            chunks.append(pd.read_parquet(path))
            continue
        result = fetch(lower, upper)
        chunk = result.to_frame() if isinstance(result, pd.Series) else result
        if cacheable:
            path.parent.mkdir(parents=True, exist_ok=True)
            chunk.to_parquet(path)
        chunks.append(chunk)
    return pd.concat(chunks).sort_index()


def build_dataset(
    start: pd.Timestamp,
    end: pd.Timestamp,
    entsoe: EntsoeClient,
    *,
    weather_source: str | None = "reanalysis",
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Hourly dataset on ``[start, end)``: price, ENTSO-E forecasts and optional weather."""
    sources: dict[str, Fetcher] = {
        "price": entsoe.day_ahead_prices,
        "load_forecast": entsoe.load_forecast,
        "wind_solar_forecast": entsoe.wind_solar_forecast,
    }
    if weather_source is not None:
        sources[f"weather_{weather_source}"] = lambda s, e: fetch_weather(s, e, weather_source)

    parts = [fetch_cached(name, fetch, start, end, cache_dir) for name, fetch in sources.items()]
    return assemble(parts, start, end)
