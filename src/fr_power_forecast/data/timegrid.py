"""Conversion between UTC hourly series and (delivery day x 24 hours) matrices.

The day-ahead auction clears 24 products per delivery day in Paris local time. Daylight saving
time breaks this: the last Sunday of March has 23 hours and the last Sunday of October has 25.
Following common EPF practice, the missing hour 02:00 is filled with the mean of 01:00 and 03:00,
and the two 02:00 hours of the October day are averaged.
"""

from collections.abc import Iterator

import numpy as np
import pandas as pd

TZ = "Europe/Paris"


def day_length(date: pd.Timestamp) -> int:
    """Number of hours in a local calendar day (23, 24 or 25). ``date`` is timezone-naive."""
    day = pd.Timestamp(date).normalize()
    start = day.tz_localize(TZ)
    end = (day + pd.Timedelta(days=1)).tz_localize(TZ)
    return int((end - start) / pd.Timedelta(hours=1))


def to_daily_matrix(series: pd.Series) -> pd.DataFrame:
    """Reshape a tz-aware hourly series into a (n_days, 24) frame indexed by local date.

    Gaps other than the DST hole stay NaN: missing data is reported, never silently filled.
    """
    if series.index.tz is None:
        raise ValueError("series index must be timezone-aware")

    local = series.index.tz_convert(TZ)
    frame = pd.DataFrame(
        {
            "value": series.to_numpy(dtype=float),
            "date": local.tz_localize(None).normalize(),
            "hour": local.hour,
        }
    )
    matrix = frame.groupby(["date", "hour"])["value"].mean().unstack("hour")
    days = pd.date_range(matrix.index.min(), matrix.index.max(), freq="D", name="date")
    matrix = matrix.reindex(index=days, columns=range(24))
    matrix.columns.name = "hour"

    for day in days:
        if day_length(day) == 23:
            matrix.loc[day, 2] = (matrix.loc[day, 1] + matrix.loc[day, 3]) / 2
    return matrix


def from_daily_matrix(matrix: pd.DataFrame, name: str | None = None) -> pd.Series:
    """Inverse of :func:`to_daily_matrix`, returning a UTC hourly series.

    The 02:00 value of a 23-hour day is dropped; on a 25-hour day it is used for both hours.
    """
    first = matrix.index.min().tz_localize(TZ)
    last = (matrix.index.max() + pd.Timedelta(days=1)).tz_localize(TZ)
    index = pd.date_range(first, last, freq="1h", inclusive="left")

    rows = matrix.index.get_indexer(index.tz_localize(None).normalize())
    if np.any(rows < 0):
        raise ValueError("matrix index must contain consecutive days")
    values = matrix.to_numpy(dtype=float)[rows, index.hour]
    return pd.Series(values, index=index.tz_convert("UTC"), name=name)


def yearly_chunks(start: pd.Timestamp, end: pd.Timestamp) -> Iterator[tuple]:
    """Split ``[start, end)`` into consecutive chunks of at most one year."""
    current = start
    while current < end:
        upper = min(current + pd.DateOffset(years=1), end)
        yield current, upper
        current = upper


def check_utc(*timestamps: pd.Timestamp) -> None:
    for ts in timestamps:
        if ts.tz is None or ts.utcoffset() != pd.Timedelta(0):
            raise ValueError(f"expected a UTC timestamp, got {ts}")
