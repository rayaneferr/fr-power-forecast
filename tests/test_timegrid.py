import numpy as np
import pandas as pd
import pytest

from fr_power_forecast.data.timegrid import day_length, from_daily_matrix, to_daily_matrix


def hourly_utc(start_local: str, end_local: str, values=None) -> pd.Series:
    index = pd.date_range(
        pd.Timestamp(start_local, tz="Europe/Paris"),
        pd.Timestamp(end_local, tz="Europe/Paris"),
        freq="1h",
        inclusive="left",
    ).tz_convert("UTC")
    if values is None:
        values = np.arange(len(index), dtype=float)
    return pd.Series(values, index=index)


def test_day_length_around_dst():
    assert day_length(pd.Timestamp("2025-03-30")) == 23
    assert day_length(pd.Timestamp("2025-10-26")) == 25
    assert day_length(pd.Timestamp("2025-06-15")) == 24


def test_regular_day_maps_local_hours_to_columns():
    matrix = to_daily_matrix(hourly_utc("2025-06-15", "2025-06-16"))
    assert matrix.shape == (1, 24)
    assert matrix.iloc[0].tolist() == list(range(24))


def test_spring_forward_fills_missing_hour_with_neighbour_mean():
    series = hourly_utc("2025-03-30", "2025-03-31")
    assert len(series) == 23
    row = to_daily_matrix(series).iloc[0]
    # Local 02:00 does not exist: hours 0,1 then 3..23 hold values 0,1,2..22.
    assert row[1] == 1.0 and row[3] == 2.0
    assert row[2] == 1.5


def test_fall_back_averages_the_repeated_hour():
    series = hourly_utc("2025-10-26", "2025-10-27")
    assert len(series) == 25
    row = to_daily_matrix(series).iloc[0]
    # Values 2 and 3 are the two local 02:00 hours.
    assert row[2] == 2.5
    assert row[3] == 4.0


def test_gaps_and_missing_days_stay_nan():
    series = hourly_utc("2025-06-15", "2025-06-18")
    series.iloc[5] = np.nan
    series = series.drop(series.index[24:48])  # remove a whole day
    matrix = to_daily_matrix(series)
    assert matrix.shape == (3, 24)
    assert np.isnan(matrix.iloc[0, 5])
    assert matrix.iloc[1].isna().all()


def test_naive_index_is_rejected():
    with pytest.raises(ValueError):
        to_daily_matrix(pd.Series([1.0], index=pd.DatetimeIndex(["2025-01-01"])))


def test_round_trip_over_both_dst_changes():
    series = hourly_utc("2025-03-29", "2025-10-28")
    back = from_daily_matrix(to_daily_matrix(series))
    assert back.index.equals(series.index)

    fall_back = pd.Timestamp("2025-10-26", tz="Europe/Paris")
    regular = (series.index < fall_back) | (series.index >= fall_back + pd.Timedelta(hours=4))
    pd.testing.assert_series_equal(
        back[regular], series[regular], check_names=False, check_freq=False
    )
