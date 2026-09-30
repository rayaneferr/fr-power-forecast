import httpx
import pandas as pd
import pytest

from fr_power_forecast.data import entsoe

NS = 'xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"'


def price_document(start: str, end: str, resolution: str, points: dict, **extra) -> str:
    curve = extra.get("curve", "A01")
    contract = extra.get("contract")
    contract_tag = (
        f"<contract_MarketAgreement.type>{contract}</contract_MarketAgreement.type>"
        if contract
        else ""
    )
    body = "".join(
        f"<Point><position>{p}</position><price.amount>{v}</price.amount></Point>"
        for p, v in points.items()
    )
    return f"""<?xml version="1.0" encoding="utf-8"?>
<Publication_MarketDocument {NS}>
  <TimeSeries>
    <mRID>1</mRID>{contract_tag}
    <curveType>{curve}</curveType>
    <Period>
      <timeInterval><start>{start}</start><end>{end}</end></timeInterval>
      <resolution>{resolution}</resolution>
      {body}
    </Period>
  </TimeSeries>
</Publication_MarketDocument>"""


def wind_solar_document() -> str:
    def series(psr: str, value: float) -> str:
        points = "".join(
            f"<Point><position>{p}</position><quantity>{value}</quantity></Point>" for p in (1, 2)
        )
        return f"""<TimeSeries><MktPSRType><psrType>{psr}</psrType></MktPSRType>
        <Period><timeInterval><start>2025-06-14T22:00Z</start><end>2025-06-15T00:00Z</end>
        </timeInterval><resolution>PT60M</resolution>{points}</Period></TimeSeries>"""

    return f"<GL_MarketDocument {NS}>{series('B16', 10)}{series('B19', 20)}</GL_MarketDocument>"


NO_DATA = f"""<Acknowledgement_MarketDocument {NS}>
  <Reason><code>999</code><text>No matching data found</text></Reason>
</Acknowledgement_MarketDocument>"""


def test_hourly_prices_are_parsed_in_utc():
    doc = price_document("2025-06-14T22:00Z", "2025-06-15T01:00Z", "PT60M", {1: 50, 2: 45, 3: -3})
    [(attrs, series)] = entsoe.parse_timeseries(doc, "price.amount")
    assert series.index[0] == pd.Timestamp("2025-06-14T22:00Z")
    assert series.tolist() == [50.0, 45.0, -3.0]
    assert attrs["contract_type"] is None


def test_quarter_hour_prices_are_averaged_to_hours():
    points = {i + 1: v for i, v in enumerate([10, 20, 30, 40, 100, 100, 100, 100])}
    doc = price_document("2025-10-01T22:00Z", "2025-10-02T00:00Z", "PT15M", points)
    [(_, series)] = entsoe.parse_timeseries(doc, "price.amount")
    assert series.tolist() == [25.0, 100.0]


def test_a03_curve_repeats_omitted_points():
    # Positions 2-4 omitted: they carry the value of position 1.
    doc = price_document("2025-10-01T22:00Z", "2025-10-01T23:00Z", "PT15M", {1: 10}, curve="A03")
    [(_, series)] = entsoe.parse_timeseries(doc, "price.amount")
    assert series.tolist() == [10.0]


def test_no_data_acknowledgement_raises_no_data_error():
    with pytest.raises(entsoe.NoDataError, match="No matching data"):
        entsoe.parse_timeseries(NO_DATA, "price.amount")


def test_unsupported_resolution_raises():
    doc = price_document("2025-06-14T22:00Z", "2025-06-15T22:00Z", "P1D", {1: 50})
    with pytest.raises(entsoe.EntsoeError, match="resolution"):
        entsoe.parse_timeseries(doc, "price.amount")


def test_yearly_chunks_cover_range_without_overlap():
    start, end = pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2021-06-01", tz="UTC")
    chunks = list(entsoe.yearly_chunks(start, end))
    assert chunks[0][0] == start and chunks[-1][1] == end
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:], strict=False))
    assert len(chunks) == 3


def mock_client(handler, **kwargs) -> entsoe.EntsoeClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return entsoe.EntsoeClient("token", http=http, sleep=lambda _: None, **kwargs)


def test_client_sends_token_and_filters_intraday_auctions():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        day_ahead = price_document(
            "2025-06-14T22:00Z", "2025-06-14T23:00Z", "PT60M", {1: 50}, contract="A01"
        )
        return httpx.Response(200, text=day_ahead)

    start, end = pd.Timestamp("2025-06-14T22:00Z"), pd.Timestamp("2025-06-14T23:00Z")
    prices = mock_client(handler).day_ahead_prices(start, end)

    params = requests[0].url.params
    assert params["securityToken"] == "token"
    assert params["documentType"] == "A44"
    assert params["periodStart"] == "202506142200"
    assert prices.name == "price" and prices.tolist() == [50.0]

    intraday = price_document(
        "2025-06-14T22:00Z", "2025-06-14T23:00Z", "PT60M", {1: 1}, contract="A07"
    )
    only_intraday = mock_client(lambda r: httpx.Response(200, text=intraday))
    assert only_intraday.day_ahead_prices(start, end).empty


def test_client_retries_on_rate_limit_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429)
        return httpx.Response(
            200, text=price_document("2025-06-14T22:00Z", "2025-06-14T23:00Z", "PT60M", {1: 7})
        )

    start, end = pd.Timestamp("2025-06-14T22:00Z"), pd.Timestamp("2025-06-14T23:00Z")
    assert mock_client(handler).day_ahead_prices(start, end).tolist() == [7.0]
    assert calls["n"] == 3


def test_client_treats_no_data_as_empty():
    client = mock_client(lambda r: httpx.Response(400, text=NO_DATA))
    start, end = pd.Timestamp("2025-06-14T22:00Z"), pd.Timestamp("2025-06-15T22:00Z")
    assert client.load_forecast(start, end).empty


def test_client_splits_wind_and_solar_by_production_type():
    client = mock_client(lambda r: httpx.Response(200, text=wind_solar_document()))
    start, end = pd.Timestamp("2025-06-14T22:00Z"), pd.Timestamp("2025-06-15T00:00Z")
    frame = client.wind_solar_forecast(start, end)
    assert frame["solar_forecast"].tolist() == [10.0, 10.0]
    assert frame["wind_onshore_forecast"].tolist() == [20.0, 20.0]
    assert frame["wind_offshore_forecast"].isna().all()


def test_client_rejects_non_utc_bounds():
    client = mock_client(lambda r: httpx.Response(200, text=NO_DATA))
    with pytest.raises(ValueError, match="UTC"):
        client.load_forecast(pd.Timestamp("2025-06-15", tz="Europe/Paris"), pd.Timestamp.now("UTC"))
