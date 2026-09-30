# Data

The modelling dataset is one row per UTC hour. By default it is built from public sources that
need no account or API key:

```bash
uv run fr-power-forecast build-dataset --start 2017-01-01 --end 2026-10-01
```

Raw downloads are cached in `data/raw/<source>/<year>.parquet` for completed years only; the
assembled dataset is written to `data/processed/epex_fr_hourly.parquet`. Nothing under `data/` is
committed. The command prints the share of missing hours and the longest gap per column: gaps are
reported, never interpolated.

## Sources

Default (`--source public`):

| Column | Provider | Content | Known at gate closure (D 12:00 CET)? |
|---|---|---|---|
| `price` | [Energy-Charts](https://api.energy-charts.info) (Fraunhofer ISE) | EPEX day-ahead price, FR zone | Target |
| `load_forecast` | [RTE eCO2mix](https://odre.opendatasoft.com) via ODRE | `prevision_j1`, RTE's D-1 consumption forecast | Yes |
| `forecast_*` | [Open-Meteo](https://open-meteo.com) historical forecast API | temperature 2 m, wind 100 m, shortwave radiation | Approximately (see below) |

Optional (`--source entsoe`, requires a free `ENTSOE_API_KEY`): ENTSO-E Transparency Platform
prices (A44), load forecast (A65) and wind/solar generation forecasts (A69). The public sources
have no wind/solar generation forecast; the forecast wind speed and solar radiation stand in as
ex-ante proxies.

Oracle only (`--weather reanalysis`): `reanalysis_*`, ERA5 observed weather from the Open-Meteo
archive. It was not known at gate closure and must never be used in a headline result; it only
measures an upper bound ("what if the weather forecast were perfect").

Weather columns are the plain mean over eight large cities (Paris, Lyon, Marseille, Toulouse,
Lille, Bordeaux, Nantes, Strasbourg). Archived weather forecasts have values from 2017 onwards.

## Leakage rules

- RTE's D-1 load forecast is the TSO forecast used as a covariate by Lago et al. (2021); it is
  the same forecast RTE reports to ENTSO-E.
- `forecast_*` is stitched from the first hours of successive weather model runs, so it is
  slightly more accurate than a true D+1 forecast issued at gate closure. Results using it are
  flagged.
- `reanalysis_*` is oracle information (see above).

## Time handling

- Everything is stored in UTC. The delivery day and the 24 products are defined in
  Europe/Paris local time (`fr_power_forecast.data.timegrid`).
- DST: the missing 02:00 of the March day is filled with the mean of 01:00 and 03:00; the two
  02:00 hours of the October day are averaged.
- Since October 2025 the day-ahead market clears 15-minute products. Quarter-hour prices are
  averaged to hourly values so that the whole history shares one grid.
- RTE files contain rows for the non-existent local hour 02:00 of the spring DST day, stamped with
  the same UTC time as 03:00 but with different values. Rows whose local and UTC timestamps
  disagree are dropped. On the autumn DST day RTE publishes only one of the two 02:00 hours; the
  other stays missing.
- ENTSO-E only: curve type A03 points are forward-filled within their period, and intraday
  auction prices returned under the same A44 document type are discarded.

## Licences and attribution

- Prices: Bundesnetzagentur | SMARD.de, CC BY 4.0, retrieved through Energy-Charts.
- Load forecast: RTE eCO2mix, Licence Ouverte v2.0 (Etalab).
- Weather: Open-Meteo, CC BY 4.0.

Any redistribution of the dataset (e.g. on Hugging Face) must carry these attributions.

## Snapshot (build of 2026-09-30, 2016-01-01 to 2026-10-01)

94,224 hours. Missing: 2 price hours (delivery day not yet published at build time), one
`load_forecast` hour per year (the autumn DST hour RTE does not publish), and the whole of 2016
for the archived weather forecasts.

| Year | Mean price (EUR/MWh) | Max price | Negative-price hours | Mean load forecast (MW) |
|---|---:|---:|---:|---:|
| 2016 | 36.8 | 874.0 | 2 | 54,493 |
| 2017 | 45.0 | 206.1 | 4 | 54,558 |
| 2018 | 50.2 | 260.0 | 11 | 53,885 |
| 2019 | 39.4 | 121.5 | 27 | 53,481 |
| 2020 | 32.2 | 200.0 | 102 | 50,726 |
| 2021 | 109.2 | 620.0 | 64 | 53,264 |
| 2022 | 275.9 | 2,987.8 | 4 | 50,813 |
| 2023 | 96.9 | 276.1 | 147 | 48,594 |
| 2024 | 58.0 | 284.2 | 352 | 48,956 |
| 2025 | 61.1 | 473.3 | 513 | 49,699 |
| 2026 (to Sep) | 81.5 | 342.2 | 563 | 48,744 |

The three regimes the benchmark reports on are visible: stable prices until 2020, the 2021-2022
energy crisis (peak of 2,987.78 EUR/MWh on 2022-04-04), and the rise of negative prices with solar
build-out from 2023.
