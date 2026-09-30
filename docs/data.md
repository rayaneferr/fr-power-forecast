# Data

The modelling dataset is one row per UTC hour. It is built with:

```bash
export ENTSOE_API_KEY=...          # free, see "Access" below
uv run fr-power-forecast build-dataset --start 2019-01-01 --end 2026-10-01 --weather forecast
```

Raw downloads are cached in `data/raw/<source>/<year>.parquet` for completed years only; the
assembled dataset is written to `data/processed/epex_fr_hourly.parquet`. Nothing under `data/` is
committed. The command prints the share of missing hours and the longest gap per column: gaps are
reported, never interpolated.

## Sources

| Column | Source | Document | Known at gate closure (D 12:00 CET)? |
|---|---|---|---|
| `price` | ENTSO-E Transparency | A44 day-ahead prices, bidding zone FR | Target |
| `load_forecast` | ENTSO-E Transparency | A65 day-ahead total load forecast | Yes |
| `solar_forecast`, `wind_onshore_forecast`, `wind_offshore_forecast` | ENTSO-E Transparency | A69 day-ahead wind and solar forecast | Yes |
| `forecast_*` | Open-Meteo historical forecast API | temperature 2 m, wind 100 m, shortwave radiation | Approximately (see below) |
| `reanalysis_*` | Open-Meteo archive (ERA5) | same variables | **No, oracle only** |

Weather columns are the plain mean over eight large cities (Paris, Lyon, Marseille, Toulouse,
Lille, Bordeaux, Nantes, Strasbourg).

## Leakage rules

- The ENTSO-E forecasts are published before gate closure and are the fair covariates used by
  Lago et al. (2021).
- `reanalysis_*` is observed weather. It is kept only to measure an upper bound ("what if the
  weather forecast were perfect") and must never be used in a headline result.
- `forecast_*` is stitched from the first hours of successive weather model runs, so it is
  slightly more accurate than a true D+1 forecast. Results using it are flagged.

## Time handling

- Everything is stored in UTC. The delivery day and the 24 products are defined in
  Europe/Paris local time (`fr_power_forecast.data.timegrid`).
- DST: the missing 02:00 of the March day is filled with the mean of 01:00 and 03:00; the two
  02:00 hours of the October day are averaged.
- Since October 2025 the day-ahead market clears 15-minute products. Quarter-hour prices are
  averaged to hourly values so that the whole history shares one grid.
- ENTSO-E curve type A03 omits points whose value repeats the previous one; they are
  forward-filled within their period.
- Intraday auction prices, returned under the same A44 document type with another contract type,
  are discarded.

## Access

The ENTSO-E API requires a free security token: create an account on
[transparency.entsoe.eu](https://transparency.entsoe.eu), then request "Restful API access" by
email to the Transparency Platform helpdesk. Open-Meteo needs no key.
