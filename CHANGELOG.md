# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `data.timegrid`: DST-safe conversion between UTC hourly series and (day x 24) matrices.
- `data.entsoe`: ENTSO-E client for day-ahead prices (A44), load forecast (A65) and wind/solar
  forecast (A69), with yearly chunking, retries, 15-minute to hourly aggregation and
  intraday-auction filtering.
- `data.weather`: Open-Meteo national weather, reanalysis (oracle) or archived forecasts.
- `data.dataset` and the `fr-power-forecast build-dataset` command: cached download, assembly on
  the full hourly grid and missing-data report.
- `docs/data.md`: sources, leakage rules and time handling.

## [0.1.0] - 2026-09-30

### Added
- Project scaffolding: uv, ruff, pytest, GitHub Actions CI.
- `metrics` module: MAE, RMSE, sMAPE, rMAE, pinball loss, CRPS from quantiles,
  interval coverage and width, multivariate one-sided Diebold-Mariano test.

[0.1.0]: https://github.com/rayaneferr/fr-power-forecast/releases/tag/v0.1.0
