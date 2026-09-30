# fr-power-forecast

[![CI](https://github.com/rayaneferr/fr-power-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/rayaneferr/fr-power-forecast/actions/workflows/ci.yml)

**Do time series foundation models beat market-specific models on French day-ahead electricity prices?**

This project benchmarks zero-shot and fine-tuned time series foundation models (Chronos-2, TimesFM,
Moirai) against the established electricity price forecasting baselines (seasonal naive, LEAR,
LightGBM, N-HiTS) on the EPEX-FR day-ahead market, along three axes:

1. **Accuracy**: MAE, rMAE, with Diebold-Mariano tests for statistical significance.
2. **Uncertainty**: quantile forecasts scored with pinball loss / CRPS, interval coverage, and
   conformal calibration on top.
3. **Cost of use**: inference latency and compute cost per forecast, plus a battery arbitrage
   backtest that turns forecast error into euros.

Results are broken down by market regime (pre-2022, 2022 energy crisis, negative-price hours).

## Protocol

The evaluation follows the best practices of
[Lago et al. (2021)](https://arxiv.org/abs/2008.08004): rolling-origin backtest, daily
recalibration, forecasts of the 24 hourly prices of day D+1 issued with the information available
at gate closure on day D.

## Roadmap

- [x] **v0.1** Evaluation metrics (MAE, rMAE, pinball, CRPS, coverage, Diebold-Mariano)
- [x] **v0.2** Data pipeline ([details](docs/data.md)): Energy-Charts prices, RTE load
      forecast, Open-Meteo weather (keyless), optional ENTSO-E, DST-safe hourly alignment
- [x] **v0.3** Baselines ([details](docs/models.md)): seasonal naive, LEAR, LightGBM
- [x] **v0.4** Rolling-origin backtest engine ([results](docs/backtest.md))
- [ ] **v0.5** Foundation models, zero-shot (Chronos-2, TimesFM, Moirai)
- [ ] **v0.6** Conformal calibration of the quantile forecasts
- [ ] **v0.7** Battery arbitrage backtest (EUR value of forecasts)
- [ ] **v0.8** Fine-tuned Chronos-2 on EPEX-FR
- [ ] **v1.0** Report, Hugging Face dataset and live leaderboard

## Development

```bash
uv sync                # macOS: LightGBM needs `brew install libomp`
uv run pytest
uv run ruff check && uv run ruff format --check
```

## References

- Lago, Marcjasz, De Schutter, Weron (2021). *Forecasting day-ahead electricity prices: A review of
  state-of-the-art algorithms, best practices and an open-access benchmark.* Applied Energy 293.
  [arXiv:2008.08004](https://arxiv.org/abs/2008.08004)
- Ansari et al. (2025). *Chronos-2: From Univariate to Universal Forecasting.*
  [arXiv:2510.15821](https://arxiv.org/abs/2510.15821)
- Shchur et al. (2025). *fev-bench: A Realistic Benchmark for Time Series Forecasting.*
  [arXiv:2509.26468](https://arxiv.org/abs/2509.26468)
- Aksu et al. (2024). *GIFT-Eval: A Benchmark For General Time Series Forecasting Model
  Evaluation.* [arXiv:2410.10393](https://arxiv.org/abs/2410.10393)
- Diebold, Mariano (1995). *Comparing Predictive Accuracy.* Journal of Business & Economic
  Statistics 13(3).

## License

MIT
