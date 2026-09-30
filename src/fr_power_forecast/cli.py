"""Command-line entry point: ``fr-power-forecast build-dataset | backtest | evaluate``."""

import argparse
import os
from pathlib import Path

import pandas as pd

from fr_power_forecast import backtest as bt
from fr_power_forecast.data.dataset import (
    build_dataset,
    entsoe_sources,
    missing_report,
    public_sources,
)
from fr_power_forecast.data.entsoe import EntsoeClient
from fr_power_forecast.models import LEAR, LightGBM, SeasonalNaive, daily_panel
from fr_power_forecast.models.lear import DEFAULT_EXOG
from fr_power_forecast.models.tsfm import Chronos2, TimesFM3

DATASET = Path("data/processed/epex_fr_hourly.parquet")
RESULTS = Path("results")


def _utc_date(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _day(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _make_model(args):
    if args.model == "naive":
        return SeasonalNaive()
    if args.model == "lear":
        return LEAR(window_days=args.window_days)
    if args.model == "lightgbm":
        return LightGBM(window_days=args.window_days, quantiles=tuple(args.quantiles))
    zero_shot = {"chronos2": Chronos2, "timesfm3": TimesFM3}[args.model]
    return zero_shot(context_days=args.context_days, exog=DEFAULT_EXOG if args.exog else ())


def _result_name(model, recalibrate_every: int) -> str:
    return model.name if recalibrate_every == 1 else f"{model.name}_every{recalibrate_every}d"


def run_backtest(args) -> None:
    panel = daily_panel(pd.read_parquet(args.dataset))
    model = _make_model(args)
    directory = args.results_dir / _result_name(model, args.recalibrate_every)

    end = args.end or panel.index[panel["price"].notna().all(axis=1)].max() + pd.Timedelta(days=1)
    days = panel.index[(panel.index >= args.start) & (panel.index < end)]
    previous = bt.load(directory) if args.resume and directory.exists() else None
    if previous is not None:
        days = days[days > previous.forecasts.index.get_level_values("date").max()]
        if len(days) == 0:
            print(f"{directory} is already up to date")
            return

    result = bt.rolling_backtest(model, panel, days, args.recalibrate_every, args.n_jobs)
    if previous is not None:
        result = bt.extend(previous, result)
    bt.save(result, directory)
    timings = result.timings
    if model.zero_shot:
        seconds = timings["predict_seconds"].sum() / timings["n_days"].sum()
        print(f"{len(days)} days written to {directory}; zero-shot, {seconds:.3f} s per day")
        return
    print(
        f"{len(days)} days written to {directory}; {len(timings)} recalibrations, "
        f"mean fit {timings['fit_seconds'].mean():.2f} s, "
        f"last recalibration on data up to {timings['calibrated_until'].max():%Y-%m-%d}"
    )


def run_evaluate(args) -> None:
    panel = daily_panel(pd.read_parquet(args.dataset))
    names = args.models or sorted(p.name for p in args.results_dir.iterdir() if p.is_dir())
    results = {name: bt.load(args.results_dir / name).forecasts for name in names}
    table = bt.evaluate(results, panel, args.reference, args.start, args.end)
    days = table.attrs["days"]
    print(f"{len(days)} common days, {days.min():%Y-%m-%d} to {days.max():%Y-%m-%d}\n")
    print(table.to_string(float_format=lambda v: f"{v:.4g}"))

    fresh = bt.freshness(results, panel)
    print("\n" + fresh.to_string())
    stale = fresh[fresh["days_not_backtested"] > 0]
    if len(stale):
        print(
            f"\nThe dataset has price data beyond the backtest of {', '.join(stale.index)}. "
            "Recalibrate with `fr-power-forecast backtest --resume` to get the latest results."
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fr-power-forecast")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-dataset", help="download and assemble the hourly dataset")
    build.add_argument("--start", type=_utc_date, default=_utc_date("2019-01-01"))
    build.add_argument("--end", type=_utc_date, required=True, help="exclusive, UTC date")
    build.add_argument(
        "--source",
        choices=["public", "entsoe"],
        default="public",
        help="public: Energy-Charts + RTE, no key; entsoe: needs ENTSOE_API_KEY",
    )
    build.add_argument("--weather", choices=["reanalysis", "forecast", "none"], default="forecast")
    build.add_argument("--cache-dir", type=Path, default=Path("data/raw"))
    build.add_argument("--out", type=Path, default=DATASET)

    backtest = commands.add_parser("backtest", help="rolling-origin backtest of one model")
    backtest.add_argument(
        "--model", choices=["naive", "lear", "lightgbm", "chronos2", "timesfm3"], required=True
    )
    backtest.add_argument("--start", type=_day, required=True, help="first test day")
    backtest.add_argument("--end", type=_day, help="exclusive; default: after the last price")
    backtest.add_argument("--recalibrate-every", type=int, default=1, help="days")
    backtest.add_argument("--window-days", type=int, default=1456)
    backtest.add_argument("--quantiles", type=float, nargs="*", default=[], help="lightgbm only")
    backtest.add_argument("--context-days", type=int, default=336, help="foundation models only")
    backtest.add_argument(
        "--exog", action="store_true", help="foundation models: use the exogenous forecasts"
    )
    backtest.add_argument("--n-jobs", type=int, default=1)
    backtest.add_argument("--resume", action="store_true", help="only forecast new days")
    backtest.add_argument("--dataset", type=Path, default=DATASET)
    backtest.add_argument("--results-dir", type=Path, default=RESULTS)

    evaluate = commands.add_parser("evaluate", help="compare backtested models")
    evaluate.add_argument("--models", nargs="*", help="result names; default: all")
    evaluate.add_argument("--reference", default="naive")
    evaluate.add_argument("--start", type=_day, help="first scored day")
    evaluate.add_argument("--end", type=_day, help="exclusive")
    evaluate.add_argument("--dataset", type=Path, default=DATASET)
    evaluate.add_argument("--results-dir", type=Path, default=RESULTS)

    args = parser.parse_args(argv)
    if args.command == "backtest":
        return run_backtest(args)
    if args.command == "evaluate":
        return run_evaluate(args)

    if args.source == "entsoe":
        api_key = os.environ.get("ENTSOE_API_KEY")
        if not api_key:
            parser.error("--source entsoe requires the ENTSOE_API_KEY environment variable")
        sources = entsoe_sources(EntsoeClient(api_key))
    else:
        sources = public_sources()

    dataset = build_dataset(
        args.start,
        args.end,
        sources,
        weather_source=None if args.weather == "none" else args.weather,
        cache_dir=args.cache_dir,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(args.out)
    print(f"{len(dataset)} hours written to {args.out}")
    print(missing_report(dataset).to_string())


if __name__ == "__main__":
    main()
