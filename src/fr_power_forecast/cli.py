"""Command-line entry point: ``fr-power-forecast build-dataset``."""

import argparse
import os
from pathlib import Path

import pandas as pd

from fr_power_forecast.data.dataset import (
    build_dataset,
    entsoe_sources,
    missing_report,
    public_sources,
)
from fr_power_forecast.data.entsoe import EntsoeClient


def _utc_date(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


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
    build.add_argument("--out", type=Path, default=Path("data/processed/epex_fr_hourly.parquet"))

    args = parser.parse_args(argv)
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
