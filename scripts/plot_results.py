"""Figures of the backtest results for the README, in light and dark variants.

uv run --group plots python scripts/plot_results.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, DayLocator

from fr_power_forecast import backtest as bt
from fr_power_forecast.models import daily_panel

OUT = Path("docs/img")

THEMES = {
    "light": {
        "surface": "#ffffff",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
        "lear": "#2a78d6",
        "naive": "#eb6834",
    },
    "dark": {
        "surface": "#0d1117",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#30302e",
        "lear": "#3987e5",
        "naive": "#d95926",
    },
}


def style(ax, theme):
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["muted"])
    ax.tick_params(colors=theme["muted"], length=0, labelsize=9)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def title(fig, theme, text, subtitle):
    fig.text(0.06, 0.94, text, fontsize=13, weight="bold", color=theme["text"])
    fig.text(0.06, 0.885, subtitle, fontsize=9.5, color=theme["muted"])


def monthly_mae(panel, results, theme, path):
    truth = panel["price"]
    errors = {}
    for name, frame in results.items():
        pred = frame["forecast"].unstack("hour")
        days = pred.index.intersection(truth.dropna().index)
        daily = (pred.loc[days] - truth.loc[days]).abs().mean(axis=1)
        errors[name] = daily.resample("MS").mean()

    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=theme["surface"])
    fig.subplots_adjust(left=0.06, right=0.9, top=0.8, bottom=0.1)
    style(ax, theme)
    series = [("naive", "Seasonal naive", "--"), ("lear_1456", "LEAR", "-")]
    color = {"naive": theme["naive"], "lear_1456": theme["lear"]}
    for name, label, dash in series:
        e = errors[name]
        ax.plot(e.index, e.values, dash, color=color[name], linewidth=2, label=label)
        ax.annotate(
            label,
            (e.index[-1], e.values[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9.5,
            color=theme["text"],
        )
    ax.set_ylim(0, None)
    ax.set_ylabel("EUR/MWh", color=theme["muted"], fontsize=9)
    ax.axvspan(
        pd.Timestamp("2022-01-01"),
        pd.Timestamp("2023-01-01"),
        color=theme["grid"],
        alpha=0.5,
        linewidth=0,
    )
    ax.text(
        pd.Timestamp("2022-07-01"),
        3,
        "2022 energy crisis",
        ha="center",
        va="bottom",
        fontsize=9,
        color=theme["muted"],
    )
    legend = ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=theme["text"])
    legend.set_in_layout(False)
    title(
        fig,
        theme,
        "Monthly forecast error, day-ahead prices FR",
        "Mean absolute error per month, rolling-origin backtest with daily recalibration",
    )
    fig.savefig(path, dpi=200, facecolor=theme["surface"])
    plt.close(fig)


def typical_week(panel, results, year=2026) -> pd.Timestamp:
    """Monday of the week of ``year`` whose LEAR rMAE is closest to the whole backtest's."""
    truth = panel["price"]
    abs_error = {}
    for name in ("naive", "lear_1456"):
        pred = results[name]["forecast"].unstack("hour")
        days = pred.index.intersection(truth.dropna().index)
        abs_error[name] = (pred.loc[days] - truth.loc[days]).abs().mean(axis=1)
    overall = abs_error["lear_1456"].mean() / abs_error["naive"].mean()
    weekly = {
        k: v.loc[str(year)].resample("W-MON", label="left", closed="left")
        for k, v in abs_error.items()
    }
    counts = weekly["naive"].count()
    ratio = (weekly["lear_1456"].mean() / weekly["naive"].mean())[counts == 7]
    return (ratio - overall).abs().idxmin()


def sample_week(panel, results, theme, path):
    start = typical_week(panel, results)
    end = start + pd.Timedelta(days=7)
    days = pd.date_range(start, end, freq="D", inclusive="left")
    hours = pd.date_range(start, end, freq="1h", inclusive="left")
    truth = panel["price"].loc[days].to_numpy().ravel()

    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=theme["surface"])
    fig.subplots_adjust(left=0.06, right=0.97, top=0.76, bottom=0.1)
    style(ax, theme)
    ax.axhline(0, color=theme["muted"], linewidth=1)
    ax.plot(hours, truth, color=theme["text"], linewidth=2, label="Actual price")
    for name, label, dash, key in [
        ("lear_1456", "LEAR", "-", "lear"),
        ("naive", "Seasonal naive", "--", "naive"),
    ]:
        pred = results[name]["forecast"].unstack("hour").loc[days].to_numpy().ravel()
        mae = np.mean(np.abs(pred - truth))
        ax.plot(hours, pred, dash, color=theme[key], linewidth=2, label=f"{label} (MAE {mae:.1f})")
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%a %d %b"))
    ax.set_ylabel("EUR/MWh", color=theme["muted"], fontsize=9)
    ax.legend(
        loc="lower left",
        ncol=3,
        frameon=False,
        fontsize=9,
        labelcolor=theme["text"],
        bbox_to_anchor=(-0.01, 1.0),
    )
    title(
        fig,
        theme,
        f"One week of forecasts, {start:%d} to {end - pd.Timedelta(days=1):%d %B %Y}",
        "A typical 2026 week: LEAR/naive error ratio closest to the backtest average",
    )
    fig.savefig(path, dpi=200, facecolor=theme["surface"])
    plt.close(fig)


def main():
    panel = daily_panel(pd.read_parquet("data/processed/epex_fr_hourly.parquet"))
    results = {name: bt.load(Path("results") / name).forecasts for name in ["naive", "lear_1456"]}
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, theme in THEMES.items():
        monthly_mae(panel, results, theme, OUT / f"monthly_mae_{mode}.png")
        sample_week(panel, results, theme, OUT / f"sample_week_{mode}.png")


if __name__ == "__main__":
    main()
