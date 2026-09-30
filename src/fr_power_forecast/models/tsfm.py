"""Time series foundation models used zero-shot: Chronos-2 and TimesFM 3.0.

The models are never trained on this dataset. To forecast day ``d`` they read a context of
``context_days`` days of hourly prices ending on day ``d - 1`` and, when ``exog`` is set, the
same window of exogenous forecasts plus their values on day ``d`` (known future covariates).
Contexts are the flattened (day x 24) panel, so DST days follow the 24-hour convention of
:mod:`fr_power_forecast.data.timegrid`. The point forecast is the median quantile.

Both libraries are optional: ``uv sync --extra tsfm``.
"""

from abc import abstractmethod

import numpy as np
import pandas as pd

from fr_power_forecast.models.base import TARGET, Forecaster, as_days

DECILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _window(panel: pd.DataFrame, variable: str, first: pd.Timestamp, last: pd.Timestamp):
    days = pd.date_range(first, last, freq="D")
    return panel[variable].reindex(days).to_numpy(dtype=float).ravel()


def _fill_gaps(values: np.ndarray) -> np.ndarray:
    """Forward-fill, then back-fill, NaN inside a context (past values only)."""
    return pd.Series(values).ffill().bfill().to_numpy()


class ZeroShot(Forecaster):
    """Base class: builds the contexts, subclasses run the model on a batch of them."""

    zero_shot = True
    label: str

    def __init__(
        self,
        context_days: int = 336,
        exog: tuple[str, ...] = (),
        quantiles: tuple[float, ...] = DECILES,
        batch_size: int = 64,
    ):
        if 0.5 not in quantiles:
            raise ValueError("quantiles must include the median")
        self.context_days = context_days
        self.exog = tuple(exog)
        self.quantiles = tuple(sorted(quantiles))
        self.batch_size = batch_size
        self.name = self.label + ("_exog" if self.exog else "")

    def contexts(self, panel: pd.DataFrame, days) -> list[dict]:
        """One input per day: ``target`` and ``past`` end on d-1, ``future`` covers day d."""
        inputs = []
        for day in as_days(days):
            first = day - pd.Timedelta(days=self.context_days)
            last = day - pd.Timedelta(days=1)
            inputs.append(
                {
                    "target": _fill_gaps(_window(panel, TARGET, first, last)),
                    "past": {v: _fill_gaps(_window(panel, v, first, last)) for v in self.exog},
                    "future": {v: _fill_gaps(_window(panel, v, day, day)) for v in self.exog},
                }
            )
        return inputs

    @abstractmethod
    def _forecast(self, inputs: list[dict]) -> np.ndarray:
        """(len(inputs), 24, n_quantiles) quantile forecasts."""

    def predict_quantiles(self, panel: pd.DataFrame, days) -> np.ndarray:
        days = as_days(days)
        inputs = self.contexts(panel, days)
        chunks = [
            self._forecast(inputs[i : i + self.batch_size])
            for i in range(0, len(inputs), self.batch_size)
        ]
        return np.sort(np.concatenate(chunks), axis=-1)

    def predict_with_quantiles(self, panel: pd.DataFrame, days):
        days = as_days(days)
        quantiles = self.predict_quantiles(panel, days)
        return self._frame(quantiles[..., self.quantiles.index(0.5)], days), quantiles

    def predict(self, panel: pd.DataFrame, days) -> pd.DataFrame:
        return self.predict_with_quantiles(panel, days)[0]


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Chronos2(ZeroShot):
    """Chronos-2 (Ansari et al., 2025), with known future covariates when ``exog`` is set."""

    label = "chronos2"

    def __init__(self, *args, checkpoint: str = "amazon/chronos-2", device=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint = checkpoint
        self.device = device
        self._pipeline = None

    def _forecast(self, inputs: list[dict]) -> np.ndarray:
        if self._pipeline is None:
            from chronos import Chronos2Pipeline

            self._pipeline = Chronos2Pipeline.from_pretrained(
                self.checkpoint, device_map=self.device or default_device()
            )
        batch = [
            {"target": x["target"], "past_covariates": x["past"], "future_covariates": x["future"]}
            if self.exog
            else {"target": x["target"]}
            for x in inputs
        ]
        quantiles, _ = self._pipeline.predict_quantiles(
            batch, prediction_length=24, quantile_levels=list(self.quantiles)
        )
        return np.stack([q[0].float().cpu().numpy() for q in quantiles])


class TimesFM3(ZeroShot):
    """TimesFM 3.0 (non-commercial weights), with past-and-future covariates when ``exog`` is set.

    The library clips forecasts at zero by default (``make_positive``), which would erase negative
    prices: it is disabled here.
    """

    label = "timesfm3"

    def __init__(self, *args, checkpoint: str = "google/timesfm-3.0-pytorch", device=None, **kw):
        super().__init__(*args, **kw)
        if self.quantiles != DECILES:
            raise ValueError("TimesFM 3.0 forecasts the nine deciles only")
        self.checkpoint = checkpoint
        self.device = device
        self._model = None

    def _forecast(self, inputs: list[dict]) -> np.ndarray:
        if self._model is None:
            from timesfm3 import ModelConfig, TimesFM3Evaluator

            config = ModelConfig(
                checkpoint_path=self.checkpoint,
                per_core_batch_size=self.batch_size,
                device=self.device or default_device(),
            )
            self._model = TimesFM3Evaluator(config)
        covariates = None
        if self.exog:
            covariates = [
                np.stack([np.concatenate([x["past"][v], x["future"][v]]) for v in self.exog])
                for x in inputs
            ]
        outputs = self._model.predict_batch(
            [x["target"].astype(np.float32) for x in inputs],
            horizon=24,
            past_future_covariates=covariates,
            return_quantiles=True,
            make_positive=False,
        )
        return np.stack([np.asarray(out.quantiles, dtype=float) for out in outputs])
