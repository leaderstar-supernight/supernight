"""TimesFM 3.0 ten-session price-path research shared by TW and US stage 2.

The output is deliberately separate from the classification ensemble and from
all trading rules. Model weights are downloaded to ignored system_data and are
never committed to the repository.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLASS_ZH = {
    "down_first": "下行先觸",
    "neutral": "盤整",
    "up_first": "上行先觸",
}


def summary_columns(market: str) -> list[str]:
    price = "TimesFM第10日預測價USD" if market == "US" else "TimesFM第10日預測價"
    return [
        "TimesFM10日路徑判讀",
        price,
        "TimesFM10日預測漲跌幅",
        "TimesFM預估觸及日",
        "TimesFM狀態",
    ]


def _empty_summary(market: str, status: str, model_id: str | None = None) -> dict[str, Any]:
    price = "TimesFM第10日預測價USD" if market == "US" else "TimesFM第10日預測價"
    return {
        "TimesFM10日路徑判讀": "資料不足",
        price: np.nan,
        "TimesFM10日預測漲跌幅": np.nan,
        "TimesFM預估觸及日": "資料不足",
        "TimesFM狀態": status,
        "TimesFM模型": model_id,
        "TimesFM訊號日期": pd.NaT,
        "TimesFM上方ATR門檻": np.nan,
        "TimesFM下方ATR門檻": np.nan,
    }


def _automatic_device(requested: str | None) -> str:
    requested = str(requested or "auto").lower()
    if requested in {"cpu", "cuda"}:
        return requested
    if requested != "auto":
        raise ValueError("TimesFM device 必須是 auto、cpu 或 cuda")
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_evaluator(settings: dict[str, Any]):
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    cache_dir = settings.get("model_cache")
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return TimesFM3Evaluator(
        ModelConfig(
            checkpoint_path=settings["model_id"],
            per_core_batch_size=int(settings.get("per_core_batch_size", 4)),
            device=_automatic_device(settings.get("device")),
            cache_dir=cache_dir,
        )
    )


def _atr_on_adjusted(prices: pd.DataFrame, period: int) -> float:
    previous = prices["Close"].shift(1)
    true_range = pd.concat(
        [
            prices["High"] - prices["Low"],
            (prices["High"] - previous).abs(),
            (prices["Low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    value = true_range.rolling(int(period), min_periods=int(period)).mean().iloc[-1]
    return float(value) if np.isfinite(value) else np.nan


def _classify_path(
    path: np.ndarray, upper: float, lower: float
) -> tuple[str, int | None]:
    for step, value in enumerate(path, start=1):
        if not np.isfinite(value):
            continue
        if value <= lower:
            return "down_first", step
        if value >= upper:
            return "up_first", step
    return "neutral", None


class TimesFMPathRunner:
    """Load the model once, then forecast one current path per candidate."""

    def __init__(self, cfg: dict[str, Any], market: str, evaluator: Any | None = None):
        if market not in {"TW", "US"}:
            raise ValueError("TimesFM market 必須是 TW 或 US")
        self.market = market
        self.settings = cfg.get("timesfm", {})
        self.ai = cfg["ai"]
        self.enabled = bool(self.settings.get("enabled", False))
        self.model_id = self.settings.get("model_id")
        self.evaluator = evaluator
        self.load_error: str | None = None
        if not self.enabled:
            return
        if not bool(self.settings.get("research_only", False)):
            self.load_error = "設定未確認 research_only，TimesFM 3.0 已停用"
            return
        if self.settings.get("backend") != "timesfm_3_pytorch":
            self.load_error = "不支援的 TimesFM backend"
            return
        if int(self.settings.get("horizon", 0)) != int(self.ai["horizon"]):
            self.load_error = "TimesFM 與 AI horizon 不一致"
            return
        if evaluator is None:
            try:
                self.evaluator = _load_evaluator(self.settings)
            except ImportError:
                self.load_error = "缺少 timesfm 3.0；請安裝 requirements"
            except Exception as exc:
                self.load_error = "模型載入失敗：" + type(exc).__name__

    def forecast(
        self, ticker: str, raw: pd.DataFrame, adjusted: pd.DataFrame | None
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        if not self.enabled:
            return _empty_summary(self.market, "未啟用", self.model_id), pd.DataFrame()
        if self.load_error or self.evaluator is None:
            return _empty_summary(
                self.market, self.load_error or "模型未載入", self.model_id
            ), pd.DataFrame()
        try:
            return self._forecast(ticker, raw, adjusted)
        except Exception as exc:
            status = "資料或預測失敗：" + type(exc).__name__
            return _empty_summary(self.market, status, self.model_id), pd.DataFrame()

    def unavailable(self, status: str) -> tuple[dict[str, Any], pd.DataFrame]:
        """Return an explicit non-fatal status without loading or predicting."""
        return _empty_summary(self.market, status, self.model_id), pd.DataFrame()

    def _forecast(
        self, ticker: str, raw: pd.DataFrame, adjusted: pd.DataFrame | None
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        if adjusted is None or adjusted.empty or raw.empty:
            raise ValueError("缺少還原行情")
        common = raw.index.intersection(adjusted.index)
        raw = raw.loc[common]
        adjusted = adjusted.loc[common]
        valid = (
            np.isfinite(raw["Close"])
            & np.isfinite(adjusted[["High", "Low", "Close"]]).all(axis=1)
            & (raw["Close"] > 0)
            & (adjusted["Close"] > 0)
        )
        if not bool(valid.iloc[-1]):
            raise ValueError("最新還原行情不完整")
        signal_date = pd.Timestamp(common[-1]).normalize()
        close_raw = float(raw["Close"].iloc[-1])
        close_adjusted = float(adjusted["Close"].iloc[-1])
        context_length = int(self.settings.get("context_length", 512))
        min_context = int(self.settings.get("min_context", 256))
        context = adjusted.loc[valid, "Close"].tail(context_length).to_numpy(dtype=np.float32)
        if len(context) < min_context:
            raise ValueError("有效歷史不足")
        context = context / context[-1] * 100.0

        outputs = list(
            self.evaluator.predict_batch(
                contexts=[context],
                horizon=int(self.settings["horizon"]),
                return_quantiles=True,
                use_symmetric_averaging=bool(
                    self.settings.get("symmetric_averaging", False)
                ),
                make_positive=True,
                sort_quantiles=True,
            )
        )
        if len(outputs) != 1:
            raise RuntimeError("模型回傳筆數錯誤")
        point = np.asarray(outputs[0].forecast, dtype=float).squeeze()
        quantiles = np.asarray(outputs[0].quantiles, dtype=float).squeeze()
        horizon = int(self.settings["horizon"])
        if point.shape != (horizon,) or quantiles.shape != (horizon, 9):
            raise RuntimeError("模型回傳形狀錯誤")
        scale = close_raw / 100.0
        point_raw = point * scale
        q10 = quantiles[:, 0] * scale
        q50 = quantiles[:, 4] * scale
        q90 = quantiles[:, 8] * scale

        atr_adjusted = _atr_on_adjusted(
            adjusted.loc[:signal_date], int(self.settings.get("atr_period", 14))
        )
        factor = close_adjusted / close_raw
        atr_raw = atr_adjusted / factor if np.isfinite(factor) and factor > 0 else np.nan
        if not np.isfinite(atr_raw) or atr_raw <= 0:
            raise ValueError("ATR 無法計算")
        upper = close_raw + float(self.ai["up_atr"]) * atr_raw
        lower = close_raw - float(self.ai["down_atr"]) * atr_raw
        path_class, hit_step = _classify_path(point_raw, upper, lower)
        price_column = (
            "TimesFM第10日預測價USD" if self.market == "US" else "TimesFM第10日預測價"
        )
        summary = {
            "TimesFM10日路徑判讀": CLASS_ZH[path_class],
            price_column: float(point_raw[-1]),
            "TimesFM10日預測漲跌幅": float(point_raw[-1] / close_raw - 1),
            "TimesFM預估觸及日": (
                "10日內未觸及" if hit_step is None else f"第{hit_step}交易日"
            ),
            "TimesFM狀態": "可用；價格路徑研究，不參與AI集成或交易建議",
            "TimesFM模型": self.model_id,
            "TimesFM訊號日期": signal_date,
            "TimesFM上方ATR門檻": upper,
            "TimesFM下方ATR門檻": lower,
        }
        estimated_dates = pd.bdate_range(
            signal_date + pd.offsets.BusinessDay(1), periods=horizon
        )
        path = pd.DataFrame(
            {
                "市場": self.market,
                "代號": ticker,
                "訊號日期": signal_date,
                "交易日序": np.arange(1, horizon + 1),
                "預估日期（未排休市）": estimated_dates,
                "TimesFM點預測": point_raw,
                "Q10": q10,
                "Q50": q50,
                "Q90": q90,
                "上方ATR門檻": upper,
                "下方ATR門檻": lower,
                "10日路徑判讀": CLASS_ZH[path_class],
                "預估觸及交易日": hit_step,
                "用途": "個人研究；不參與AI集成、評分或交易建議",
            }
        )
        return summary, path
