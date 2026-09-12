from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "Stock_price_prediction"))

from prediction_audit import (_timesfm_table, _training_dataset, _walk_forward_table,
                              review_output_path)
from timing_timesfm import TimesFMPathRunner, _quantile_class_probabilities


def test_review_output_path_uses_unified_reports_folder():
    actual = review_output_path(datetime(2026, 9, 12, 23, 4, 5))
    assert actual == (
        PROJECT
        / "reports"
        / "30日評估"
        / "2026"
        / "09"
        / "WHID_30日驗證_20260912_230405.xlsx"
    )


def test_timesfm_is_evaluated_as_hard_class_and_point_return():
    classes = ["down_first", "neutral", "up_first"]
    labels = ["下行先觸", "盤整", "上行先觸"]
    rows = []
    for index in range(30):
        predicted_return = (index % 5 - 2) / 100
        rows.append(
            {
                "market": "TW",
                "ticker": f"TEST{index}",
                "signal_date": "2026-01-01",
                "snapshot_json": json.dumps(
                    {
                        "TimesFM10日路徑判讀": labels[index % 3],
                        "TimesFM10日預測漲跌幅": predicted_return,
                        "TimesFM狀態": "可用；價格路徑研究",
                        "TimesFM模型": "google/timesfm-test",
                        "TimesFM估計下行先觸機率": 0.8 if index % 3 == 0 else 0.1,
                        "TimesFM估計盤整機率": 0.8 if index % 3 == 1 else 0.1,
                        "TimesFM估計上行先觸機率": 0.8 if index % 3 == 2 else 0.1,
                    },
                    ensure_ascii=False,
                ),
                "actual_class": classes[index % 3],
                "return_10": predicted_return - 0.005,
            }
        )

    result = _timesfm_table(pd.DataFrame(rows))

    assert len(result) == 1
    assert result.iloc[0]["模型"] == "google/timesfm-test"
    assert result.iloc[0]["成熟樣本"] == 30
    assert result.iloc[0]["點路徑正確率"] == 1.0
    assert result.iloc[0]["機率最高類正確率"] == 1.0
    assert result.iloc[0]["平衡正確率"] == 1.0
    assert result.iloc[0]["10日預測報酬MAE"] == pytest.approx(0.005)
    assert result.iloc[0]["Brier Skill"] > 0
    assert result.iloc[0]["判定"] == "估計機率初步有參考性"


def test_timesfm_unavailable_rows_are_not_counted():
    frame = pd.DataFrame(
        [
            {
                "market": "US",
                "ticker": "TEST",
                "signal_date": "2026-01-01",
                "snapshot_json": json.dumps(
                    {
                        "TimesFM10日路徑判讀": "資料不足",
                        "TimesFM狀態": "未執行：資料不足",
                    },
                    ensure_ascii=False,
                ),
                "actual_class": "neutral",
                "return_10": 0.0,
            }
        ]
    )

    result = _timesfm_table(frame)

    assert result.empty


def test_quantile_paths_produce_three_class_scenario_shares():
    paths = pd.DataFrame(
        {
            "down_1": [99, 98], "down_2": [99, 97], "down_3": [98, 96],
            "flat_1": [100, 100], "flat_2": [100, 101], "flat_3": [101, 100],
            "up_1": [101, 103], "up_2": [102, 104], "up_3": [103, 105],
        }
    ).to_numpy()

    result = _quantile_class_probabilities(paths, upper=102.5, lower=98.5)

    assert result == {"down_first": 3 / 9, "neutral": 3 / 9, "up_first": 3 / 9}


def test_timesfm_runner_exports_probabilities_and_all_quantile_paths():
    horizon = 10
    quantiles = pd.DataFrame(
        {
            **{f"down{index}": pd.Series(range(horizon)).map(lambda step: 100 - step / 3)
               for index in range(3)},
            **{f"flat{index}": [100.0] * horizon for index in range(3)},
            **{f"up{index}": pd.Series(range(horizon)).map(lambda step: 100 + step / 2)
               for index in range(3)},
        }
    ).to_numpy()

    class Evaluator:
        def predict_batch(self, **_):
            return [SimpleNamespace(
                forecast=pd.Series(range(horizon)).map(lambda step: 100 + step / 2).to_numpy(),
                quantiles=quantiles,
            )]

    cfg = {
        "ai": {"horizon": horizon, "up_atr": 1.5, "down_atr": 1.0},
        "timesfm": {
            "enabled": True, "research_only": True, "backend": "timesfm_3_pytorch",
            "model_id": "google/timesfm-test", "horizon": horizon,
            "context_length": 256, "min_context": 256, "atr_period": 14,
        },
    }
    index = pd.bdate_range("2025-01-01", periods=300)
    prices = pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0}, index=index
    )
    runner = TimesFMPathRunner(cfg, "TW", evaluator=Evaluator())

    summary, path = runner.forecast("TEST", prices, prices)

    assert summary["TimesFM估計下行先觸機率"] == pytest.approx(3 / 9)
    assert summary["TimesFM估計盤整機率"] == pytest.approx(3 / 9)
    assert summary["TimesFM估計上行先觸機率"] == pytest.approx(3 / 9)
    assert set(f"Q{value}" for value in range(10, 100, 10)).issubset(path.columns)


def test_training_features_use_only_current_and_past_snapshots():
    outcome_columns = [
        "return_5", "return_10", "return_20", "return_60", "return_126", "return_252",
        "benchmark_return_5", "benchmark_return_10", "benchmark_return_20",
        "abnormal_return_5", "abnormal_return_10", "abnormal_return_20",
        "max_up_10", "max_down_10",
    ]
    rows = []
    for day, rsi in enumerate([40.0, 42.0, 45.0, 44.0], start=1):
        row = {
            "market": "TW", "ticker": "TEST", "signal_date": f"2026-01-0{day}",
            "actual_class": "neutral",
            "snapshot_json": json.dumps({"RSI": rsi, "短期趨勢": "偏多"}, ensure_ascii=False),
        }
        row.update({column: 0.01 for column in outcome_columns})
        rows.append(row)

    training, raw, derived, categories = _training_dataset(pd.DataFrame(rows))

    assert raw == ["RSI"]
    assert categories == ["短期趨勢"]
    assert "變化1次｜RSI" in derived
    assert pd.isna(training.loc[0, "變化1次｜RSI"])
    assert training.loc[1, "變化1次｜RSI"] == 2.0
    assert training.loc[2, "斜率5次｜RSI"] == pytest.approx(2.5)


def test_walk_forward_screen_keeps_an_embargo_between_train_and_test():
    rows = []
    classes = ["down_first", "neutral", "up_first"]
    for day in range(40):
        for ticker_index in range(4):
            class_index = (day + ticker_index) % 3
            rows.append(
                {
                    "market": "TW",
                    "ticker": f"T{ticker_index}",
                    "signal_date": str((pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)).date()),
                    "actual_class": classes[class_index],
                    "RSI": 20.0 + class_index * 30 + ticker_index,
                }
            )

    result = _walk_forward_table(pd.DataFrame(rows), ["RSI"], embargo_dates=3)

    assert len(result) == 1
    assert result.iloc[0]["狀態"] == "完成時間序列樣本外驗證"
    assert result.iloc[0]["隔離日期數"] == 3
    assert pd.Timestamp(result.iloc[0]["訓練截止日"]) < pd.Timestamp(result.iloc[0]["測試起始日"])
