from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "Stock_price_prediction"))

from prediction_audit import _timesfm_table, review_output_path


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
    assert result.iloc[0]["正確率"] == 1.0
    assert result.iloc[0]["平衡正確率"] == 1.0
    assert result.iloc[0]["10日預測報酬MAE"] == pytest.approx(0.005)
    assert result.iloc[0]["判定"] == "初步有參考性"


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
