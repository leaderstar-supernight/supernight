from pathlib import Path
import sys

import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1] / 'Stock_price_prediction'
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from timing_decision import apply_shadow_decision


def sample_frame():
    return pd.DataFrame({
        '代號': list('ABCD'),
        '未持有建議': ['符合研究進場條件', '等待', '符合研究進場條件', '等待'],
        '持有情境建議': ['若持有：續抱觀察'] * 4,
        '風險': ['未觸發風險門檻', '未觸發風險門檻', '波動偏高', '未觸發風險門檻'],
        '新聞情緒分數': [1.0, 0.5, -0.5, -1.0],
        'RSI': [20.0, 40.0, 60.0, 80.0],
        '個股情緒分數': [20.0, 40.0, 60.0, 80.0],
        'ATR比例': [0.01, 0.02, 0.03, 0.04],
        'AI上行先觸機率': [0.6] * 4,
        'AI下行先觸機率': [0.3] * 4,
        'AI盤整機率': [0.1] * 4,
        'TimesFM估計上行先觸機率': [0.2] * 4,
        'TimesFM估計下行先觸機率': [0.7] * 4,
        'TimesFM估計盤整機率': [0.1] * 4,
    })


def test_shadow_decision_never_changes_formal_advice():
    source = sample_frame()
    expected_unheld = source['未持有建議'].copy()
    expected_held = source['持有情境建議'].copy()
    result = apply_shadow_decision(source, 'TW', {'decision': {}})
    pd.testing.assert_series_equal(result['未持有建議'], expected_unheld)
    pd.testing.assert_series_equal(result['持有情境建議'], expected_held)


def test_shadow_decision_confirms_only_with_factors_and_no_risk():
    result = apply_shadow_decision(sample_frame(), 'TW', {'decision': {}})
    assert result.loc[0, '研究因子分數'] == 2
    assert result.loc[0, '研究決策'] == '觀察候選（基本條件＋因子確認）'
    assert result.loc[2, '研究決策'] == '風險否決（研究）'
    assert result.loc[3, '研究因子分數'] == -2
    assert result.loc[3, '研究決策'] == '偏空警示（研究）'


def test_ai_and_timesfm_are_display_only():
    source = sample_frame()
    source.loc[1, ['AI上行先觸機率', 'AI下行先觸機率', 'AI盤整機率']] = [0.99, 0.005, 0.005]
    result = apply_shadow_decision(source, 'US', {'decision': {}})
    assert result.loc[1, '研究決策'] == '等待／觀察'
    assert '不計入決策' in result.loc[1, '研究模型參考']
    assert 'TimesFM下行最高 70.0%' in result.loc[1, '研究模型參考']
    assert '不改正式' in result.loc[1, '研究決策狀態']
