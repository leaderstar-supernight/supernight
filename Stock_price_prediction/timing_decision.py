"""Cross-sectional shadow decision layer for stage-2 research outputs.

This module deliberately leaves the formal rule advice unchanged.  Its output is
an auditable research candidate/warning layer that can be promoted only after
the monthly time-series validation gate is satisfied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


DECISION_COLUMNS = [
    '研究決策', '研究因子方向', '研究因子分數', '研究因子依據',
    '研究模型參考', '研究決策狀態',
]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')


def _cutoffs(series: pd.Series, minimum_count: int) -> tuple[float, float] | None:
    values = _numeric(series).dropna()
    if len(values) < minimum_count or values.nunique() < 2:
        return None
    low = float(values.quantile(0.25))
    high = float(values.quantile(0.75))
    return (low, high) if high > low else None


def _probability_reference(row: pd.Series) -> str:
    labels = {
        'AI上行先觸機率': 'AI上行', 'AI下行先觸機率': 'AI下行', 'AI盤整機率': 'AI盤整',
        'TimesFM估計上行先觸機率': 'TimesFM上行',
        'TimesFM估計下行先觸機率': 'TimesFM下行',
        'TimesFM估計盤整機率': 'TimesFM盤整',
    }
    ai = []
    timesfm = []
    for column, label in labels.items():
        value = pd.to_numeric(pd.Series([row.get(column)]), errors='coerce').iloc[0]
        if pd.isna(value):
            continue
        item = (float(value), label)
        (timesfm if column.startswith('TimesFM') else ai).append(item)
    parts = []
    if ai:
        value, label = max(ai)
        parts.append(f'{label}最高 {value:.1%}')
    else:
        parts.append('AI資料不足')
    if timesfm:
        value, label = max(timesfm)
        parts.append(f'{label}最高 {value:.1%}')
    else:
        parts.append('TimesFM資料不足')
    return '；'.join(parts) + '；目前均不計入決策'


def apply_shadow_decision(detail: pd.DataFrame, market: str, cfg: dict) -> pd.DataFrame:
    """Add a non-production research decision without changing formal advice.

    The first monthly review has fewer than ten independent signal dates.  Only
    news sentiment and an RSI chase filter are therefore scored.  Stock
    sentiment and ATR are disclosed as observations because their current
    economic meaning is unstable or overlaps other inputs.
    """
    out = detail.copy()
    settings = cfg.get('decision', {})
    if out.empty:
        return out.reindex(columns=[*out.columns, *[c for c in DECISION_COLUMNS if c not in out]])

    enabled = bool(settings.get('enabled', True))
    mode = str(settings.get('mode', 'shadow')).lower()
    minimum_count = int(settings.get('min_cross_section_count', 4))
    confirm_score = float(settings.get('min_confirm_score', 2.0))
    news_cut = _cutoffs(out.get('新聞情緒分數', pd.Series(index=out.index, dtype=float)), minimum_count)
    rsi_cut = _cutoffs(out.get('RSI', pd.Series(index=out.index, dtype=float)), minimum_count)
    sentiment_cut = _cutoffs(out.get('個股情緒分數', pd.Series(index=out.index, dtype=float)), minimum_count)
    atr_cut = _cutoffs(out.get('ATR比例', pd.Series(index=out.index, dtype=float)), minimum_count)

    rows = []
    for _, row in out.iterrows():
        score = 0.0
        scored = []
        observations = []

        news = pd.to_numeric(pd.Series([row.get('新聞情緒分數')]), errors='coerce').iloc[0]
        if news_cut and pd.notna(news):
            if news >= news_cut[1]:
                score += 1.0; scored.append('新聞情緒位於本次候選高檔（+1）')
            elif news <= news_cut[0]:
                score -= 1.0; scored.append('新聞情緒位於本次候選低檔（-1）')

        rsi = pd.to_numeric(pd.Series([row.get('RSI')]), errors='coerce').iloc[0]
        if rsi_cut and pd.notna(rsi):
            if rsi >= rsi_cut[1]:
                score -= 1.0; scored.append('RSI位於本次候選高檔，避免追價（-1）')
            elif rsi <= rsi_cut[0]:
                score += 1.0; scored.append('RSI位於本次候選低檔，僅作反彈觀察（+1）')

        stock_sentiment = pd.to_numeric(pd.Series([row.get('個股情緒分數')]), errors='coerce').iloc[0]
        if sentiment_cut and pd.notna(stock_sentiment):
            if stock_sentiment >= sentiment_cut[1]:
                observations.append('個股情緒高檔；月報目前呈不穩定／反向關係，不計分')
            elif stock_sentiment <= sentiment_cut[0]:
                observations.append('個股情緒低檔；月報目前呈不穩定／反向關係，不計分')

        atr = pd.to_numeric(pd.Series([row.get('ATR比例')]), errors='coerce').iloc[0]
        if market.upper() == 'US' and atr_cut and pd.notna(atr) and atr >= atr_cut[1]:
            observations.append('ATR位於候選高檔；只作風險／部位提醒，不視為看多')

        formal = str(row.get('未持有建議') or '')
        risk = str(row.get('風險') or '')
        safe = risk == '未觸發風險門檻'
        formal_entry = formal == '符合研究進場條件'
        if not enabled:
            decision = '未啟用'
        elif not safe:
            decision = '風險否決（研究）'
        elif score >= confirm_score:
            decision = '觀察候選（基本條件＋因子確認）' if formal_entry else '因子偏多但基本條件未成立'
        elif score <= -confirm_score:
            decision = '偏空警示（研究）'
        elif formal_entry:
            decision = '基本條件成立但因子未確認'
        else:
            decision = '等待／觀察'

        direction = '偏多確認' if score >= confirm_score else ('偏空警示' if score <= -confirm_score else '分歧／中性')
        basis = '；'.join(scored + observations) or '本次可用因子無明確分化'
        status = ('影子模式；不改正式未持有／持有建議；AI與TimesFM不計分'
                  if enabled and mode == 'shadow' else '研究設定停用或非影子模式')
        rows.append({
            '研究決策': decision,
            '研究因子方向': direction,
            '研究因子分數': score,
            '研究因子依據': basis,
            '研究模型參考': _probability_reference(row),
            '研究決策狀態': status,
        })

    decision_frame = pd.DataFrame(rows, index=out.index)
    for column in DECISION_COLUMNS:
        out[column] = decision_frame[column]
    return out
