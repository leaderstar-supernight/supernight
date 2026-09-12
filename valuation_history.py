"""Historical valuation context shared by the TW and US notebooks.

Historical PE percentiles never enter the fundamental score.  They may be
used by the separate price recommendation only when fair-value upside also
confirms the historical valuation signal.
"""

from __future__ import annotations

import math
import statistics
from typing import Mapping, Sequence

import pandas as pd


HISTORICAL_PE_WINDOWS = (1, 3, 5)
MIN_VALID_OBSERVATIONS = 8
HISTORICAL_RATIO_PERCENTILES = {
    "p05": 0.05,
    "p25": 0.25,
    "p50": 0.50,
    "p75": 0.75,
    "p95": 0.95,
}


def _finite_positive(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def calculate_historical_pe_context(
    ratios: pd.DataFrame | None,
    current_pe: float | None,
) -> dict[str, float | int | None]:
    """Return independent 1Y/3Y/5Y PE medians and empirical percentiles.

    The source is expected to contain monthly point-in-time PE observations.
    A window is available only when the valid PE history reaches its calendar
    boundary and the window has at least eight observations. Missing longer
    windows remain ``None``; shorter windows are never substituted.
    """
    windows = calculate_historical_ratio_windows(ratios, current_pe, "pe")
    result: dict[str, float | int | None] = {}
    for years in HISTORICAL_PE_WINDOWS:
        suffix = f"{years}y"
        statistics_for_window = windows.get(suffix)
        result[f"historical_pe_median_{suffix}"] = (
            statistics_for_window["p50"] if statistics_for_window else None
        )
        result[f"historical_pe_percentile_{suffix}"] = (
            statistics_for_window["current_percentile"] if statistics_for_window else None
        )
        result[f"historical_pe_sample_count_{suffix}"] = (
            int(statistics_for_window["count"])
            if statistics_for_window
            else int(windows.get(f"{suffix}_sample_count", 0))
        )
    result["pe_median_trend_1y_vs_3y"] = None
    result["pe_median_trend_1y_vs_5y"] = None

    median_1y = result["historical_pe_median_1y"]
    for years in (3, 5):
        comparison = result[f"historical_pe_median_{years}y"]
        if median_1y is not None and comparison not in (None, 0):
            result[f"pe_median_trend_1y_vs_{years}y"] = float(median_1y / comparison - 1.0)

    return result


def calculate_historical_ratio_windows(
    ratios: pd.DataFrame | None,
    current_value: float | None,
    column: str,
) -> dict[str, dict[str, float] | int | None]:
    """Calculate complete, independent 1Y/3Y/5Y ratio distributions.

    The result is intentionally not a weighted or averaged composite.  Each
    available window contains its own P05/P25/P50/P75/P95 distribution and
    the current ratio's empirical percentile.  A longer window is never
    filled with a shorter one.
    """
    result: dict[str, dict[str, float] | int | None] = {}
    for years in HISTORICAL_PE_WINDOWS:
        result[f"{years}y"] = None
        result[f"{years}y_sample_count"] = 0

    if ratios is None or ratios.empty or not {"date", column}.issubset(ratios.columns):
        return result

    valid = ratios[["date", column]].copy()
    valid["date"] = pd.to_datetime(valid["date"], errors="coerce", utc=True).dt.tz_convert(None)
    latest = valid["date"].dropna().max()
    if pd.isna(latest):
        return result
    valid[column] = valid[column].map(_finite_positive)
    valid = valid.dropna(subset=["date", column]).sort_values("date")
    valid = valid.drop_duplicates("date", keep="last")
    if valid.empty:
        return result

    usable_current = _finite_positive(current_value)
    for years in HISTORICAL_PE_WINDOWS:
        suffix = f"{years}y"
        cutoff = latest - pd.DateOffset(years=years)
        window = valid.loc[valid["date"] > cutoff]
        sample_count = len(window)
        result[f"{suffix}_sample_count"] = sample_count

        # Monthly data may start at the first month-end immediately after cutoff.
        full_coverage = valid["date"].min() <= cutoff + pd.offsets.MonthEnd(1)
        if not full_coverage or sample_count < MIN_VALID_OBSERVATIONS:
            continue

        values = window[column].astype(float)
        statistics_for_window = {
            key: float(values.quantile(quantile))
            for key, quantile in HISTORICAL_RATIO_PERCENTILES.items()
        }
        statistics_for_window["mean"] = float(values.mean())
        statistics_for_window["min"] = float(values.min())
        statistics_for_window["max"] = float(values.max())
        statistics_for_window["count"] = float(sample_count)
        statistics_for_window["current_percentile"] = (
            float(100.0 * (values <= usable_current).sum() / sample_count)
            if usable_current is not None
            else None
        )
        result[suffix] = statistics_for_window

    return result


def select_historical_valuation_reference(
    windows: Mapping[str, Mapping[str, float] | int | None] | None,
) -> dict[str, object]:
    """Select one explainable price-band window without blending time scales.

    Three years is the normal anchor.  One year is selected only when the
    1Y, 3Y and 5Y medians all confirm a continuing re-rating in the same
    direction.  Five years remains structural context and is not averaged
    into the active band.
    """
    unavailable = {
        "reference_period": None,
        "environment": "資料不足",
        "confidence": "資料不足",
        "statistics": None,
        "current_percentile": None,
    }
    if not windows:
        return unavailable

    stats_1y = windows.get("1y")
    stats_3y = windows.get("3y")
    stats_5y = windows.get("5y")
    if not isinstance(stats_3y, Mapping) or _finite_positive(stats_3y.get("p50")) is None:
        return unavailable

    reference_period = "3Y"
    reference_stats = stats_3y
    environment = "以3年中樞為主"
    confidence = "中"

    med_1y = _finite_positive(stats_1y.get("p50")) if isinstance(stats_1y, Mapping) else None
    med_3y = _finite_positive(stats_3y.get("p50"))
    med_5y = _finite_positive(stats_5y.get("p50")) if isinstance(stats_5y, Mapping) else None

    if med_1y is not None and med_3y is not None and med_5y is not None:
        change_1y_3y = med_1y / med_3y - 1.0
        change_3y_5y = med_3y / med_5y - 1.0
        continuing_shift = (
            change_1y_3y * change_3y_5y > 0
            and abs(change_1y_3y) >= 0.15
            and abs(change_3y_5y) >= 0.10
        )
        if continuing_shift:
            reference_period = "1Y"
            reference_stats = stats_1y
            direction = "升評" if change_1y_3y > 0 else "降評"
            environment = f"估值中樞持續{direction}"
            confidence = "中"
        elif abs(change_1y_3y) <= 0.15 and abs(change_3y_5y) <= 0.15:
            environment = "估值中樞穩定"
            confidence = "高"
        elif abs(change_1y_3y) <= 0.15:
            environment = "近三年中樞已轉移並趨穩"
            confidence = "中"
        else:
            environment = "一年估值偏離三年中樞"
            confidence = "中"
    elif med_1y is not None:
        environment = "五年資料不足；採三年中樞"

    return {
        "reference_period": reference_period,
        "environment": environment,
        "confidence": confidence,
        "statistics": reference_stats,
        "current_percentile": reference_stats.get("current_percentile"),
    }


def assess_pe_valuation_applicability(
    eps_history: Sequence[float | None] | None,
    current_eps: float | None,
) -> str:
    """Return a caution label for using PE on the company's earnings history."""
    if _finite_positive(current_eps) is None:
        return "不適用（目前EPS非正值）"
    raw = list(eps_history or [])[:5]
    finite = []
    has_non_positive = False
    for value in raw:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            continue
        if number <= 0:
            has_non_positive = True
        else:
            finite.append(number)
    if len(finite) < 3:
        return "資料不足"
    if has_non_positive:
        return "有限（近年EPS含非正值）"
    mean = statistics.fmean(finite)
    coefficient_of_variation = statistics.pstdev(finite) / mean if mean > 0 else math.inf
    if coefficient_of_variation >= 0.60:
        return "留意（近年EPS波動大）"
    return "正常"


def calculate_price_recommendation(
    current_price: float | None,
    fair_price: float | None,
    historical_percentile: float | None,
    applicability: str,
    confidence: str,
    reference_period: str | None,
) -> dict[str, str | float | None]:
    """Combine historical position and fair-value upside into separate advice.

    This advice is independent from ``Fundamental Score``.  A cheap historical
    percentile alone cannot produce a buy recommendation; fair-value upside
    must confirm it.
    """
    price = _finite_positive(current_price)
    fair = _finite_positive(fair_price)
    try:
        percentile = float(historical_percentile) if historical_percentile is not None else None
    except (TypeError, ValueError):
        percentile = None
    if (
        price is None
        or fair is None
        or percentile is None
        or not math.isfinite(percentile)
        or reference_period is None
    ):
        return {
            "recommendation": "資料不足",
            "confidence": "資料不足",
            "upside": None,
            "basis": "缺少完整3年估值分布、現價或合理價",
        }

    upside = fair / price - 1.0
    if percentile <= 15 and upside >= 0.25:
        recommendation = "強買"
    elif percentile < 50 and upside >= 0.10:
        recommendation = "買進"
    elif percentile >= 95 and upside <= -0.25:
        recommendation = "強賣"
    elif percentile >= 85 and upside <= -0.10:
        recommendation = "賣出"
    else:
        recommendation = "持有"

    final_confidence = confidence
    if applicability.startswith(("不適用", "資料不足")):
        recommendation = "持有（估值適用性不足）"
        final_confidence = "低"
    elif applicability.startswith(("有限", "留意")):
        if recommendation in {"強買", "買進"}:
            recommendation = "持有（需檢查盈餘循環）"
        final_confidence = "低"

    return {
        "recommendation": recommendation,
        "confidence": final_confidence,
        "upside": upside,
        "basis": (
            f"{reference_period}估值百分位{percentile:.1f}%；"
            f"模型合理價上行空間{upside:.1%}；估值適用性：{applicability}"
        ),
    }


def calculate_fundamental_score(
    quality: float | None,
    growth: float | None,
    buffett: float | None,
    weights: Mapping[str, float],
) -> float | None:
    """Combine fundamental components; historical valuation is excluded."""
    components = (("quality", quality), ("growth", growth), ("buffett", buffett))
    usable = [
        (float(weights.get(name, 0.0)), float(value))
        for name, value in components
        if value is not None and float(weights.get(name, 0.0)) > 0
    ]
    weight_sum = sum(weight for weight, _ in usable)
    if not usable or weight_sum <= 0:
        return None
    return sum(weight * value for weight, value in usable) / weight_sum
