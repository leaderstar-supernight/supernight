"""Conservative research fair-value model for stage-one TW/US analysis.

This module is deliberately separate from the production model. Historical
valuation and analyst targets are not inputs. The V2 research model normalizes
growth, reduces reliance on divergent forward estimates, and expresses quality
as bounded PE points instead of multiplying growth by a PEG factor.
"""
from __future__ import annotations

import math
from typing import Any


MODEL_VERSION = "Fundamental-Fair-Value-V2-Research"


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _positive(value: Any) -> float | None:
    result = _number(value)
    return result if result is not None and result > 0 else None


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def earnings_trend_multiplier(score: Any) -> float | None:
    """Use a narrow trend adjustment so one quarter cannot dominate value."""
    value = _number(score)
    if value is None:
        return None
    if value >= 80:
        return 1.05
    if value >= 60:
        return 1.025
    if value >= 40:
        return 1.00
    if value >= 20:
        return 0.95
    return 0.90


def _roic_adjustment(value: Any) -> float | None:
    value = _number(value)
    if value is None:
        return None
    if value >= 0.25:
        return 0.10
    if value >= 0.15:
        return 0.05
    if value >= 0.08:
        return 0.0
    return -0.10


def _fcf_margin_adjustment(value: Any) -> float | None:
    value = _number(value)
    if value is None:
        return None
    if value >= 0.20:
        return 0.10
    if value >= 0.10:
        return 0.05
    if value >= 0.05:
        return 0.0
    if value >= 0:
        return -0.05
    return -0.15


def _cash_conversion_adjustment(value: Any) -> float | None:
    value = _number(value)
    if value is None:
        return None
    if value >= 1.0:
        return 0.05
    if value >= 0.8:
        return 0.0
    if value >= 0.6:
        return -0.05
    return -0.10


def _normalized_growth_pair(
    forward_growth: float | None,
    eps_cagr_3y: float | None,
    growth_cap: float,
) -> tuple[float | None, float | None]:
    """Cap both growth inputs and limit either from outrunning the other."""
    if (
        forward_growth is None
        or eps_cagr_3y is None
        or forward_growth <= 0
        or eps_cagr_3y <= 0
    ):
        return None, None
    forward_capped = min(forward_growth, growth_cap)
    cagr_capped = min(eps_cagr_3y, growth_cap)
    forward_limit = max(cagr_capped * 1.5, cagr_capped + 0.05)
    cagr_limit = max(forward_capped * 1.5, forward_capped + 0.05)
    return min(forward_capped, forward_limit), min(cagr_capped, cagr_limit)


def _forward_weight(
    forward_growth: float | None,
    eps_cagr_3y: float | None,
) -> tuple[float, str, float]:
    """Return Forward EPS weight, explanation, and PE risk penalty."""
    if forward_growth is None or eps_cagr_3y is None:
        return 0.0, "Forward或3Y CAGR缺值，僅採TTM EPS", 5.0
    gap = abs(forward_growth - eps_cagr_3y)
    if gap <= 0.15 and forward_growth <= 0.50:
        return 0.70, "Forward與3Y CAGR差距不超過15個百分點", 0.0
    if gap <= 0.30 and forward_growth <= 0.75:
        return 0.50, "Forward與3Y CAGR有中度差距", 2.0
    return 0.30, "Forward與3Y CAGR高度分歧，降低Forward權重", 5.0


def calculate_fundamental_fair_value_v2(
    *,
    ttm_eps: Any,
    forward_eps: Any,
    eps_cagr_3y: Any,
    earnings_trend_score: Any,
    roic: Any,
    fcf_margin: Any,
    ocf_to_net_income: Any,
    fcf_positive_streak: Any,
    forward_eps_label: str = "Forward EPS",
    growth_cap: float = 0.30,
    base_pe: float = 15.0,
    growth_pe_slope: float = 0.50,
    pe_floor: float = 8.0,
    pe_base_cap: float = 35.0,
    pe_high_cap: float = 40.0,
) -> dict[str, Any]:
    """Calculate a conservative, auditable research fair-value range.

    Required inputs for a general-company PE value are positive TTM EPS,
    Forward EPS, Forward EPS growth, and 3Y EPS CAGR. Missing required inputs
    return N/A rather than a fixed-PE buy/sell signal.
    """
    ttm = _positive(ttm_eps)
    forward = _positive(forward_eps)
    cagr = _number(eps_cagr_3y)
    forward_growth = forward / ttm - 1.0 if ttm is not None and forward is not None else None
    normalized_forward, normalized_cagr = _normalized_growth_pair(
        forward_growth, cagr, growth_cap
    )
    trend_multiplier = earnings_trend_multiplier(earnings_trend_score)
    sustainable_growth = (
        0.60 * normalized_forward + 0.40 * normalized_cagr
        if normalized_forward is not None and normalized_cagr is not None
        else None
    )
    adjusted_sustainable_growth = (
        min(growth_cap, sustainable_growth * (trend_multiplier or 1.0))
        if sustainable_growth is not None
        else None
    )

    roic_adjustment = _roic_adjustment(roic)
    fcf_margin_adjustment = _fcf_margin_adjustment(fcf_margin)
    cash_conversion_adjustment = _cash_conversion_adjustment(ocf_to_net_income)
    streak = _number(fcf_positive_streak)
    fcf_consistency_adjustment = (
        0.05 if streak is not None and streak >= 4
        else 0.0 if streak is not None
        else None
    )
    quality_adjustments = (
        roic_adjustment,
        fcf_margin_adjustment,
        cash_conversion_adjustment,
        fcf_consistency_adjustment,
    )
    valuation_quality_peg = _clip(
        1.0 + sum(value for value in quality_adjustments if value is not None),
        0.80,
        1.40,
    )
    quality_pe_adjustment = 10.0 * (valuation_quality_peg - 1.0)

    forward_weight, forward_weight_note, forecast_risk_penalty = _forward_weight(
        forward_growth, cagr
    )
    if ttm is not None and forward is not None:
        model_eps = (1.0 - forward_weight) * ttm + forward_weight * forward
        model_eps_weight_basis = (
            f"TTM EPS {1.0-forward_weight:.0%} + {forward_eps_label} {forward_weight:.0%}；"
            f"{forward_weight_note}"
        )
    elif ttm is not None:
        model_eps = ttm
        model_eps_weight_basis = f"TTM EPS 100%（{forward_eps_label}缺值）"
    else:
        model_eps = None
        model_eps_weight_basis = "TTM EPS無有效正值"

    fair_pe_unclipped = (
        base_pe
        + adjusted_sustainable_growth * 100.0 * growth_pe_slope
        + quality_pe_adjustment
        - forecast_risk_penalty
        if adjusted_sustainable_growth is not None
        else None
    )
    fair_pe_base = (
        _clip(fair_pe_unclipped, pe_floor, pe_base_cap)
        if fair_pe_unclipped is not None
        else None
    )
    fair_pe_low = max(pe_floor, fair_pe_base - 4.0) if fair_pe_base is not None else None
    fair_pe_high = min(pe_high_cap, fair_pe_base + 4.0) if fair_pe_base is not None else None

    missing = []
    if ttm is None:
        missing.append("正值TTM EPS")
    if forward is None:
        missing.append(f"正值{forward_eps_label}")
    if forward_growth is None or forward_growth <= 0:
        missing.append("正向Forward EPS Growth")
    if cagr is None or cagr <= 0:
        missing.append("正向3Y EPS CAGR")
    if trend_multiplier is None:
        missing.append("盈餘趨勢分數")
    quality_labels = ("ROIC", "FCF Margin", "OCF／淨利", "FCF連續正值年數")
    missing_quality = [
        label for label, value in zip(quality_labels, quality_adjustments)
        if value is None
    ]

    if fair_pe_base is None or model_eps is None:
        status = "資料不足"
        note = "研究版V2不以固定PE補出買賣價；缺少：" + "、".join(missing)
    elif missing or missing_quality or forward_weight < 0.70:
        status = "部分資料"
        reasons = missing + missing_quality
        if forward_weight < 0.70:
            reasons.append(forward_weight_note)
        note = "研究版V2已估值但降低信心：" + "、".join(reasons)
    else:
        status = "完整"
        note = "研究版V2：基礎PE＋正常化成長溢價＋品質PE調整－預估分歧折價"

    def value(pe: float | None) -> float | None:
        return model_eps * pe if model_eps is not None and pe is not None else None

    return {
        "valuation_model_version": MODEL_VERSION,
        "forward_eps_growth": forward_growth,
        "normalized_forward_eps_growth": normalized_forward,
        "adjusted_3y_eps_cagr": normalized_cagr,
        "sustainable_eps_growth": sustainable_growth,
        "earnings_trend_multiplier": trend_multiplier,
        "adjusted_sustainable_eps_growth": adjusted_sustainable_growth,
        "valuation_quality_peg": valuation_quality_peg,
        "quality_pe_adjustment": quality_pe_adjustment,
        "forecast_risk_penalty": forecast_risk_penalty,
        "forward_eps_weight": forward_weight,
        "roic_adjustment": roic_adjustment,
        "fcf_margin_adjustment": fcf_margin_adjustment,
        "cash_conversion_adjustment": cash_conversion_adjustment,
        "fcf_consistency_adjustment": fcf_consistency_adjustment,
        "fundamental_fair_pe_unclipped": fair_pe_unclipped,
        "fundamental_fair_pe_low": fair_pe_low,
        "fundamental_fair_pe_base": fair_pe_base,
        "fundamental_fair_pe_high": fair_pe_high,
        "model_eps": model_eps,
        "model_eps_weight_basis": model_eps_weight_basis,
        "fundamental_value_low": value(fair_pe_low),
        "fundamental_value_base": value(fair_pe_base),
        "fundamental_value_high": value(fair_pe_high),
        "valuation_model_status": status,
        "valuation_model_note": note,
    }
