"""Historical valuation context shared by the TW and US notebooks.

Historical PE percentiles in this module are descriptive fields only.  The
fundamental score helper deliberately has no valuation-percentile input.
"""

from __future__ import annotations

import math
from typing import Mapping

import pandas as pd


HISTORICAL_PE_WINDOWS = (1, 3, 5)
MIN_VALID_OBSERVATIONS = 8


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
    result: dict[str, float | int | None] = {}
    for years in HISTORICAL_PE_WINDOWS:
        suffix = f"{years}y"
        result[f"historical_pe_median_{suffix}"] = None
        result[f"historical_pe_percentile_{suffix}"] = None
        result[f"historical_pe_sample_count_{suffix}"] = 0
    result["pe_median_trend_1y_vs_3y"] = None
    result["pe_median_trend_1y_vs_5y"] = None

    if ratios is None or ratios.empty or not {"date", "pe"}.issubset(ratios.columns):
        return result

    valid = ratios[["date", "pe"]].copy()
    valid["date"] = pd.to_datetime(valid["date"], errors="coerce", utc=True).dt.tz_convert(None)
    latest = valid["date"].dropna().max()
    if pd.isna(latest):
        return result
    valid["pe"] = valid["pe"].map(_finite_positive)
    valid = valid.dropna(subset=["date", "pe"]).sort_values("date")
    valid = valid.drop_duplicates("date", keep="last")
    if valid.empty:
        return result

    usable_current_pe = _finite_positive(current_pe)

    for years in HISTORICAL_PE_WINDOWS:
        suffix = f"{years}y"
        cutoff = latest - pd.DateOffset(years=years)
        window = valid.loc[valid["date"] > cutoff]
        sample_count = len(window)
        result[f"historical_pe_sample_count_{suffix}"] = sample_count

        # Monthly data may start at the month-end immediately after cutoff.
        full_coverage = valid["date"].min() <= cutoff + pd.offsets.MonthEnd(1)
        if not full_coverage or sample_count < MIN_VALID_OBSERVATIONS:
            continue

        values = window["pe"].astype(float)
        result[f"historical_pe_median_{suffix}"] = float(values.median())
        if usable_current_pe is not None:
            result[f"historical_pe_percentile_{suffix}"] = float(
                100.0 * (values <= usable_current_pe).sum() / sample_count
            )

    median_1y = result["historical_pe_median_1y"]
    for years in (3, 5):
        comparison = result[f"historical_pe_median_{years}y"]
        if median_1y is not None and comparison not in (None, 0):
            result[f"pe_median_trend_1y_vs_{years}y"] = float(median_1y / comparison - 1.0)

    return result


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
