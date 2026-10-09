import pytest

from fundamental_valuation_v2 import calculate_fundamental_fair_value_v2


def nflx_inputs():
    return {
        "ttm_eps": 3.177,
        "forward_eps": 3.821,
        "eps_cagr_3y": 0.365,
        "earnings_trend_score": 47.5,
        "roic": 0.312,
        "fcf_margin": 0.231,
        "ocf_to_net_income": 0.877,
        "fcf_positive_streak": 4,
    }


def test_nflx_v2_is_bounded_and_auditable():
    result = calculate_fundamental_fair_value_v2(**nflx_inputs())
    assert result["normalized_forward_eps_growth"] == pytest.approx(3.821 / 3.177 - 1)
    assert result["adjusted_3y_eps_cagr"] == pytest.approx(0.30)
    assert result["forward_eps_weight"] == 0.50
    assert result["fundamental_fair_pe_base"] == pytest.approx(27.5812, abs=1e-3)
    assert result["fundamental_fair_pe_low"] < result["fundamental_fair_pe_base"]
    assert result["fundamental_fair_pe_high"] > result["fundamental_fair_pe_base"]
    assert result["valuation_model_status"] == "部分資料"


def test_extreme_forward_growth_is_normalized_and_downweighted():
    result = calculate_fundamental_fair_value_v2(
        **(nflx_inputs() | {"forward_eps": 12.0, "eps_cagr_3y": 0.20})
    )
    assert result["normalized_forward_eps_growth"] == pytest.approx(0.30)
    assert result["adjusted_sustainable_eps_growth"] <= 0.30
    assert result["forward_eps_weight"] == 0.30
    assert result["forecast_risk_penalty"] == 5.0
    assert result["fundamental_fair_pe_base"] <= 35.0
    assert result["fundamental_fair_pe_high"] > result["fundamental_fair_pe_base"]


def test_nonpositive_growth_returns_na_instead_of_fixed_pe_value():
    result = calculate_fundamental_fair_value_v2(
        **(nflx_inputs() | {"forward_eps": 2.0})
    )
    assert result["fundamental_fair_pe_base"] is None
    assert result["fundamental_value_base"] is None
    assert result["valuation_model_status"] == "資料不足"


def test_missing_forward_returns_na_and_ttm_is_only_diagnostic_base():
    result = calculate_fundamental_fair_value_v2(
        **(nflx_inputs() | {"forward_eps": None})
    )
    assert result["model_eps"] == pytest.approx(3.177)
    assert result["fundamental_value_base"] is None
    assert result["valuation_model_status"] == "資料不足"


def test_historical_and_analyst_values_are_not_model_inputs():
    result = calculate_fundamental_fair_value_v2(**nflx_inputs())
    assert "historical" not in " ".join(result).lower()
    assert "analyst" not in " ".join(result).lower()
