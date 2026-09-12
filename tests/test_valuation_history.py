import math
import json
import unittest
from pathlib import Path

import pandas as pd

from valuation_history import (
    assess_pe_valuation_applicability,
    calculate_fundamental_score,
    calculate_historical_pe_context,
    calculate_historical_ratio_windows,
    calculate_price_recommendation,
    select_historical_valuation_reference,
)


def monthly_ratios(periods: int, values=None) -> pd.DataFrame:
    dates = pd.date_range("2021-01-31", periods=periods, freq="ME")
    pe_values = list(values) if values is not None else list(range(1, periods + 1))
    return pd.DataFrame({"date": dates, "pe": pe_values})


class HistoricalPEContextTests(unittest.TestCase):
    def test_1y_3y_5y_are_independent_and_not_averaged(self):
        result = calculate_historical_pe_context(monthly_ratios(60), current_pe=30)

        self.assertEqual(result["historical_pe_median_1y"], 54.5)
        self.assertEqual(result["historical_pe_median_3y"], 42.5)
        self.assertEqual(result["historical_pe_median_5y"], 30.5)
        self.assertAlmostEqual(result["historical_pe_percentile_1y"], 0.0)
        self.assertAlmostEqual(result["historical_pe_percentile_3y"], 6 / 36 * 100)
        self.assertAlmostEqual(result["historical_pe_percentile_5y"], 30 / 60 * 100)
        self.assertAlmostEqual(result["pe_median_trend_1y_vs_3y"], 54.5 / 42.5 - 1)
        self.assertAlmostEqual(result["pe_median_trend_1y_vs_5y"], 54.5 / 30.5 - 1)
        self.assertNotIn("historical_pe_percentile", result)

    def test_insufficient_history_does_not_substitute_shorter_window(self):
        result = calculate_historical_pe_context(monthly_ratios(29), current_pe=25)

        self.assertIsNotNone(result["historical_pe_median_1y"])
        self.assertIsNotNone(result["historical_pe_percentile_1y"])
        self.assertIsNone(result["historical_pe_median_3y"])
        self.assertIsNone(result["historical_pe_percentile_3y"])
        self.assertIsNone(result["historical_pe_median_5y"])
        self.assertIsNone(result["historical_pe_percentile_5y"])

    def test_invalid_pe_does_not_pollute_distribution(self):
        values = list(range(1, 61))
        values[-1] = -10
        values[-2] = 0
        values[-3] = math.inf
        values[-4] = math.nan
        result = calculate_historical_pe_context(monthly_ratios(60, values), current_pe=30)

        self.assertEqual(result["historical_pe_sample_count_1y"], 8)
        self.assertEqual(result["historical_pe_median_1y"], 52.5)
        self.assertEqual(result["historical_pe_percentile_1y"], 0.0)
        self.assertEqual(result["historical_pe_sample_count_3y"], 32)
        self.assertEqual(result["historical_pe_median_3y"], 40.5)
        self.assertAlmostEqual(result["historical_pe_percentile_3y"], 6 / 32 * 100)
        self.assertEqual(result["historical_pe_sample_count_5y"], 56)
        self.assertEqual(result["historical_pe_median_5y"], 28.5)
        self.assertAlmostEqual(result["historical_pe_percentile_5y"], 30 / 56 * 100)

    def test_invalid_current_pe_keeps_median_but_not_percentile(self):
        result = calculate_historical_pe_context(monthly_ratios(60), current_pe=-2)
        self.assertEqual(result["historical_pe_median_1y"], 54.5)
        self.assertIsNone(result["historical_pe_percentile_1y"])

    def test_reference_defaults_to_3y_when_valuation_centre_is_stable(self):
        ratios = monthly_ratios(60, [20.0] * 60)
        windows = calculate_historical_ratio_windows(ratios, 15.0, "pe")
        selected = select_historical_valuation_reference(windows)

        self.assertEqual(selected["reference_period"], "3Y")
        self.assertEqual(selected["environment"], "估值中樞穩定")
        self.assertEqual(selected["statistics"]["p50"], 20.0)

    def test_reference_uses_1y_only_for_confirmed_continuing_rerating(self):
        ratios = monthly_ratios(60, list(range(60, 0, -1)))
        windows = calculate_historical_ratio_windows(ratios, 5.0, "pe")
        selected = select_historical_valuation_reference(windows)

        self.assertEqual(windows["1y"]["p50"], 6.5)
        self.assertEqual(windows["3y"]["p50"], 18.5)
        self.assertEqual(windows["5y"]["p50"], 30.5)
        self.assertEqual(selected["reference_period"], "1Y")
        self.assertEqual(selected["environment"], "估值中樞持續降評")

    def test_reference_does_not_use_1y_as_substitute_when_3y_is_missing(self):
        windows = calculate_historical_ratio_windows(monthly_ratios(29), 20.0, "pe")
        selected = select_historical_valuation_reference(windows)

        self.assertIsNotNone(windows["1y"])
        self.assertIsNone(windows["3y"])
        self.assertIsNone(selected["reference_period"])

    def test_low_historical_percentile_alone_cannot_produce_buy(self):
        advice = calculate_price_recommendation(
            current_price=100.0,
            fair_price=105.0,
            historical_percentile=5.0,
            applicability="正常",
            confidence="高",
            reference_period="3Y",
        )
        self.assertEqual(advice["recommendation"], "持有")

    def test_price_recommendation_requires_percentile_and_upside_confirmation(self):
        strong_buy = calculate_price_recommendation(
            current_price=100.0,
            fair_price=140.0,
            historical_percentile=10.0,
            applicability="正常",
            confidence="高",
            reference_period="3Y",
        )
        strong_sell = calculate_price_recommendation(
            current_price=100.0,
            fair_price=70.0,
            historical_percentile=97.0,
            applicability="正常",
            confidence="高",
            reference_period="3Y",
        )
        self.assertEqual(strong_buy["recommendation"], "強買")
        self.assertEqual(strong_sell["recommendation"], "強賣")

    def test_volatile_eps_downgrades_a_bullish_price_signal(self):
        applicability = assess_pe_valuation_applicability([10, 2, 12, 1, 8], 10)
        advice = calculate_price_recommendation(
            current_price=100.0,
            fair_price=140.0,
            historical_percentile=10.0,
            applicability=applicability,
            confidence="高",
            reference_period="3Y",
        )
        self.assertTrue(applicability.startswith("留意"))
        self.assertEqual(advice["recommendation"], "持有（需檢查盈餘循環）")
        self.assertEqual(advice["confidence"], "低")

    def test_historical_percentile_cannot_change_fundamental_score(self):
        weights = {"valuation": 25, "quality": 35, "growth": 30, "buffett": 10}
        low_percentile = calculate_historical_pe_context(monthly_ratios(60), current_pe=2)
        high_percentile = calculate_historical_pe_context(monthly_ratios(60), current_pe=59)
        score_before = calculate_fundamental_score(80, 60, 50, weights)
        score_after = calculate_fundamental_score(80, 60, 50, weights)

        self.assertNotEqual(
            low_percentile["historical_pe_percentile_5y"],
            high_percentile["historical_pe_percentile_5y"],
        )
        self.assertEqual(score_before, score_after)
        self.assertAlmostEqual(score_before, (35 * 80 + 30 * 60 + 10 * 50) / 75)

    def test_tw_and_us_notebook_engines_use_period_fields_and_ignore_valuation_weight(self):
        project = Path(__file__).resolve().parents[1]
        for notebook_name in (
            "Stock_Valuation_ReDesgin_TW.ipynb",
            "Stock_Valuation_ReDesgin_US.ipynb",
        ):
            notebook = json.loads((project / notebook_name).read_text(encoding="utf-8"))
            namespace = {}
            for cell_index in (1, 2, 3, 5, 6):
                source = "".join(notebook["cells"][cell_index].get("source", []))
                exec(compile(source, f"<{notebook_name}:cell{cell_index}>", "exec"), namespace)

            config = {
                "valuation": {
                    "earnings_availability_lag_days": 0,
                    "historical_windows_years": [1, 3, 5],
                    "historical_window_months": [],
                },
                "scoring": {
                    # This deliberately extreme legacy value must have no effect.
                    "weights": {"valuation": 99, "quality": 35, "growth": 30, "buffett": 10}
                },
            }
            snapshot = namespace["FinancialSnapshot"](
                ticker="TEST",
                source="test",
                as_of=pd.Timestamp("2025-12-31").date(),
                current_price=30.0,
                ttm_eps=1.0,
                eps_ttm_nowcast=1.5,
                forward_eps=2.0,
                book_value_per_share=10.0,
                financial_history={"eps": [1.0, 1.0, 1.0, 1.0, 1.0]},
            )
            prices = pd.DataFrame(
                {"Close": range(1, 61)},
                index=pd.date_range("2021-01-31", periods=60, freq="ME"),
            )
            fundamentals = pd.DataFrame(
                {"report_date": [pd.Timestamp("2020-12-31")], "ttm_eps": [1.0], "bvps": [10.0]}
            )

            valuation = namespace["ValuationEngine"](config).calculate(
                snapshot, prices, fundamentals
            )
            score = namespace["ScoringEngine"](config).total_score(80, 60, 50)
            target = namespace["PriceTargetEngine"]().calculate(
                snapshot,
                {
                    "valuation": {
                        "primary_metric": "pe",
                        "pe_cheap": 10.0,
                        "pe_expensive": 60.0,
                    }
                },
                valuation,
            )

            self.assertEqual(valuation.historical_pe_median_1y, 54.5)
            self.assertEqual(valuation.historical_pe_median_3y, 42.5)
            self.assertEqual(valuation.historical_pe_median_5y, 30.5)
            self.assertAlmostEqual(score, 68.0)
            self.assertEqual(target.historical_reference_period, "1Y")
            self.assertEqual(target.historical_median_price_1y, 54.5)
            self.assertEqual(target.historical_median_price_3y, 42.5)
            self.assertEqual(target.historical_median_price_5y, 30.5)
            expected_estimated_eps = 1.5 if notebook_name.endswith("TW.ipynb") else 2.0
            expected_model_eps = (1.0 + expected_estimated_eps) / 2
            self.assertEqual(target.model_fair_price, 54.5 * expected_model_eps)
            self.assertEqual(target.valuation_eps_base, expected_model_eps)
            self.assertEqual(target.ttm_eps_fair_price, 54.5)
            self.assertEqual(
                target.estimated_eps_fair_price,
                54.5 * expected_estimated_eps,
            )
            self.assertIn("50%", target.model_earnings_basis)
            self.assertEqual(target.price_recommendation, "強買")

            estimate_attribute = (
                "eps_ttm_nowcast" if notebook_name.endswith("TW.ipynb") else "forward_eps"
            )
            original_estimate = getattr(snapshot, estimate_attribute)
            setattr(snapshot, estimate_attribute, None)
            missing_estimate_target = namespace["PriceTargetEngine"]().calculate(
                snapshot,
                {
                    "valuation": {
                        "primary_metric": "pe",
                        "pe_cheap": 10.0,
                        "pe_expensive": 60.0,
                    }
                },
                valuation,
            )
            self.assertEqual(missing_estimate_target.model_fair_price, 54.5)
            self.assertIn("TTM EPS 100%", missing_estimate_target.model_earnings_basis)
            self.assertEqual(missing_estimate_target.price_recommendation_confidence, "低")
            setattr(snapshot, estimate_attribute, original_estimate)

            short_prices = pd.DataFrame(
                {"Close": range(1, 30)},
                index=pd.date_range("2023-08-31", periods=29, freq="ME"),
            )
            short_valuation = namespace["ValuationEngine"](config).calculate(
                snapshot, short_prices, fundamentals
            )
            short_target = namespace["PriceTargetEngine"]().calculate(
                snapshot,
                {
                    "valuation": {
                        "primary_metric": "pe",
                        "pe_cheap": 10.0,
                        "pe_expensive": 30.0,
                    }
                },
                short_valuation,
            )
            self.assertIsNone(short_target.historical_reference_period)
            self.assertIsNone(short_target.historical_fair_price)
            self.assertEqual(short_target.model_fair_price, 20.0 * expected_model_eps)
            self.assertEqual(short_target.price_recommendation, "資料不足")


if __name__ == "__main__":
    unittest.main()
