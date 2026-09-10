import math
import json
import unittest
from pathlib import Path

import pandas as pd

from valuation_history import calculate_fundamental_score, calculate_historical_pe_context


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
                book_value_per_share=10.0,
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

            self.assertEqual(valuation.historical_pe_median_1y, 54.5)
            self.assertEqual(valuation.historical_pe_median_3y, 42.5)
            self.assertEqual(valuation.historical_pe_median_5y, 30.5)
            self.assertAlmostEqual(score, 68.0)


if __name__ == "__main__":
    unittest.main()
