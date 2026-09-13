"""Verify the unified WHID report layout without downloading market data."""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parent
MODULE_ROOT = PROJECT / "Stock_price_prediction"
sys.path.insert(0, str(MODULE_ROOT))

REPORT_PATTERN = re.compile(
    r"^(TW|US)_20\d{6}_\d{6}_(?:Step[12]|Combined)_Report\.xlsx$"
)
REVIEW_PATTERN = re.compile(r"^WHID_30日驗證_20\d{6}_\d{6}\.xlsx$")


def verify_notebooks() -> dict:
    files = [
        PROJECT / "Stock_Valuation_ReDesgin_TW.ipynb",
        PROJECT / "Stock_Valuation_ReDesgin_US.ipynb",
        PROJECT / "Stock_Valuation_SMART.ipynb",
        MODULE_ROOT / "TW_Timing_第二階段.ipynb",
        MODULE_ROOT / "US_Timing_第二階段.ipynb",
    ]
    result = {}
    for path in files:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        sources = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        for index, source in enumerate(sources):
            compile(source, f"<{path.name}:cell{index}>", "exec")
        analyst_order = None
        if "ReDesgin" in path.name:
            combined = "\n".join(sources)
            marker = "TW_SIMPLE_COLUMNS" if "_TW" in path.name else "US_SIMPLE_COLUMNS"
            simple_block = combined[combined.index(marker):combined.index("def build_simple_report", combined.index(marker))]
            labels = ["分析師目標價最低", "分析師目標價中位數", "分析師目標價平均", "分析師目標價最高"]
            positions = [simple_block.index(label) for label in labels]
            if positions != sorted(positions):
                raise AssertionError(f"{path.name} 分析師目標價欄位順序錯誤")
            analyst_order = labels
        result[path.name] = {
            "code_cells": len(sources),
            "step1_naming": ("Step1_Report.xlsx" in "\n".join(sources))
                if "ReDesgin" in path.name else None,
            "simple_analyst_order": analyst_order,
        }
    return result


def verify_historical_pe_migration() -> dict:
    files = [
        PROJECT / "Stock_Valuation_ReDesgin_TW.ipynb",
        PROJECT / "Stock_Valuation_ReDesgin_US.ipynb",
        PROJECT / "Stock_Valuation_SMART.ipynb",
    ]
    required = {
        "historical_pe_median_1y",
        "historical_pe_percentile_1y",
        "historical_pe_median_3y",
        "historical_pe_percentile_3y",
        "historical_pe_median_5y",
        "historical_pe_percentile_5y",
        "pe_median_trend_1y_vs_3y",
        "pe_median_trend_1y_vs_5y",
        "historical_pe_window_stats",
        "historical_pb_window_stats",
        "historical_reference_period",
        "historical_reference_percentile",
        "historical_median_price_1y",
        "historical_median_price_3y",
        "historical_median_price_5y",
        "valuation_eps_base",
        "ttm_eps_fair_price",
        "estimated_eps_fair_price",
        "model_earnings_basis",
        '"Price Recommendation"',
        '"Price Recommendation Basis"',
    }
    forbidden = {
        "historical_pe_percentile": re.compile(r"\bhistorical_pe_percentile\b"),
        "pe_percentile_5y": re.compile(r"\bpe_percentile_5y\b"),
        "nowcast_pe_percentile": re.compile(r"\bnowcast_pe_percentile\b"),
        "valuation_score": re.compile(r"\bvaluation_score\b"),
        "Valuation Percentile": re.compile(r'"Valuation Percentile"'),
        "Buy Price Recommendation": re.compile(r'"Buy Price Recommendation"'),
        "買價投資建議": re.compile(r'買價投資建議'),
        "historical_pe_stats": re.compile(r"\bhistorical_pe_stats\b"),
        "historical_pb_stats": re.compile(r"\bhistorical_pb_stats\b"),
        "historical_pe legacy average": re.compile(r"\bhistorical_pe\b"),
        "historical_pb legacy average": re.compile(r"\bhistorical_pb\b"),
        "_windowed_averages": re.compile(r"\b_windowed_averages\b"),
        "_historical_ratio_statistics": re.compile(r"\b_historical_ratio_statistics\b"),
        "_fair_multiple": re.compile(r"\b_fair_multiple\b"),
    }
    result = {}
    for path in files:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        missing = sorted(name for name in required if name not in source)
        legacy = sorted(name for name, pattern in forbidden.items() if pattern.search(source))
        if missing or legacy:
            raise AssertionError({"notebook": path.name, "missing": missing, "legacy": legacy})
        score_block = source[source.index("def total_score("):source.index("class BuffettChecklist")]
        if (re.search(r"\bvaluation\s*:", score_block)
                or 'weights["valuation"]' in score_block
                or "historical_pe" in score_block.lower()):
            raise AssertionError(f"{path.name} Fundamental Score still references valuation")
        result[path.name] = {"required_fields": len(required), "legacy_references": 0}
    return result


def _empty_result() -> dict:
    frame = pd.DataFrame({"代號": ["TEST"]})
    empty = pd.DataFrame()
    return {
        "detail": frame,
        "simple": frame,
        "data_status": empty,
        "backtest": empty,
        "trades": empty,
        "equity": empty,
        "ai_latest": empty,
        "ai_metrics": empty,
        "ai_comparison": empty,
        "ai_model_summary": empty,
        "ai_calibration": empty,
        "ai_status": empty,
        "ai_predictions": empty,
        "signals": empty,
        "timesfm_path": empty,
        "news": empty,
        "market_context": empty,
        "candidates": frame,
        "provenance": {"來源": "版型測試"},
    }


def verify_export_functions() -> dict:
    from timing_tw import export_reports as export_tw
    from timing_us import export_reports as export_us

    system_parent = PROJECT / "system_data"
    system_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=system_parent) as temporary:
        root = Path(temporary)
        cfg = {
            "output": {"root": str(root / "reports"),
                       "system_root": str(root / "system_data")},
            "rules": {}, "sentiment": {}, "ai": {}, "timesfm": {}, "backtest": {},
        }
        paths = {}
        consumed_sources = {}
        for market, exporter in (("TW", export_tw), ("US", export_us)):
            source = root / "reports" / "簡化版" / market / \
                f"{market}_20260912_200000_Step1_Report.xlsx"
            source.parent.mkdir(parents=True, exist_ok=True)
            candidates = pd.DataFrame({
                "代號": ["TEST"],
                "現價": [100.0],
                "分析師目標價最低": [90.0],
                "分析師目標價中位數": [105.0],
                "分析師目標價平均": [110.0],
                "分析師目標價最高": [130.0],
            })
            candidates.to_excel(source, sheet_name="Report", index=False)
            payload = _empty_result()
            payload["candidates"] = candidates
            price_column = "現價" if market == "TW" else "分析日收盤價USD"
            payload["simple"] = pd.DataFrame({
                "代號": ["TEST"], price_column: [101.0], "短期趨勢": ["偏多"]
            })
            payload["provenance"] = {"來源": str(source), "WHID評估日期": "2026-09-12"}
            paths[market] = exporter(payload, cfg)
            consumed_sources[market] = source
        result = {}
        for market, market_paths in paths.items():
            for version in ("詳細版", "簡化版"):
                path = Path(market_paths[version])
                if not path.exists() or not REPORT_PATTERN.fullmatch(path.name):
                    raise AssertionError(f"{market} {version} 檔名或輸出失敗：{path}")
                if path.parent != root / "reports" / version / market:
                    raise AssertionError(f"{market} {version} 路徑錯誤：{path.parent}")
            with pd.ExcelFile(market_paths["詳細版"]) as book:
                if "TimesFMPath" not in book.sheet_names:
                    raise AssertionError(f"{market} 詳細版缺少TimesFMPath工作表")
            simple_path = Path(market_paths["簡化版"])
            if "_Combined_Report.xlsx" not in simple_path.name:
                raise AssertionError(f"{market} 簡化版不是整合檔：{simple_path.name}")
            with pd.ExcelFile(simple_path) as book:
                if book.sheet_names != ["Report"]:
                    raise AssertionError(f"{market} 簡化版工作表錯誤：{book.sheet_names}")
                combined = pd.read_excel(book, sheet_name="Report")
                expected = ["分析師目標價最低", "分析師目標價中位數", "分析師目標價平均", "分析師目標價最高"]
                actual = [column for column in combined.columns if column in expected]
                if actual != expected:
                    raise AssertionError(f"{market} 分析師目標價欄位錯誤：{actual}")
                if list(combined.columns[:len(candidates.columns)]) != list(candidates.columns):
                    raise AssertionError(f"{market} Step1欄位未排在前方")
                if combined.columns[-1] != "短期趨勢" or combined.columns.duplicated().any():
                    raise AssertionError(f"{market} Step2欄位順序或重複欄位錯誤")
                if market == "US" and "分析日收盤價USD" in combined.columns:
                    raise AssertionError("US簡化版重複保留Step2收盤價")
            if consumed_sources[market].exists():
                raise AssertionError(f"{market} 本次Step1簡化暫存檔未移除")
            result[market] = market_paths
        return result


def verify_existing_layout() -> dict:
    report_root = PROJECT / "reports"
    files = list(report_root.rglob("*.xlsx")) if report_root.exists() else []
    review_root = report_root / "30日評估"
    ordinary = [path for path in files if review_root not in path.parents]
    reviews = [path for path in files if review_root in path.parents]
    invalid = [str(path) for path in ordinary if not REPORT_PATTERN.fullmatch(path.name)]
    invalid.extend(str(path) for path in reviews if not REVIEW_PATTERN.fullmatch(path.name))
    nested = [str(path.parent) for path in ordinary
              if path.parent.parent.parent != report_root]
    nested.extend(str(path.parent) for path in reviews
                  if path.parent.parent.parent != review_root
                  or not re.fullmatch(r"20\d{2}", path.parent.parent.name)
                  or not re.fullmatch(r"0[1-9]|1[0-2]", path.parent.name))
    if invalid or nested:
        raise AssertionError({"invalid_names": invalid, "nested_folders": nested})
    old_runtime = MODULE_ROOT / "reports"
    old_files = list(old_runtime.rglob("*")) if old_runtime.exists() else []
    old_files = [str(path) for path in old_files if path.is_file()]
    if old_files:
        raise AssertionError({"legacy_runtime_files": old_files})
    return {"xlsx_files": len(files), "review_files": len(reviews),
            "invalid_names": 0, "nested_folders": 0}


def verify_review_output() -> dict:
    from prediction_audit import review_output_path

    expected = PROJECT / "reports" / "30日評估" / "2026" / "09" / \
        "WHID_30日驗證_20260912_230405.xlsx"
    actual = review_output_path(datetime(2026, 9, 12, 23, 4, 5))
    if actual != expected:
        raise AssertionError({"expected": str(expected), "actual": str(actual)})
    return {"path": str(actual)}


def verify_database() -> dict:
    path = PROJECT / "system_data" / "validation" / "prediction_audit.sqlite3"
    if not path.exists():
        return {"status": "not_found"}
    connection = sqlite3.connect(path)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        forecasts = connection.execute("SELECT count(1) FROM forecasts").fetchone()[0]
        indicators = connection.execute("SELECT count(1) FROM indicators").fetchone()[0]
        old_paths = sum(
            connection.execute(
                f"SELECT count(1) FROM {table} "
                "WHERE instr(source_report, 'Stock_price_prediction') > 0"
            ).fetchone()[0]
            for table in ("forecasts", "indicators")
        )
    finally:
        connection.close()
    if integrity != "ok" or old_paths:
        raise AssertionError({"integrity": integrity, "old_source_paths": old_paths})
    return {"status": "ok", "forecasts": forecasts, "indicators": indicators,
            "old_source_paths": old_paths}


def main() -> None:
    result = {
        "notebooks": verify_notebooks(),
        "historical_pe": verify_historical_pe_migration(),
        "export_functions": verify_export_functions(),
        "review_output": verify_review_output(),
        "existing_layout": verify_existing_layout(),
        "database": verify_database(),
        "status": "ok",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
