"""One-time, idempotent migration to the unified WHID report layout."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
REPORT_ROOT = PROJECT / "reports"
SYSTEM_ROOT = PROJECT / "system_data"
OLD_RUNTIME_ROOT = PROJECT / "Stock_price_prediction" / "reports"

OLD_STEP1 = re.compile(
    r"^(TW|US)_Report_(20\d{6})_(\d{6})(?:_\d+)?\.xlsx$", re.IGNORECASE
)
OLD_STEP2 = re.compile(
    r"^(TW|US)_Timing_(20\d{6})_(\d{6})(?:_\d+)?_(詳細|簡化)\.xlsx$",
    re.IGNORECASE,
)
NEW_REPORT = re.compile(
    r"^(TW|US)_20\d{6}_\d{6}_Step[12]_Report\.xlsx$", re.IGNORECASE
)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _move_file(source: Path, destination: Path, dry_run: bool, changes: list[dict]) -> None:
    if not _inside(source, PROJECT) or not _inside(destination, PROJECT):
        raise ValueError(f"拒絕搬移專案外路徑：{source} -> {destination}")
    old = str(source.resolve())
    new = str(destination.resolve())
    if source.resolve() == destination.resolve():
        return
    if destination.exists():
        if _digest(source) != _digest(destination):
            raise FileExistsError(f"目的檔已存在且內容不同：{destination}")
        action = "deduplicated"
        if not dry_run:
            source.unlink()
    else:
        action = "moved"
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
    changes.append({"source": old, "destination": new, "action": action})


def _version_from_path(path: Path) -> str | None:
    for value in ("詳細版", "簡化版"):
        if value in path.parts:
            return value
    return None


def migrate_excel(dry_run: bool, changes: list[dict]) -> None:
    if REPORT_ROOT.exists():
        for source in list(REPORT_ROOT.rglob("*.xlsx")):
            if source.name.startswith("~$") or NEW_REPORT.fullmatch(source.name):
                continue
            match = OLD_STEP1.fullmatch(source.name)
            version = _version_from_path(source)
            if match and version:
                market, date_value, time_value = match.groups()
                destination = REPORT_ROOT / version / market.upper() / (
                    f"{market.upper()}_{date_value}_{time_value}_Step1_Report.xlsx"
                )
                _move_file(source, destination, dry_run, changes)

    if OLD_RUNTIME_ROOT.exists():
        for source in list(OLD_RUNTIME_ROOT.rglob("*.xlsx")):
            if source.name.startswith("~$"):
                continue
            match = OLD_STEP2.fullmatch(source.name)
            version = _version_from_path(source)
            if match and version:
                market, date_value, time_value, _ = match.groups()
                destination = REPORT_ROOT / version / market.upper() / (
                    f"{market.upper()}_{date_value}_{time_value}_Step2_Report.xlsx"
                )
                _move_file(source, destination, dry_run, changes)


def _move_tree(source: Path, destination: Path, dry_run: bool,
               changes: list[dict], flatten: bool = False) -> None:
    if not source.exists():
        return
    for item in list(source.rglob("*")):
        if not item.is_file():
            continue
        relative = Path(item.name) if flatten else item.relative_to(source)
        _move_file(item, destination / relative, dry_run, changes)


def migrate_runtime(dry_run: bool, changes: list[dict]) -> None:
    for market in ("TW", "US"):
        _move_tree(
            OLD_RUNTIME_ROOT / f"stage2_{market}" / "研究資料",
            SYSTEM_ROOT / "research" / market,
            dry_run,
            changes,
            flatten=True,
        )
    _move_tree(OLD_RUNTIME_ROOT / "cache_stage2", SYSTEM_ROOT / "cache" / "TW",
               dry_run, changes, flatten=True)
    _move_tree(OLD_RUNTIME_ROOT / "cache_stage2_US", SYSTEM_ROOT / "cache" / "US",
               dry_run, changes, flatten=True)
    _move_tree(OLD_RUNTIME_ROOT / "validation", SYSTEM_ROOT / "validation",
               dry_run, changes)
    _move_tree(OLD_RUNTIME_ROOT / "short_term_ai",
               SYSTEM_ROOT / "legacy_reports" / "short_term_ai", dry_run, changes)
    _move_tree(PROJECT / "Stock_price_prediction" / "docx_render_check",
               SYSTEM_ROOT / "docx_render_check", dry_run, changes)

    for version in ("詳細版", "簡化版"):
        _move_tree(REPORT_ROOT / version / "history",
                   SYSTEM_ROOT / "history" / version, dry_run, changes)


def _replace_paths(value, mapping: dict[str, str]):
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_replace_paths(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _replace_paths(item, mapping) for key, item in value.items()}
    return value


def update_ledger(changes: list[dict], dry_run: bool) -> dict:
    db_path = SYSTEM_ROOT / "validation" / "prediction_audit.sqlite3"
    if dry_run or not db_path.exists():
        return {"database": str(db_path), "updated_rows": 0,
                "status": "dry_run" if dry_run else "not_found"}
    mapping = {row["source"]: row["destination"] for row in changes}
    connection = sqlite3.connect(db_path)
    updated = 0
    try:
        for table in ("forecasts", "indicators"):
            for old, new in mapping.items():
                cursor = connection.execute(
                    f"UPDATE {table} SET source_report=? WHERE source_report=?", (new, old)
                )
                updated += cursor.rowcount
        if connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='audit_runs'"
        ).fetchone():
            for run_utc, details in connection.execute(
                "SELECT run_utc, details FROM audit_runs WHERE details IS NOT NULL"
            ).fetchall():
                try:
                    parsed = json.loads(details)
                except (TypeError, json.JSONDecodeError):
                    continue
                replaced = _replace_paths(parsed, mapping)
                if replaced != parsed:
                    connection.execute(
                        "UPDATE audit_runs SET details=? WHERE run_utc=?",
                        (json.dumps(replaced, ensure_ascii=False), run_utc),
                    )
                    updated += 1
        connection.commit()
    finally:
        connection.close()
    return {"database": str(db_path), "updated_rows": updated, "status": "ok"}


def remove_empty_directories() -> int:
    removed = 0
    if OLD_RUNTIME_ROOT.exists() and not any(p.is_file() for p in OLD_RUNTIME_ROOT.rglob("*")):
        try:
            shutil.rmtree(OLD_RUNTIME_ROOT)
            removed += 1
        except OSError:
            pass
    for version in ("詳細版", "簡化版"):
        for market in ("TW", "US"):
            folder = REPORT_ROOT / version / market
            if not folder.exists():
                continue
            for child in list(folder.iterdir()):
                if (child.is_dir() and not child.is_symlink()
                        and not any(p.is_file() for p in child.rglob("*"))):
                    try:
                        shutil.rmtree(child)
                        removed += 1
                    except OSError:
                        pass
    return removed


def run(dry_run: bool = False, verbose: bool = False) -> dict:
    changes: list[dict] = []
    migrate_excel(dry_run, changes)
    migrate_runtime(dry_run, changes)
    ledger = update_ledger(changes, dry_run)
    removed = 0 if dry_run else remove_empty_directories()
    result = {
        "status": "dry_run" if dry_run else "ok",
        "project": str(PROJECT),
        "files": len(changes),
        "reports": sum(1 for row in changes if "Step" in Path(row["destination"]).name),
        "ledger": ledger,
        "empty_directories_removed": removed,
    }
    if verbose:
        result["changes"] = changes
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(args.dry_run, args.verbose)


if __name__ == "__main__":
    main()
