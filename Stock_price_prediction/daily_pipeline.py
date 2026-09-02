"""Run WHID stage 1 then stage 2 for TW/US, archive forecasts, and settle mature rows."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent
LOG_ROOT = BASE / 'reports' / 'validation' / '每日執行紀錄'

NOTEBOOKS = {
    'TW': [PROJECT / 'Stock_Valuation_ReDesgin_TW.ipynb', BASE / 'TW_Timing_第二階段.ipynb'],
    'US': [PROJECT / 'Stock_Valuation_ReDesgin_US.ipynb', BASE / 'US_Timing_第二階段.ipynb'],
}


def execute_notebook(path, output_path, timeout=10800):
    import nbformat
    from nbclient import NotebookClient
    notebook = nbformat.read(path, as_version=4)
    kernel = notebook.get('metadata', {}).get('kernelspec', {}).get('name', 'python3')
    client = NotebookClient(
        notebook, timeout=timeout, kernel_name=kernel, allow_errors=False,
        resources={'metadata': {'path': str(path.parent)}})
    client.execute()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output_path)
    return output_path


def run(markets=('TW', 'US'), dry_run=False):
    from prediction_audit import archive_latest, settle_pending, status
    started = datetime.now(timezone.utc)
    run_id = started.strftime('%Y%m%d_%H%M%S')
    log_dir = LOG_ROOT / datetime.now().strftime('%Y/%m/%d') / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    summary = {'run_id': run_id, 'started_utc': started.isoformat(), 'python': sys.executable,
               'markets': {}, 'settlement': None, 'ledger_status': None}
    failures = []
    for market in markets:
        market_result = {'notebooks': [], 'archive': None, 'status': 'running'}
        summary['markets'][market] = market_result
        market_ok = True
        for order, notebook in enumerate(NOTEBOOKS[market], 1):
            item = {'path': str(notebook), 'stage': order, 'started_utc': datetime.now(timezone.utc).isoformat()}
            market_result['notebooks'].append(item)
            if dry_run:
                item['status'] = 'dry_run'
                continue
            try:
                if not notebook.exists():
                    raise FileNotFoundError(notebook)
                output = log_dir / f'{market}_stage{order}_executed.ipynb'
                execute_notebook(notebook, output)
                item.update({'status': 'ok', 'output': str(output),
                             'finished_utc': datetime.now(timezone.utc).isoformat()})
            except Exception as exc:
                item.update({'status': 'failed', 'error': f'{type(exc).__name__}: {str(exc)[:500]}',
                             'traceback': traceback.format_exc(limit=8)})
                failures.append(f'{market} stage {order}: {type(exc).__name__}: {str(exc)[:180]}')
                market_ok = False
                break
        if dry_run:
            market_result['status'] = 'dry_run'
        elif market_ok:
            try:
                second_stage_start = market_result['notebooks'][-1]['started_utc']
                market_result['archive'] = archive_latest(market, not_before=second_stage_start)
                market_result['status'] = 'ok'
            except Exception as exc:
                market_result['status'] = 'archive_failed'
                market_result['archive_error'] = f'{type(exc).__name__}: {str(exc)[:500]}'
                failures.append(f'{market} archive: {type(exc).__name__}: {str(exc)[:180]}')
        else:
            market_result['status'] = 'failed'
    if not dry_run:
        try:
            summary['settlement'] = settle_pending()
        except Exception as exc:
            failures.append(f'settlement: {type(exc).__name__}: {str(exc)[:180]}')
            summary['settlement'] = {'status': 'failed', 'error': f'{type(exc).__name__}: {str(exc)[:500]}'}
    summary['ledger_status'] = status()
    summary['finished_utc'] = datetime.now(timezone.utc).isoformat()
    summary['status'] = 'failed' if failures else ('dry_run' if dry_run else 'ok')
    summary['failures'] = failures
    summary_path = log_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    summary['summary_path'] = str(summary_path)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', choices=['TW', 'US', 'ALL'], default='ALL')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    markets = ('TW', 'US') if args.market == 'ALL' else (args.market,)
    result = run(markets, args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
