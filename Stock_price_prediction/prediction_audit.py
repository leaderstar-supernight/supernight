"""Immutable daily forecast ledger, delayed outcome settlement, and 30-day review."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
VALIDATION_ROOT = BASE / 'reports' / 'validation'
DB_PATH = VALIDATION_ROOT / 'prediction_audit.sqlite3'
CLASS_ORDER = ['down_first', 'neutral', 'up_first']


def _json_value(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def connect(db_path=DB_PATH):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute('PRAGMA journal_mode=WAL')
    connection.executescript('''
    CREATE TABLE IF NOT EXISTS forecasts (
        market TEXT NOT NULL,
        ticker TEXT NOT NULL,
        signal_date TEXT NOT NULL,
        model TEXT NOT NULL,
        recorded_utc TEXT NOT NULL,
        source_report TEXT NOT NULL,
        horizon INTEGER NOT NULL,
        p_down REAL NOT NULL,
        p_neutral REAL NOT NULL,
        p_up REAL NOT NULL,
        signal_close REAL NOT NULL,
        signal_atr REAL NOT NULL,
        upper_barrier REAL NOT NULL,
        lower_barrier REAL NOT NULL,
        actual_class TEXT,
        maturity_date TEXT,
        settled_utc TEXT,
        settlement_status TEXT NOT NULL DEFAULT 'pending',
        PRIMARY KEY (market, ticker, signal_date, model)
    );
    CREATE TABLE IF NOT EXISTS indicators (
        market TEXT NOT NULL,
        ticker TEXT NOT NULL,
        signal_date TEXT NOT NULL,
        recorded_utc TEXT NOT NULL,
        source_report TEXT NOT NULL,
        snapshot_json TEXT NOT NULL,
        horizon INTEGER NOT NULL,
        signal_close REAL,
        signal_atr REAL,
        upper_barrier REAL,
        lower_barrier REAL,
        actual_class TEXT,
        maturity_date TEXT,
        return_5 REAL,
        return_10 REAL,
        return_20 REAL,
        return_60 REAL,
        return_126 REAL,
        return_252 REAL,
        max_up_10 REAL,
        max_down_10 REAL,
        settled_utc TEXT,
        settlement_status TEXT NOT NULL DEFAULT 'pending',
        PRIMARY KEY (market, ticker, signal_date)
    );
    CREATE TABLE IF NOT EXISTS audit_runs (
        run_utc TEXT PRIMARY KEY,
        action TEXT NOT NULL,
        status TEXT NOT NULL,
        details TEXT
    );
    ''')
    existing = {row[1] for row in connection.execute('PRAGMA table_info(indicators)')}
    for column in ['return_60', 'return_126', 'return_252']:
        if column not in existing:
            connection.execute(f'ALTER TABLE indicators ADD COLUMN {column} REAL')
    connection.commit()
    return connection


def latest_detailed_report(market, not_before=None):
    folder = BASE / 'reports' / f'stage2_{market}' / '詳細版'
    files = [p for p in folder.rglob('*.xlsx') if not p.name.startswith('~$')] if folder.exists() else []
    if not_before is not None:
        threshold = pd.Timestamp(not_before).timestamp()
        files = [p for p in files if p.stat().st_mtime >= threshold]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _first(row, names):
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return None


def archive_report(market, report_path, db_path=DB_PATH):
    """Archive the first forecast for each market/ticker/signal date; never overwrite it."""
    report_path = Path(report_path).resolve()
    with pd.ExcelFile(report_path) as book:
        report = pd.read_excel(book, sheet_name='Report')
        latest = pd.read_excel(book, sheet_name='AILatest')
    required = {'代號', 'prediction_date', 'model', 'class', 'probability', 'signal_close',
                'signal_atr', 'upper_barrier', 'lower_barrier', 'horizon'}
    missing = required - set(latest.columns)
    warnings = []
    if missing:
        warnings.append(f'AILatest無可封存的新版三分類結果，缺少：{sorted(missing)}')
        latest = pd.DataFrame(columns=sorted(required))
    recorded = datetime.now(timezone.utc).isoformat()
    inserted_forecasts = 0
    inserted_indicators = 0
    connection = connect(db_path)
    try:
        for (ticker, signal_date, model), group in latest.groupby(['代號', 'prediction_date', 'model'], dropna=False):
            probability = group.set_index('class').probability.reindex(CLASS_ORDER)
            if probability.isna().any() or abs(float(probability.sum()) - 1.0) > 1e-5:
                continue
            first = group.iloc[0]
            cursor = connection.execute('''
                INSERT OR IGNORE INTO forecasts
                (market,ticker,signal_date,model,recorded_utc,source_report,horizon,
                 p_down,p_neutral,p_up,signal_close,signal_atr,upper_barrier,lower_barrier)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
                market, str(ticker), str(pd.Timestamp(signal_date).date()), str(model), recorded,
                str(report_path), int(first.horizon), float(probability.down_first),
                float(probability.neutral), float(probability.up_first), float(first.signal_close),
                float(first.signal_atr), float(first.upper_barrier), float(first.lower_barrier)))
            inserted_forecasts += cursor.rowcount
        report_date_col = '訊號日期' if market == 'TW' else '訊號日期NY'
        close_col = '現價' if market == 'TW' else '分析日收盤價USD'
        ensemble_source = latest.loc[latest.model == 'ensemble'].copy()
        ensemble_source['prediction_date'] = pd.to_datetime(ensemble_source.prediction_date).dt.strftime('%Y-%m-%d')
        ensemble = ensemble_source.groupby(['代號', 'prediction_date']).first()
        for _, row in report.iterrows():
            ticker = str(row.get('代號', '')).strip()
            if not ticker or pd.isna(row.get(report_date_col)):
                continue
            signal_date = str(pd.Timestamp(row[report_date_col]).date())
            key = (ticker, signal_date)
            ai_row = ensemble.loc[key] if key in ensemble.index else None
            close = _number(ai_row.signal_close) if ai_row is not None else _number(row.get(close_col))
            atr = _number(ai_row.signal_atr) if ai_row is not None else None
            if atr is None and close is not None:
                atr_pct = _number(row.get('ATR比例'))
                atr = close * atr_pct if atr_pct is not None else None
            upper = _number(ai_row.upper_barrier) if ai_row is not None else (close + 1.5 * atr if close and atr else None)
            lower = _number(ai_row.lower_barrier) if ai_row is not None else (close - atr if close and atr else None)
            snapshot = {str(k): _json_value(v) for k, v in row.to_dict().items()}
            cursor = connection.execute('''
                INSERT OR IGNORE INTO indicators
                (market,ticker,signal_date,recorded_utc,source_report,snapshot_json,horizon,
                 signal_close,signal_atr,upper_barrier,lower_barrier)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)''', (
                market, ticker, signal_date, recorded, str(report_path),
                json.dumps(snapshot, ensure_ascii=False), 10, close, atr, upper, lower))
            inserted_indicators += cursor.rowcount
        connection.execute('INSERT INTO audit_runs VALUES (?,?,?,?)', (
            recorded, f'archive_{market}', 'ok', json.dumps({
                'report': str(report_path), 'forecasts': inserted_forecasts,
                'indicators': inserted_indicators, 'warnings': warnings}, ensure_ascii=False)))
        connection.commit()
    finally:
        connection.close()
    return {'market': market, 'report': str(report_path), 'forecasts': inserted_forecasts,
            'indicators': inserted_indicators, 'warnings': warnings}


def archive_latest(market, not_before=None, db_path=DB_PATH):
    path = latest_detailed_report(market, not_before)
    if path is None:
        raise FileNotFoundError(f'找不到{market}本次新產生的第二階段詳細報表')
    return archive_report(market, path, db_path)


def _provider_prices(market, ticker):
    if market == 'TW':
        from timing_data import Provider, load_config
        cfg = load_config(BASE / 'config' / 'timing_TW.yaml')
        raw, adjusted, _, _ = Provider(cfg).bundle(ticker)
    else:
        from timing_us_data import Provider, load_config
        cfg = load_config(BASE / 'config' / 'timing_US.yaml')
        raw, adjusted, _, _ = Provider(cfg).bundle(ticker)
    if raw is None or adjusted is None:
        raise RuntimeError('原始價或還原價不可用')
    return raw.sort_index(), adjusted.sort_index()


def _outcome(raw, adjusted, signal_date, horizon, upper, lower):
    signal_date = pd.Timestamp(signal_date)
    future_raw = raw.loc[raw.index > signal_date].head(max(252, horizon))
    if len(future_raw) < horizon:
        return None
    signal_adjusted = adjusted.loc[adjusted.index <= signal_date]
    future_adjusted = adjusted.loc[adjusted.index > signal_date].head(252)
    if signal_adjusted.empty or len(future_adjusted) < horizon:
        return None
    actual = 'neutral'
    for _, bar in future_raw.head(horizon).iterrows():
        if np.isfinite(bar.Low) and bar.Low <= lower:
            actual = 'down_first'
            break
        if np.isfinite(bar.High) and bar.High >= upper:
            actual = 'up_first'
            break
    base = float(signal_adjusted.Close.iloc[-1])
    def forward_return(days):
        return (float(future_adjusted.Close.iloc[days - 1]) / base - 1) if len(future_adjusted) >= days else None
    signal_raw = raw.loc[raw.index <= signal_date]
    raw_base = float(signal_raw.Close.iloc[-1]) if not signal_raw.empty else np.nan
    ten = future_raw.head(horizon)
    return {
        'actual_class': actual, 'maturity_date': str(ten.index[-1].date()),
        'return_5': forward_return(5), 'return_10': forward_return(10),
        'return_20': forward_return(20),
        'return_60': forward_return(60), 'return_126': forward_return(126),
        'return_252': forward_return(252),
        'max_up_10': float(ten.High.max() / raw_base - 1) if np.isfinite(raw_base) else None,
        'max_down_10': float(ten.Low.min() / raw_base - 1) if np.isfinite(raw_base) else None,
    }


def settle_pending(db_path=DB_PATH):
    connection = connect(db_path)
    forecasts = pd.read_sql_query(
        "SELECT * FROM forecasts WHERE settlement_status='pending'", connection)
    indicators = pd.read_sql_query(
        "SELECT * FROM indicators WHERE settlement_status='pending' OR return_252 IS NULL", connection)
    keys = sorted(set(zip(forecasts.market, forecasts.ticker)) |
                  set(zip(indicators.market, indicators.ticker)))
    settled_forecasts = settled_indicators = enriched_indicators = 0
    errors = []
    now = datetime.now(timezone.utc).isoformat()
    for market, ticker in keys:
        try:
            raw, adjusted = _provider_prices(market, ticker)
        except Exception as exc:
            errors.append(f'{market}/{ticker}: {type(exc).__name__}: {str(exc)[:140]}')
            continue
        subset = forecasts.loc[(forecasts.market == market) & (forecasts.ticker == ticker)]
        for _, row in subset.iterrows():
            result = _outcome(raw, adjusted, row.signal_date, int(row.horizon),
                              float(row.upper_barrier), float(row.lower_barrier))
            if result is None:
                continue
            connection.execute('''UPDATE forecasts SET actual_class=?,maturity_date=?,settled_utc=?,
                settlement_status='settled' WHERE market=? AND ticker=? AND signal_date=? AND model=?''',
                (result['actual_class'], result['maturity_date'], now, market, ticker,
                 row.signal_date, row.model))
            settled_forecasts += 1
        subset = indicators.loc[(indicators.market == market) & (indicators.ticker == ticker)]
        for _, row in subset.iterrows():
            if not all(np.isfinite(_number(v) or np.nan) for v in [row.upper_barrier, row.lower_barrier]):
                connection.execute('''UPDATE indicators SET settlement_status='missing_barrier'
                    WHERE market=? AND ticker=? AND signal_date=?''', (market, ticker, row.signal_date))
                continue
            result = _outcome(raw, adjusted, row.signal_date, int(row.horizon),
                              float(row.upper_barrier), float(row.lower_barrier))
            if result is None:
                continue
            connection.execute('''UPDATE indicators SET actual_class=?,maturity_date=?,return_5=?,return_10=?,
                return_20=?,return_60=?,return_126=?,return_252=?,max_up_10=?,max_down_10=?,
                settled_utc=?,settlement_status='settled'
                WHERE market=? AND ticker=? AND signal_date=?''',
                (result['actual_class'], result['maturity_date'], result['return_5'], result['return_10'],
                 result['return_20'], result['return_60'], result['return_126'], result['return_252'],
                 result['max_up_10'], result['max_down_10'], now,
                 market, ticker, row.signal_date))
            if row.settlement_status == 'pending':
                settled_indicators += 1
            elif any(pd.isna(row.get(column)) and result[column] is not None
                     for column in ['return_20', 'return_60', 'return_126', 'return_252']):
                enriched_indicators += 1
    connection.execute('INSERT INTO audit_runs VALUES (?,?,?,?)', (
        now, 'settle', 'ok' if not errors else 'partial', json.dumps({
            'forecasts': settled_forecasts, 'indicators': settled_indicators,
            'long_horizon_updates': enriched_indicators,
            'errors': errors}, ensure_ascii=False)))
    connection.commit()
    connection.close()
    return {'forecasts': settled_forecasts, 'indicators': settled_indicators,
            'long_horizon_updates': enriched_indicators, 'errors': errors}


def _ai_tables(frame):
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
    rows, calibration = [], []
    label_map = {name: i for i, name in enumerate(CLASS_ORDER)}
    for (market, model), group in frame.groupby(['market', 'model']):
        y = group.actual_class.map(label_map).to_numpy(dtype=int)
        probability = group[['p_down', 'p_neutral', 'p_up']].to_numpy(dtype=float)
        probability = np.clip(probability, 1e-9, 1)
        probability /= probability.sum(axis=1, keepdims=True)
        pred = probability.argmax(axis=1)
        one_hot = np.eye(3)[y]
        base_probability = np.bincount(y, minlength=3) / len(y)
        base = np.tile(base_probability, (len(y), 1))
        brier = float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))
        base_brier = float(np.mean(np.sum((base - one_hot) ** 2, axis=1)))
        marginal_error = float(np.mean(np.abs(probability.mean(axis=0) - one_hot.mean(axis=0))))
        macro_f1 = float(f1_score(y, pred, labels=[0, 1, 2], average='macro', zero_division=0))
        verdict = '資料不足'
        if len(y) >= 30:
            skill = 1 - brier / base_brier if base_brier else np.nan
            if skill > 0 and marginal_error <= .10:
                verdict = '初步有參考性'
            elif skill > 0:
                verdict = '有區辨力；校準需修正'
            else:
                verdict = '未優於基準；需修正'
        rows.append({
            '市場': market, '模型': model, '成熟樣本': len(y),
            '正確率': float(accuracy_score(y, pred)),
            '平衡正確率': float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else np.nan,
            'Macro F1': macro_f1, 'Log Loss': float(log_loss(y, probability, labels=[0, 1, 2])),
            '基準Log Loss': float(log_loss(y, base, labels=[0, 1, 2])),
            'Brier': brier, '基準Brier': base_brier,
            'Brier Skill': 1 - brier / base_brier if base_brier else np.nan,
            '邊際校準誤差': marginal_error, '判定': verdict,
        })
        for class_id, class_name in enumerate(CLASS_ORDER):
            for lower in np.arange(0, 1, .1):
                upper = lower + .1
                selected = (probability[:, class_id] >= lower) & ((probability[:, class_id] < upper) if upper < 1 else True)
                if not selected.any():
                    continue
                calibration.append({
                    '市場': market, '模型': model, '分類': class_name,
                    '機率區間': f'{lower:.0%}–{upper:.0%}', '樣本數': int(selected.sum()),
                    '平均預測機率': float(probability[selected, class_id].mean()),
                    '實際發生率': float((y[selected] == class_id).mean()),
                })
    return pd.DataFrame(rows), pd.DataFrame(calibration)


POSITIVE = ('買進', '強買', '偏多', '多頭', '偏強', '正向', '安全', '優良', '增加', '成長')
NEGATIVE = ('賣出', '減碼', '偏空', '空頭', '轉弱', '偏弱', '危險', '過熱', '衰退')
CATEGORY_HINTS = ('短期趨勢', '中期趨勢', '長期趨勢', '籌碼', '量價參考', '市場環境', '風險',
                  '未持有建議', '基本面投資建議', '買價投資建議', '盈餘投資建議',
                  '護城河', '財務安全性', '獲利品質')
NUMBER_HINTS = ('歷史估值百分位', '成長分數', '品質分數', '巴菲特檢核分數', '基本面分數',
                'RSI', 'ATR比例', 'CMF', '籌碼K線分數', '投入資本報酬率', '股東權益報酬率',
                '自由現金流殖利率')


def _expected_sign(value):
    text = str(value)
    if any(token in text for token in POSITIVE):
        return 1
    if any(token in text for token in NEGATIVE):
        return -1
    return 0


def _evaluation_target(column):
    name = str(column)
    if name.startswith('WHID_'):
        return 'return_126', '126交易日'
    if '短期趨勢' in name:
        return 'return_5', '5交易日'
    if '長期趨勢' in name:
        return 'return_60', '60交易日'
    return 'return_10', '10交易日'


def _indicator_tables(frame):
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    from scipy.stats import spearmanr
    snapshots = pd.DataFrame([json.loads(value) for value in frame.snapshot_json])
    outcome = frame[['market', 'ticker', 'signal_date', 'return_5', 'return_10', 'return_20',
                     'return_60', 'return_126', 'return_252', 'actual_class',
                     'max_up_10', 'max_down_10']].reset_index(drop=True)
    snapshots = snapshots.reset_index(drop=True)
    categorical_rows, numeric_rows = [], []
    category_columns = [c for c in snapshots if any(hint in str(c) for hint in CATEGORY_HINTS)
                        and not pd.api.types.is_numeric_dtype(snapshots[c])]
    number_columns = [c for c in snapshots if any(hint in str(c) for hint in NUMBER_HINTS)]
    for column in category_columns:
        values = snapshots[column]
        target, target_label = _evaluation_target(column)
        for value, indexes in values.groupby(values, dropna=True).groups.items():
            selected = outcome.loc[list(indexes)].dropna(subset=[target])
            if selected.empty:
                categorical_rows.append({
                    '指標': column, '分類值': value, '評估期間': target_label,
                    '預期方向': '尚未判定', '成熟樣本': 0, '判定': f'需累積{target_label}結果',
                })
                continue
            sign = _expected_sign(value)
            returns = selected[target]
            hit = ((returns > 0) if sign > 0 else
                   (returns < 0) if sign < 0 else pd.Series(dtype=bool))
            hit_rate = float(hit.mean()) if sign else np.nan
            verdict = '僅描述／待定義方向'
            if sign and len(selected) < 30:
                verdict = '資料不足'
            elif sign and hit_rate >= .55:
                verdict = '初步可信'
            elif sign and hit_rate <= .45:
                verdict = '方向相反；需修正'
            elif sign:
                verdict = '尚不明確'
            categorical_rows.append({
                '指標': column, '分類值': value, '評估期間': target_label,
                '預期方向': '正向' if sign > 0 else '負向' if sign < 0 else '未定義',
                '成熟樣本': len(selected), '期間平均報酬': returns.mean(),
                '期間上漲比例': (returns > 0).mean(),
                '方向命中率': hit_rate, '上行先觸率': (selected.actual_class == 'up_first').mean(),
                '下行先觸率': (selected.actual_class == 'down_first').mean(), '判定': verdict,
            })
    for column in number_columns:
        values = pd.to_numeric(snapshots[column], errors='coerce')
        target, target_label = _evaluation_target(column)
        selected = pd.DataFrame({'value': values, 'future_return': outcome[target]}).dropna()
        if len(selected) < 10 or selected.value.nunique() < 3:
            numeric_rows.append({'指標': column, '評估期間': target_label, '成熟樣本': len(selected),
                                 '判定': f'需累積{target_label}結果'})
            continue
        rho, pvalue = spearmanr(selected.value, selected.future_return)
        verdict = '資料不足' if len(selected) < 30 else (
            '初步有關聯' if abs(rho) >= .15 and pvalue <= .10 else
            '關聯偏弱／需修正' if abs(rho) < .08 else '尚不明確')
        numeric_rows.append({
            '指標': column, '評估期間': target_label, '成熟樣本': len(selected), 'Spearman相關': rho,
            'p值': pvalue, '低值組期間平均報酬': selected.loc[selected.value <= selected.value.quantile(.25), 'future_return'].mean(),
            '高值組期間平均報酬': selected.loc[selected.value >= selected.value.quantile(.75), 'future_return'].mean(),
            '判定': verdict,
        })
    return pd.DataFrame(categorical_rows), pd.DataFrame(numeric_rows)


def build_review(db_path=DB_PATH, minimum_calendar_days=30):
    connection = connect(db_path)
    forecasts = pd.read_sql_query(
        "SELECT * FROM forecasts WHERE settlement_status='settled'", connection)
    indicators = pd.read_sql_query(
        "SELECT * FROM indicators WHERE settlement_status='settled'", connection)
    coverage = pd.read_sql_query('''SELECT market, MIN(signal_date) AS first_signal_date,
        MAX(signal_date) AS last_signal_date, COUNT(*) AS snapshots,
        SUM(CASE WHEN settlement_status='settled' THEN 1 ELSE 0 END) AS settled
        FROM indicators GROUP BY market''', connection)
    connection.close()
    ai, calibration = _ai_tables(forecasts)
    categorical, numeric = _indicator_tables(indicators)
    span_days = 0
    if not coverage.empty:
        span_days = max((pd.to_datetime(coverage.last_signal_date) - pd.to_datetime(coverage.first_signal_date)).dt.days.max(), 0)
    readiness = '已達30日，可進行第一輪判讀' if span_days >= minimum_calendar_days else f'累積{span_days}日，尚未達{minimum_calendar_days}日'
    overview = pd.DataFrame([
        {'項目': '資料狀態', '結果': readiness},
        {'項目': 'AI成熟預測數', '結果': len(forecasts)},
        {'項目': '指標成熟快照數', '結果': len(indicators)},
        {'項目': '判讀原則', '結果': '至少30筆成熟樣本；AI須優於類別基準，方向型指標命中率以55%為初步門檻'},
        {'項目': '期限分工', '結果': '短期趨勢5日；中期／籌碼／量價／風險10日；長期趨勢60日；WHID基本面與估值126日'},
        {'項目': '限制', '結果': '30日只足以初查第二階段；第一階段須等待約6個月，不可用短期漲跌判定'},
    ])
    method = pd.DataFrame([
        {'主題': '封存', '說明': '同一市場、股票、訊號日與模型只保留首次預測，後續重跑不覆寫。'},
        {'主題': '成熟', '說明': '訊號日後第10個交易日完成才核對；同日上下雙觸採下行先觸。'},
        {'主題': 'AI', '說明': '比較正確率、平衡正確率、Macro F1、Log Loss、Brier Skill與校準誤差。'},
        {'主題': '分類指標', '說明': '短期趨勢看5日，中期／籌碼／風險看10日，長期趨勢看60日，WHID基本面與估值看126日。'},
        {'主題': '數值指標', '說明': '依指標期限使用Spearman相關及高低四分位報酬差；多重比較結果只作篩選。'},
    ])
    output_dir = VALIDATION_ROOT / '30日評估' / datetime.now().strftime('%Y/%m')
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = output_dir / f'WHID_30日驗證_{stamp}.xlsx'
    sheets = {'總覽': overview, '資料涵蓋': coverage, 'AI模型': ai, 'AI校準': calibration,
              '分類指標': categorical, '數值指標': numeric, '方法': method}
    from openpyxl.styles import Font, PatternFill
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for name, frame in sheets.items():
            (frame if not frame.empty else pd.DataFrame({'狀態': ['尚無成熟資料']})).to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes = 'A2'; ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(color='FFFFFF', bold=True)
                cell.fill = PatternFill('solid', fgColor='17365D')
            for column in ws.columns:
                width = min(55, max(12, max(len(str(cell.value or '')) for cell in column) + 2))
                ws.column_dimensions[column[0].column_letter].width = width
    return {'path': str(output), 'readiness': readiness, 'ai_rows': len(ai),
            'categorical_rows': len(categorical), 'numeric_rows': len(numeric)}


def status(db_path=DB_PATH):
    connection = connect(db_path)
    tables = {}
    for table in ['forecasts', 'indicators']:
        tables[table] = pd.read_sql_query(
            f'''SELECT market, COUNT(*) AS total,
            SUM(CASE WHEN settlement_status='settled' THEN 1 ELSE 0 END) AS settled,
            MIN(signal_date) AS first_date, MAX(signal_date) AS last_date
            FROM {table} GROUP BY market''', connection).to_dict('records')
    connection.close()
    return tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['archive', 'settle', 'review', 'status'])
    parser.add_argument('--market', choices=['TW', 'US'])
    parser.add_argument('--report')
    parser.add_argument('--db', default=str(DB_PATH))
    args = parser.parse_args()
    if args.action == 'archive':
        if not args.market:
            parser.error('archive需要--market')
        result = archive_report(args.market, args.report, args.db) if args.report else archive_latest(args.market, db_path=args.db)
    elif args.action == 'settle':
        result = settle_pending(args.db)
    elif args.action == 'review':
        result = build_review(args.db)
    else:
        result = status(args.db)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
