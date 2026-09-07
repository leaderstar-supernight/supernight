"""TW second-stage inputs. Credentials are neither cached nor included in errors."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

VERSION = 'TW-timing-1.7'
TW_ZONE = timezone(timedelta(hours=8))


def number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def symbol(value):
    value = str(value).strip().upper()
    if not re.fullmatch(r'\d{4,6}\.(TW|TWO)', value):
        raise ValueError('台股代號須包含市場後綴，例如 2330.TW 或 3324.TWO')
    return value


def load_config(path):
    path = Path(path).resolve()
    c = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    for group in ['data', 'candidates', 'rules', 'ai', 'timesfm', 'backtest', 'output']:
        if not isinstance(c.get(group), dict):
            raise ValueError(f'設定缺少 {group}')
    for group, key in [('data','finmind_config'), ('candidates','report_path'),
                       ('candidates','holdings_path'), ('timesfm','model_cache'), ('output','root'),
                       ('output','system_root')]:
        if c[group].get(key):
            c[group][key] = str((path.parent / c[group][key]).resolve())
    c['_config_path'] = str(path)
    c.setdefault('sentiment', {})
    s=c['sentiment']
    defaults={'enabled':True,'news_enabled':True,'news_count':20,'news_min_articles':2,
              'news_positive_threshold':0.15,'news_negative_threshold':-0.15,
              'market_history_days':365,'margin_pressure_growth':0.05,
              'margin_utilization_high':0.60}
    for key,value in defaults.items():s.setdefault(key,value)
    if not isinstance(s['enabled'],bool) or not isinstance(s['news_enabled'],bool):
        raise ValueError('情緒研究開關須為布林值')
    if int(s['news_count'])<1 or int(s['news_min_articles'])<1 or int(s['market_history_days'])<90:
        raise ValueError('情緒研究篇數或歷史期間設定錯誤')
    if not -1<s['news_negative_threshold']<s['news_positive_threshold']<1:
        raise ValueError('新聞情緒門檻設定錯誤')
    if not 0<s['margin_pressure_growth']<1 or not 0<s['margin_utilization_high']<=1:
        raise ValueError('融資壓力門檻設定錯誤')
    r, a, tf, b = c['rules'], c['ai'], c['timesfm'], c['backtest']
    for key in ['short_fast_ma','short_slow_ma','short_slope_days','fast_ma','slow_ma','slope_days',
                'long_fast_ma','long_slow_ma','long_slope_days','chip_days','chip_history_days','branch_top_n']:
        if int(r[key]) < 1: raise ValueError(f'{key} 必須大於零')
    if (r['short_fast_ma'] >= r['short_slow_ma'] or r['fast_ma'] >= r['slow_ma'] or
            r['long_fast_ma'] >= r['long_slow_ma'] or r['chip_lag_sessions'] < 1):
        raise ValueError('均線順序或籌碼資料延遲設定錯誤')
    for key in ['margin_overheat_growth','lending_volume_ratio_high','branch_pressure_threshold']:
        if number(r[key]) is None or not 0 < r[key] < 1: raise ValueError(f'{key} 必須介於零與一')
    for key in ['chip_kline_enabled','branch_data_enabled']:
        if not isinstance(r[key],bool): raise ValueError(f'{key} 必須為布林值')
    for key in ['horizon','sequence_length','min_train','calibration_size','validation_size','final_test',
                'min_class_count','lstm_hidden','lstm_epochs','lstm_batch_size','lstm_patience','lstm_validation_size']:
        if int(a[key]) < 1: raise ValueError(f'AI {key} 必須大於零')
    for key in ['up_atr','down_atr','lstm_learning_rate']:
        if number(a[key]) is None or a[key] <= 0: raise ValueError(f'AI {key} 必須大於零')
    if not 0 <= a['lstm_dropout'] < 1: raise ValueError('AI lstm_dropout 必須介於零與一')
    if not isinstance(a['lstm_walk_forward'],bool): raise ValueError('AI lstm_walk_forward 必須為布林值')
    supported={'logistic','xgboost','lstm','momentum_benchmark'}
    if not a['models'] or not set(a['models']).issubset(supported): raise ValueError('AI模型設定錯誤')
    if not a.get('ensemble_models') or not set(a['ensemble_models']).issubset(set(a['models'])):
        raise ValueError('AI集成模型必須是models的非空子集合')
    if not isinstance(tf.get('enabled'), bool) or not isinstance(tf.get('research_only'), bool):
        raise ValueError('TimesFM開關須為布林值')
    if tf.get('enabled') and not tf.get('research_only'):
        raise ValueError('TimesFM 3.0只允許本專案以research_only模式使用')
    if tf.get('backend') != 'timesfm_3_pytorch' or not str(tf.get('model_id', '')).strip():
        raise ValueError('TimesFM模型設定錯誤')
    if str(tf.get('device', 'auto')).lower() not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('TimesFM device必須是auto、cpu或cuda')
    for key in ['context_length','min_context','horizon','atr_period','per_core_batch_size']:
        if not isinstance(tf.get(key), int) or tf[key] < 1:
            raise ValueError(f'TimesFM {key}須為正整數')
    if tf['min_context'] > tf['context_length']:
        raise ValueError('TimesFM min_context不可大於context_length')
    if tf['horizon'] != a['horizon']:
        raise ValueError('TimesFM與AI horizon必須一致')
    if not isinstance(tf.get('symmetric_averaging'), bool):
        raise ValueError('TimesFM symmetric_averaging須為布林值')
    for key in ['fee_rate','sell_tax_rate','slippage','stop_loss','take_profit']:
        if not 0 <= b[key] < 1: raise ValueError(f'{key} 設定超出範圍')
    if b['initial_cash'] <= 0 or b['max_holding_sessions'] < 1 or b['evaluation_sessions'] < 2:
        raise ValueError('回測資金或期間設定錯誤')
    limit = c['candidates']['max_stocks']
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError('max_stocks 必須為正整數或 null')
    return c


def read_table(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path, dtype={'代號': str, 'Ticker': str, 'ticker': str})
    with pd.ExcelFile(path) as book:
        name = 'Report' if 'Report' in book.sheet_names else book.sheet_names[0]
        return pd.read_excel(book, sheet_name=name)


def report_date(path):
    hit = re.search(r'(20\d{6})(?:[_\-.]|$)', Path(path).stem)
    if not hit: return None
    try: return pd.to_datetime(hit.group(1), format='%Y%m%d').date().isoformat()
    except ValueError: return None


def find_report(project):
    report_root = Path(project) / 'reports'
    folder = report_root / '簡化版' / 'TW'
    paths = [p for p in folder.glob('TW_*_Step1_Report.xlsx') if not p.name.startswith('~$')]
    if not paths:
        folder = report_root / '詳細版' / 'TW'
        paths = [p for p in folder.glob('TW_*_Step1_Report.xlsx') if not p.name.startswith('~$')]
    if not paths: raise ValueError('找不到 WHID 台股報表；請填 report_path 或 tickers')
    return max(paths, key=lambda p: (report_date(p) or '', p.stat().st_mtime))


def candidates(c, project):
    settings = c['candidates']
    manual = settings.get('tickers') or []
    if manual:
        frame = pd.DataFrame({'代號': list(dict.fromkeys(symbol(t) for t in manual))})
        provenance = {'來源': '手動候選名單；未宣稱通過WHID篩選', 'WHID評估日期': None}
    else:
        path = Path(settings['report_path']) if settings.get('report_path') else find_report(project)
        frame = read_table(path)
        frame = frame.rename(columns={'Ticker': '代號', 'ticker': '代號'})
        if '代號' not in frame: raise ValueError('WHID 報表缺少代號欄位')
        frame = frame.loc[frame['代號'].astype(str).str.fullmatch(r'\d{4,6}\.(TW|TWO)', na=False)].copy()
        frame['代號'] = frame['代號'].map(symbol)
        frame = frame.drop_duplicates('代號')
        date_value = report_date(path)
        # The report creation date is provenance, not a point-in-time history of selection.
        provenance = {'來源': str(path), 'WHID評估日期': date_value}
    if frame.empty: raise ValueError('候選名單没有有效台股代號')
    count = len(frame)
    if settings.get('max_stocks'): frame = frame.head(settings['max_stocks']).copy()
    provenance.update({'候選總數': count, '本次處理數': len(frame), '選取方式': '報表原順序；未自動按評分排除股票'})
    return frame.reset_index(drop=True), provenance


def read_holdings(path):
    if not path: return {}
    f = read_table(path)
    required = {'代號','股數','平均成本'}
    if not required.issubset(f): raise ValueError('持股表需要代號、股數、平均成本')
    holdings = {}
    for row in f.to_dict('records'):
        t = symbol(row['代號']); qty, cost = number(row['股數']), number(row['平均成本'])
        if t in holdings: raise ValueError('持股表代號重複，請先合併')
        if qty is None or cost is None or qty <= 0 or cost <= 0 or qty != int(qty):
            raise ValueError('持股股數須為正整數，成本須為正數')
        holdings[t] = {'股數': int(qty), '平均成本': cost}
    return holdings


def clean_prices(frame):
    f = frame.copy()
    required = ['Open','High','Low','Close','Volume']
    if not set(required).issubset(f): raise ValueError('行情欄位不足')
    f.index = pd.to_datetime(f.index)
    if f.index.tz is not None: f.index = f.index.tz_localize(None)
    f.index = f.index.normalize()
    if f.index.duplicated().any(): raise ValueError('行情日期重複')
    f = f.sort_index()
    f[required] = f[required].apply(pd.to_numeric, errors='coerce')
    inactive = (f.Volume == 0) & ((f[['Open','High','Low','Close']] == 0).all(axis=1) |
                                  f[['Open','High','Low','Close']].isna().all(axis=1))
    # A reported no-trade day stays on the calendar, but is never a price/label.
    f.loc[inactive,['Open','High','Low','Close']] = np.nan
    tolerance = f[['Open','High','Low','Close']].abs().max(axis=1) * 1e-6
    max_oc=f[['Open','Close']].max(axis=1); min_oc=f[['Open','Close']].min(axis=1)
    high_rounding=(f.High<max_oc)&((max_oc-f.High)<=tolerance)
    low_rounding=(f.Low>min_oc)&((f.Low-min_oc)<=tolerance)
    f.loc[high_rounding,'High']=max_oc[high_rounding]
    f.loc[low_rounding,'Low']=min_oc[low_rounding]
    valid = np.isfinite(f[required]).all(axis=1) & (f[['Open','High','Low','Close']] > 0).all(axis=1)
    valid &= (f['Volume'] >= 0) & (f['High'] >= f[['Open','Close','Low']].max(axis=1))
    valid &= f['Low'] <= f[['Open','Close','High']].min(axis=1)
    if not (valid|inactive).all(): raise ValueError('行情有缺值或OHLC範圍異常，不自動跨缺口計算')
    return f


class Provider:
    def __init__(self, config, now=None):
        self.config = config
        self.now = now or datetime.now(TW_ZONE)
        self.report_check_date = pd.Timestamp(self.now.astimezone(TW_ZONE).date()) if self.now.tzinfo else pd.Timestamp(self.now.date())
        # Daily signals are complete only after conservative 20:00 Taiwan cutoff.
        self.asof = pd.Timestamp(self.now.date())
        if self.now.hour < 20: self.asof -= pd.Timedelta(days=1)
        self.start = (self.asof - pd.DateOffset(years=config['data']['years'])).strftime('%Y-%m-%d')
        self.cache = Path(config['output']['system_root']) / 'cache' / 'TW'
        self.cache.mkdir(parents=True, exist_ok=True)
        f = Path(config['data']['finmind_config'])
        settings = yaml.safe_load(f.read_text(encoding='utf-8-sig'))
        fm = settings.get('data_sources',{}).get('finmind',{})
        self._token = fm.get('token') if fm.get('enabled') else None
        self._branch_unavailable_reason=None
        if not self._token: raise ValueError('現有 FinMind 設定未啟用或缺少金鑰')

    def fetch(self, dataset, code=None, start_date=None, end_date=None):
        params = {'dataset': dataset, 'start_date': start_date or self.start,
                  'end_date': end_date or self.asof.strftime('%Y-%m-%d')}
        if code is not None: params['data_id']=code
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:24]
        path = self.cache / (key + '.json')
        if path.exists() and (datetime.now().timestamp() - path.stat().st_mtime) < self.config['data']['cache_hours']*3600:
            return pd.DataFrame(json.loads(path.read_text(encoding='utf-8')))
        try:
            r = requests.get('https://api.finmindtrade.com/api/v4/data', params=params,
                             headers={'Authorization': 'Bearer ' + self._token}, timeout=30)
            if r.status_code != 200: raise ValueError(f'HTTP {r.status_code}')
            body = r.json()
            if body.get('status',200) != 200: raise ValueError(f'API {body.get("status")}')
            rows = body.get('data',[])
            if not rows: raise ValueError('無資料')
        except Exception as exc:
            # Requests exceptions may contain headers/URLs; never propagate their content.
            status = str(exc) if isinstance(exc,ValueError) and re.fullmatch(r'(HTTP|API) \d+|無資料', str(exc)) else type(exc).__name__
            raise RuntimeError(f'{dataset}: {status}') from None
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
        return pd.DataFrame(rows)

    def fetch_branch_days(self, code, dates):
        """Fetch licensed branch details one day at a time; never fail the core bundle."""
        if self._branch_unavailable_reason:
            raise RuntimeError('TaiwanStockTradingDailyReport: '+self._branch_unavailable_reason)
        frames=[]
        for day in dates:
            params={'data_id':code,'date':pd.Timestamp(day).strftime('%Y-%m-%d')}
            key=hashlib.sha256(('branch:'+json.dumps(params,sort_keys=True)).encode()).hexdigest()[:24]
            path=self.cache/(key+'.json')
            if path.exists() and (datetime.now().timestamp()-path.stat().st_mtime)<self.config['data']['cache_hours']*3600:
                rows=json.loads(path.read_text(encoding='utf-8'))
            else:
                try:
                    response=requests.get('https://api.finmindtrade.com/api/v4/taiwan_stock_trading_daily_report',
                        params=params,headers={'Authorization':'Bearer '+self._token},timeout=30)
                    if response.status_code!=200:raise ValueError(f'HTTP {response.status_code}')
                    body=response.json()
                    if body.get('status',200)!=200:raise ValueError(f'API {body.get("status")}')
                    rows=body.get('data',[])
                    if not rows:raise ValueError('無資料')
                except Exception as exc:
                    status=str(exc) if isinstance(exc,ValueError) and re.fullmatch(r'(HTTP|API) \d+|無資料',str(exc)) else type(exc).__name__
                    if status!='無資料':self._branch_unavailable_reason=status
                    raise RuntimeError(f'TaiwanStockTradingDailyReport: {status}') from None
                path.write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
            frame=pd.DataFrame(rows)
            if not frame.empty:frames.append(frame)
        return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

    def prices(self, dataset, code):
        f = self.fetch(dataset,code)
        if not (f['stock_id'].astype(str) == code).all(): raise ValueError('行情代號不一致')
        f = f.rename(columns={'open':'Open','max':'High','min':'Low','close':'Close',
                              'Trading_Volume':'Volume','Trading_money':'Turnover'})
        return clean_prices(f.set_index('date')).loc[:self.asof]

    def market_context(self):
        """Return public market inputs once per run; never infer an account maintenance ratio."""
        days=int(self.config['sentiment']['market_history_days'])
        start=(self.asof-pd.Timedelta(days=days)).strftime('%Y-%m-%d')
        context={'index':pd.DataFrame(),'margin':pd.DataFrame(),
                 'meta':{'市場情緒資料來源':'FinMind TAIEX＋TaiwanStockTotalMarginPurchaseShortSale',
                         '大盤融資維持率':'未提供；公開資料無法還原投資人帳戶擔保品與負債'}}
        errors=[]
        try: context['index']=self.prices('TaiwanStockPrice','TAIEX').loc[start:]
        except Exception as exc: errors.append('TAIEX '+(str(exc) if isinstance(exc,RuntimeError) else type(exc).__name__))
        try: context['margin']=self.fetch('TaiwanStockTotalMarginPurchaseShortSale',None,start_date=start)
        except Exception as exc: errors.append('大盤融資 '+(str(exc) if isinstance(exc,RuntimeError) else type(exc).__name__))
        context['meta']['市場情緒資料狀態']='可用' if not errors else '部分資料不足：'+'；'.join(errors)
        return context

    def news(self,ticker):
        """Fetch recent Yahoo headlines for archiving; failure never blocks price analysis."""
        if not self.config['sentiment']['news_enabled']:return pd.DataFrame()
        ticker=symbol(ticker)
        try:
            import yfinance as yf
            items=yf.Ticker(ticker).get_news(count=int(self.config['sentiment']['news_count']),tab='all') or []
        except Exception as exc:
            raise RuntimeError('Yahoo新聞: '+type(exc).__name__) from None
        rows=[];retrieved=datetime.now(timezone.utc).isoformat()
        for item in items:
            content=item.get('content',item) if isinstance(item,dict) else {}
            provider=content.get('provider') or {}
            canonical=content.get('canonicalUrl') or content.get('clickThroughUrl') or {}
            rows.append({'代號':ticker,'新聞ID':content.get('id') or item.get('id'),
                         '發布時間UTC':content.get('pubDate') or content.get('providerPublishTime'),
                         '來源':provider.get('displayName') if isinstance(provider,dict) else provider,
                         '標題':content.get('title'),'摘要':content.get('summary') or content.get('description'),
                         '網址':canonical.get('url') if isinstance(canonical,dict) else canonical,
                         '相關代號':content.get('relatedTickers') or item.get('relatedTickers'),
                         '取得時間UTC':retrieved,'新聞資料源':'Yahoo Finance'})
        return pd.DataFrame(rows).drop_duplicates(subset=['新聞ID','標題'],keep='first') if rows else pd.DataFrame()

    def bundle(self, ticker):
        ticker = symbol(ticker); code = ticker.split('.')[0]
        raw = self.prices('TaiwanStockPrice', code)
        meta = {'原始價格來源':'FinMind TaiwanStockPrice',
                '籌碼來源':'FinMind 法人＋融資融券＋借券；券商分點視sponsor權限',
                '還原價格來源':None,'還原價格狀態':'資料不足','籌碼狀態':'資料不足',
                '取得時間UTC':datetime.now(timezone.utc).isoformat(), '資料截止日':str(self.asof.date())}
        adjusted = None
        try:
            adjusted = self.prices('TaiwanStockPriceAdj',code)
            meta.update({'還原價格來源':'FinMind TaiwanStockPriceAdj','還原價格狀態':'可用'})
        except Exception as exc:
            meta['FinMind還原價格註記'] = str(exc)
            if self.config['data']['yahoo_adjusted_fallback']:
                try:
                    import yfinance as yf
                    f = yf.Ticker(ticker).history(start=self.start, end=(self.asof+pd.Timedelta(days=1)).strftime('%Y-%m-%d'), auto_adjust=True)
                    adjusted = clean_prices(f)
                    meta.update({'還原價格來源':'Yahoo auto_adjust=True；整段研究資料','還原價格狀態':'可用'})
                except Exception as error:
                    meta['Yahoo還原價格註記'] = type(error).__name__
        if adjusted is not None:
            # Align by raw trading sessions, never forward-fill missing OHLC or mix price sources.
            adjusted = adjusted.reindex(raw.index)
            inactive=(raw.Volume==0)&raw.Close.isna()
            adjusted.loc[inactive,['Open','High','Low','Close']]=np.nan
            adjusted.loc[inactive,'Volume']=0
            missing=adjusted[['Open','High','Low','Close']].isna().any(axis=1)&~inactive
            factor = adjusted['Close']/raw['Close']
            normalized = (adjusted[['Open','High','Low','Close']].div(raw[['Open','High','Low','Close']])).div(factor,axis=0)
            inconsistent=((normalized-1).abs()>0.01).any(axis=1)&~inactive
            excluded=missing|inconsistent
            # Quarantine individual dates instead of blending providers or inventing quotes.
            # Labels spanning these dates and rolling feature windows are excluded naturally.
            adjusted.loc[excluded,['Open','High','Low','Close','Volume']]=np.nan
            meta['還原缺漏日數']=int(missing.sum())
            meta['跨來源差異隔離日數']=int(inconsistent.sum())
            meta['研究隔離日期']=[str(d.date()) for d in raw.index[excluded]]
            meta['還原價格狀態']='可用；異常日期已隔離，回測另限完整區段' if excluded.any() else '可用'
            if adjusted.Close.notna().sum()<self.config['rules']['slow_ma']:
                adjusted=None
                meta['還原價格狀態']='有效還原行情不足，AI及回測停用'
        chips = pd.DataFrame()
        try:
            chips = self.fetch('TaiwanStockInstitutionalInvestorsBuySell',code)
            if not (chips['stock_id'].astype(str)==code).all(): raise ValueError('籌碼代號不一致')
            meta['籌碼狀態'] = '已取得；逐日完整性另檢查'
        except Exception as exc:
            meta['籌碼註記'] = str(exc) if isinstance(exc,RuntimeError) else type(exc).__name__
        chip_start=(self.asof-pd.Timedelta(days=self.config['rules']['chip_history_days'])).strftime('%Y-%m-%d')
        margin=lending=branches=pd.DataFrame()
        if self.config['rules']['chip_kline_enabled']:
            try:
                margin=self.fetch('TaiwanStockMarginPurchaseShortSale',code,start_date=chip_start)
                if not (margin['stock_id'].astype(str)==code).all():raise ValueError('融資融券代號不一致')
                meta['融資融券狀態']='可用'
            except Exception as exc:
                meta['融資融券狀態']='資料不足';meta['融資融券註記']=str(exc) if isinstance(exc,RuntimeError) else type(exc).__name__
            try:
                lending=self.fetch('TaiwanStockSecuritiesLending',code,start_date=chip_start)
                if not (lending['stock_id'].astype(str)==code).all():raise ValueError('借券代號不一致')
                meta['借券狀態']='可用'
            except Exception as exc:
                meta['借券狀態']='資料不足';meta['借券註記']=str(exc) if isinstance(exc,RuntimeError) else type(exc).__name__
            if self.config['rules']['branch_data_enabled']:
                n=self.config['rules']['chip_days'];lag=self.config['rules']['chip_lag_sessions']
                eligible=raw.index[:-lag] if lag else raw.index
                requested=list(eligible[-n:])
                try:
                    branches=self.fetch_branch_days(code,requested)
                    if not (branches['stock_id'].astype(str)==code).all():raise ValueError('分點代號不一致')
                    meta['券商分點狀態']='可用（FinMind sponsor）'
                except Exception as exc:
                    meta['券商分點狀態']='資料不足／可能需要FinMind sponsor權限'
                    meta['券商分點註記']=str(exc) if isinstance(exc,RuntimeError) else type(exc).__name__
                branches.attrs['requested_dates']=[pd.Timestamp(d).normalize() for d in requested]
        chips.attrs['margin']=margin;chips.attrs['lending']=lending;chips.attrs['branches']=branches
        if raw.empty: raise ValueError('無完整日線')
        meta['無成交空白行情日數']=int(raw.Close.isna().sum())
        return raw, adjusted, chips, meta
