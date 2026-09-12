"""Independent US-only Yahoo data and WHID input handling. No credentials."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

VERSION='US-timing-1.8'
NY=ZoneInfo('America/New_York')
TAIPEI=ZoneInfo('Asia/Taipei')


def number(value):
    try:
        value=float(value)
        return value if np.isfinite(value) else None
    except (TypeError,ValueError): return None


def symbol(value):
    value=str(value).strip().upper()
    if not re.fullmatch(r'[A-Z][A-Z0-9]{0,5}(?:[.-][A-Z])?',value):
        raise ValueError('需要美股Yahoo代號，例如 AAPL、NVDA、BRK-B')
    return value.replace('.','-')


def load_config(path):
    path=Path(path).resolve();c=yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    for g in ['data','candidates','rules','ai','timesfm','backtest','output']:
        if not isinstance(c.get(g),dict): raise ValueError(f'設定缺少{g}')
    for g,k in [('candidates','report_path'),('candidates','holdings_path'),('timesfm','model_cache'),('output','root'),('output','system_root')]:
        if c[g].get(k):c[g][k]=str((path.parent/c[g][k]).resolve())
    c['_config_path']=str(path)
    c.setdefault('sentiment',{})
    s=c['sentiment']
    defaults={'enabled':True,'news_enabled':True,'news_count':20,'news_min_articles':2,
              'news_positive_threshold':0.15,'news_negative_threshold':-0.15,
              'vix_symbol':'^VIX'}
    for key,value in defaults.items():s.setdefault(key,value)
    if not isinstance(s['enabled'],bool) or not isinstance(s['news_enabled'],bool):raise ValueError('情緒研究開關須為布林值')
    if int(s['news_count'])<1 or int(s['news_min_articles'])<1:raise ValueError('新聞篇數設定錯誤')
    if not -1<s['news_negative_threshold']<s['news_positive_threshold']<1:raise ValueError('新聞情緒門檻設定錯誤')
    r,a,tf,b=c['rules'],c['ai'],c['timesfm'],c['backtest']
    for k in ['short_fast_ma','short_slow_ma','short_slope_days','fast_ma','slow_ma','slope_days',
              'long_fast_ma','long_slow_ma','long_slope_days','flow_days']:
        if not isinstance(r[k],int) or r[k]<1: raise ValueError(f'{k}須為正整數')
    if (r['short_fast_ma']>=r['short_slow_ma'] or r['fast_ma']>=r['slow_ma'] or
            r['long_fast_ma']>=r['long_slow_ma']): raise ValueError('各層短均線須小於長均線')
    if not 0<=r['cmf_positive']<1: raise ValueError('CMF門檻須在0至1之間')
    for k in ['volume_ratio_min','max_extension','max_atr_pct','min_daily_turnover_usd','atr_stop_multiple']:
        if number(r[k]) is None or r[k]<=0: raise ValueError(f'{k}須為正數')
    for k in ['horizon','sequence_length','min_train','calibration_size','validation_size','final_test',
              'min_class_count','lstm_hidden','lstm_epochs','lstm_batch_size','lstm_patience','lstm_validation_size']:
        if not isinstance(a[k],int) or a[k]<1:raise ValueError(f'AI {k}須為正整數')
    for k in ['up_atr','down_atr','lstm_learning_rate']:
        if number(a[k]) is None or a[k]<=0:raise ValueError(f'AI {k}須為正數')
    if not 0<=a['lstm_dropout']<1:raise ValueError('AI lstm_dropout設定錯誤')
    if not isinstance(a['lstm_walk_forward'],bool):raise ValueError('AI lstm_walk_forward須為布林值')
    supported={'logistic','xgboost','lstm','momentum_benchmark'}
    if not a['models'] or not set(a['models']).issubset(supported): raise ValueError('AI模型設定錯誤')
    if not a.get('ensemble_models') or not set(a['ensemble_models']).issubset(set(a['models'])):
        raise ValueError('AI集成模型必須是models的非空子集合')
    if not isinstance(tf.get('enabled'),bool) or not isinstance(tf.get('research_only'),bool):raise ValueError('TimesFM開關須為布林值')
    if tf.get('enabled') and not tf.get('research_only'):raise ValueError('TimesFM 3.0只允許本專案以research_only模式使用')
    if tf.get('backend')!='timesfm_3_pytorch' or not str(tf.get('model_id','')).strip():raise ValueError('TimesFM模型設定錯誤')
    if str(tf.get('device','auto')).lower() not in {'auto','cpu','cuda'}:raise ValueError('TimesFM device必須是auto、cpu或cuda')
    for k in ['context_length','min_context','horizon','atr_period','per_core_batch_size']:
        if not isinstance(tf.get(k),int) or tf[k]<1:raise ValueError(f'TimesFM {k}須為正整數')
    if tf['min_context']>tf['context_length']:raise ValueError('TimesFM min_context不可大於context_length')
    if tf['horizon']!=a['horizon']:raise ValueError('TimesFM與AI horizon必須一致')
    if not isinstance(tf.get('symmetric_averaging'),bool):raise ValueError('TimesFM symmetric_averaging須為布林值')
    for k in ['fee_rate','sell_fee_rate','slippage','stop_loss','take_profit']:
        if number(b[k]) is None or not 0<=b[k]<1:raise ValueError(f'{k}設定錯誤')
    if b['fee_rate']+b['sell_fee_rate']>=1:raise ValueError('總費率設定錯誤')
    if b['initial_cash']<=0 or b['max_holding_sessions']<1 or b['evaluation_sessions']<2:raise ValueError('回測資金／期間設定錯誤')
    if not 17<=c['data']['cutoff_hour_new_york']<=23:raise ValueError('收盤資料截止小時須為紐約17至23點')
    if c['data']['years']<1 or c['data']['max_price_age_days']<0:raise ValueError('資料期間設定錯誤')
    c['data'].setdefault('allow_recent_reference',True)
    c['data'].setdefault('max_reference_gap_sessions',2)
    if not isinstance(c['data']['allow_recent_reference'],bool):raise ValueError('allow_recent_reference須為布林值')
    if not isinstance(c['data']['max_reference_gap_sessions'],int) or c['data']['max_reference_gap_sessions']<0:raise ValueError('參考資料間隔須為非負整數')
    if c['candidates']['max_stocks'] is not None and (not isinstance(c['candidates']['max_stocks'],int) or c['candidates']['max_stocks']<1):raise ValueError('max_stocks須為正整數或null')
    c['data']['benchmark']=symbol(c['data']['benchmark'])
    return c


def read_table(path):
    path=Path(path)
    if path.suffix.lower()=='.csv':return pd.read_csv(path,dtype={'代號':str,'Ticker':str,'ticker':str})
    with pd.ExcelFile(path) as book:
        return pd.read_excel(book,sheet_name='Report' if 'Report' in book.sheet_names else book.sheet_names[0])


def report_date(path):
    hit=re.search(r'(20\d{6})(?:[_\-.]|$)',Path(path).stem)
    try:return pd.to_datetime(hit.group(1),format='%Y%m%d').date().isoformat() if hit else None
    except ValueError:return None


def find_report(project):
    report_root=Path(project)/'reports'
    folder=report_root/'簡化版'/'US'
    paths=[p for p in folder.glob('US_*_Step1_Report.xlsx') if not p.name.startswith('~$')]
    if not paths:
        folder=report_root/'詳細版'/'US'
        paths=[p for p in folder.glob('US_*_Step1_Report.xlsx') if not p.name.startswith('~$')]
    if not paths:raise ValueError('找不到WHID美股報表；請指定report_path或tickers')
    return max(paths,key=lambda p:(report_date(p) or '',p.stat().st_mtime))


def candidates(c,project):
    s=c['candidates']
    if s.get('tickers'):
        f=pd.DataFrame({'代號':list(dict.fromkeys(symbol(t) for t in s['tickers']))})
        meta={'來源':'手動候選名單；未宣稱通過WHID篩選','WHID評估日期':None}
    else:
        path=Path(s['report_path']) if s.get('report_path') else find_report(project)
        f=read_table(path).rename(columns={'Ticker':'代號','ticker':'代號'})
        if '代號' not in f:raise ValueError('報表缺少代號')
        if '市場' in f:f=f.loc[f['市場'].astype(str).str.upper().eq('US')].copy()
        valid=f['代號'].astype(str).str.strip().str.upper().str.fullmatch(r'[A-Z][A-Z0-9]{0,5}(?:[.-][A-Z])?',na=False)
        f=f.loc[valid].copy();f['代號']=f['代號'].map(symbol);f=f.drop_duplicates('代號')
        meta={'來源':str(path),'WHID評估日期':report_date(path)}
    if f.empty:raise ValueError('候選名單沒有有效美股代號')
    total=len(f)
    if s.get('max_stocks'):f=f.head(s['max_stocks']).copy()
    meta.update({'候選總數':total,'本次處理數':len(f),'選取方式':'報表原順序；未重新排名或按投資建議篩選'})
    return f.reset_index(drop=True),meta


def read_holdings(path):
    if not path:return {}
    f=read_table(path)
    if not {'代號','股數','平均成本'}.issubset(f):raise ValueError('持股CSV須有代號、股數、平均成本USD')
    result={}
    for r in f.to_dict('records'):
        t=symbol(r['代號']);qty,cost=number(r['股數']),number(r['平均成本'])
        if t in result:raise ValueError('持股代號重複')
        if qty is None or cost is None or qty<=0 or cost<=0:raise ValueError('股數與每股USD成本須為正數')
        result[t]={'股數':qty,'平均成本':cost}
    return result


def clean_prices(frame):
    f=frame.copy();cols=['Open','High','Low','Close','Volume']
    if not set(cols).issubset(f):raise ValueError('Yahoo行情缺OHLCV')
    f.index=pd.to_datetime(f.index)
    if f.index.tz is not None:f.index=f.index.tz_convert(NY).tz_localize(None)
    f.index=f.index.normalize()
    if f.index.duplicated().any():raise ValueError('行情日期重複')
    f=f.sort_index();f[cols]=f[cols].apply(pd.to_numeric,errors='coerce')
    inactive=(f.Volume==0)&((f[cols[:4]]==0).all(axis=1)|f[cols[:4]].isna().all(axis=1))
    f.loc[inactive,cols[:4]]=np.nan
    max_oc=f[['Open','Close']].max(axis=1);min_oc=f[['Open','Close']].min(axis=1)
    tol=f[cols[:4]].abs().max(axis=1)*1e-6
    hi=(f.High<max_oc)&((max_oc-f.High)<=tol);lo=(f.Low>min_oc)&((f.Low-min_oc)<=tol)
    f.loc[hi,'High']=max_oc[hi];f.loc[lo,'Low']=min_oc[lo]
    valid=np.isfinite(f[cols]).all(axis=1)&(f[cols[:4]]>0).all(axis=1)&(f.Volume>=0)
    valid&=(f.High>=f[['Open','Close','Low']].max(axis=1))&(f.Low<=f[['Open','Close','High']].min(axis=1))
    # Retain incomplete sessions as gaps, including a missing latest close.
    # No partial bar may enter indicators or model labels.
    f.loc[~valid,cols[:4]]=np.nan
    if 'Adj Close' in f:f.loc[~valid,'Adj Close']=np.nan
    return f


def adjusted_from_raw(raw):
    if 'Adj Close' not in raw:return None
    adj_close=pd.to_numeric(raw['Adj Close'],errors='coerce')
    factor=adj_close/raw.Close
    valid=np.isfinite(factor)&(factor>0)
    adjusted=raw[['Open','High','Low','Close','Volume']].copy()
    adjusted.loc[:,['Open','High','Low','Close']]=raw[['Open','High','Low','Close']].mul(factor,axis=0)
    adjusted.loc[~valid,['Open','High','Low','Close']]=np.nan
    if not valid.any():return None
    return adjusted


class Provider:
    def __init__(self,config,now=None):
        self.config=config
        now=now or datetime.now(timezone.utc)
        if now.tzinfo is None:raise ValueError('now需含時區，避免夏令時間偏移')
        self.now=now.astimezone(NY)
        self.report_check_date=pd.Timestamp(now.astimezone(TAIPEI).date())
        self.asof=pd.Timestamp(self.now.date())
        if self.now.hour<config['data']['cutoff_hour_new_york']:self.asof-=pd.Timedelta(days=1)
        self.start=(self.asof-pd.DateOffset(years=config['data']['years'])).strftime('%Y-%m-%d')
        self.cache=Path(config['output']['system_root'])/'cache'/'US'
        self.cache.mkdir(parents=True,exist_ok=True)
        self._memo={}

    def history(self,ticker):
        ticker=symbol(ticker)
        if ticker in self._memo:
            f,m=self._memo[ticker];return f.copy(),dict(m)
        key=hashlib.sha256(f'{ticker}|{self.start}|{self.asof.date()}|US-v2'.encode()).hexdigest()[:24]
        path=self.cache/(key+'.json');body=None;downloaded=False
        if path.exists():
            try:
                cached=json.loads(path.read_text(encoding='utf-8'))
                age=(datetime.now(timezone.utc)-datetime.fromisoformat(cached['retrieved_utc'])).total_seconds()
                ttl=self.config['data']['cache_hours']*3600
                # Incomplete latest candles should be refreshed after 15 minutes.
                if not cached['data'][-1].get('Close') or not cached['data'][-1].get('Adj Close'):ttl=min(ttl,900)
                if 0<=age<ttl:body=cached
            except (OSError,ValueError,KeyError,IndexError,TypeError):body=None
        if body is None:
            try:
                import yfinance as yf
                obj=yf.Ticker(ticker)
                frame=obj.history(start=self.start,end=(self.asof+pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                                  auto_adjust=False,actions=True,prepost=False)
                metadata=obj.get_history_metadata()
                if frame.empty:raise ValueError('下載為空；可能是連線、限流或Yahoo未回傳行情')
                meta={k:metadata.get(k) for k in ['currency','exchangeName','instrumentType','symbol']}
                meta['latest_missing_fields']=[col for col in ['Open','High','Low','Close','Adj Close','Volume'] if col not in frame or pd.isna(frame[col].iloc[-1])]
                frame=clean_prices(frame).loc[:self.asof]
                rows=frame.reset_index(names='date')
                rows['date']=rows.date.dt.strftime('%Y-%m-%d')
                body={'metadata':meta,'data':json.loads(rows.to_json(orient='records')),
                      'retrieved_utc':datetime.now(timezone.utc).isoformat()}
                downloaded=True
            except Exception as exc:
                reason=str(exc) if isinstance(exc,ValueError) else ('缺少yfinance或其必要套件' if isinstance(exc,ImportError) else '請檢查連線或稍後重試')
                raise RuntimeError(f'Yahoo {ticker}: {type(exc).__name__}；{reason[:120]}') from None
        m=body['metadata']
        if m.get('currency')!='USD' or m.get('instrumentType') not in {'EQUITY','ETF'}:
            raise ValueError('本版只接受已確認USD的美股股票／ETF；不推測缺漏幣別')
        if m.get('exchangeName') not in {'NMS','NGM','NCM','NYQ','ASE','PCX','BTS','NAS','NYS'}:
            raise ValueError('無法確認受支援的美國交易所；OTC/外國交易所不自動套用')
        if m.get('symbol') and symbol(m['symbol'])!=ticker:raise ValueError('Yahoo回傳代號不符')
        f=clean_prices(pd.DataFrame(body['data']).set_index('date')).loc[:self.asof]
        if f.empty:raise ValueError('截止日之前沒有完整行情')
        # Reading a cache must not reset its age indefinitely.
        if downloaded:path.write_text(json.dumps(body,ensure_ascii=False),encoding='utf-8')
        m={**m,'retrieved_utc':body['retrieved_utc']}
        self._memo[ticker]=(f.copy(),dict(m));return f,m

    def bundle(self,ticker):
        raw,info=self.history(ticker)
        benchmark_raw,binfo=self.history(self.config['data']['benchmark'])
        benchmark=adjusted_from_raw(benchmark_raw)
        # Benchmark dates supply the US session calendar; retain missing stock days.
        calendar=benchmark_raw.index[benchmark_raw.index>=raw.index[0]].union(raw.index)
        raw=raw.reindex(calendar).sort_index()
        adjusted=adjusted_from_raw(raw)
        raw['Turnover']=raw.Close*raw.Volume
        excluded=raw.index[raw.Close.isna()|(adjusted.Close.isna() if adjusted is not None else True)]
        meta={'價格來源':'Yahoo OHLCV（auto_adjust=False）','還原價格來源':'Yahoo Adj Close / Close 因子',
              '幣別':'USD','交易所':info['exchangeName'],'商品類型':info['instrumentType'],
              '基準':self.config['data']['benchmark'],'基準來源':'Yahoo還原日線',
              '資料截止日NY':str(self.asof.date()),'取得時間UTC':info['retrieved_utc'],
              '紐約執行時間':self.now.isoformat(),'最後有效行情日':str(raw.Close.dropna().index[-1].date()) if raw.Close.notna().any() else None,
              'Yahoo最新列缺少欄位':info.get('latest_missing_fields',[]),'基準最新列缺少欄位':binfo.get('latest_missing_fields',[]),
              '研究隔離日期':[str(d.date()) for d in excluded],
              '基準隔離日期':[] if benchmark is None else [str(d.date()) for d in benchmark.index[benchmark.Close.isna()]],
              '量價資料說明':'CMF及帶方向成交量僅為代理指標，不是機構或法人實際資金流',
              '機構籌碼狀態':'未使用13F；季度持股不當作每日買賣超',
              '還原價格狀態':'可用' if adjusted is not None else '缺還原價格，AI及回測停用'}
        return raw,adjusted,benchmark,meta

    def market_context(self):
        """Yahoo SPY/VIX market context. FINRA margin is monthly and is not treated as daily data."""
        benchmark_raw,_=self.history(self.config['data']['benchmark'])
        context={'benchmark':adjusted_from_raw(benchmark_raw),'vix':pd.DataFrame(),
                 'meta':{'市場情緒資料來源':'Yahoo SPY＋CBOE VIX行情（Yahoo轉載）',
                         '美股個股融資維持率':'無公開逐股等價資料',
                         'FINRA融資資料':'僅月度市場總額，未混入每日情緒分數'}}
        try:
            import yfinance as yf
            f=yf.Ticker(str(self.config['sentiment']['vix_symbol'])).history(
                start=self.start,end=(self.asof+pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                auto_adjust=False,actions=False,prepost=False)
            context['vix']=clean_prices(f).loc[:self.asof]
            context['meta']['市場情緒資料狀態']='可用'
        except Exception as exc:
            context['meta']['市場情緒資料狀態']='VIX資料不足：'+type(exc).__name__
        return context

    def news(self,ticker):
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
