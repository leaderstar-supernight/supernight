"""Independent US research calculations; no dependency on TW modules."""
from __future__ import annotations
import numpy as np
import pandas as pd


def whid_freshness(provenance,fallback_date,max_age_days):
    """Validate the Taipei-dated WHID run against the stage-2 run date."""
    whid_day=provenance.get('WHID評估日期');reference=provenance.get('第二階段執行日期台北') or fallback_date
    if not whid_day:
        manual='手動' in provenance.get('來源','')
        return manual,None,str(reference),'手動名單未驗證WHID日期' if manual else '無法確認WHID評估日期'
    try:
        whid=pd.Timestamp(whid_day).normalize();checked=pd.Timestamp(reference).normalize()
        if pd.isna(whid) or pd.isna(checked):raise ValueError
        age=(checked-whid).days
    except (TypeError,ValueError,OverflowError):return False,None,str(reference),'WHID評估日期無法解析'
    fresh=0<=age<=max_age_days
    return fresh,age,str(checked.date()),None if fresh else ('WHID評估日期晚於本次執行日期' if age<0 else 'WHID候選名單已超過有效天數')


def flow_features(raw,adjusted,rules):
    f=pd.DataFrame(index=raw.index)
    if adjusted is None:
        f['cmf']=np.nan;f['signed_volume_ratio']=np.nan
    else:
        c,h,l=(adjusted[k] for k in ['Close','High','Low'])
        v=raw.Volume
        span=h-l
        mult=((2*c-h-l)/span.replace(0,np.nan)).mask(span.eq(0)&c.notna(),0.)
        n=rules['flow_days'];denom=v.rolling(n,min_periods=n).sum().replace(0,np.nan)
        f['cmf']=(mult*v).rolling(n,min_periods=n).sum()/denom
        f['signed_volume_ratio']=(np.sign(c.diff())*v).rolling(n,min_periods=n).sum()/denom
    f['flow_ready']=f[['cmf','signed_volume_ratio']].notna().all(axis=1)
    f['flow_positive']=f.flow_ready&(f.cmf>rules['cmf_positive'])&(f.signed_volume_ratio>0)
    f['flow_negative']=f.flow_ready&(f.cmf<-rules['cmf_positive'])&(f.signed_volume_ratio<0)
    return f


def signals(raw,adjusted,benchmark,cfg):
    r=cfg['rules'];p=adjusted if adjusted is not None else raw
    x=technical(p,r).join(flow_features(raw,adjusted,r))
    x['turnover_20']=(raw.Close*raw.Volume).rolling(20,min_periods=20).mean()
    x['risk_ok']=(x.atr_pct<=r['max_atr_pct'])&(x.turnover_20>=r['min_daily_turnover_usd'])
    x['technical_entry']=x.trend_ready&x.trend_up&x.risk_ok&(x.volume_ratio>=r['volume_ratio_min'])&(x.extension<=r['max_extension'])
    if benchmark is not None:
        market=technical(benchmark,r).reindex(raw.index)
        x['market_ready']=market.trend_ready.fillna(False).astype(bool)
        x['market_up']=x.market_ready&market.trend_up.fillna(False).astype(bool)
        x['market_down']=x.market_ready&market.trend_down.fillna(False).astype(bool)
        x['relative_return_20']=x.return_20-market.return_20
    else:
        x['market_ready']=False;x['market_up']=False;x['market_down']=False;x['relative_return_20']=np.nan
    flow_ok=x.flow_positive if r['require_flow_for_entry'] else ~x.flow_negative
    market_ok=x.market_up if r['require_market_for_entry'] else pd.Series(True,index=x.index)
    x['entry']=x.technical_entry&flow_ok&market_ok
    x['exit']=x.trend_ready&(x.trend_down|x.flow_negative|(x.market_down if r['require_market_for_entry'] else False))
    if adjusted is None:x['entry']=False;x['technical_entry']=False
    return x


def latest_assessment(ticker,raw,adjusted,x,cfg,provenance,asof,holding=None):
    r=cfg['rules'];last=x.iloc[-1];day=raw.index[-1];close=float(raw.Close.iloc[-1])
    actual=raw.Close.dropna()
    age=(pd.Timestamp(asof).normalize()-actual.index[-1]).days if len(actual) else 9999
    ready=bool(last.trend_ready) and adjusted is not None and 0<=age<=cfg['data']['max_price_age_days'] and np.isfinite(close)
    notes=[]
    if not ready:notes.append('行情過期、缺最新日、還原價格或指標不足')
    if not last.market_ready:notes.append('市場基準資料不足')
    if not last.flow_ready:notes.append('量價資料不足')
    whid=provenance.get('WHID評估日期')
    whid_fresh,age_whid,whid_reference,whid_reason=whid_freshness(provenance,asof,cfg['candidates']['max_whid_age_days'])
    stale_whid=not whid_fresh
    if stale_whid:notes.append((whid_reason or 'WHID名單日期異常')+'，進場前需重評')
    risks=[]
    if pd.isna(last.atr_pct) or pd.isna(last.turnover_20):risks.append('風險資料不足')
    else:
        if last.atr_pct>r['max_atr_pct']:risks.append('波動偏高')
        if last.turnover_20<r['min_daily_turnover_usd']:risks.append('流動性偏低')
    if pd.notna(last.extension) and last.extension>r['max_extension']:risks.append('偏離均線過大')
    advice='等待'
    if not ready:advice='資料不足／暫不判斷'
    elif stale_whid:advice='先更新WHID評估'
    elif r['require_market_for_entry'] and not last.market_ready:advice='等待市場資料'
    elif r['require_flow_for_entry'] and not last.flow_ready:advice='等待量價資料'
    elif bool(last.entry):advice='符合研究進場條件'
    held='若持有：續抱觀察'
    if not ready:held='若持有：資料不足，人工確認'
    elif bool(last.exit):held='若持有：符合退出條件'
    elif risks or (r['require_market_for_entry'] and not last.market_ready):held='若持有：檢查風險／部位'
    factor=float(adjusted.Close.iloc[-1]/close) if adjusted is not None and np.isfinite(close) and close>0 else np.nan
    atr_raw=float(last.atr)/factor if np.isfinite(factor) and factor>0 else np.nan
    stop=close-r['atr_stop_multiple']*atr_raw
    def trend_text(prefix):
        if not bool(last[f'{prefix}_trend_ready']):return '資料不足'
        return '多頭' if bool(last[f'{prefix}_trend_up']) else ('轉弱' if bool(last[f'{prefix}_trend_down']) else '整理')
    short_trend=trend_text('short');medium_trend=trend_text('medium');long_trend=trend_text('long')
    stock_score=50.
    if bool(last.flow_positive):stock_score+=20
    elif bool(last.flow_negative):stock_score-=20
    if medium_trend=='多頭':stock_score+=15
    elif medium_trend=='轉弱':stock_score-=15
    if pd.notna(last.relative_return_20):stock_score+=10 if last.relative_return_20>0 else -10
    stock_score=float(np.clip(stock_score,0,100)) if bool(last.flow_ready) else np.nan
    stock_sentiment='資料不足' if pd.isna(stock_score) else ('偏多' if stock_score>=65 else ('偏空' if stock_score<=35 else '中性'))
    row={'代號':ticker,'訊號日期NY':str(day.date()),'現價USD':close,
        '趨勢':medium_trend,'短期趨勢':short_trend,'中期趨勢':medium_trend,'長期趨勢':long_trend,
        '量價參考':'資料不足' if not last.flow_ready else ('偏強' if last.flow_positive else ('偏弱' if last.flow_negative else '分歧／中性')),
        '市場環境':'資料不足' if not last.market_ready else ('偏多' if last.market_up else ('偏弱' if last.market_down else '整理')),
        '個股情緒':stock_sentiment,'個股情緒分數':stock_score,
        '個股情緒形成依據':'CMF／帶方向成交量＋中期趨勢＋相對SPY20日報酬；研究參考，不參與交易建議',
        '風險':'／'.join(risks) or '未觸發風險門檻','未持有建議':advice,'持有情境建議':held,
        'ATR風險參考價USD':stop if stop>0 else np.nan,'支撐參考USD':last.support/factor,'壓力參考USD':last.resistance/factor,
        '資料狀態':'；'.join(notes) or '可用；研究規則未驗證獲利','WHID評估日期':whid,
        'WHID名單日齡':age_whid,'WHID日期比較基準':whid_reference,'WHID日期判定':whid_reason or '有效',
        'CMF':last.cmf,'帶方向成交量比':last.signed_volume_ratio,'RSI':last.rsi,'ATR比例':last.atr_pct,
        '成交量比':last.volume_ratio,'20日估計均成交金額USD':last.turnover_20,'均線乖離':last.extension,
        '相對SPY20日報酬差':last.relative_return_20,'持股資料':'未提供；持有建議僅為情境分析',
        '股數':np.nan,'平均成本USD':np.nan,'未實現損益USD':np.nan,'成本報酬率':np.nan}
    if holding:
        qty,cost=holding['股數'],holding['平均成本']
        row.update({'持股資料':'已提供','股數':qty,'平均成本USD':cost,'未實現損益USD':qty*(close-cost),'成本報酬率':close/cost-1})
    return row


POSITIVE_WORDS=('beat','beats','upgrade','upgraded','growth','surge','record','profit','profits','raise','raises',
                'strong','outperform','buyback','dividend','approval','win','wins')
NEGATIVE_WORDS=('miss','misses','downgrade','downgraded','decline','drop','loss','losses','cut','cuts','warning',
                'lawsuit','probe','recall','fraud','weak','underperform')


def news_sentiment(news,ticker,asof,cfg):
    base={'新聞情緒':'資料不足','新聞情緒分數':np.nan,'新聞篇數':0,'新聞正面篇數':0,
          '新聞負面篇數':0,'新聞中性篇數':0,'新聞資料狀態':'無可用新聞',
          '新聞情緒用途':'研究參考；不參與交易建議或AI特徵'}
    if news is None or news.empty:return base,pd.DataFrame()
    f=news.copy();scores=[];labels=[]
    for rec in f.to_dict('records'):
        text=(str(rec.get('標題') or '')+' '+str(rec.get('摘要') or '')).lower()
        pos=sum(text.count(w) for w in POSITIVE_WORDS);neg=sum(text.count(w) for w in NEGATIVE_WORDS)
        score=(pos-neg)/max(1,pos+neg);scores.append(float(score));labels.append('正面' if score>0 else ('負面' if score<0 else '中性'))
    f['新聞情緒分數']=scores;f['新聞情緒標籤']=labels
    def related(value):
        if isinstance(value,(list,tuple,set)):values=[str(v).upper() for v in value]
        elif value is None or (isinstance(value,float) and pd.isna(value)):values=[]
        else:values=[s.strip().upper() for s in str(value).replace('[','').replace(']','').replace("'",'').split(',') if s.strip()]
        return not values or ticker.upper() in values
    f['與個股直接相關']=f.get('相關代號',pd.Series([None]*len(f),index=f.index)).map(related)
    used=f.loc[f['與個股直接相關']].copy()
    if used.empty:used=f
    score=float(used['新聞情緒分數'].mean());minimum=int(cfg['sentiment']['news_min_articles'])
    used_labels=used['新聞情緒標籤'].tolist()
    label='資料不足' if len(used)<minimum else ('偏多' if score>=cfg['sentiment']['news_positive_threshold'] else
          ('偏空' if score<=cfg['sentiment']['news_negative_threshold'] else '中性'))
    base.update({'新聞情緒':label,'新聞情緒分數':score,'新聞篇數':len(used),'新聞取得總篇數':len(f),
                 '新聞正面篇數':used_labels.count('正面'),'新聞負面篇數':used_labels.count('負面'),
                 '新聞中性篇數':used_labels.count('中性'),
                 '新聞資料狀態':'可用；僅標題／摘要字典基準，尚未證明對報酬有預測力'})
    return base,f


def market_sentiment(context,cfg):
    result={'市場情緒':'資料不足','市場情緒分數':np.nan,'SPY近20日報酬率':np.nan,
            'SPY相對60日線':np.nan,'VIX':np.nan,'VIX近20日變化率':np.nan,
            '市場情緒用途':'研究參考；不參與交易建議或AI特徵'}
    if not context:return result
    benchmark=context.get('benchmark');vix=context.get('vix')
    if benchmark is None or benchmark.empty:return {**result,**context.get('meta',{})}
    close=benchmark.Close.dropna();ret20=close.pct_change(20,fill_method=None).iloc[-1] if len(close)>20 else np.nan
    ma60=close.rolling(60).mean().iloc[-1] if len(close)>=60 else np.nan
    distance=close.iloc[-1]/ma60-1 if pd.notna(ma60) else np.nan
    vix_now=vix_change=np.nan
    if vix is not None and not vix.empty:
        vc=vix.Close.dropna();vix_now=float(vc.iloc[-1]) if len(vc) else np.nan
        if len(vc)>20:vix_change=vc.iloc[-1]/vc.iloc[-21]-1
    score=50
    if pd.notna(ret20):score+=15 if ret20>0 else -15
    if pd.notna(distance):score+=15 if distance>0 else -15
    if pd.notna(vix_now):score+=10 if vix_now<20 else (-15 if vix_now>=30 else 0)
    if pd.notna(vix_change) and vix_change>=0.20:score-=10
    score=float(np.clip(score,0,100));label='偏多' if score>=65 else ('偏空' if score<=35 else '中性')
    result.update({'市場情緒':label,'市場情緒分數':score,'SPY近20日報酬率':ret20,
                   'SPY相對60日線':distance,'VIX':vix_now,'VIX近20日變化率':vix_change})
    result.update(context.get('meta',{}));return result


def compare_strategies(raw,adjusted,x,cfg,benchmark=None):
    if adjusted is None:return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    unknown=adjusted.Close.isna()&~raw.Volume.eq(0)
    bench=benchmark.reindex(raw.index) if benchmark is not None else None
    if bench is not None:unknown|=bench.Close.isna()
    # A missing latest session blocks current advice, but history ending before
    # that gap remains usable. Report the actual historical end explicitly.
    complete=np.flatnonzero((~unknown & adjusted.Close.notna()).to_numpy())
    if len(complete)==0:return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    end=int(complete[-1])+1
    raw,adjusted,x,unknown=raw.iloc[:end],adjusted.iloc[:end],x.iloc[:end],unknown.iloc[:end]
    if bench is not None:bench=bench.iloc[:end]
    ready=np.flatnonzero(x.trend_ready.to_numpy())
    if unknown.any():ready=ready[ready>int(np.flatnonzero(unknown.to_numpy())[-1])]
    if len(ready)==0:return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    start=max(int(ready[0])+1,len(raw)-cfg['backtest']['evaluation_sessions'])
    defs={'buy_hold':(pd.Series(True,index=x.index),pd.Series(False,index=x.index)),
          'trend':(x.technical_entry,x.trend_down),'trend_flow_market':(x.entry,x.exit)}
    summaries=[];trades=[];curves=[]
    for name,(ent,ext) in defs.items():
        s,t,e=simulate(raw,adjusted,ent,ext,cfg,name,start)
        if s:summaries.append(s)
        if not t.empty:trades.append(t)
        if not e.empty:curves.append(e)
    if bench is not None:
        s,t,e=simulate(bench,bench,pd.Series(True,index=x.index),pd.Series(False,index=x.index),cfg,'buy_hold',start)
        if s:s['strategy']='benchmark_buy_hold';s['benchmark']=cfg['data']['benchmark'];summaries.append(s)
        for frame,collection in [(t,trades),(e,curves)]:
            if not frame.empty:frame['strategy']='benchmark_buy_hold';collection.append(frame)
    return pd.DataFrame(summaries),pd.concat(trades,ignore_index=True) if trades else pd.DataFrame(),pd.concat(curves,ignore_index=True) if curves else pd.DataFrame()


def technical(prices, rules):
    c,h,l,v = (prices[k] for k in ['Close','High','Low','Volume'])
    x = pd.DataFrame(index=prices.index)
    for d in [1,5,20,60]: x[f'return_{d}'] = c.pct_change(d, fill_method=None)
    periods={20,60,rules['short_fast_ma'],rules['short_slow_ma'],rules['fast_ma'],rules['slow_ma'],
             rules['long_fast_ma'],rules['long_slow_ma']}
    for d in sorted(periods):
        x[f'ma_{d}'] = c.rolling(d).mean()
        x[f'distance_{d}'] = c/x[f'ma_{d}']-1
    delta=c.diff()
    gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rsi=100-100/(1+gain/loss.replace(0,np.nan))
    x['rsi'] = rsi.mask((loss==0)&(gain>0),100).mask((loss==0)&(gain==0),50)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    x['atr']=tr.rolling(14).mean(); x['atr_pct']=x.atr/c
    x['volume_ratio']=v/v.rolling(20).mean().replace(0,np.nan)
    x['volatility_20']=x.return_1.rolling(20).std()
    x['range_position']=(c-l.rolling(20).min())/(h.rolling(20).max()-l.rolling(20).min()).replace(0,np.nan)
    x['support']=l.shift().rolling(20).min(); x['resistance']=h.shift().rolling(20).max()
    def add_trend(prefix, fast_days, slow_days, slope_days):
        fast=x[f'ma_{fast_days}'];slow=x[f'ma_{slow_days}']
        x[f'{prefix}_trend_ready']=slow.shift(slope_days).notna()&slow.notna()&c.notna()
        x[f'{prefix}_trend_up']=(c>fast)&(fast>slow)&(slow>slow.shift(slope_days))
        x[f'{prefix}_trend_down']=(c<slow)|(fast<slow)
    add_trend('short',rules['short_fast_ma'],rules['short_slow_ma'],rules['short_slope_days'])
    add_trend('medium',rules['fast_ma'],rules['slow_ma'],rules['slope_days'])
    add_trend('long',rules['long_fast_ma'],rules['long_slow_ma'],rules['long_slope_days'])
    # Existing entry, exit, AI and backtest behavior remains tied to the medium trend.
    x['trend_ready']=x['medium_trend_ready'];x['trend_up']=x['medium_trend_up'];x['trend_down']=x['medium_trend_down']
    fast=x[f'ma_{rules["fast_ma"]}']
    x['extension']=c/fast-1
    return x.replace([np.inf,-np.inf],np.nan)

def simulate(raw, adjusted, entry, exit_signal, cfg, strategy, start_index):
    """Single-symbol total-return research simulation, fractional adjusted units.

    Prior-close signals -> next open. Estimated fees/slippage are charged. Same-bar
    stop and profit uses stop first; locked-price/zero-volume bars do not fill.
    Not an executable order simulator; adjusted units assume reinvested payouts.
    """
    b=cfg['backtest']; p=adjusted
    cash=float(b['initial_cash']); units=0.; cost=0.; entry_price=0.; entry_i=None
    trades=[]; equity=[]; skipped=0
    if p is None: return {},pd.DataFrame(),pd.DataFrame()
    if not raw.index.equals(p.index): raise ValueError('回測原始及還原日期未對齊')
    if not raw.index.equals(entry.index) or not raw.index.equals(exit_signal.index): raise ValueError('回測訊號日期未對齊')
    start_index=max(1,int(start_index))
    mark=float(p.Close.iloc[:start_index].dropna().iloc[-1])
    for i in range(start_index,len(p)):
        date=p.index[i]; row=p.iloc[i]; original=raw.iloc[i]
        can_fill=original.Volume>0 and original.High>original.Low
        closed=False
        def close_position(price, reason):
            nonlocal cash,units,closed
            effective=price*(1-b['slippage'])
            proceeds=units*effective*(1-b['fee_rate']-b['sell_fee_rate'])
            cash+=proceeds
            trades.append({'strategy':strategy,'entry_date':str(p.index[entry_i].date()),
                'exit_date':str(date.date()),'entry_adjusted':entry_price,'exit_adjusted':effective,
                'holding_sessions':i-entry_i+1,'return_after_cost':proceeds/cost-1,
                'pnl_after_cost':proceeds-cost,'reason':reason})
            units=0.; closed=True
        if units>0 and can_fill:
            if bool(exit_signal.iloc[i-1]): close_position(row.Open,'前日退出訊號')
            elif strategy!='buy_hold' and i-entry_i>=b['max_holding_sessions']:
                close_position(row.Open,'持有期上限')
        if units==0 and not closed and bool(entry.iloc[i-1]) and can_fill:
            entry_price=row.Open*(1+b['slippage'])
            cost=cash; units=cash/(entry_price*(1+b['fee_rate'])); cash=0.; entry_i=i
        elif not can_fill and (units>0 or bool(entry.iloc[i-1])): skipped+=1
        if units>0 and can_fill and strategy!='buy_hold':
            stop=entry_price*(1-b['stop_loss']); take=entry_price*(1+b['take_profit'])
            # Opening gaps have known precedence. For ambiguous intraday order, stop wins.
            if i>entry_i and row.Open<=stop: close_position(row.Open,'跳空停損')
            elif i>entry_i and row.Open>=take: close_position(row.Open,'跳空停利')
            elif row.Low<=stop: close_position(stop,'停損；同日雙觸價採停損優先')
            elif row.High>=take: close_position(take,'停利')
        if pd.notna(row.Close): mark=float(row.Close)
        # A no-trade day is marked at the last available close only for wealth;
        # it remains missing in feature/label creation and cannot fill an order.
        value=cash+units*mark*(1-b['slippage'])*(1-b['fee_rate']-b['sell_fee_rate'])
        equity.append({'date':date,'strategy':strategy,'equity':value,'in_position':units>0})
    e=pd.DataFrame(equity)
    if e.empty: return {},pd.DataFrame(trades),e
    wealth=pd.Series([b['initial_cash']]+e.equity.tolist())
    dd=wealth/wealth.cummax()-1
    elapsed=max(1,(pd.Timestamp(e.date.iloc[-1])-pd.Timestamp(e.date.iloc[0])).days)
    total=wealth.iloc[-1]/wealth.iloc[0]-1
    result={'strategy':strategy,'start':str(pd.Timestamp(e.date.iloc[0]).date()),
        'end':str(pd.Timestamp(e.date.iloc[-1]).date()),'return_after_cost':total,
        'annualized_return':(1+total)**(365.25/elapsed)-1 if elapsed>=365 else np.nan,
        'max_drawdown':dd.min(),'closed_trades':len(trades),
        'win_rate':np.mean([t['pnl_after_cost']>0 for t in trades]) if trades else np.nan,
        'open_position':bool(units>0),'blocked_sessions':skipped,
        'note':'USD單檔還原價研究；分數股/股息再投資，成本為估計而非券商實價；非WHID歷史選股及投資組合績效'}
    return result,pd.DataFrame(trades),e
