"""Transparent research rules and a single-stock, long-only daily simulator."""
from __future__ import annotations

import numpy as np
import pandas as pd


def whid_freshness(provenance, fallback_date, max_age_days):
    """Validate a WHID report against its run date, never the last market bar.

    WHID filenames use the Taipei calendar date. Comparing that date with a
    previous trading session incorrectly rejects reports after weekends,
    holidays, or before the US close.
    """
    whid_day=provenance.get('WHID評估日期')
    reference=provenance.get('第二階段執行日期台北') or fallback_date
    if not whid_day:
        return ('手動' in provenance.get('來源',''),None,str(reference),'手動名單未驗證WHID日期' if '手動' in provenance.get('來源','') else '無法確認WHID評估日期')
    try:
        whid=pd.Timestamp(whid_day).normalize();checked=pd.Timestamp(reference).normalize()
        if pd.isna(whid) or pd.isna(checked):raise ValueError
        age=(checked-whid).days
    except (TypeError,ValueError,OverflowError):
        return False,None,str(reference),'WHID評估日期無法解析'
    fresh=0<=age<=max_age_days
    return fresh,age,str(checked.date()),None if fresh else ('WHID評估日期晚於本次執行日期' if age<0 else 'WHID候選名單已超過有效天數')


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


def chip_features(records, raw, rules):
    out=pd.DataFrame(index=raw.index)
    out['foreign_net']=np.nan; out['trust_net']=np.nan;out['dealer_net']=np.nan
    if not records.empty:
        required={'date','name','buy','sell'}
        if not required.issubset(records): raise ValueError('法人籌碼欄位不足')
        f=records.copy()
        f['date']=pd.to_datetime(f.date).dt.normalize()
        if f.duplicated(['date','name']).any(): raise ValueError('法人籌碼重複，避免重複計入')
        f['net']=pd.to_numeric(f.buy,errors='coerce')-pd.to_numeric(f.sell,errors='coerce')
        wide=f.loc[f.name.isin(['Foreign_Investor','Investment_Trust'])].pivot(
            index='date',columns='name',values='net').reindex(raw.index)
        for name,col in [('Foreign_Investor','foreign_net'),('Investment_Trust','trust_net')]:
            if name in wide: out[col]=wide[name]
        dealers=f.loc[f.name.astype(str).str.startswith('Dealer')].groupby('date').net.sum()
        if not dealers.empty:out['dealer_net']=dealers.reindex(raw.index)
    n,lag=rules['chip_days'],rules['chip_lag_sessions']
    # Full n-session coverage is mandatory. A missing row is not zero net buying.
    for col in ['foreign_net','trust_net','dealer_net']:
        out[col+'_sum']=out[col].rolling(n,min_periods=n).sum().shift(lag)
    valid=out[['foreign_net_sum','trust_net_sum']].notna().all(axis=1)
    out['chip_ready']=valid
    out['chip_positive']=valid & (out.foreign_net_sum>0) & (out.trust_net_sum>0)
    out['chip_negative']=valid & (out.foreign_net_sum<0) & (out.trust_net_sum<0)
    denom=raw.Volume.rolling(n).sum().shift(lag).replace(0,np.nan)
    out['institutional_net_ratio']=(out.foreign_net_sum+out.trust_net_sum)/denom
    out['chip_date']=pd.Series(raw.index,index=raw.index).shift(lag).where(valid)
    return out


def margin_features(frame,raw,rules):
    out=pd.DataFrame(index=raw.index)
    for col in ['margin_balance','short_balance','margin_growth','short_growth','short_margin_ratio']:
        out[col]=np.nan
    out['margin_ready']=False;out['margin_overheat']=False
    required={'date','MarginPurchaseTodayBalance','ShortSaleTodayBalance'}
    if frame is None or frame.empty or not required.issubset(frame):return out
    f=frame.copy();f['date']=pd.to_datetime(f.date).dt.normalize()
    if f.duplicated('date').any():raise ValueError('融資融券日期重複')
    f=f.set_index('date').reindex(raw.index)
    margin=pd.to_numeric(f.MarginPurchaseTodayBalance,errors='coerce')
    short=pd.to_numeric(f.ShortSaleTodayBalance,errors='coerce')
    n,lag=rules['chip_days'],rules['chip_lag_sessions']
    out['margin_balance']=margin.shift(lag);out['short_balance']=short.shift(lag)
    out['margin_growth']=margin.shift(lag)/margin.shift(lag+n).replace(0,np.nan)-1
    out['short_growth']=short.shift(lag)/short.shift(lag+n).replace(0,np.nan)-1
    out['short_margin_ratio']=short.shift(lag)/margin.shift(lag).replace(0,np.nan)
    price_growth=raw.Close.shift(lag)/raw.Close.shift(lag+n)-1
    out['margin_ready']=out[['margin_balance','short_balance','margin_growth']].notna().all(axis=1)
    out['margin_overheat']=out.margin_ready&(out.margin_growth>=rules['margin_overheat_growth'])&(price_growth>0)
    return out.replace([np.inf,-np.inf],np.nan)


def lending_features(frame,raw,rules):
    out=pd.DataFrame(index=raw.index)
    out['lending_volume_sum']=np.nan;out['lending_volume_ratio']=np.nan
    out['lending_ready']=False;out['lending_high']=False
    required={'date','volume'}
    if frame is None or frame.empty or not required.issubset(frame):return out
    f=frame.copy();f['date']=pd.to_datetime(f.date).dt.normalize()
    daily=pd.to_numeric(f.volume,errors='coerce').groupby(f.date).sum().reindex(raw.index,fill_value=0)
    n,lag=rules['chip_days'],rules['chip_lag_sessions']
    out['lending_volume_sum']=daily.rolling(n,min_periods=n).sum().shift(lag)
    market_volume=raw.Volume.rolling(n,min_periods=n).sum().shift(lag).replace(0,np.nan)
    out['lending_volume_ratio']=out.lending_volume_sum/market_volume
    out['lending_ready']=out.lending_volume_ratio.notna()
    out['lending_high']=out.lending_ready&(out.lending_volume_ratio>=rules['lending_volume_ratio_high'])
    return out.replace([np.inf,-np.inf],np.nan)


def branch_features(frame,raw,rules):
    out=pd.DataFrame(index=raw.index)
    for col in ['branch_buy_concentration','branch_sell_concentration','branch_pressure','branch_buy_avg_price']:
        out[col]=np.nan
    out['branch_ready']=False;out['branch_positive']=False;out['branch_negative']=False
    required={'date','securities_trader_id','price','buy','sell'}
    if frame is None or frame.empty or not required.issubset(frame):return out
    f=frame.copy();f['date']=pd.to_datetime(f.date).dt.normalize()
    for col in ['price','buy','sell']:f[col]=pd.to_numeric(f[col],errors='coerce')
    requested=[pd.Timestamp(d).normalize() for d in frame.attrs.get('requested_dates',[])]
    available={pd.Timestamp(d).normalize() for d in f.date.dropna().unique()}
    if requested and not set(requested).issubset(available):return out
    by=f.groupby('securities_trader_id',as_index=False).agg(buy=('buy','sum'),sell=('sell','sum'))
    by['net']=by.buy-by.sell
    n=max(1,int(rules['branch_top_n']))
    top_ids=by.nlargest(n,'net').loc[lambda z:z.net>0,'securities_trader_id']
    bottom=by.nsmallest(n,'net').loc[lambda z:z.net<0]
    top=by.loc[by.securities_trader_id.isin(top_ids)]
    top_buy=float(top.net.sum());top_sell=float((-bottom.net).sum())
    dates=requested or sorted(f.date.dropna().unique())
    total_volume=float(raw.Volume.reindex(dates).sum())
    if not np.isfinite(total_volume) or total_volume<=0:return out
    top_rows=f.loc[f.securities_trader_id.isin(top_ids)&(f.buy>0)&(f.price>0)]
    avg=float((top_rows.price*top_rows.buy).sum()/top_rows.buy.sum()) if top_rows.buy.sum()>0 else np.nan
    idx=out.index[-1]
    out.loc[idx,'branch_buy_concentration']=top_buy/total_volume
    out.loc[idx,'branch_sell_concentration']=top_sell/total_volume
    out.loc[idx,'branch_pressure']=(top_buy-top_sell)/total_volume
    out.loc[idx,'branch_buy_avg_price']=avg
    out.loc[idx,'branch_ready']=True
    out.loc[idx,'branch_positive']=out.loc[idx,'branch_pressure']>=rules['branch_pressure_threshold']
    out.loc[idx,'branch_negative']=out.loc[idx,'branch_pressure']<=-rules['branch_pressure_threshold']
    return out


def chip_kline_features(records,raw,rules):
    base=chip_features(records,raw,rules)
    enabled=rules.get('chip_kline_enabled',True)
    margin=margin_features(records.attrs.get('margin') if enabled else None,raw,rules)
    lending=lending_features(records.attrs.get('lending') if enabled else None,raw,rules)
    branches=branch_features(records.attrs.get('branches') if enabled else None,raw,rules)
    out=base.join(margin).join(lending).join(branches)
    score=pd.Series(50.,index=out.index)
    score+=np.where(out.foreign_net_sum>0,15,np.where(out.foreign_net_sum<0,-15,0))
    score+=np.where(out.trust_net_sum>0,15,np.where(out.trust_net_sum<0,-15,0))
    score+=np.where(out.dealer_net_sum>0,5,np.where(out.dealer_net_sum<0,-5,0))
    score+=np.where(out.branch_positive,10,np.where(out.branch_negative,-10,0))
    score-=np.where(out.margin_overheat,10,0);score-=np.where(out.lending_high,5,0)
    ready=out.chip_ready|out.margin_ready|out.lending_ready|out.branch_ready
    out['chip_kline_score']=score.clip(0,100).where(ready&enabled)
    return out


def signals(raw, adjusted, records, cfg):
    r=cfg['rules']
    p=adjusted if adjusted is not None else raw
    x=technical(p,r).join(chip_kline_features(records,raw,r))
    money=raw['Turnover'] if 'Turnover' in raw else raw.Close*raw.Volume
    x['turnover_20']=money.rolling(20).mean()
    x['risk_ok']=(x.atr_pct<=r['max_atr_pct'])&(x.turnover_20>=r['min_daily_turnover'])
    x['technical_entry']=x.trend_ready & x.trend_up & x.risk_ok & (x.volume_ratio>=r['volume_ratio_min']) & (x.extension<=r['max_extension'])
    x['entry']=x.technical_entry & (x.chip_positive if r['require_chip_for_entry'] else ~x.chip_negative)
    x['exit']=x.trend_ready & (x.trend_down | x.chip_negative)
    if adjusted is None:
        x['entry']=False; x['technical_entry']=False
    return x


def latest_assessment(ticker, raw, adjusted, x, cfg, provenance, asof, holding=None):
    r=cfg['rules']; last=x.iloc[-1]; close=float(raw.Close.iloc[-1]); day=raw.index[-1]
    notes=[]
    age=(pd.Timestamp(asof).normalize()-day).days
    old=age>cfg['data']['max_price_age_days'] or age<0
    if old: notes.append('行情過期或日期異常')
    if adjusted is None: notes.append('無完整還原價格，暫停進場判斷／AI／回測')
    ready=bool(last.trend_ready) and adjusted is not None and not old
    def trend_text(prefix):
        if not bool(last[f'{prefix}_trend_ready']): return '資料不足'
        return '多頭' if bool(last[f'{prefix}_trend_up']) else ('轉弱' if bool(last[f'{prefix}_trend_down']) else '整理')
    short_trend=trend_text('short');trend=trend_text('medium');long_trend=trend_text('long')
    chip='資料不足' if not last.chip_ready else ('同步買超' if last.chip_positive else ('同步賣超' if last.chip_negative else '分歧／中性'))
    coverage=[]
    if bool(last.chip_ready):coverage.append('法人')
    if bool(last.margin_ready):coverage.append('融資融券')
    if bool(last.lending_ready):coverage.append('借券')
    if bool(last.branch_ready):coverage.append('券商分點')
    if not coverage:chip_kline='資料不足'
    elif (bool(last.chip_positive) and bool(last.branch_negative)) or (bool(last.chip_negative) and bool(last.branch_positive)):
        chip_kline='法人／分點方向分歧'
    elif bool(last.chip_positive) and bool(last.margin_overheat):chip_kline='法人偏多／融資過熱'
    elif bool(last.chip_positive) and bool(last.branch_positive):chip_kline='偏多共振'
    elif bool(last.chip_negative) and bool(last.branch_negative):chip_kline='偏空共振'
    elif bool(last.chip_positive) and not bool(last.branch_negative):chip_kline='法人偏多'
    elif bool(last.chip_negative) and not bool(last.branch_positive):chip_kline='法人偏空'
    elif bool(last.branch_positive):chip_kline='分點偏多／法人未同步'
    elif bool(last.branch_negative):chip_kline='分點偏空／法人未同步'
    elif bool(last.margin_overheat):chip_kline='散戶槓桿偏熱'
    else:chip_kline='籌碼分歧／中性'
    risks=[]
    if pd.isna(last.turnover_20) or pd.isna(last.atr_pct): risks.append('風險資料不足')
    else:
        if last.turnover_20<r['min_daily_turnover']: risks.append('流動性偏低')
        if last.atr_pct>r['max_atr_pct']: risks.append('波動偏高')
    if pd.notna(last.extension) and last.extension>r['max_extension']: risks.append('偏離均線過大')
    whid_day=provenance.get('WHID評估日期')
    whid_fresh,whid_age,whid_reference,whid_reason=whid_freshness(provenance,asof,cfg['candidates']['max_whid_age_days'])
    whid_stale=not whid_fresh
    if whid_stale: notes.append((whid_reason or 'WHID候選名單日期異常')+'，進場前需重評')
    entry='等待'
    if not ready: entry='資料不足／暫不判斷'
    elif whid_stale: entry='先更新WHID評估'
    elif cfg['rules']['require_chip_for_entry'] and not last.chip_ready: entry='等待籌碼資料'
    elif bool(last.entry): entry='符合研究進場條件'
    held='若持有：續抱觀察'
    if not ready: held='若持有：資料不足，人工確認'
    elif bool(last.exit): held='若持有：符合退出條件'
    elif risks: held='若持有：檢查風險／部位'
    factor=float(adjusted.Close.iloc[-1]/close) if adjusted is not None else 1.0
    stop=close-r['atr_stop_multiple']*float(last.atr)/factor if pd.notna(last.atr) else np.nan
    row={'代號':ticker,'訊號日期':str(day.date()),'現價':close,'趨勢':trend,
         '短期趨勢':short_trend,'中期趨勢':trend,'長期趨勢':long_trend,'籌碼':chip,
         '籌碼K線':chip_kline,'籌碼K線分數':last.chip_kline_score,
         '籌碼K線資料涵蓋':'+'.join(coverage) or '無','籌碼K線用途':'研究參考；尚未參與交易建議',
         '風險':'／'.join(risks) or '未觸發風險門檻','未持有建議':entry,'持有情境建議':held,
         'ATR風險參考價':stop if stop>0 else np.nan,
         '支撐參考':last.support/factor,'壓力參考':last.resistance/factor,
         '資料狀態':'；'.join(notes) or '可用；研究規則未驗證獲利',
         'WHID評估日期':whid_day,'WHID名單日齡':whid_age,'WHID日期比較基準':whid_reference,
         'WHID日期判定':whid_reason or '有效','籌碼採用日期':last.chip_date,
         '外資近N日淨買股數':last.foreign_net_sum,'投信近N日淨買股數':last.trust_net_sum,
         '自營商近N日淨買股數':last.dealer_net_sum,
         '融資餘額':last.margin_balance,'融券餘額':last.short_balance,
         '融資餘額近N日變化率':last.margin_growth,'融券餘額近N日變化率':last.short_growth,
         '券資比':last.short_margin_ratio,'融資過熱':bool(last.margin_overheat) if bool(last.margin_ready) else np.nan,
         '借券成交近N日股數':last.lending_volume_sum,'借券成交近N日占量':last.lending_volume_ratio,
         '借券成交偏高':bool(last.lending_high) if bool(last.lending_ready) else np.nan,
         '分點買超集中度':last.branch_buy_concentration,
         '分點賣超集中度':last.branch_sell_concentration,'分點集中壓力':last.branch_pressure,
         '主要買超分點買進均價':last.branch_buy_avg_price,
         'RSI':last.rsi,'ATR比例':last.atr_pct,'成交量比':last.volume_ratio,
         '20日均成交金額':last.turnover_20,'均線乖離':last.extension,
         '持股資料':'未提供；持有建議僅為情境分析','股數':np.nan,'平均成本':np.nan,
         '未實現損益':np.nan,'成本報酬率':np.nan}
    if holding:
        qty,cost=holding['股數'],holding['平均成本']
        row.update({'持股資料':'已提供','股數':qty,'平均成本':cost,
                    '未實現損益':qty*(close-cost),'成本報酬率':close/cost-1})
    return row


def simulate(raw, adjusted, entry, exit_signal, cfg, strategy, start_index):
    """Single-symbol total-return research simulation, fractional adjusted units.

    Prior-close signals -> next open. Fees/tax/slippage are charged. Same-bar
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
            proceeds=units*effective*(1-b['fee_rate']-b['sell_tax_rate'])
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
        value=cash+units*mark*(1-b['slippage'])*(1-b['fee_rate']-b['sell_tax_rate'])
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
        'note':'單檔還原價研究模擬；分數股/股息再投資假設，期末扣估計退出成本；非WHID歷史選股績效；非投資組合'}
    return result,pd.DataFrame(trades),e


def compare_strategies(raw, adjusted, x, cfg):
    if adjusted is None: return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    ready=np.flatnonzero(x.trend_ready.to_numpy())
    if len(ready)==0: return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    # Never simulate an open position across an unknown traded-session price.
    # Known zero-volume/no-trade dates can be carried for mark-to-market only.
    unknown=adjusted.Close.isna() & (raw.Volume>0)
    if unknown.any():
        after=int(np.flatnonzero(unknown.to_numpy())[-1])
        ready=ready[ready>after]
        if len(ready)==0:return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    start=max(int(ready[0])+1,len(raw)-cfg['backtest']['evaluation_sessions'])
    definitions={
        'buy_hold':(pd.Series(True,index=x.index),pd.Series(False,index=x.index)),
        'trend':(x.technical_entry,x.trend_down),
        'trend_chip':(x.entry,x.exit),
    }
    summaries=[]; trades=[]; curves=[]
    for name,(entry,exit_signal) in definitions.items():
        s,t,e=simulate(raw,adjusted,entry,exit_signal,cfg,name,start)
        if s: summaries.append(s)
        if not t.empty: trades.append(t)
        if not e.empty: curves.append(e)
    return pd.DataFrame(summaries),pd.concat(trades,ignore_index=True) if trades else pd.DataFrame(),pd.concat(curves,ignore_index=True) if curves else pd.DataFrame()
