from __future__ import annotations
import io, json, zipfile, hashlib
from datetime import datetime, date, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests

UTC=ZoneInfo('UTC'); CT=ZoneInfo('America/Chicago'); ET=ZoneInfo('America/New_York')
OUT=Path('quick_btc_check/results'); OUT.mkdir(parents=True, exist_ok=True)
MONTHS=[f'2026-{m:02d}' for m in range(1,7)]
BASE='https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m'
FEE=0.0004

def uts(d,hh,mm,z): return pd.Timestamp(datetime.combine(d,time(hh,mm),tzinfo=z).astimezone(UTC))

def load():
    fs=[]; mani=[]
    for m in MONTHS:
        r=requests.get(f'{BASE}/BTCUSDT-1m-{m}.zip',timeout=90); r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            raw=z.read([n for n in z.namelist() if n.endswith('.csv')][0])
        x=pd.read_csv(io.BytesIO(raw),header=None,low_memory=False)
        if pd.to_numeric(pd.Series([x.iloc[0,0]]),errors='coerce').isna().iloc[0]: x=x.iloc[1:]
        x=x.iloc[:,:6]; x.columns=['open_time','open','high','low','close','volume']
        for c in x.columns: x[c]=pd.to_numeric(x[c],errors='coerce')
        x=x.dropna()
        unit='us' if float(x.open_time.max())>1e14 else 'ms'
        x['ts']=pd.to_datetime(x.open_time.astype('int64'),unit=unit,utc=True)
        x=x.set_index('ts')[['open','high','low','close','volume']].astype(float)
        fs.append(x); mani.append({'month':m,'rows':len(x),'zip_sha256':hashlib.sha256(r.content).hexdigest()})
    return pd.concat(fs).sort_index(), mani

def first_bos(df,start,cutoff,anchor):
    w=df.loc[(df.index>=start)&(df.index<cutoff)]
    for ts,row in w.iterrows():
        if row.low<=anchor<=row.high: return None, 'ANCHOR_FIRST'
        pos=df.index.get_indexer([ts])[0]
        if pos<3: continue
        p=df.iloc[pos-3:pos]
        if row.close<anchor and row.close>p.high.max(): return (ts,'LONG'),'SIGNAL'
        if row.close>anchor and row.close<p.low.min(): return (ts,'SHORT'),'SIGNAL'
    return None,'NONE'

def simulate_stop(df,d,anchor,stop_bps):
    start=uts(d,9,35,ET); cutoff=uts(d,10,30,ET)
    sig,status=first_bos(df,start,cutoff,anchor)
    if not sig: return None
    ts,side=sig; pos=df.index.get_indexer([ts])[0]
    if pos+1>=len(df): return None
    ets=df.index[pos+1]
    if ets>=cutoff: return None
    entry=float(df.iloc[pos+1].open); sd=entry*stop_bps/10000.0
    room=(anchor-entry) if side=='LONG' else (entry-anchor)
    if room<sd: return None
    target=anchor-1 if side=='LONG' else anchor+1
    stop=entry-sd if side=='LONG' else entry+sd
    exit_px=float(df.at[cutoff,'open']); reason='CUTOFF'; xt=cutoff
    for t,b in df.loc[(df.index>=ets)&(df.index<cutoff)].iterrows():
        hs=(b.low<=stop) if side=='LONG' else (b.high>=stop)
        ht=(b.high>=target) if side=='LONG' else (b.low<=target)
        if hs: exit_px=stop; reason='STOP'; xt=t; break
        if ht: exit_px=target; reason='TARGET'; xt=t; break
    gross=((exit_px-entry) if side=='LONG' else (entry-exit_px))/sd
    fee=(FEE*(entry+exit_px))/sd
    return {'date':str(d),'bps':stop_bps,'side':side,'entry':entry,'anchor':anchor,'exit':exit_px,'reason':reason,'gross_r':gross,'fee_r':fee,'net_r':gross-fee}

def mae_to_target(df,d,anchor):
    start=uts(d,9,35,ET); cutoff=uts(d,10,30,ET)
    sig,status=first_bos(df,start,cutoff,anchor)
    if not sig: return None
    ts,side=sig; pos=df.index.get_indexer([ts])[0]
    if pos+1>=len(df): return None
    ets=df.index[pos+1]; entry=float(df.iloc[pos+1].open)
    target=anchor-1 if side=='LONG' else anchor+1
    bars=df.loc[(df.index>=ets)&(df.index<cutoff)]
    worst=0.0
    for t,b in bars.iterrows():
        if side=='LONG':
            worst=max(worst,max(0.0,entry-float(b.low)))
            if b.high>=target: return worst/entry*10000.0
        else:
            worst=max(worst,max(0.0,float(b.high)-entry))
            if b.low<=target: return worst/entry*10000.0
    return None

def main():
    df,mani=load(); rows=[]; stoprows=[]; maes=[]
    first=df.index.min().tz_convert(ET).date()+timedelta(days=1); last=df.index.max().tz_convert(ET).date()
    d=first
    while d<=last:
        if d.weekday()<5:
            a=uts(d-timedelta(days=1),17,0,CT); o=uts(d,9,30,ET); s=uts(d,9,35,ET); c1030=uts(d,10,30,ET); c1100=uts(d,11,0,ET)
            if all(t in df.index for t in [a,o,s,c1030,c1100]):
                anchor=float(df.at[a,'open'])
                pre=df.loc[(df.index>=o)&(df.index<s)]
                pre_touch=bool(((pre.low<=anchor)&(pre.high>=anchor)).any())
                w1030=df.loc[(df.index>=s)&(df.index<c1030)]
                w1100=df.loc[(df.index>=s)&(df.index<c1100)]
                t1030=bool(((w1030.low<=anchor)&(w1030.high>=anchor)).any())
                t1100=bool(((w1100.low<=anchor)&(w1100.high>=anchor)).any())
                rows.append({'date':str(d),'pre_935_touch':pre_touch,'touch_by_1030':t1030,'touch_by_1100':t1100})
                if not pre_touch:
                    m=mae_to_target(df,d,anchor)
                    if m is not None: maes.append(m)
                    for bps in [10,20,30,50]:
                        r=simulate_stop(df,d,anchor,bps)
                        if r: stoprows.append(r)
        d+=timedelta(days=1)
    fp=pd.DataFrame(rows); st=pd.DataFrame(stoprows)
    eligible=len(fp); cond=fp[~fp.pre_935_touch]
    summary={'eligible_weekdays':eligible,'pre_935_touch_rate':float(fp.pre_935_touch.mean()),'conditional_sessions':len(cond),
             'fair_price_touch_rate_935_1030_given_not_pre_touched':float(cond.touch_by_1030.mean()),
             'fair_price_touch_rate_935_1100_given_not_pre_touched':float(cond.touch_by_1100.mean()),
             'mae_bps_before_successful_first_bos_return':{'n':len(maes),'median':float(np.median(maes)) if maes else None,'p75':float(np.percentile(maes,75)) if maes else None,'p90':float(np.percentile(maes,90)) if maes else None,'max':float(np.max(maes)) if maes else None},
             'stop_ladder':{}}
    if not st.empty:
        for bps,g in st.groupby('bps'):
            summary['stop_ladder'][str(int(bps))]={'trades':len(g),'target_rate':float((g.reason=='TARGET').mean()),'stop_rate':float((g.reason=='STOP').mean()),'gross_avg_r':float(g.gross_r.mean()),'net_avg_r_4bp_side':float(g.net_r.mean()),'gross_total_r':float(g.gross_r.sum()),'net_total_r_4bp_side':float(g.net_r.sum()),'avg_fee_r':float(g.fee_r.mean())}
    (OUT/'quick_summary.json').write_text(json.dumps({'data':mani,'summary':summary},indent=2))
    fp.to_csv(OUT/'first_passage.csv',index=False); st.to_csv(OUT/'stop_ladder.csv',index=False)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
