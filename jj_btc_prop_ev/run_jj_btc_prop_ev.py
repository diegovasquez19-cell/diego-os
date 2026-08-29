from __future__ import annotations

import hashlib
import io
import json
import math
import random
import zipfile
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

OUT = Path("jj_btc_prop_ev/results")
OUT.mkdir(parents=True, exist_ok=True)

UTC = ZoneInfo("UTC")
CT = ZoneInfo("America/Chicago")
ET = ZoneInfo("America/New_York")

MONTHS = [f"2026-{m:02d}" for m in range(1, 7)]
BASE = "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m"
FEE_SIDE = 0.0004  # Breakout/Kraken Pro: 0.04% per buy/sell order
FRONT_RUN_USD = 1.0
MC_PATHS = 5000
EVAL_HORIZON_DAYS = 90
FUNDED_HORIZON_DAYS = 30
SEED = 20260828


@dataclass
class Trade:
    trading_date: str
    variant: str
    mode: str
    weekday_only: bool
    attempt: int
    side: str
    signal_time_utc: str
    entry_time_utc: str
    exit_time_utc: str
    anchor: float
    entry: float
    exit: float
    stop_distance: float
    gross_r: float
    fee_r_4bp_side: float
    net_r_4bp_side: float
    exit_reason: str


def download_month(month: str):
    url = f"{BASE}/BTCUSDT-1m-{month}.zip"
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    sha = hashlib.sha256(r.content).hexdigest()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"No CSV in {url}")
        raw = z.read(names[0])
    return raw, {"month": month, "url": url, "zip_sha256": sha, "csv_bytes": len(raw)}


def parse_month(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), header=None, low_memory=False)
    # Binance Vision files may or may not contain a header row.
    if not pd.to_numeric(pd.Series([df.iloc[0, 0]]), errors="coerce").notna().iloc[0]:
        df = df.iloc[1:].copy()
    if df.shape[1] < 6:
        raise RuntimeError(f"Unexpected kline schema with {df.shape[1]} columns")
    df = df.iloc[:, :6].copy()
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().copy()
    max_ts = float(df["open_time"].max())
    unit = "us" if max_ts > 1e14 else "ms"
    df["ts"] = pd.to_datetime(df["open_time"].astype("int64"), unit=unit, utc=True)
    df = df.set_index("ts")[["open", "high", "low", "close", "volume"]]
    return df.astype(float)


def load_data():
    frames, manifests = [], []
    for m in MONTHS:
        raw, manifest = download_month(m)
        d = parse_month(raw)
        manifest["rows"] = int(len(d))
        manifest["first_ts"] = str(d.index.min())
        manifest["last_ts"] = str(d.index.max())
        frames.append(d)
        manifests.append(manifest)
        print(f"loaded {m}: {len(d):,} rows")
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df, manifests


def utc_ts(d: date, hh: int, mm: int, zone: ZoneInfo) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(d, time(hh, mm), tzinfo=zone).astimezone(UTC))


def stop_distance_for(variant: str, fill: float) -> float:
    if variant == "literal30":
        return 30.0
    if variant == "pct10bp":
        return fill * 0.001  # 10 basis points = 0.10%
    raise ValueError(variant)


def simulate_day(df: pd.DataFrame, d: date, variant: str, mode: str, weekday_only: bool):
    if weekday_only and d.weekday() >= 5:
        return []
    max_attempts = 1 if mode == "F1" else 3
    cutoff_hm = (10, 30) if mode == "F1" else (11, 0)

    anchor_date = d - timedelta(days=1)
    anchor_ts = utc_ts(anchor_date, 17, 0, CT)
    if anchor_ts not in df.index:
        return []
    anchor = float(df.at[anchor_ts, "open"])

    ny_open = utc_ts(d, 9, 30, ET)
    scan_start = utc_ts(d, 9, 35, ET)
    cutoff = utc_ts(d, cutoff_hm[0], cutoff_hm[1], ET)

    # If fair price was already touched during the first five NY minutes, thesis is complete.
    pre = df.loc[(df.index >= ny_open) & (df.index < scan_start)]
    if len(pre) < 5:
        return []
    if ((pre["low"] <= anchor) & (pre["high"] >= anchor)).any():
        return []

    trades = []
    cursor = scan_start
    attempts = 0

    while cursor < cutoff and attempts < max_attempts:
        window = df.loc[(df.index >= cursor) & (df.index < cutoff)]
        if window.empty:
            break
        signal_ts = None
        side = None

        for ts, row in window.iterrows():
            # Anchor touch while flat ends the thesis/day.
            if row["low"] <= anchor <= row["high"]:
                return trades
            pos = df.index.get_indexer([ts])[0]
            if pos < 3:
                continue
            prior = df.iloc[pos - 3:pos]
            close = float(row["close"])
            if close < anchor and close > float(prior["high"].max()):
                signal_ts, side = ts, "LONG"
                break
            if close > anchor and close < float(prior["low"].min()):
                signal_ts, side = ts, "SHORT"
                break

        if signal_ts is None:
            break
        sig_pos = df.index.get_indexer([signal_ts])[0]
        if sig_pos + 1 >= len(df):
            break
        entry_ts = df.index[sig_pos + 1]
        if entry_ts >= cutoff:
            break
        entry = float(df.iloc[sig_pos + 1]["open"])
        stop_dist = stop_distance_for(variant, entry)

        # Frozen gate: at least 1R room to the fair-price anchor from actual fill.
        room = (anchor - entry) if side == "LONG" else (entry - anchor)
        if room < stop_dist:
            cursor = entry_ts
            continue

        target = anchor - FRONT_RUN_USD if side == "LONG" else anchor + FRONT_RUN_USD
        stop = entry - stop_dist if side == "LONG" else entry + stop_dist
        attempts += 1

        exit_ts = cutoff
        if cutoff in df.index:
            exit_px = float(df.at[cutoff, "open"])
        else:
            before = df.loc[df.index < cutoff]
            if before.empty:
                break
            exit_ts = before.index[-1]
            exit_px = float(before.iloc[-1]["close"])
        reason = "CUTOFF"

        trade_bars = df.loc[(df.index >= entry_ts) & (df.index < cutoff)]
        for ts2, bar in trade_bars.iterrows():
            if side == "LONG":
                hit_stop = float(bar["low"]) <= stop
                hit_target = float(bar["high"]) >= target
            else:
                hit_stop = float(bar["high"]) >= stop
                hit_target = float(bar["low"]) <= target
            # Conservative ambiguity: stop first if both occur in one 1m bar.
            if hit_stop:
                exit_ts, exit_px, reason = ts2, stop, "STOP"
                break
            if hit_target:
                exit_ts, exit_px, reason = ts2, target, "TARGET"
                break

        gross_p = (exit_px - entry) if side == "LONG" else (entry - exit_px)
        gross_r = gross_p / stop_dist
        fee_per_btc = FEE_SIDE * (entry + exit_px)
        fee_r = fee_per_btc / stop_dist
        net_r = gross_r - fee_r

        trades.append(Trade(
            trading_date=str(d), variant=variant, mode=mode, weekday_only=weekday_only,
            attempt=attempts, side=side, signal_time_utc=signal_ts.isoformat(),
            entry_time_utc=entry_ts.isoformat(), exit_time_utc=exit_ts.isoformat(),
            anchor=anchor, entry=entry, exit=float(exit_px), stop_distance=stop_dist,
            gross_r=float(gross_r), fee_r_4bp_side=float(fee_r), net_r_4bp_side=float(net_r),
            exit_reason=reason,
        ))

        if reason == "CUTOFF":
            break
        # A completely fresh BOS must form after a stopped/closed trade.
        cursor = exit_ts + pd.Timedelta(minutes=1)

    return trades


def metrics(trades: list[Trade], eligible_days: int):
    if not trades:
        return {"trades": 0, "eligible_days": eligible_days}
    g = np.array([t.gross_r for t in trades], dtype=float)
    n = np.array([t.net_r_4bp_side for t in trades], dtype=float)
    gross_wins = g[g > 0]
    gross_losses = g[g < 0]
    eq = np.cumsum(n)
    peak = np.maximum.accumulate(np.r_[0.0, eq])
    dd = np.r_[0.0, eq] - peak
    return {
        "trades": len(trades),
        "eligible_days": eligible_days,
        "trades_per_eligible_day": len(trades) / eligible_days if eligible_days else None,
        "target_rate": sum(t.exit_reason == "TARGET" for t in trades) / len(trades),
        "stop_rate": sum(t.exit_reason == "STOP" for t in trades) / len(trades),
        "cutoff_rate": sum(t.exit_reason == "CUTOFF" for t in trades) / len(trades),
        "gross_win_rate": float(np.mean(g > 0)),
        "gross_avg_r": float(np.mean(g)),
        "gross_median_r": float(np.median(g)),
        "gross_total_r": float(np.sum(g)),
        "net_avg_r_4bp_side": float(np.mean(n)),
        "net_median_r_4bp_side": float(np.median(n)),
        "net_total_r_4bp_side": float(np.sum(n)),
        "net_win_rate_4bp_side": float(np.mean(n > 0)),
        "avg_fee_r": float(np.mean([t.fee_r_4bp_side for t in trades])),
        "net_max_drawdown_r": float(abs(np.min(dd))),
        "gross_profit_factor": float(gross_wins.sum() / abs(gross_losses.sum())) if gross_losses.size else None,
    }


def build_daily_map(trades: list[Trade], all_dates: list[str]):
    m = {d: [] for d in all_dates}
    for t in trades:
        m.setdefault(t.trading_date, []).append(t)
    return m


def apply_trade_to_account(balance, initial, mdd, leverage, desired_risk, t: Trade, fee_side=FEE_SIDE):
    stop_dist = t.stop_distance
    qty_risk = desired_risk / stop_dist
    qty_leverage = (max(balance, 0.0) * leverage) / t.entry
    qty = max(0.0, min(qty_risk, qty_leverage))
    signed_move = (t.exit - t.entry) if t.side == "LONG" else (t.entry - t.exit)
    gross = signed_move * qty
    friction = fee_side * (t.entry + t.exit) * qty
    return gross - friction, qty, gross, friction


def mc_eval(daily_map, plan, paths=MC_PATHS, horizon=EVAL_HORIZON_DAYS, seed=SEED):
    rng = random.Random(seed)
    keys = list(daily_map.keys())
    pass_days, breaches = [], 0
    final_pnls = []
    for _ in range(paths):
        bal = plan["balance"]
        initial = plan["balance"]
        static_floor = initial - plan["mdd"]
        passed = False
        breached = False
        for day_n in range(1, horizon + 1):
            day_start = bal
            daily_floor = day_start * (1.0 - plan["daily_loss_pct"])
            day = rng.choice(keys)
            for t in daily_map[day]:
                pnl, _, _, _ = apply_trade_to_account(
                    bal, initial, plan["mdd"], plan["leverage"], plan["mdd"] / 4.0, t, plan["fee_side"]
                )
                bal += pnl
                if bal <= static_floor or bal <= daily_floor:
                    breached = True
                    break
                if bal >= initial + plan["target"]:
                    passed = True
                    pass_days.append(day_n)
                    break
            if breached or passed:
                break
        if breached:
            breaches += 1
        final_pnls.append(bal - initial)
    p = len(pass_days) / paths
    return {
        "pass_rate_90d": p,
        "breach_rate_90d": breaches / paths,
        "unresolved_rate_90d": 1.0 - p - breaches / paths,
        "median_days_to_pass": float(np.median(pass_days)) if pass_days else None,
        "mean_final_pnl": float(np.mean(final_pnls)),
        "evaluation_fee": plan["eval_fee"],
        "expected_fee_cost_per_pass": (plan["eval_fee"] / p) if p > 0 else None,
    }


def mc_funded(daily_map, plan, paths=MC_PATHS, horizon=FUNDED_HORIZON_DAYS, seed=SEED + 1):
    rng = random.Random(seed)
    keys = list(daily_map.keys())
    payouts, survived = [], 0
    for _ in range(paths):
        bal = plan["balance"]
        initial = plan["balance"]
        static_floor = initial - plan["mdd"]
        breached = False
        for _day_n in range(1, horizon + 1):
            day_start = bal
            daily_floor = day_start * (1.0 - plan["daily_loss_pct"])
            day = rng.choice(keys)
            for t in daily_map[day]:
                pnl, _, _, _ = apply_trade_to_account(
                    bal, initial, plan["mdd"], plan["leverage"], plan["mdd"] / 4.0, t, plan["fee_side"]
                )
                bal += pnl
                if bal <= static_floor or bal <= daily_floor:
                    breached = True
                    break
            if breached:
                break
        if not breached:
            survived += 1
            payouts.append(max(0.0, bal - initial) * plan["split"])
        else:
            payouts.append(0.0)
    return {
        "funded_survival_30d": survived / paths,
        "mean_payout_30d_unconditional": float(np.mean(payouts)),
        "median_payout_30d_unconditional": float(np.median(payouts)),
        "payout_probability_30d": float(np.mean(np.array(payouts) >= 50.0)),
    }


def main():
    df, manifests = load_data()
    start_d = df.index.min().tz_convert(ET).date() + timedelta(days=1)
    end_d = df.index.max().tz_convert(ET).date()
    dates = []
    d = start_d
    while d <= end_d:
        dates.append(d)
        d += timedelta(days=1)

    configs = [
        ("literal30", "F1", True),
        ("literal30", "F2", True),
        ("pct10bp", "F1", True),
        ("pct10bp", "F2", True),
        # Crypto trades 24/7; pre-specified weekend sensitivity only for normalized stop.
        ("pct10bp", "F1", False),
        ("pct10bp", "F2", False),
    ]

    all_trades = []
    metrics_by_config = {}
    trades_by_config = {}
    eligible_dates_by_config = {}
    for variant, mode, weekday_only in configs:
        key = f"{variant}_{mode}_{'weekdays' if weekday_only else 'all_days'}"
        tlist = []
        eligible_dates = []
        for day in dates:
            if weekday_only and day.weekday() >= 5:
                continue
            # only dates with required anchor + session data are considered eligible
            anchor_ts = utc_ts(day - timedelta(days=1), 17, 0, CT)
            open_ts = utc_ts(day, 9, 30, ET)
            cutoff_hm = (10, 30) if mode == "F1" else (11, 0)
            cutoff_ts = utc_ts(day, cutoff_hm[0], cutoff_hm[1], ET)
            if anchor_ts not in df.index or open_ts not in df.index or cutoff_ts not in df.index:
                continue
            eligible_dates.append(str(day))
            tlist.extend(simulate_day(df, day, variant, mode, weekday_only))
        trades_by_config[key] = tlist
        eligible_dates_by_config[key] = eligible_dates
        metrics_by_config[key] = metrics(tlist, len(eligible_dates))
        all_trades.extend(tlist)
        print(key, metrics_by_config[key])

    plans = {
        "Breakout_10K_Classic": {"balance": 10000.0, "target": 1000.0, "mdd": 600.0, "daily_loss_pct": 0.03, "eval_fee": 85.0, "leverage": 10.0, "fee_side": 0.0004, "split": 0.80},
        "Breakout_10K_Pro": {"balance": 10000.0, "target": 1200.0, "mdd": 500.0, "daily_loss_pct": 0.03, "eval_fee": 65.0, "leverage": 10.0, "fee_side": 0.0004, "split": 0.80},
        "Breakout_10K_Turbo": {"balance": 10000.0, "target": 900.0, "mdd": 300.0, "daily_loss_pct": 0.03, "eval_fee": 40.0, "leverage": 10.0, "fee_side": 0.0004, "split": 0.80},
        # Kraken Consumer Prop: 1x and 4bp spread each side; modeled as equivalent per-side friction.
        "Kraken_Consumer_10K": {"balance": 10000.0, "target": 1200.0, "mdd": 500.0, "daily_loss_pct": 0.03, "eval_fee": 90.0, "leverage": 1.0, "fee_side": 0.0004, "split": 0.80},
    }

    ev = {}
    # Prop EV on the weekday configs; weekend sensitivity stays in signal layer first.
    for cfg in ["literal30_F1_weekdays", "literal30_F2_weekdays", "pct10bp_F1_weekdays", "pct10bp_F2_weekdays"]:
        daily = build_daily_map(trades_by_config[cfg], eligible_dates_by_config[cfg])
        ev[cfg] = {}
        for name, plan in plans.items():
            e = mc_eval(daily, plan)
            f = mc_funded(daily, plan)
            proxy = e["pass_rate_90d"] * f["mean_payout_30d_unconditional"] - plan["eval_fee"]
            ev[cfg][name] = {**e, **f, "one_eval_plus_30d_funded_ev_proxy": float(proxy)}
            print(cfg, name, ev[cfg][name])

    trade_df = pd.DataFrame([asdict(t) for t in all_trades])
    trade_df.to_csv(OUT / "trades.csv", index=False)

    summary = {
        "run": {
            "name": "JJ-BTC PROP EV pilot",
            "generated_utc": datetime.now(tz=UTC).isoformat(),
            "market": "Binance BTCUSDT USD-M perpetual 1m",
            "months": MONTHS,
            "seed": SEED,
            "mc_paths": MC_PATHS,
            "eval_horizon_days": EVAL_HORIZON_DAYS,
            "funded_horizon_days": FUNDED_HORIZON_DAYS,
        },
        "locked_port": {
            "anchor": "17:00 America/Chicago previous calendar day, 1m candle OPEN",
            "ny_observation": "09:30-09:35 America/New_York; anchor touch ends thesis",
            "entry_start": "09:35 America/New_York",
            "bos": "1m close beyond prior 3 closed bars extreme; next-bar-open fill",
            "direction": "toward anchor",
            "room_gate": ">=1R room from actual fill to anchor",
            "target": "anchor front-run by $1",
            "F1": "max 1 attempt; flatten 10:30 ET",
            "F2": "max 3 fresh BOS attempts; flatten 11:00 ET",
            "intrabar_ambiguity": "stop-first",
            "stop_variants_predeclared": {"literal30": "$30 fixed", "pct10bp": "10 bps of entry price"},
            "prop_friction": "0.04% per side",
        },
        "data_manifest": manifests,
        "signal_metrics": metrics_by_config,
        "prop_ev": ev,
        "caveats": [
            "Pilot is Jan-Jun 2026 only; it is not a frozen lockbox result.",
            "Binance perpetual candles proxy the tradable BTC path; prop terminal spreads/slippage can differ.",
            "Open-position size caps beyond headline leverage are not modeled.",
            "Monte Carlo resamples observed eligible days and is exploratory, not evidence of independent future returns.",
            "Funded EV proxy assumes one 30-day funded period with an 80% split and no intermediate withdrawals.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    rows = []
    for cfg, m in metrics_by_config.items():
        rows.append({"config": cfg, **m})
    pd.DataFrame(rows).to_csv(OUT / "signal_summary.csv", index=False)

    evrows = []
    for cfg, plans_out in ev.items():
        for pname, vals in plans_out.items():
            evrows.append({"config": cfg, "plan": pname, **vals})
    evdf = pd.DataFrame(evrows).sort_values("one_eval_plus_30d_funded_ev_proxy", ascending=False)
    evdf.to_csv(OUT / "prop_ev_summary.csv", index=False)

    best_sig = max(metrics_by_config.items(), key=lambda kv: kv[1].get("net_avg_r_4bp_side", -999))
    best_ev = evdf.iloc[0].to_dict() if len(evdf) else {}
    report = [
        "# JJ-BTC PROP EV — Jan-Jun 2026 Pilot\n",
        "**Research only. Separate from ATLAS.**\n",
        "## Headline\n",
        f"- Best fee-adjusted signal configuration: **{best_sig[0]}** — net avg R {best_sig[1].get('net_avg_r_4bp_side', float('nan')):.4f}, {best_sig[1].get('trades', 0)} trades.\n",
    ]
    if best_ev:
        report += [
            f"- Best exploratory prop-EV cell: **{best_ev['config']} × {best_ev['plan']}** — 90d pass rate {best_ev['pass_rate_90d']:.1%}, 30d funded survival {best_ev['funded_survival_30d']:.1%}, EV proxy ${best_ev['one_eval_plus_30d_funded_ev_proxy']:.2f}.\n"
        ]
    report += [
        "\n## Interpretation guardrails\n",
        "- The $30 stop is the literal JJ numeric port; the 10bp stop is a pre-declared cross-asset normalization, not an optimized parameter.\n",
        "- Fee-adjusted metrics include 4bp per side. This is critical because BTC prop friction is large relative to tight stops.\n",
        "- Weekend variants are sensitivity tests only; prop EV currently uses weekday variants to preserve the JJ NY-session context.\n",
        "- Do not deploy from this pilot. Expand to multi-year train/validation/lockbox only if the pilot clears basic edge and EV sanity checks.\n",
        "\n## Files\n- `trades.csv`\n- `signal_summary.csv`\n- `prop_ev_summary.csv`\n- `summary.json`\n",
    ]
    (OUT / "REPORT.md").write_text("".join(report))
    print("\n" + "".join(report))


if __name__ == "__main__":
    main()
