import os
import sys
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.market_data import market_data_service

def optimize_high_winrate_fast(symbol="^NSEI", interval="15m", period="60d"):
    df = market_data_service.get_historical_candles(symbol, period=period, interval=interval)
    if df is None or len(df) < 50:
        return None

    close_arr = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    open_arr = df["open"].values
    vol_arr = df["volume"].values

    close_s = df["close"]
    high_s = df["high"]
    low_s = df["low"]
    open_s = df["open"]
    vol_s = df["volume"]

    # Indicators
    ema9_arr = close_s.ewm(span=9, adjust=False).mean().values
    ema21_arr = close_s.ewm(span=21, adjust=False).mean().values
    ema50_arr = close_s.ewm(span=50, adjust=False).mean().values

    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_arr = tr.rolling(14, min_periods=1).mean().values

    delta = close_s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, 1e-4)
    rsi_arr = (100 - (100 / (1 + rs))).fillna(50.0).values

    sum_tr_14 = tr.rolling(14, min_periods=1).sum()
    max_h_14 = high_s.rolling(14, min_periods=1).max()
    min_l_14 = low_s.rolling(14, min_periods=1).min()
    denom = (max_h_14 - min_l_14).replace(0, 1e-4)
    chop_arr = (100.0 * np.log10(np.maximum(1e-4, sum_tr_14 / denom)) / np.log10(14)).fillna(50.0).clip(0.0, 100.0).values

    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14, min_periods=1).mean() / tr.rolling(14, min_periods=1).mean().replace(0, 1e-4)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14, min_periods=1).mean() / tr.rolling(14, min_periods=1).mean().replace(0, 1e-4)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-4)
    adx_arr = dx.rolling(14, min_periods=1).mean().fillna(20.0).values

    bar_range = (high_s - low_s).replace(0, 1e-4)
    bar_delta = ((close_s - open_s) / bar_range) * vol_s
    cvd_s = bar_delta.cumsum()
    cvd_slope_arr = cvd_s.diff(3).fillna(0.0).values
    vol_sma_arr = vol_s.rolling(20, min_periods=1).mean().values

    hl2 = (high_arr + low_arr) / 2.0
    st_dir = np.where(close_arr >= hl2, 1, -1)

    pullback_bull_arr = ((low_arr <= ema9_arr * 1.002) & (close_arr >= ema9_arr))
    pullback_bear_arr = ((high_arr >= ema9_arr * 0.998) & (close_arr <= ema9_arr))

    swing_low_3_arr = low_s.rolling(3, min_periods=1).min().values
    swing_high_3_arr = high_s.rolling(3, min_periods=1).max().values

    n_bars = len(df)
    results = []

    # Target: explore high-accuracy institutional setups
    for chop_cut in [45.0, 50.0, 52.0, 55.0]:
        for adx_cut in [18.0, 20.0, 24.0]:
            for rsi_min, rsi_max in [(40.0, 70.0), (45.0, 68.0), (42.0, 65.0)]:
                for t1_mult in [0.75, 1.0, 1.2]:
                    for t2_mult in [2.0, 2.5]:
                        for sl_mult in [0.8, 1.0, 1.2]:
                            for req_pb in [False, True]:
                                trades = []
                                active_type = None # 1: CALL, -1: PUT
                                active_entry = 0.0
                                active_t1 = 0.0
                                active_t2 = 0.0
                                active_sl = 0.0
                                t1_hit = False
                                cooldown = 0

                                for i in range(25, n_bars):
                                    bh = high_arr[i]
                                    bl = low_arr[i]
                                    bc = close_arr[i]
                                    bo = open_arr[i]
                                    c_atr = max(bc * 0.003, atr_arr[i])

                                    if active_type == 1:
                                        if bh >= active_t2:
                                            trades.append(active_t2 - active_entry)
                                            active_type = None
                                            cooldown = 4
                                        elif bh >= active_t1 and not t1_hit:
                                            t1_hit = True
                                            active_sl = active_entry # Trail to BE
                                        elif bl <= active_sl:
                                            if t1_hit:
                                                trades.append(active_t1 - active_entry)
                                            else:
                                                trades.append(active_sl - active_entry)
                                            active_type = None

                                    elif active_type == -1:
                                        if bl <= active_t2:
                                            trades.append(active_entry - active_t2)
                                            active_type = None
                                            cooldown = 4
                                        elif bl <= active_t1 and not t1_hit:
                                            t1_hit = True
                                            active_sl = active_entry # Trail to BE
                                        elif bh >= active_sl:
                                            if t1_hit:
                                                trades.append(active_entry - active_t1)
                                            else:
                                                trades.append(active_entry - active_sl)
                                            active_type = None

                                    if cooldown > 0:
                                        cooldown -= 1
                                        continue

                                    if active_type is None and chop_arr[i] <= chop_cut and adx_arr[i] >= adx_cut:
                                        dist_21 = abs(bc - ema21_arr[i]) / c_atr
                                        if dist_21 <= 2.2:
                                            e9 = ema9_arr[i]
                                            e21 = ema21_arr[i]
                                            e50 = ema50_arr[i]
                                            rsi_val = rsi_arr[i]

                                            is_bull = (e9 >= e21) and (bc >= e50) and (bc >= e9) and (rsi_min <= rsi_val <= rsi_max) and (st_dir[i] == 1)
                                            if req_pb:
                                                is_bull = is_bull and pullback_bull_arr[i]

                                            is_bear = (e9 <= e21) and (bc <= e50) and (bc <= e9) and ((100 - rsi_max) <= rsi_val <= (100 - rsi_min)) and (st_dir[i] == -1)
                                            if req_pb:
                                                is_bear = is_bear and pullback_bear_arr[i]

                                            vol_bull = (cvd_slope_arr[i] >= 0) and (vol_arr[i] >= 1.0 * vol_sma_arr[i] or bc >= bo)
                                            vol_bear = (cvd_slope_arr[i] <= 0) and (vol_arr[i] >= 1.0 * vol_sma_arr[i] or bc <= bo)

                                            if is_bull and vol_bull:
                                                active_type = 1
                                                active_entry = bc
                                                active_t1 = bc + (t1_mult * c_atr)
                                                active_t2 = bc + (t2_mult * c_atr)
                                                active_sl = bc - (sl_mult * c_atr)
                                                t1_hit = False
                                            elif is_bear and vol_bear:
                                                active_type = -1
                                                active_entry = bc
                                                active_t1 = bc - (t1_mult * c_atr)
                                                active_t2 = bc - (t2_mult * c_atr)
                                                active_sl = bc + (sl_mult * c_atr)
                                                t1_hit = False

                                if len(trades) >= 10:
                                    wins = [g for g in trades if g > 0]
                                    losses = [g for g in trades if g < 0]
                                    wr = round((len(wins) / len(trades)) * 100.0, 1)
                                    net_pts = round(sum(trades), 2)
                                    tot_g = sum(wins)
                                    tot_l = abs(sum(losses))
                                    pf = round(tot_g / max(1.0, tot_l), 2) if tot_l > 0 else 99.0

                                    results.append({
                                        "chop": chop_cut,
                                        "adx": adx_cut,
                                        "rsi": (rsi_min, rsi_max),
                                        "pb": req_pb,
                                        "t1": t1_mult,
                                        "t2": t2_mult,
                                        "sl": sl_mult,
                                        "trades": len(trades),
                                        "wins": len(wins),
                                        "losses": len(losses),
                                        "wr": wr,
                                        "net_pts": net_pts,
                                        "pf": pf,
                                    })

    if results:
        # Sort by Win Rate desc, then Profit Factor desc
        results.sort(key=lambda x: (x["wr"], x["pf"], x["net_pts"]), reverse=True)
        top = results[0]
        print(f"\n==================================================================")
        print(f"🎯 OPTIMAL HIGH WIN-RATE FORMULA FOR {symbol} ({period} {interval})")
        print(f"🏆 WIN RATE: {top['wr']}% ({top['wins']} Wins / {top['losses']} Losses / {top['trades']} Trades)")
        print(f"💰 Net Gain: {top['net_pts']:+,.2f} pts | Profit Factor: {top['pf']}")
        print(f"⚙️ Calibrated Rules:")
        print(f"   • Chop Index: <={top['chop']} | ADX: >={top['adx']}")
        print(f"   • RSI Sweet Spot: {top['rsi'][0]} - {top['rsi'][1]}")
        print(f"   • Target 1: {top['t1']}x ATR | Target 2: {top['t2']}x ATR | SL: {top['sl']}x ATR")
        print(f"   • Pullback Required: {top['pb']}")
        print(f"==================================================================")

        print("Top 5 Calibrations:")
        for idx, r in enumerate(results[:5], 1):
            print(f"  #{idx} -> WR: {r['wr']}% | PF: {r['pf']} | Net: {r['net_pts']:+,.2f} pts | N={r['trades']} (Chop<={r['chop']}, ADX>={r['adx']}, RSI={r['rsi']}, T1={r['t1']}, T2={r['t2']}, SL={r['sl']}, PB={r['pb']})")
        return top

if __name__ == "__main__":
    print("=" * 65)
    print("🚀 QUANT ACCELERATOR: OPTIMIZING WIN RATE TO 70% - 85%+")
    print("=" * 65)
    optimize_high_winrate_fast("^BSESN", "1d", "1y")
    optimize_high_winrate_fast("^BSESN", "15m", "60d")
    optimize_high_winrate_fast("^NSEI", "1d", "1y")
    optimize_high_winrate_fast("^NSEI", "15m", "60d")
