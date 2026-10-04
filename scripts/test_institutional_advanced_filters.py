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

def test_institutional_advanced_filters(symbol="^NSEI", interval="15m", period="60d"):
    df = market_data_service.get_historical_candles(symbol, period=period, interval=interval)
    if df is None or len(df) < 50:
        return None

    close_arr = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    open_arr = df["open"].values
    vol_arr = df["volume"].values
    times = [str(t) for t in df.index]

    close_s = df["close"]
    high_s = df["high"]
    low_s = df["low"]
    open_s = df["open"]
    vol_s = df["volume"]

    # 1. EMAs
    ema9_arr = close_s.ewm(span=9, adjust=False).mean().values
    ema21_arr = close_s.ewm(span=21, adjust=False).mean().values
    ema50_arr = close_s.ewm(span=50, adjust=False).mean().values
    ema200_arr = close_s.ewm(span=min(200, len(df)), adjust=False).mean().values

    # 2. ATR
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_arr = tr.rolling(14, min_periods=1).mean().values

    # 3. RSI
    delta = close_s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, 1e-4)
    rsi_arr = (100 - (100 / (1 + rs))).fillna(50.0).values

    # 4. CHOP
    sum_tr_14 = tr.rolling(14, min_periods=1).sum()
    max_h_14 = high_s.rolling(14, min_periods=1).max()
    min_l_14 = low_s.rolling(14, min_periods=1).min()
    denom = (max_h_14 - min_l_14).replace(0, 1e-4)
    chop_arr = (100.0 * np.log10(np.maximum(1e-4, sum_tr_14 / denom)) / np.log10(14)).fillna(50.0).clip(0.0, 100.0).values

    # 5. ADX
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14, min_periods=1).mean() / atr_arr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14, min_periods=1).mean() / atr_arr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-4)
    adx_arr = dx.rolling(14, min_periods=1).mean().fillna(20.0).values

    # 6. CVD & Volume
    bar_range = (high_s - low_s).replace(0, 1e-4)
    bar_delta = ((close_s - open_s) / bar_range) * vol_s
    cvd_s = bar_delta.cumsum()
    cvd_slope_arr = cvd_s.diff(3).fillna(0.0).values
    vol_sma_arr = vol_s.rolling(20, min_periods=1).mean().values

    # 7. Supertrend 10, 3
    hl2 = (high_arr + low_arr) / 2.0
    st_dir = np.where(close_arr >= hl2, 1, -1)

    # 8. HTF Bias
    htf_bull_bias = close_arr >= ema200_arr
    htf_bear_bias = close_arr <= ema200_arr

    n_bars = len(df)
    results = []

    for killzone_filter in [True, False]:
        for htf_bias_req in [True, False]:
            for chop_cut in [50.0, 52.0, 55.0]:
                for adx_cut in [18.0, 20.0, 22.0]:
                    for rsi_min, rsi_max in [(42.0, 68.0), (45.0, 65.0)]:
                        for t1_mult in [0.75, 1.0]:
                            for t2_mult in [2.0, 2.5]:
                                for sl_mult in [0.8, 1.0, 1.2]:
                                    trades = []
                                    active_type = None
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
                                        bt = times[i]
                                        c_atr = max(bc * 0.003, atr_arr[i])

                                        is_killzone = True
                                        if killzone_filter and " " in bt:
                                            time_part = bt.split(" ")[1] if len(bt.split(" ")) > 1 else ""
                                            if "11:30" <= time_part <= "13:30":
                                                is_killzone = False

                                        if active_type == 1:
                                            if bh >= active_t2:
                                                trades.append(active_t2 - active_entry)
                                                active_type = None
                                                cooldown = 4
                                            elif bh >= active_t1 and not t1_hit:
                                                t1_hit = True
                                                active_sl = active_entry
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
                                                active_sl = active_entry
                                            elif bh >= active_sl:
                                                if t1_hit:
                                                    trades.append(active_entry - active_t1)
                                                else:
                                                    trades.append(active_entry - active_sl)
                                                active_type = None

                                        if cooldown > 0:
                                            cooldown -= 1
                                            continue

                                        if active_type is None and is_killzone and chop_arr[i] <= chop_cut and adx_arr[i] >= adx_cut:
                                            dist_21 = abs(bc - ema21_arr[i]) / c_atr
                                            if dist_21 <= 2.2:
                                                e9 = ema9_arr[i]
                                                e21 = ema21_arr[i]
                                                e50 = ema50_arr[i]
                                                rsi_val = rsi_arr[i]

                                                is_bull = (e9 >= e21) and (bc >= e50) and (bc >= e9) and (rsi_min <= rsi_val <= rsi_max) and (st_dir[i] == 1)
                                                is_bear = (e9 <= e21) and (bc <= e50) and (bc <= e9) and ((100 - rsi_max) <= rsi_val <= (100 - rsi_min)) and (st_dir[i] == -1)

                                                if htf_bias_req:
                                                    is_bull = is_bull and htf_bull_bias[i]
                                                    is_bear = is_bear and htf_bear_bias[i]

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

                                    if len(trades) >= 8:
                                        wins = [g for g in trades if g > 0]
                                        losses = [g for g in trades if g < 0]
                                        wr = round((len(wins) / len(trades)) * 100.0, 1)
                                        net_pts = round(sum(trades), 2)
                                        tot_g = sum(wins)
                                        tot_l = abs(sum(losses))
                                        pf = round(tot_g / max(1.0, tot_l), 2) if tot_l > 0 else 99.0

                                        results.append({
                                            "kz": killzone_filter,
                                            "htf": htf_bias_req,
                                            "chop": chop_cut,
                                            "adx": adx_cut,
                                            "rsi": (rsi_min, rsi_max),
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
        results.sort(key=lambda x: (x["wr"], x["pf"], x["net_pts"]), reverse=True)
        top = results[0]
        print(f"\n==================================================================")
        print(f"🏛️ INSTITUTIONAL CALIBRATION RESULT FOR {symbol} ({period} {interval})")
        print(f"🏆 WIN RATE: {top['wr']}% ({top['wins']} Wins / {top['losses']} Losses / {top['trades']} Trades)")
        print(f"💰 Net Gain: {top['net_pts']:+,.2f} pts | Profit Factor: {top['pf']}")
        print(f"⚙️ Key Active Filters:")
        print(f"   • Midday Lunch Chop Filter (11:30-13:30 IST): {top['kz']}")
        print(f"   • HTF 200 EMA Direction Bias: {top['htf']}")
        print(f"   • Chop Index Limit: <={top['chop']} | ADX: >={top['adx']}")
        print(f"   • RSI Sweet Spot: {top['rsi'][0]} - {top['rsi'][1]}")
        print(f"   • Target 1: {top['t1']}x ATR | Target 2: {top['t2']}x ATR | SL: {top['sl']}x ATR")
        print(f"==================================================================")

        print("Top 5 Calibrations:")
        for idx, r in enumerate(results[:5], 1):
            print(f"  #{idx} -> WR: {r['wr']}% | PF: {r['pf']} | Net: {r['net_pts']:+,.2f} pts | N={r['trades']} (KZ={r['kz']}, HTF={r['htf']}, Chop<={r['chop']}, ADX>={r['adx']}, RSI={r['rsi']}, T1={r['t1']}, T2={r['t2']}, SL={r['sl']})")
        return top

if __name__ == "__main__":
    test_institutional_advanced_filters("^BSESN", "1d", "1y")
    test_institutional_advanced_filters("^BSESN", "15m", "60d")
    test_institutional_advanced_filters("^NSEI", "1d", "1y")
    test_institutional_advanced_filters("^NSEI", "15m", "60d")
