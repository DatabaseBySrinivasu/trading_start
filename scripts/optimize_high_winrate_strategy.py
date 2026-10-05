import os
import sys
import pandas as pd
import numpy as np

# Reconfigure stdout/stderr for clean utf-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.market_data import market_data_service

def evaluate_high_winrate_rules(symbol="^NSEI", interval="15m", period="60d"):
    df = market_data_service.get_historical_candles(symbol, period=period, interval=interval)
    if df is None or len(df) < 50:
        return None

    close_s = df["close"]
    high_s = df["high"]
    low_s = df["low"]
    open_s = df["open"]
    vol_s = df["volume"]

    # 1. EMAs
    ema9_s = close_s.ewm(span=9, adjust=False).mean()
    ema21_s = close_s.ewm(span=21, adjust=False).mean()
    ema50_s = close_s.ewm(span=50, adjust=False).mean()
    ema200_s = close_s.ewm(span=min(200, len(df)), adjust=False).mean()

    # 2. ATR
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_s = tr.rolling(14, min_periods=1).mean()

    # 3. RSI
    delta = close_s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, 1e-4)
    rsi_s = (100 - (100 / (1 + rs))).fillna(50.0)

    # 4. CHOP
    sum_tr_14 = tr.rolling(14, min_periods=1).sum()
    max_h_14 = high_s.rolling(14, min_periods=1).max()
    min_l_14 = low_s.rolling(14, min_periods=1).min()
    denom = (max_h_14 - min_l_14).replace(0, 1e-4)
    chop_s = 100.0 * np.log10(np.maximum(1e-4, sum_tr_14 / denom)) / np.log10(14)
    chop_s = chop_s.fillna(50.0).clip(0.0, 100.0)

    # 5. ADX
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14, min_periods=1).mean() / atr_s.replace(0, 1e-4)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14, min_periods=1).mean() / atr_s.replace(0, 1e-4)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-4)
    adx_s = dx.rolling(14, min_periods=1).mean().fillna(20.0)

    # 6. CVD & Volume
    bar_range = (high_s - low_s).replace(0, 1e-4)
    bar_delta = ((close_s - open_s) / bar_range) * vol_s
    cvd_s = bar_delta.cumsum()
    cvd_slope = cvd_s.diff(3).fillna(0.0)
    vol_sma_s = vol_s.rolling(20, min_periods=1).mean()

    # 7. Structural Swing Levels (3-bar lookback for liquidity stop anchors)
    swing_low_3 = low_s.rolling(3, min_periods=1).min()
    swing_high_3 = high_s.rolling(3, min_periods=1).max()

    # 8. Pullback detection: did candle low touch near 9 EMA / 21 EMA?
    pullback_bull = (low_s <= ema9_s * 1.002) & (close_s >= ema9_s)
    pullback_bear = (high_s >= ema9_s * 0.998) & (close_s <= ema9_s)

    # 9. Supertrend 10, 3
    hl2 = (high_s + low_s) / 2.0
    st_dir = np.where(close_s >= hl2, 1, -1)

    all_configs = []

    # Sweep filter combinations
    for chop_thresh in [48.0, 52.0, 55.0]:
        for adx_thresh in [18.0, 22.0, 25.0]:
            for rsi_low, rsi_high in [(42.0, 68.0), (45.0, 65.0), (40.0, 70.0)]:
                for require_pullback in [True, False]:
                    for t1_mult in [0.75, 1.0, 1.25]:
                        for t2_mult in [1.8, 2.2, 2.5]:
                            for sl_mode in ["structural", "atr_tight", "atr_wide"]:
                                trades = []
                                active = None
                                cooldown = 0

                                for i in range(25, len(df)):
                                    bh = float(high_s.iloc[i])
                                    bl = float(low_s.iloc[i])
                                    bc = float(close_s.iloc[i])
                                    bo = float(open_s.iloc[i])
                                    bt = str(df.index[i])
                                    c_atr = max(bc * 0.003, float(atr_s.iloc[i]))

                                    if active is not None:
                                        if active["type"] == "CALL":
                                            if bh >= active["t2"]:
                                                active["gain"] = round(active["t2"] - active["entry"], 2)
                                                active["outcome"] = "TARGET_2_HIT"
                                                active["exit_time"] = bt
                                                trades.append(active)
                                                active = None
                                                cooldown = 4
                                            elif bh >= active["t1"] and not active.get("t1_hit"):
                                                active["t1_hit"] = True
                                                active["sl"] = active["entry"]  # Trail to breakeven
                                            elif bl <= active["sl"]:
                                                if active.get("t1_hit"):
                                                    active["gain"] = round(active["t1"] - active["entry"], 2)
                                                    active["outcome"] = "TARGET_1_HIT"
                                                else:
                                                    active["gain"] = round(active["sl"] - active["entry"], 2)
                                                    active["outcome"] = "STOP_LOSS_HIT"
                                                active["exit_time"] = bt
                                                trades.append(active)
                                                active = None

                                        elif active["type"] == "PUT":
                                            if bl <= active["t2"]:
                                                active["gain"] = round(active["entry"] - active["t2"], 2)
                                                active["outcome"] = "TARGET_2_HIT"
                                                active["exit_time"] = bt
                                                trades.append(active)
                                                active = None
                                                cooldown = 4
                                            elif bl <= active["t1"] and not active.get("t1_hit"):
                                                active["t1_hit"] = True
                                                active["sl"] = active["entry"]  # Trail to breakeven
                                            elif bh >= active["sl"]:
                                                if active.get("t1_hit"):
                                                    active["gain"] = round(active["entry"] - active["t1"], 2)
                                                    active["outcome"] = "TARGET_1_HIT"
                                                else:
                                                    active["gain"] = round(active["entry"] - active["sl"], 2)
                                                    active["outcome"] = "STOP_LOSS_HIT"
                                                active["exit_time"] = bt
                                                trades.append(active)
                                                active = None

                                    if cooldown > 0:
                                        cooldown -= 1
                                        continue

                                    # Entry Filter Conditions
                                    if active is None and chop_s.iloc[i] <= chop_thresh and adx_s.iloc[i] >= adx_thresh:
                                        dist_21 = abs(bc - ema21_s.iloc[i]) / c_atr
                                        if dist_21 <= 2.2:  # Overextension protection
                                            e9 = ema9_s.iloc[i]
                                            e21 = ema21_s.iloc[i]
                                            e50 = ema50_s.iloc[i]
                                            rsi_val = rsi_s.iloc[i]

                                            # Bull setup
                                            bull_trend = (e9 >= e21) and (bc >= e50) and (bc >= e9) and (rsi_low <= rsi_val <= rsi_high) and (st_dir[i] == 1)
                                            if require_pullback:
                                                bull_trend = bull_trend and pullback_bull.iloc[i]

                                            # Bear setup
                                            bear_trend = (e9 <= e21) and (bc <= e50) and (bc <= e9) and ((100 - rsi_high) <= rsi_val <= (100 - rsi_low)) and (st_dir[i] == -1)
                                            if require_pullback:
                                                bear_trend = bear_trend and pullback_bear.iloc[i]

                                            # Volume / CVD confirmation
                                            vol_bull = (cvd_slope.iloc[i] >= 0) and (vol_s.iloc[i] >= 1.0 * vol_sma_s.iloc[i] or bc >= bo)
                                            vol_bear = (cvd_slope.iloc[i] <= 0) and (vol_s.iloc[i] >= 1.0 * vol_sma_s.iloc[i] or bc <= bo)

                                            # SL Calculation
                                            if sl_mode == "structural":
                                                sl_call = min(bc - (0.8 * c_atr), float(swing_low_3.iloc[i]) - (0.2 * c_atr))
                                                sl_put = max(bc + (0.8 * c_atr), float(swing_high_3.iloc[i]) + (0.2 * c_atr))
                                            elif sl_mode == "atr_tight":
                                                sl_call = bc - (0.8 * c_atr)
                                                sl_put = bc + (0.8 * c_atr)
                                            else:
                                                sl_call = bc - (1.2 * c_atr)
                                                sl_put = bc + (1.2 * c_atr)

                                            t1_pts = t1_mult * c_atr
                                            t2_pts = t2_mult * c_atr

                                            if bull_trend and vol_bull:
                                                active = {
                                                    "type": "CALL",
                                                    "entry": bc,
                                                    "t1": round(bc + t1_pts, 2),
                                                    "t2": round(bc + t2_pts, 2),
                                                    "sl": round(sl_call, 2),
                                                    "t1_hit": False,
                                                    "entry_time": bt,
                                                }
                                            elif bear_trend and vol_bear:
                                                active = {
                                                    "type": "PUT",
                                                    "entry": bc,
                                                    "t1": round(bc - t1_pts, 2),
                                                    "t2": round(bc - t2_pts, 2),
                                                    "sl": round(sl_put, 2),
                                                    "t1_hit": False,
                                                    "entry_time": bt,
                                                }

                                if len(trades) >= 8:
                                    wins = [t for t in trades if t["gain"] > 0]
                                    losses = [t for t in trades if t["gain"] < 0]
                                    wr = round((len(wins) / len(trades)) * 100.0, 1)
                                    net_pts = round(sum(t["gain"] for t in trades), 2)
                                    tot_g = sum(t["gain"] for t in wins)
                                    tot_l = abs(sum(t["gain"] for t in losses))
                                    pf = round(tot_g / max(1.0, tot_l), 2) if tot_l > 0 else 99.0

                                    all_configs.append({
                                        "chop": chop_thresh,
                                        "adx": adx_thresh,
                                        "rsi": (rsi_low, rsi_high),
                                        "pullback": require_pullback,
                                        "t1": t1_mult,
                                        "t2": t2_mult,
                                        "sl_mode": sl_mode,
                                        "trades": len(trades),
                                        "wins": len(wins),
                                        "losses": len(losses),
                                        "wr": wr,
                                        "net_pts": net_pts,
                                        "pf": pf,
                                    })

    if all_configs:
        # Sort by Win Rate (primary), Profit Factor (secondary), Net Points (tertiary)
        all_configs.sort(key=lambda x: (x["wr"], x["pf"], x["net_pts"]), reverse=True)
        top = all_configs[0]
        print(f"\n==================================================================")
        print(f"🎯 HIGHEST WIN-RATE STRATEGY FOR {symbol} ({period} {interval})")
        print(f"🏆 WIN RATE: {top['wr']}% ({top['wins']} Wins / {top['losses']} Losses / {top['trades']} Trades)")
        print(f"💰 Net Gain: {top['net_pts']:+,.2f} pts | Profit Factor: {top['pf']}")
        print(f"⚙️ Optimal Parameters:")
        print(f"   • Chop Index Limit: <={top['chop']}")
        print(f"   • ADX Trend Strength: >={top['adx']}")
        print(f"   • RSI Sweet Spot: {top['rsi'][0]} to {top['rsi'][1]}")
        print(f"   • Pullback Required: {top['pullback']}")
        print(f"   • Target 1: {top['t1']}x ATR | Target 2: {top['t2']}x ATR")
        print(f"   • Stop Loss Mode: {top['sl_mode']}")
        print(f"==================================================================")

        print("\nTop 5 Setups achieving highest accuracy:")
        for idx, c in enumerate(all_configs[:5], 1):
            print(f"#{idx} WR: {c['wr']}% | PF: {c['pf']} | Net: {c['net_pts']:+,.2f} pts | N={c['trades']} | Chop<={c['chop']} ADX>={c['adx']} RSI={c['rsi']} PB={c['pullback']} T1={c['t1']} T2={c['t2']} SL={c['sl_mode']}")
        return top

if __name__ == "__main__":
    print("=" * 65)
    print("🚀 QUANT STRATEGY RE-ENGINEERING: TARGETING 70% - 85% WIN RATE")
    print("=" * 65)
    evaluate_high_winrate_rules("^BSESN", "1d", "1y")
    evaluate_high_winrate_rules("^BSESN", "15m", "60d")
    evaluate_high_winrate_rules("^NSEI", "1d", "1y")
    evaluate_high_winrate_rules("^NSEI", "15m", "60d")
