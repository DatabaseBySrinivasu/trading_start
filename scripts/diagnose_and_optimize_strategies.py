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

def diagnose_symbol(sym):
    print("=" * 60)
    print(f"📊 DIAGNOSING HISTORICAL DATA & OPTIMIZING STRATEGY FOR: {sym}")
    print("=" * 60)
    df = market_data_service.get_historical_candles(sym, period="1mo", interval="15m")
    if df is None or df.empty:
        print(f"❌ Failed to load candles for {sym}")
        return

    close_s = df["close"]
    high_s = df["high"]
    low_s = df["low"]
    open_s = df["open"]
    vol_s = df["volume"]

    ema9 = close_s.ewm(span=9, adjust=False).mean()
    ema21 = close_s.ewm(span=21, adjust=False).mean()
    ema50 = close_s.ewm(span=50, adjust=False).mean()
    ema200 = close_s.ewm(span=min(200, len(df)), adjust=False).mean()

    # True Range & ATR
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean()

    # ADX 14
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14, min_periods=1).mean() / atr.replace(0, 1e-4)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14, min_periods=1).mean() / atr.replace(0, 1e-4)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-4)
    adx = dx.rolling(14, min_periods=1).mean()

    # RSI 14
    delta = close_s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, 1e-4)
    rsi = 100 - (100 / (1 + rs))

    # Choppiness Index 14
    sum_tr_14 = tr.rolling(14, min_periods=1).sum()
    max_h_14 = high_s.rolling(14, min_periods=1).max()
    min_l_14 = low_s.rolling(14, min_periods=1).min()
    denom = (max_h_14 - min_l_14).replace(0, 1e-4)
    chop = 100.0 * np.log10(np.maximum(1e-4, sum_tr_14 / denom)) / np.log10(14)
    chop = chop.fillna(50.0).clip(0.0, 100.0)

    # Volume & CVD
    bar_range = (high_s - low_s).replace(0, 1e-4)
    bar_delta = ((close_s - open_s) / bar_range) * vol_s
    cvd = bar_delta.cumsum()
    cvd_slope = cvd.diff(3).fillna(0.0)
    vol_sma = vol_s.rolling(20, min_periods=1).mean()

    # Supertrend
    hl2 = (high_s + low_s) / 2.0
    st_dir = np.where(close_s >= hl2, 1, -1)

    min_move = 1.0 if sym in ["NATURALGAS", "NG=F"] else 5.0

    best_config = None
    best_score = -999999

    for min_adx_val in [15.0, 18.0, 20.0, 22.0]:
        for max_chop_val in [45.0, 50.0, 55.0, 60.0]:
            for t1_mult in [0.9, 1.0, 1.2]:
                for sl_mult in [0.8, 1.0, 1.2]:
                    for rsi_filter in [True, False]:
                        for overext_mult in [2.2, 2.8, 3.5]:
                            trades = []
                            active = None
                            cooldown = 0

                            for i in range(20, len(df)):
                                bh = float(high_s.iloc[i])
                                bl = float(low_s.iloc[i])
                                bc = float(close_s.iloc[i])
                                bo = float(open_s.iloc[i])
                                bt = str(df.index[i])
                                c_atr = max(bc * 0.0025, float(atr.iloc[i]))

                                # Active Trade Management
                                if active is not None:
                                    if active["type"] == "CALL":
                                        if bh >= active["t2"]:
                                            active["gain"] = round(active["t2"] - active["entry"], 2)
                                            active["outcome"] = "TARGET_2_HIT"
                                            active["exit_time"] = bt
                                            trades.append(active)
                                            active = None
                                            cooldown = 3
                                        elif bh >= active["t1"] and not active["t1_hit"]:
                                            active["t1_hit"] = True
                                            active["sl"] = active["entry"]  # Trail to BE
                                        elif bl <= active["sl"]:
                                            if active["t1_hit"]:
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
                                            cooldown = 3
                                        elif bl <= active["t1"] and not active["t1_hit"]:
                                            active["t1_hit"] = True
                                            active["sl"] = active["entry"]  # Trail to BE
                                        elif bh >= active["sl"]:
                                            if active["t1_hit"]:
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

                                # Entry Trigger Conditions
                                if active is None and chop.iloc[i] <= max_chop_val and adx.iloc[i] >= min_adx_val:
                                    dist_21 = abs(bc - ema21.iloc[i]) / c_atr
                                    if dist_21 <= overext_mult:
                                        # Technical Confluence
                                        is_bull = (bc >= ema9.iloc[i]) and (ema9.iloc[i] >= ema21.iloc[i]) and (bc >= ema50.iloc[i]) and (st_dir[i] == 1)
                                        is_bear = (bc <= ema9.iloc[i]) and (ema9.iloc[i] <= ema21.iloc[i]) and (bc <= ema50.iloc[i]) and (st_dir[i] == -1)

                                        if rsi_filter:
                                            is_bull = is_bull and (rsi.iloc[i] <= 70.0)
                                            is_bear = is_bear and (rsi.iloc[i] >= 30.0)

                                        vol_ok_bull = (cvd_slope.iloc[i] >= 0) or (vol_s.iloc[i] >= 0.95 * vol_sma.iloc[i] and bc >= bo)
                                        vol_ok_bear = (cvd_slope.iloc[i] <= 0) or (vol_s.iloc[i] >= 0.95 * vol_sma.iloc[i] and bc <= bo)

                                        exp_gain = t1_mult * c_atr
                                        if exp_gain >= min_move:
                                            if is_bull and vol_ok_bull:
                                                active = {
                                                    "type": "CALL",
                                                    "entry": bc,
                                                    "t1": round(bc + t1_mult * c_atr, 2),
                                                    "t2": round(bc + 2.0 * c_atr, 2),
                                                    "sl": round(bc - sl_mult * c_atr, 2),
                                                    "t1_hit": False,
                                                    "entry_time": bt,
                                                }
                                            elif is_bear and vol_ok_bear:
                                                active = {
                                                    "type": "PUT",
                                                    "entry": bc,
                                                    "t1": round(bc - t1_mult * c_atr, 2),
                                                    "t2": round(bc - 2.0 * c_atr, 2),
                                                    "sl": round(bc + sl_mult * c_atr, 2),
                                                    "t1_hit": False,
                                                    "entry_time": bt,
                                                }

                            if len(trades) >= 5:
                                wins = [t for t in trades if t["gain"] > 0]
                                losses = [t for t in trades if t["gain"] < 0]
                                wr = (len(wins) / len(trades)) * 100.0
                                net_pts = sum(t["gain"] for t in trades)
                                t1_cnt = sum(1 for t in trades if t["outcome"] in ["TARGET_1_HIT", "TARGET_2_HIT"])
                                t2_cnt = sum(1 for t in trades if t["outcome"] == "TARGET_2_HIT")

                                # Scoring formula balancing high win rate and total net points
                                score = (wr * 10.0) + (net_pts / (c_atr * 0.1))

                                if wr >= 70.0 and score > best_score:
                                    best_score = score
                                    best_config = {
                                        "min_adx": min_adx_val,
                                        "max_chop": max_chop_val,
                                        "t1_mult": t1_mult,
                                        "sl_mult": sl_mult,
                                        "rsi_filter": rsi_filter,
                                        "overext_mult": overext_mult,
                                        "total_trades": len(trades),
                                        "winning_trades": len(wins),
                                        "losing_trades": len(losses),
                                        "target_1_hits": t1_cnt,
                                        "target_2_hits": t2_cnt,
                                        "win_rate": round(wr, 1),
                                        "net_points": round(net_pts, 2),
                                    }

    if best_config:
        print(f"✅ OPTIMAL STRATEGY CALIBRATED FOR {sym}:")
        print(f"   🏆 Win Rate: {best_config['win_rate']}% ({best_config['winning_trades']}/{best_config['total_trades']})")
        print(f"   💰 Net Points: {best_config['net_points']:+,.2f} pts")
        print(f"   🎯 Target 1 Hits: {best_config['target_1_hits']} | Target 2 Hits: {best_config['target_2_hits']}")
        print(f"   ⚙️ Parameters: ADX>={best_config['min_adx']} | CHOP<={best_config['max_chop']} | T1={best_config['t1_mult']}*ATR | SL={best_config['sl_mult']}*ATR | RSI Filter: {best_config['rsi_filter']} | Overextension: {best_config['overext_mult']}*ATR")
    else:
        print(f"⚠️ No configuration exceeded 70% threshold with >=5 trades on baseline sweep.")

if __name__ == "__main__":
    for s in ["^NSEI", "^NSEBANK", "^BSESN", "CRUDEOIL", "NATURALGAS"]:
        diagnose_symbol(s)
