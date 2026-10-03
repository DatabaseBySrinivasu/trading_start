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

def sweep_weighted_confluence(symbol="^NSEI", interval="15m", period="60d"):
    df = market_data_service.get_historical_candles(symbol, period=period, interval=interval)
    if df is None or len(df) < 50:
        return

    close_s = df["close"]
    high_s = df["high"]
    low_s = df["low"]
    open_s = df["open"]
    vol_s = df["volume"]

    ema9_s = close_s.ewm(span=9, adjust=False).mean()
    ema21_s = close_s.ewm(span=21, adjust=False).mean()
    ema50_s = close_s.ewm(span=min(50, len(df)), adjust=False).mean()
    ema200_s = close_s.ewm(span=min(200, len(df)), adjust=False).mean()

    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_s = tr.rolling(14, min_periods=1).mean()

    delta = close_s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, 1e-4)
    rsi_s = (100 - (100 / (1 + rs))).fillna(50.0)

    sum_tr_14 = tr.rolling(14, min_periods=1).sum()
    max_h_14 = high_s.rolling(14, min_periods=1).max()
    min_l_14 = low_s.rolling(14, min_periods=1).min()
    denom = (max_h_14 - min_l_14).replace(0, 1e-4)
    chop_s = 100.0 * np.log10(np.maximum(1e-4, sum_tr_14 / denom)) / np.log10(14)
    chop_s = chop_s.fillna(50.0).clip(0.0, 100.0)

    bar_range = (high_s - low_s).replace(0, 1e-4)
    bar_delta = ((close_s - open_s) / bar_range) * vol_s
    cvd_s = bar_delta.cumsum()
    cvd_slope = cvd_s.diff(3).fillna(0.0)
    vol_sma_s = vol_s.rolling(20, min_periods=1).mean()

    hl2 = (high_s + low_s) / 2.0
    st_dir = np.where(close_s >= hl2, 1, -1)

    results = []
    lot_size = 20 if "BSESN" in symbol or "SENSEX" in symbol else 65

    # Grid search for highest accuracy
    for conf_cut in [50.0, 55.0, 60.0, 65.0, 70.0]:
        for max_chop in [45.0, 50.0, 55.0, 60.0]:
            for t1_atr in [0.8, 1.0, 1.2]:
                for sl_atr in [0.6, 0.8, 1.0]:
                    trades = []
                    active = None
                    cooldown = 0

                    for i in range(20, len(df)):
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
                                    active["sl"] = active["entry"]  # Trail to BE
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
                                    active["sl"] = active["entry"]  # Trail to BE
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

                        if active is None and chop_s.iloc[i] <= max_chop:
                            dist_21 = abs(bc - ema21_s.iloc[i]) / c_atr
                            if dist_21 <= 2.8:
                                # Weighted Scoring
                                # Pillar 1: Trend Alignment
                                p1_bull = (bc >= ema9_s.iloc[i]) and (ema9_s.iloc[i] >= ema21_s.iloc[i]) and (st_dir[i] == 1)
                                p1_bear = (bc <= ema9_s.iloc[i]) and (ema9_s.iloc[i] <= ema21_s.iloc[i]) and (st_dir[i] == -1)

                                # Pillar 2: Volume & Order Flow
                                p2_bull = (cvd_slope.iloc[i] >= 0) or (vol_s.iloc[i] >= 1.05 * vol_sma_s.iloc[i] and bc >= bo)
                                p2_bear = (cvd_slope.iloc[i] <= 0) or (vol_s.iloc[i] >= 1.05 * vol_sma_s.iloc[i] and bc <= bo)

                                # Pillar 3: Location / Momentum
                                p3_bull = bc >= ema9_s.iloc[i] and bc >= bo and rsi_s.iloc[i] <= 70.0
                                p3_bear = bc <= ema9_s.iloc[i] and bc <= bo and rsi_s.iloc[i] >= 30.0

                                bull_score = (30.0 if p1_bull else 0.0) + (25.0 if p2_bull else 0.0) + (25.0 if p3_bull else 0.0) + (10.0 if chop_s.iloc[i] < 45.0 else 0.0)
                                bear_score = (30.0 if p1_bear else 0.0) + (25.0 if p2_bear else 0.0) + (25.0 if p3_bear else 0.0) + (10.0 if chop_s.iloc[i] < 45.0 else 0.0)

                                if bull_score >= conf_cut and bull_score > bear_score:
                                    active = {
                                        "type": "CALL",
                                        "entry": bc,
                                        "t1": round(bc + t1_atr * c_atr, 2),
                                        "t2": round(bc + 2.0 * c_atr, 2),
                                        "sl": round(bc - sl_atr * c_atr, 2),
                                        "t1_hit": False,
                                        "entry_time": bt,
                                    }
                                elif bear_score >= conf_cut and bear_score > bull_score:
                                    active = {
                                        "type": "PUT",
                                        "entry": bc,
                                        "t1": round(bc - t1_atr * c_atr, 2),
                                        "t2": round(bc - 2.0 * c_atr, 2),
                                        "sl": round(bc + sl_atr * c_atr, 2),
                                        "t1_hit": False,
                                        "entry_time": bt,
                                    }

                    if len(trades) >= 10:
                        wins = [t for t in trades if t["gain"] > 0]
                        wr = round((len(wins) / len(trades)) * 100.0, 1)
                        net_pts = round(sum(t["gain"] for t in trades), 2)
                        p_gain = sum(t["gain"] for t in wins)
                        p_loss = abs(sum(t["gain"] for t in trades if t["gain"] < 0))
                        pf = round(p_gain / max(1.0, p_loss), 2)
                        results.append({
                            "conf_cut": conf_cut,
                            "max_chop": max_chop,
                            "t1_atr": t1_atr,
                            "sl_atr": sl_atr,
                            "trades": len(trades),
                            "wins": len(wins),
                            "wr": wr,
                            "net_pts": net_pts,
                            "pf": pf,
                        })

    if results:
        # Sort by Win Rate and Profit Factor
        results.sort(key=lambda x: (x["wr"], x["pf"], x["net_pts"]), reverse=True)
        top = results[0]
        print(f"\n🏆 TOP OPTIMIZED CALIBRATION FOR {symbol} ({period} {interval}):")
        print(f"   Win Rate: {top['wr']}% ({top['wins']}/{top['trades']} trades)")
        print(f"   Net Points: {top['net_pts']:+,.2f} pts | Profit Factor: {top['pf']}")
        print(f"   Best Parameters: Conf>={top['conf_cut']}% | Chop<={top['max_chop']} | T1={top['t1_atr']}*ATR | SL={top['sl_atr']}*ATR")

        print("\nTop 5 Candidates:")
        for r in results[:5]:
            print(f"   Conf:{r['conf_cut']}% | Chop:{r['max_chop']} | T1:{r['t1_atr']} | SL:{r['sl_atr']} -> WR: {r['wr']}% | Net: {r['net_pts']:+,.2f} pts | PF: {r['pf']} | N={r['trades']}")
        return top
    return None

if __name__ == "__main__":
    print("=" * 65)
    print("🚀 SWEEPING WEIGHTED CONFLUENCE PARAMETERS ACROSS SENSEX & NIFTY 50")
    print("=" * 65)
    sweep_weighted_confluence("^BSESN", "15m", "60d")
    sweep_weighted_confluence("^NSEI", "15m", "60d")
    sweep_weighted_confluence("^BSESN", "1d", "1y")
    sweep_weighted_confluence("^NSEI", "1d", "1y")
