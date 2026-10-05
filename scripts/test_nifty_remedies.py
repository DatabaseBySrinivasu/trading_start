import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.market_data import market_data_service

def test_remedies():
    df = market_data_service.get_historical_candles("^NSEI", period="60d", interval="15m")
    if df is None:
        return

    close_s = df["close"]
    high_s = df["high"]
    low_s = df["low"]
    open_s = df["open"]
    vol_s = df["volume"]
    times = [str(t) for t in df.index]

    ema9_s = close_s.ewm(span=9, adjust=False).mean()
    ema21_s = close_s.ewm(span=21, adjust=False).mean()
    ema50_s = close_s.ewm(span=50, adjust=False).mean()
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
    chop_s = (100.0 * np.log10(np.maximum(1e-4, sum_tr_14 / denom)) / np.log10(14)).fillna(50.0).clip(0.0, 100.0)

    hl2 = (high_s + low_s) / 2.0
    st_dir = np.where(close_s >= hl2, 1, -1)

    # EMA 9 3-bar slope
    ema9_slope = (ema9_s - ema9_s.shift(3)) / atr_s

    # Candle body ratio
    bar_range = (high_s - low_s).replace(0, 1e-4)
    body_ratio = (close_s - open_s).abs() / bar_range

    best_results = []

    for min_gap in [0.15, 0.20, 0.25, 0.30]:
        for min_body in [0.35, 0.45, 0.55]:
            for t1_mult in [0.75, 0.9, 1.0]:
                for sl_mult in [0.9, 1.1, 1.3]:
                    trades = []
                    active = None

                    for i in range(25, len(df)):
                        bh = float(high_s.iloc[i])
                        bl = float(low_s.iloc[i])
                        bc = float(close_s.iloc[i])
                        bo = float(open_s.iloc[i])
                        bt = times[i]
                        c_atr = max(bc * 0.003, float(atr_s.iloc[i]))

                        time_part = bt.split(" ")[1] if " " in bt else ""
                        # Timing filter: Avoid 11:30 - 13:00 and after 15:00
                        if "11:30" <= time_part <= "13:00" or time_part >= "15:05":
                            is_good_time = False
                        else:
                            is_good_time = True

                        if active is not None:
                            if active["type"] == "CALL":
                                if bh >= active["t2"]:
                                    active["gain"] = round(active["t2"] - active["entry"], 2)
                                    trades.append(active)
                                    active = None
                                elif bh >= active["t1"] and not active.get("t1_hit"):
                                    active["t1_hit"] = True
                                    active["sl"] = active["entry"]
                                elif bl <= active["sl"]:
                                    if active.get("t1_hit"):
                                        active["gain"] = round(active["t1"] - active["entry"], 2)
                                    else:
                                        active["gain"] = round(active["sl"] - active["entry"], 2)
                                    trades.append(active)
                                    active = None
                            elif active["type"] == "PUT":
                                if bl <= active["t2"]:
                                    active["gain"] = round(active["entry"] - active["t2"], 2)
                                    trades.append(active)
                                    active = None
                                elif bl <= active["t1"] and not active.get("t1_hit"):
                                    active["t1_hit"] = True
                                    active["sl"] = active["entry"]
                                elif bh >= active["sl"]:
                                    if active.get("t1_hit"):
                                        active["gain"] = round(active["entry"] - active["t1"], 2)
                                    else:
                                        active["gain"] = round(active["entry"] - active["sl"], 2)
                                    trades.append(active)
                                    active = None

                        if active is None and is_good_time and chop_s.iloc[i] <= 52.0:
                            dist_21 = abs(bc - ema21_s.iloc[i]) / c_atr
                            if dist_21 <= 2.2:
                                e9 = ema9_s.iloc[i]
                                e21 = ema21_s.iloc[i]
                                e50 = ema50_s.iloc[i]
                                rsi_val = rsi_s.iloc[i]
                                slope_val = ema9_slope.iloc[i]
                                gap_val = abs(e9 - e21) / c_atr
                                b_ratio = body_ratio.iloc[i]

                                is_bull = (
                                    (e9 > e21)
                                    and (bc > e50)
                                    and (bc >= e9)
                                    and (45.0 <= rsi_val <= 68.0)
                                    and (st_dir[i] == 1)
                                    and (gap_val >= min_gap)
                                    and (slope_val > 0.1)
                                    and (b_ratio >= min_body)
                                    and (bc > bo)
                                )

                                is_bear = (
                                    (e9 < e21)
                                    and (bc < e50)
                                    and (bc <= e9)
                                    and (32.0 <= rsi_val <= 55.0)
                                    and (st_dir[i] == -1)
                                    and (gap_val >= min_gap)
                                    and (slope_val < -0.1)
                                    and (b_ratio >= min_body)
                                    and (bc < bo)
                                )

                                if is_bull:
                                    active = {
                                        "type": "CALL",
                                        "entry": bc,
                                        "t1": round(bc + t1_mult * c_atr, 2),
                                        "t2": round(bc + 2.5 * c_atr, 2),
                                        "sl": round(bc - sl_mult * c_atr, 2),
                                        "t1_hit": False,
                                    }
                                elif is_bear:
                                    active = {
                                        "type": "PUT",
                                        "entry": bc,
                                        "t1": round(bc - t1_mult * c_atr, 2),
                                        "t2": round(bc - 2.5 * c_atr, 2),
                                        "sl": round(bc + sl_mult * c_atr, 2),
                                        "t1_hit": False,
                                    }

                    if len(trades) >= 8:
                        wins = [t for t in trades if t["gain"] > 0]
                        losses = [t for t in trades if t["gain"] < 0]
                        wr = round((len(wins) / len(trades)) * 100.0, 1)
                        net_pts = round(sum(t["gain"] for t in trades), 2)
                        tot_g = sum(t["gain"] for t in wins)
                        tot_l = abs(sum(t["gain"] for t in losses))
                        pf = round(tot_g / max(1.0, tot_l), 2) if tot_l > 0 else 99.0
                        best_results.append({
                            "min_gap": min_gap,
                            "min_body": min_body,
                            "t1": t1_mult,
                            "sl": sl_mult,
                            "trades": len(trades),
                            "wins": len(wins),
                            "losses": len(losses),
                            "wr": wr,
                            "net_pts": net_pts,
                            "pf": pf,
                        })

    if best_results:
        best_results.sort(key=lambda x: (x["wr"], x["pf"], x["net_pts"]), reverse=True)
        print("🎯 TOP NIFTY 50 CALIBRATED REMEDIES (Targeting 70-80%+ WR):")
        for idx, r in enumerate(best_results[:8], 1):
            print(f"#{idx} -> 🏆 WIN RATE: {r['wr']}% ({r['wins']}W / {r['losses']}L / {r['trades']}T) | Net: {r['net_pts']:+,.2f} pts | PF: {r['pf']} (Gap>={r['min_gap']}, Body>={r['min_body']}, T1={r['t1']}, SL={r['sl']})")

if __name__ == "__main__":
    test_remedies()
