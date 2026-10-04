import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.market_data import market_data_service

def diagnose_nifty_daily():
    df = market_data_service.get_historical_candles("^NSEI", period="1y", interval="1d")
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

    bar_range = (high_s - low_s).replace(0, 1e-4)
    body_ratio = (close_s - open_s).abs() / bar_range
    ema9_slope = (ema9_s - ema9_s.shift(3)) / atr_s

    best = []
    for min_gap in [0.10, 0.15, 0.20, 0.25]:
        for min_body in [0.30, 0.40, 0.50]:
            for t1_mult in [0.75, 0.9, 1.0]:
                for sl_mult in [0.8, 1.0, 1.2]:
                    for rsi_lo, rsi_hi in [(40.0, 70.0), (45.0, 68.0), (38.0, 72.0)]:
                        trades = []
                        active = None
                        for i in range(20, len(df)):
                            bh = float(high_s.iloc[i])
                            bl = float(low_s.iloc[i])
                            bc = float(close_s.iloc[i])
                            bo = float(open_s.iloc[i])
                            bt = times[i]
                            c_atr = max(bc * 0.003, float(atr_s.iloc[i]))

                            if active is not None:
                                if active["type"] == "CALL":
                                    if bh >= active["t2"]:
                                        trades.append(active["t2"] - active["entry"])
                                        active = None
                                    elif bh >= active["t1"] and not active.get("t1_hit"):
                                        active["t1_hit"] = True
                                        active["sl"] = active["entry"]
                                    elif bl <= active["sl"]:
                                        if active.get("t1_hit"):
                                            trades.append(active["t1"] - active["entry"])
                                        else:
                                            trades.append(active["sl"] - active["entry"])
                                        active = None
                                elif active["type"] == "PUT":
                                    if bl <= active["t2"]:
                                        trades.append(active["entry"] - active["t2"])
                                        active = None
                                    elif bl <= active["t1"] and not active.get("t1_hit"):
                                        active["t1_hit"] = True
                                        active["sl"] = active["entry"]
                                    elif bh >= active["sl"]:
                                        if active.get("t1_hit"):
                                            trades.append(active["entry"] - active["t1"])
                                        else:
                                            trades.append(active["entry"] - active["sl"])
                                        active = None

                            if active is None and chop_s.iloc[i] <= 55.0:
                                dist_21 = abs(bc - ema21_s.iloc[i]) / c_atr
                                if dist_21 <= 2.2:
                                    e9 = ema9_s.iloc[i]
                                    e21 = ema21_s.iloc[i]
                                    e50 = ema50_s.iloc[i]
                                    rsi_val = rsi_s.iloc[i]
                                    gap = abs(e9 - e21) / c_atr
                                    slp = ema9_slope.iloc[i]
                                    br = body_ratio.iloc[i]

                                    is_bull = (e9 >= e21) and (bc >= e50) and (bc >= e9) and (rsi_lo <= rsi_val <= rsi_hi) and (st_dir[i] == 1) and (gap >= min_gap) and (slp > 0.05) and (br >= min_body) and (bc > bo)
                                    is_bear = (e9 <= e21) and (bc <= e50) and (bc <= e9) and ((100 - rsi_hi) <= rsi_val <= (100 - rsi_lo)) and (st_dir[i] == -1) and (gap >= min_gap) and (slp < -0.05) and (br >= min_body) and (bc < bo)

                                    if is_bull:
                                        active = {"type": "CALL", "entry": bc, "t1": bc + t1_mult * c_atr, "t2": bc + 2.5 * c_atr, "sl": bc - sl_mult * c_atr, "t1_hit": False}
                                    elif is_bear:
                                        active = {"type": "PUT", "entry": bc, "t1": bc - t1_mult * c_atr, "t2": bc - 2.5 * c_atr, "sl": bc + sl_mult * c_atr, "t1_hit": False}

                        if len(trades) >= 5:
                            wins = [t for t in trades if t > 0]
                            losses = [t for t in trades if t < 0]
                            wr = round(len(wins) / len(trades) * 100, 1)
                            net = round(sum(trades), 2)
                            best.append({"wr": wr, "wins": len(wins), "losses": len(losses), "n": len(trades), "net": net, "gap": min_gap, "body": min_body, "t1": t1_mult, "sl": sl_mult, "rsi": (rsi_lo, rsi_hi)})

    best.sort(key=lambda x: (x["wr"], x["net"]), reverse=True)
    print("TOP NIFTY DAILY CALIBRATIONS:")
    for b in best[:10]:
        print(f"WR: {b['wr']}% ({b['wins']}W / {b['losses']}L / {b['n']}T) | Net: {b['net']:+,.2f} | Gap>={b['gap']}, Body>={b['body']}, T1={b['t1']}, SL={b['sl']}, RSI={b['rsi']}")

if __name__ == "__main__":
    diagnose_nifty_daily()
