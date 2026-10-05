import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.market_data import market_data_service

def diagnose_nifty_losses():
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

    bar_range = (high_s - low_s).replace(0, 1e-4)
    bar_delta = ((close_s - open_s) / bar_range) * vol_s
    cvd_s = bar_delta.cumsum()
    cvd_slope = cvd_s.diff(3).fillna(0.0)
    vol_sma_s = vol_s.rolling(20, min_periods=1).mean()

    # VWAP (Daily anchored)
    # Estimate VWAP
    cum_vol = vol_s.cumsum()
    cum_vol_price = (close_s * vol_s).cumsum()
    vwap_s = cum_vol_price / cum_vol.replace(0, 1)

    trades = []
    active = None

    for i in range(25, len(df)):
        bh = float(high_s.iloc[i])
        bl = float(low_s.iloc[i])
        bc = float(close_s.iloc[i])
        bo = float(open_s.iloc[i])
        bt = times[i]
        c_atr = max(bc * 0.003, float(atr_s.iloc[i]))

        if active is not None:
            if active["type"] == "CALL":
                if bh >= active["t2"]:
                    active["gain"] = round(active["t2"] - active["entry"], 2)
                    active["outcome"] = "TARGET_2_HIT"
                    trades.append(active)
                    active = None
                elif bh >= active["t1"] and not active.get("t1_hit"):
                    active["t1_hit"] = True
                    active["sl"] = active["entry"]
                elif bl <= active["sl"]:
                    if active.get("t1_hit"):
                        active["gain"] = round(active["t1"] - active["entry"], 2)
                        active["outcome"] = "TARGET_1_HIT"
                    else:
                        active["gain"] = round(active["sl"] - active["entry"], 2)
                        active["outcome"] = "STOP_LOSS_HIT"
                    trades.append(active)
                    active = None
            elif active["type"] == "PUT":
                if bl <= active["t2"]:
                    active["gain"] = round(active["entry"] - active["t2"], 2)
                    active["outcome"] = "TARGET_2_HIT"
                    trades.append(active)
                    active = None
                elif bl <= active["t1"] and not active.get("t1_hit"):
                    active["t1_hit"] = True
                    active["sl"] = active["entry"]
                elif bh >= active["sl"]:
                    if active.get("t1_hit"):
                        active["gain"] = round(active["entry"] - active["t1"], 2)
                        active["outcome"] = "TARGET_1_HIT"
                    else:
                        active["gain"] = round(active["entry"] - active["sl"], 2)
                        active["outcome"] = "STOP_LOSS_HIT"
                    trades.append(active)
                    active = None

        if active is None and chop_s.iloc[i] <= 50.0:
            dist_21 = abs(bc - ema21_s.iloc[i]) / c_atr
            if dist_21 <= 2.2:
                e9 = ema9_s.iloc[i]
                e21 = ema21_s.iloc[i]
                e50 = ema50_s.iloc[i]
                rsi_val = rsi_s.iloc[i]

                is_bull = (e9 >= e21) and (bc >= e50) and (bc >= e9) and (42.0 <= rsi_val <= 68.0) and (st_dir[i] == 1)
                is_bear = (e9 <= e21) and (bc <= e50) and (bc <= e9) and (32.0 <= rsi_val <= 58.0) and (st_dir[i] == -1)

                vol_bull = (cvd_slope.iloc[i] >= 0) and (vol_s.iloc[i] >= 1.0 * vol_sma_s.iloc[i] or bc >= bo)
                vol_bear = (cvd_slope.iloc[i] <= 0) and (vol_s.iloc[i] >= 1.0 * vol_sma_s.iloc[i] or bc <= bo)

                if is_bull and vol_bull:
                    active = {
                        "type": "CALL",
                        "entry": bc,
                        "entry_time": bt,
                        "t1": round(bc + 0.75 * c_atr, 2),
                        "t2": round(bc + 2.5 * c_atr, 2),
                        "sl": round(bc - 1.2 * c_atr, 2),
                        "t1_hit": False,
                        "rsi": rsi_val,
                        "chop": chop_s.iloc[i],
                        "ema9_21_gap": round((e9 - e21) / c_atr, 2),
                        "vol_ratio": round(vol_s.iloc[i] / max(1, vol_sma_s.iloc[i]), 2),
                    }
                elif is_bear and vol_bear:
                    active = {
                        "type": "PUT",
                        "entry": bc,
                        "entry_time": bt,
                        "t1": round(bc - 0.75 * c_atr, 2),
                        "t2": round(bc - 2.5 * c_atr, 2),
                        "sl": round(bc + 1.2 * c_atr, 2),
                        "t1_hit": False,
                        "rsi": rsi_val,
                        "chop": chop_s.iloc[i],
                        "ema9_21_gap": round((e21 - e9) / c_atr, 2),
                        "vol_ratio": round(vol_s.iloc[i] / max(1, vol_sma_s.iloc[i]), 2),
                    }

    print(f"Total trades: {len(trades)}")
    losses = [t for t in trades if t["gain"] < 0]
    wins = [t for t in trades if t["gain"] > 0]
    print(f"Wins: {len(wins)}, Losses: {len(losses)}, Win Rate: {len(wins)/len(trades)*100:.1f}%")
    print("\nSAMPLE LOSSES ON NIFTY 50 (Why did they fail?):")
    for l in losses[:6]:
        print(f"Loss at {l['entry_time']}: {l['type']} @ {l['entry']}, Loss: {l['gain']} pts | RSI: {l['rsi']:.1f}, Chop: {l['chop']:.1f}, EMA 9/21 Gap: {l['ema9_21_gap']}, Vol Ratio: {l['vol_ratio']}")

if __name__ == "__main__":
    diagnose_nifty_losses()
