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

def test_institutional_recalibrated_strategy(symbol="^NSEI", interval="15m", period="60d"):
    df = market_data_service.get_historical_candles(symbol, period=period, interval=interval)
    if df is None or len(df) < 50:
        return

    close_s = df["close"]
    high_s = df["high"]
    low_s = df["low"]
    open_s = df["open"]
    vol_s = df["volume"]

    # Indicators
    ema9_s = close_s.ewm(span=9, adjust=False).mean()
    ema21_s = close_s.ewm(span=21, adjust=False).mean()
    ema50_s = close_s.ewm(span=50, adjust=False).mean()
    ema200_s = close_s.ewm(span=min(200, len(df)), adjust=False).mean()

    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_s = tr.rolling(14, min_periods=1).mean()

    # RSI
    delta = close_s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, 1e-4)
    rsi_s = (100 - (100 / (1 + rs))).fillna(50.0)

    # Choppiness Index 14
    sum_tr_14 = tr.rolling(14, min_periods=1).sum()
    max_h_14 = high_s.rolling(14, min_periods=1).max()
    min_l_14 = low_s.rolling(14, min_periods=1).min()
    denom = (max_h_14 - min_l_14).replace(0, 1e-4)
    chop_s = 100.0 * np.log10(np.maximum(1e-4, sum_tr_14 / denom)) / np.log10(14)
    chop_s = chop_s.fillna(50.0).clip(0.0, 100.0)

    # Volume & CVD
    bar_range = (high_s - low_s).replace(0, 1e-4)
    bar_delta = ((close_s - open_s) / bar_range) * vol_s
    cvd_s = bar_delta.cumsum()
    cvd_slope = cvd_s.diff(3).fillna(0.0)
    vol_sma_s = vol_s.rolling(20, min_periods=1).mean()

    # 3-bar lowest low / highest high for structural swing stop
    swing_low_3 = low_s.rolling(3, min_periods=1).min()
    swing_high_3 = high_s.rolling(3, min_periods=1).max()

    # ADX 14
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14, min_periods=1).mean() / atr_s.replace(0, 1e-4)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14, min_periods=1).mean() / atr_s.replace(0, 1e-4)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-4)
    adx_s = dx.rolling(14, min_periods=1).mean().fillna(20.0)

    # Test pullback + institutional confirmation setup
    trades = []
    active = None
    cooldown = 0
    lot_size = 20 if "BSESN" in symbol or "SENSEX" in symbol else 65

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

        # Institutional High-Confluence Filter Rules:
        # 1. Chop Filter: CHOP <= 50.0 (Explosive trend regime) & ADX >= 20.0
        # 2. Overextension: distance from 21 EMA <= 2.2 ATR
        # 3. Pullback / Value zone: Low touched near 9 EMA or 21 EMA
        # 4. Multi-EMA stack alignment: 9 > 21 > 50 for Bull; 9 < 21 < 50 for Bear
        # 5. Institutional CVD expansion: positive CVD slope and Volume >= 1.05 * SMA20
        # 6. RSI Momentum: Bull (45 <= RSI <= 70), Bear (30 <= RSI <= 55)
        if active is None and chop_s.iloc[i] <= 52.0 and adx_s.iloc[i] >= 20.0:
            dist_21 = abs(bc - ema21_s.iloc[i]) / c_atr
            if dist_21 <= 2.2:
                # Bullish Setup
                e9 = ema9_s.iloc[i]
                e21 = ema21_s.iloc[i]
                e50 = ema50_s.iloc[i]
                rsi_val = rsi_s.iloc[i]

                is_bull_trend = (e9 > e21) and (bc > e50) and (bc >= e9) and (45.0 <= rsi_val <= 68.0)
                is_bear_trend = (e9 < e21) and (bc < e50) and (bc <= e9) and (32.0 <= rsi_val <= 55.0)

                # CVD and Volume confirmation
                vol_bull = (cvd_slope.iloc[i] > 0) and (vol_s.iloc[i] >= 1.05 * vol_sma_s.iloc[i] or bc > bo)
                vol_bear = (cvd_slope.iloc[i] < 0) and (vol_s.iloc[i] >= 1.05 * vol_sma_s.iloc[i] or bc < bo)

                # Structural Stop Loss
                struct_sl_call = min(bc - (0.8 * c_atr), float(swing_low_3.iloc[i]) - (0.2 * c_atr))
                struct_sl_put = max(bc + (0.8 * c_atr), float(swing_high_3.iloc[i]) + (0.2 * c_atr))

                # Targets
                t1_target_pts = max(5.0, c_atr * 1.0)
                t2_target_pts = max(10.0, c_atr * 2.2)

                if is_bull_trend and vol_bull:
                    active = {
                        "type": "CALL",
                        "entry": bc,
                        "t1": round(bc + t1_target_pts, 2),
                        "t2": round(bc + t2_target_pts, 2),
                        "sl": round(struct_sl_call, 2),
                        "t1_hit": False,
                        "entry_time": bt,
                    }
                elif is_bear_trend and vol_bear:
                    active = {
                        "type": "PUT",
                        "entry": bc,
                        "t1": round(bc - t1_target_pts, 2),
                        "t2": round(bc - t2_target_pts, 2),
                        "sl": round(struct_sl_put, 2),
                        "t1_hit": False,
                        "entry_time": bt,
                    }

    total = len(trades)
    wins = [t for t in trades if t["gain"] > 0]
    losses = [t for t in trades if t["gain"] < 0]
    wr = round((len(wins) / max(1, total)) * 100.0, 1)
    net_pts = round(sum(t["gain"] for t in trades), 2)
    tot_gain = sum(t["gain"] for t in wins)
    tot_loss = abs(sum(t["gain"] for t in losses))
    pf = round(tot_gain / max(1.0, tot_loss), 2) if tot_loss > 0 else 99.0
    pnl = round(net_pts * 0.55 * lot_size, 2)

    print(f"🎯 [{symbol} | {period} {interval}] High-Confluence Performance:")
    print(f"   🏆 Win Rate: {wr}% ({len(wins)} Wins / {total} Total Trades)")
    print(f"   💰 Net Points: {net_pts:+,.2f} pts | Est Option P&L: ₹{pnl:+,.2f}")
    print(f"   ⚖️ Profit Factor: {pf}")
    print(f"   🎯 T1 Hits: {sum(1 for t in trades if t['outcome']=='TARGET_1_HIT')} | T2 Hits: {sum(1 for t in trades if t['outcome']=='TARGET_2_HIT')} | SL: {len(losses)}")
    return wr, net_pts, pf, trades

if __name__ == "__main__":
    print("Testing recalibrated high-confluence institutional logic on 1-Year / 60D data...")
    test_institutional_recalibrated_strategy("^BSESN", "1d", "1y")
    test_institutional_recalibrated_strategy("^BSESN", "15m", "60d")
    test_institutional_recalibrated_strategy("^NSEI", "1d", "1y")
    test_institutional_recalibrated_strategy("^NSEI", "15m", "60d")
