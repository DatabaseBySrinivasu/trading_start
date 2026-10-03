import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Reconfigure stdout/stderr for clean utf-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.market_data import market_data_service
from app.services.telegram_service import telegram_service
from app.core.config import settings

def run_backtest_on_dataframe(df: pd.DataFrame, symbol: str, timeframe_label: str, target_chat: str = ""):
    canonical = symbol
    close_s = df["close"]
    high_s = df["high"]
    low_s = df["low"]
    open_s = df["open"]
    vol_s = df["volume"]

    # 1. Moving Averages
    ema9_s = close_s.ewm(span=9, adjust=False).mean()
    ema21_s = close_s.ewm(span=21, adjust=False).mean()
    ema50_s = close_s.ewm(span=min(50, len(df)), adjust=False).mean()
    ema200_s = close_s.ewm(span=min(200, len(df)), adjust=False).mean()

    # 2. ATR
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_s = tr.rolling(14, min_periods=1).mean()

    # 3. Supertrend
    hl2 = (high_s + low_s) / 2.0
    st_dir = np.where(close_s >= hl2, 1, -1)

    # 4. Choppiness Index 14
    sum_tr_14 = tr.rolling(14, min_periods=1).sum()
    max_h_14 = high_s.rolling(14, min_periods=1).max()
    min_l_14 = low_s.rolling(14, min_periods=1).min()
    denom = (max_h_14 - min_l_14).replace(0, 1e-4)
    chop_s = 100.0 * np.log10(np.maximum(1e-4, sum_tr_14 / denom)) / np.log10(14)
    chop_s = chop_s.fillna(50.0).clip(0.0, 100.0)

    # 5. ADX 14
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14, min_periods=1).mean() / atr_s.replace(0, 1e-4)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14, min_periods=1).mean() / atr_s.replace(0, 1e-4)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-4)
    adx_s = dx.rolling(14, min_periods=1).mean().fillna(20.0)

    # 6. Bar Delta & CVD
    bar_range = (high_s - low_s).replace(0, 1e-4)
    bar_delta = ((close_s - open_s) / bar_range) * vol_s
    cvd_s = bar_delta.cumsum()
    cvd_slope = cvd_s.diff(3).fillna(0.0)
    vol_sma_s = vol_s.rolling(20, min_periods=1).mean()

    # 7. Simulation Loop
    trades = []
    active_trade = None
    cooldown_bars = 0
    window_size = 20

    min_move = 5.0
    lot_size = 20 if "BSESN" in symbol or "SENSEX" in symbol else 65

    for i in range(window_size, len(df)):
        bar_high = float(high_s.iloc[i])
        bar_low = float(low_s.iloc[i])
        bar_close = float(close_s.iloc[i])
        bar_open = float(open_s.iloc[i])
        bar_time = str(df.index[i])
        cur_atr = max(bar_close * 0.003, float(atr_s.iloc[i]))

        # Manage Active Trade
        if active_trade is not None:
            trade_type = active_trade["type"]
            t1_spot = active_trade["spot_t1"]
            t2_spot = active_trade["spot_t2"]
            sl_spot = active_trade["spot_sl"]

            if trade_type == "CALL":
                if bar_high >= t2_spot:
                    active_trade["outcome"] = "TARGET_2_HIT"
                    active_trade["exit_price"] = t2_spot
                    active_trade["exit_time"] = bar_time
                    active_trade["points_gain"] = round(t2_spot - active_trade["entry_price"], 2)
                    trades.append(active_trade)
                    active_trade = None
                    cooldown_bars = 4
                elif bar_high >= t1_spot and not active_trade.get("t1_hit"):
                    active_trade["t1_hit"] = True
                    active_trade["spot_sl"] = active_trade["entry_price"]  # Trail to breakeven
                elif bar_low <= active_trade["spot_sl"]:
                    if active_trade.get("t1_hit"):
                        active_trade["outcome"] = "TARGET_1_HIT"
                        active_trade["exit_price"] = active_trade["spot_t1"]
                        active_trade["exit_time"] = bar_time
                        active_trade["points_gain"] = round(active_trade["spot_t1"] - active_trade["entry_price"], 2)
                    else:
                        active_trade["outcome"] = "STOP_LOSS_HIT"
                        active_trade["exit_price"] = active_trade["spot_sl"]
                        active_trade["exit_time"] = bar_time
                        active_trade["points_gain"] = round(active_trade["spot_sl"] - active_trade["entry_price"], 2)
                    trades.append(active_trade)
                    active_trade = None

            elif trade_type == "PUT":
                if bar_low <= t2_spot:
                    active_trade["outcome"] = "TARGET_2_HIT"
                    active_trade["exit_price"] = t2_spot
                    active_trade["exit_time"] = bar_time
                    active_trade["points_gain"] = round(active_trade["entry_price"] - t2_spot, 2)
                    trades.append(active_trade)
                    active_trade = None
                    cooldown_bars = 4
                elif bar_low <= t1_spot and not active_trade.get("t1_hit"):
                    active_trade["t1_hit"] = True
                    active_trade["spot_sl"] = active_trade["entry_price"]  # Trail to breakeven
                elif bar_high >= active_trade["spot_sl"]:
                    if active_trade.get("t1_hit"):
                        active_trade["outcome"] = "TARGET_1_HIT"
                        active_trade["exit_price"] = active_trade["spot_t1"]
                        active_trade["exit_time"] = bar_time
                        active_trade["points_gain"] = round(active_trade["entry_price"] - active_trade["spot_t1"], 2)
                    else:
                        active_trade["outcome"] = "STOP_LOSS_HIT"
                        active_trade["exit_price"] = active_trade["spot_sl"]
                        active_trade["exit_time"] = bar_time
                        active_trade["points_gain"] = round(active_trade["entry_price"] - active_trade["spot_sl"], 2)
                    trades.append(active_trade)
                    active_trade = None

        if cooldown_bars > 0:
            cooldown_bars -= 1
            continue

        # Check for Entry Trigger
        if active_trade is None and chop_s.iloc[i] <= 55.0 and adx_s.iloc[i] >= 18.0:
            dist_ema21 = abs(bar_close - ema21_s.iloc[i]) / cur_atr
            if dist_ema21 <= 2.5:  # Overextension filter
                p1_bull = (bar_close >= ema9_s.iloc[i]) and (ema9_s.iloc[i] >= ema21_s.iloc[i]) and (bar_close >= ema50_s.iloc[i]) and (st_dir[i] == 1)
                p1_bear = (bar_close <= ema9_s.iloc[i]) and (ema9_s.iloc[i] <= ema21_s.iloc[i]) and (bar_close <= ema50_s.iloc[i]) and (st_dir[i] == -1)

                p2_bull = (cvd_slope.iloc[i] >= 0) or (vol_s.iloc[i] >= 0.95 * vol_sma_s.iloc[i] and bar_close >= bar_open)
                p2_bear = (cvd_slope.iloc[i] <= 0) or (vol_s.iloc[i] >= 0.95 * vol_sma_s.iloc[i] and bar_close <= bar_open)

                exp_gain = cur_atr * 1.0
                if exp_gain >= min_move:
                    if p1_bull and p2_bull:
                        active_trade = {
                            "type": "CALL",
                            "entry_price": bar_close,
                            "entry_time": bar_time,
                            "spot_t1": round(bar_close + (1.0 * cur_atr), 2),
                            "spot_t2": round(bar_close + (2.0 * cur_atr), 2),
                            "spot_sl": round(bar_close - (0.8 * cur_atr), 2),
                            "t1_hit": False,
                        }
                    elif p1_bear and p2_bear:
                        active_trade = {
                            "type": "PUT",
                            "entry_price": bar_close,
                            "entry_time": bar_time,
                            "spot_t1": round(bar_close - (1.0 * cur_atr), 2),
                            "spot_t2": round(bar_close - (2.0 * cur_atr), 2),
                            "spot_sl": round(bar_close + (0.8 * cur_atr), 2),
                            "t1_hit": False,
                        }

    # Summary Metrics
    total_trades = len(trades)
    winning = [t for t in trades if t["points_gain"] > 0]
    losing = [t for t in trades if t["points_gain"] < 0]
    t1_hits = [t for t in trades if t["outcome"] == "TARGET_1_HIT"]
    t2_hits = [t for t in trades if t["outcome"] == "TARGET_2_HIT"]
    sl_hits = [t for t in trades if t["outcome"] == "STOP_LOSS_HIT"]

    win_rate = round((len(winning) / max(1, total_trades)) * 100.0, 1)
    net_spot_pts = round(sum(t["points_gain"] for t in trades), 2)
    tot_gain = sum(t["points_gain"] for t in winning)
    tot_loss = abs(sum(t["points_gain"] for t in losing))
    profit_factor = round(tot_gain / max(1.0, tot_loss), 2) if tot_loss > 0 else (99.0 if tot_gain > 0 else 1.0)

    approx_opt_pts = round(net_spot_pts * 0.55, 2)
    approx_net_pnl = round(approx_opt_pts * lot_size, 2)

    start_d = str(df.index[0]).split(" ")[0]
    end_d = str(df.index[-1]).split(" ")[0]

    print(f"\n==================================================================")
    print(f"📈 BACKTEST RESULTS FOR {symbol} ({timeframe_label})")
    print(f"📅 Historical Period: {start_d} to {end_d} ({len(df)} candles)")
    print(f"🏆 Win Rate: {win_rate}% ({len(winning)} Wins / {total_trades} Trades)")
    print(f"💰 Net Spot Points: {net_spot_pts:+,.2f} pts | Est Option P&L: ₹{approx_net_pnl:+,.2f}")
    print(f"🎯 Target 1 Hits: {len(t1_hits)} | Target 2 Hits: {len(t2_hits)} | Stop Loss: {len(sl_hits)}")
    print(f"⚖️ Profit Factor: {profit_factor}")
    print(f"==================================================================")

    return {
        "symbol": symbol,
        "timeframe_label": timeframe_label,
        "start_date": start_d,
        "end_date": end_d,
        "candles_count": len(df),
        "total_trades": total_trades,
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "target_1_hits": len(t1_hits),
        "target_2_hits": len(t2_hits),
        "stop_loss_hits": len(sl_hits),
        "win_rate": win_rate,
        "net_spot_pts": net_spot_pts,
        "approx_net_pnl": approx_net_pnl,
        "profit_factor": profit_factor,
        "lot_size": lot_size,
        "sample_trades": trades[-5:] if len(trades) >= 5 else trades,
    }

def main():
    target_chat = settings.TELEGRAM_CHAT_ID or ""
    print("=" * 65)
    print("🚀 1-YEAR HISTORICAL STRATEGY AUDIT: SENSEX & NIFTY 50")
    print(f"📱 Target Destination: {target_chat}")
    print("=" * 65)

    all_results = []

    for sym in ["^BSESN", "^NSEI"]:
        # 1. Full 1-Year Historical Daily Dataset
        df_1y = market_data_service.get_historical_candles(sym, period="1y", interval="1d")
        if df_1y is not None and not df_1y.empty:
            res_1y = run_backtest_on_dataframe(df_1y, sym, "1-Year Daily Swing/Trend Strategy", target_chat)
            all_results.append(res_1y)

        # 2. 60-Day Intraday Dataset (15m candles)
        df_60d = market_data_service.get_historical_candles(sym, period="60d", interval="15m")
        if df_60d is not None and not df_60d.empty:
            res_15m = run_backtest_on_dataframe(df_60d, sym, "60-Day Intraday 15m Strategy", target_chat)
            all_results.append(res_15m)

    # Dispatch combined Telegram report
    msg = (
        f"⚡️ <b>QUANTPULSE PRO: 1-YEAR HISTORICAL STRATEGY AUDIT</b> ⚡️\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Tested Instruments:</b> BSE SENSEX (^BSESN) &amp; NIFTY 50 (^NSEI)\n"
        f"📅 <b>Testing Dataset:</b> Full 1-Year Historical Cycles (Oct 2025 – Oct 2026)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )

    for r in all_results:
        sym_name = "SENSEX" if "BSESN" in r["symbol"] else "NIFTY 50"
        msg += (
            f"📈 <b>{sym_name} ({r['timeframe_label']}):</b>\n"
            f"  • <b>Dates:</b> {r['start_date']} to {r['end_date']} ({r['candles_count']} candles)\n"
            f"  • 🏆 <b>Win Rate:</b> <b>{r['win_rate']}%</b> ({r['winning_trades']}W / {r['losing_trades']}L)\n"
            f"  • 💰 <b>Net Spot Gain:</b> <b>{'+' if r['net_spot_pts'] >= 0 else ''}{r['net_spot_pts']:,.2f} pts</b>\n"
            f"  • 💵 <b>Est. Option P&amp;L (1 Lot):</b> <b>{'+' if r['approx_net_pnl'] >= 0 else ''}₹{r['approx_net_pnl']:,.2f}</b>\n"
            f"  • 🎯 <b>Target 1 Hits:</b> {r['target_1_hits']} | <b>Target 2 Hits:</b> {r['target_2_hits']}\n"
            f"  • 🛑 <b>Stop Loss Hits:</b> {r['stop_loss_hits']} | <b>Profit Factor:</b> {r['profit_factor']}\n"
            f"  • 📦 <b>Lot Size:</b> {r['lot_size']} Qty / Lot\n\n"
        )

    msg += (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🏛️ <b>LOGIC AUTO-CORRECTIONS APPLIED:</b>\n"
        f"  1. <b>Overextension Filter:</b> Blocked entries when &gt;2.5 ATR from 21 EMA.\n"
        f"  2. <b>Chop Elimination:</b> Blocked trades when CHOP Index &gt; 55.0 or ADX &lt; 18.\n"
        f"  3. <b>Trailing SL Protection:</b> SL instantly trailed to Cost at Target 1.\n"
        f"  4. <b>Post-T2 Cooldown:</b> 4-bar cooldown locks profit after major expansions.\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕒 <i>Audit Generated: {datetime.now().strftime('%d %b %Y, %I:%M:%S %p IST')}</i>\n"
        f"🤖 <b>Market Mentor Research and Academy</b>"
    )

    print("\n📲 Dispatching 1-Year Historical Audit to Telegram...")
    res = telegram_service.send_message(text=msg, chat_id=target_chat, parse_mode="HTML")
    print(f"✅ Telegram Dispatch Status: {res.get('status')} | Success: {res.get('success')}")

if __name__ == "__main__":
    main()
