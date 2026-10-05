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

from app.services.unified_strategy_service import unified_strategy_service
from app.services.telegram_service import telegram_service
from app.core.config import settings

def run_2year_audit():
    target_chat = settings.TELEGRAM_CHAT_ID or "6817447645"
    print("=" * 70)
    print("🚀 2-YEAR HISTORICAL STRATEGY AUDIT (OCT 2024 - OCT 2026)")
    print(f"📱 Target Destination: {target_chat}")
    print("=" * 70)

    test_matrix = [
        ("^BSESN", "1d", "2y", "2-Year Daily Trend & Swing Strategy"),
        ("^NSEI", "1d", "2y", "2-Year Daily Trend & Swing Strategy"),
        ("^BSESN", "15m", "60d", "60-Day Intraday 15m Strategy"),
        ("^NSEI", "15m", "60d", "60-Day Intraday 15m Strategy"),
    ]

    all_results = []

    for sym, interval, period, label in test_matrix:
        res = unified_strategy_service.backtest_and_auto_correct(sym, interval=interval, period=period)
        perf = res["performance"]
        date_range = res.get("date_range", {})
        lot_size = 20 if "BSESN" in sym or "SENSEX" in sym else 65

        approx_opt_pts = round(perf["total_points_gained"] * 0.55, 2)
        approx_net_pnl = round(approx_opt_pts * lot_size, 2)

        res_item = {
            "symbol": sym,
            "timeframe_label": label,
            "period": period,
            "interval": interval,
            "start_date": date_range.get("start_date_formatted", "N/A"),
            "end_date": date_range.get("end_date_formatted", "N/A"),
            "candles_count": date_range.get("candles_count", len(res.get("candles_analyzed", [])) if isinstance(res.get("candles_analyzed"), list) else res.get("candles_analyzed", 0)),
            "total_trades": perf["total_trades"],
            "winning_trades": perf["winning_trades"],
            "losing_trades": perf["losing_trades"],
            "target_1_hits": perf["target_1_hits"],
            "target_2_hits": perf["target_2_hits"],
            "stop_loss_hits": perf["stop_loss_hits"],
            "win_rate": perf["win_rate_pct"],
            "net_spot_pts": perf["total_points_gained"],
            "approx_net_pnl": approx_net_pnl,
            "profit_factor": perf["profit_factor"],
            "lot_size": lot_size,
            "auto_corrections": res.get("auto_corrections_applied", []),
        }
        all_results.append(res_item)

        print(f"\n📈 BACKTEST: {sym} ({label})")
        print(f"📅 Range: {res_item['start_date']} to {res_item['end_date']} ({res_item['candles_count']} candles)")
        print(f"🏆 Win Rate: {res_item['win_rate']}% ({res_item['winning_trades']}W / {res_item['losing_trades']}L / {res_item['total_trades']} Trades)")
        print(f"💰 Net Gain: {res_item['net_spot_pts']:+,.2f} pts | Profit Factor: {res_item['profit_factor']}")
        print(f"💵 Est Option P&L: ₹{approx_net_pnl:+,.2f} (Lot size: {lot_size})")

    # Format Telegram Message
    msg = (
        f"⚡️ <b>QUANTPULSE PRO: 2-YEAR HISTORICAL STRATEGY AUDIT</b> ⚡️\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Instruments Tested:</b> BSE SENSEX (^BSESN) &amp; NIFTY 50 (^NSEI)\n"
        f"📅 <b>Testing Dataset:</b> 2-Year Comprehensive Market Cycles (Oct 2024 – Oct 2026)\n"
        f"🎯 <b>Confluence Engine:</b> 7 Institutional Rules + Parameter Auto-Correction\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    )

    for r in all_results:
        sym_name = "BSE SENSEX" if "BSESN" in r["symbol"] else "NIFTY 50"
        msg += (
            f"📈 <b>{sym_name} — {r['timeframe_label']}:</b>\n"
            f"  • 📅 <b>Period:</b> {r['start_date']} to {r['end_date']} ({r['candles_count']} candles)\n"
            f"  • 🏆 <b>Win Rate:</b> <b>{r['win_rate']}%</b> ({r['winning_trades']} Wins / {r['losing_trades']} Losses / {r['total_trades']} Trades)\n"
            f"  • 💰 <b>Net Spot Gain:</b> <b>{'+' if r['net_spot_pts'] >= 0 else ''}{r['net_spot_pts']:,.2f} pts</b>\n"
            f"  • 💵 <b>Est. Option P&amp;L (1 Lot):</b> <b>{'+' if r['approx_net_pnl'] >= 0 else ''}₹{r['approx_net_pnl']:,.2f}</b>\n"
            f"  • 🎯 <b>Target 1 Hits:</b> {r['target_1_hits']} | <b>Target 2 Hits:</b> {r['target_2_hits']}\n"
            f"  • 🛑 <b>Stop Loss Hits:</b> {r['stop_loss_hits']} | <b>Profit Factor:</b> {r['profit_factor']}\n"
            f"  • 📦 <b>Lot Size:</b> {r['lot_size']} Qty / Lot\n\n"
        )

    msg += (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🏛️ <b>7 INSTITUTIONAL RULES ENFORCED:</b>\n"
        f"  1. <b>EMA 9/21 Gap &amp; Slope:</b> Flat whipsaw defense (gap &gt;= 0.15x ATR, slope &gt; 0.05x ATR).\n"
        f"  2. <b>Solid Candle Ratio:</b> Minimum 0.35-0.55 body ratio to eliminate Dojis/wicks.\n"
        f"  3. <b>Killzone Timing:</b> Active in 09:15-11:30 &amp; 13:30-15:05; suppresses 11:30-13:00 lunch chop.\n"
        f"  4. <b>Overextension Filter:</b> Restricts buying &gt;2.2 ATR away from 21 EMA.\n"
        f"  5. <b>RSI Sweet Spot:</b> 40.0-70.0 for CALLs, 30.0-60.0 for PUTs.\n"
        f"  6. <b>Choppiness Gate:</b> Suppresses entries when CHOP &gt; 52.0.\n"
        f"  7. <b>Dynamic Breakeven Trailing:</b> Trailed SL to entry upon Target 1 (+0.75x ATR).\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕒 <i>Audit Generated: {datetime.now().strftime('%d %b %Y, %I:%M:%S %p IST')}</i>\n"
        f"🤖 <b>Market Mentor Research and Academy</b>"
    )

    print("\n📲 Dispatching 2-Year Historical Audit to Telegram...")
    res = telegram_service.send_message(text=msg, chat_id=target_chat, parse_mode="HTML")
    print(f"✅ Telegram Dispatch Status: {res.get('status')} | Success: {res.get('success')}")

if __name__ == "__main__":
    run_2year_audit()
