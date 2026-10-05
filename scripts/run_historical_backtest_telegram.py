import os
import sys
from datetime import datetime

# Reconfigure stdout/stderr for clean utf-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.unified_strategy_service import unified_strategy_service
from app.services.telegram_service import telegram_service
from app.core.config import settings

def main():
    symbols = ["^NSEI", "^NSEBANK", "^BSESN", "CRUDEOIL", "NATURALGAS"]
    target_chat = settings.TELEGRAM_CHAT_ID or "9100040008"

    print("=" * 65)
    print("🚀 RUNNING HISTORICAL CANDLE BACKTESTING & TELEGRAM DISPATCH")
    print(f"📱 Target Telegram Destination: {target_chat}")
    print(f"⏰ Execution Time: {datetime.now().strftime('%d %b %Y, %I:%M:%S %p IST')}")
    print("=" * 65)

    all_reports = []

    for symbol in symbols:
        print(f"\n📊 [1/2] Processing Historical Backtest for: {symbol}...")
        res = unified_strategy_service.backtest_and_auto_correct(
            symbol=symbol,
            period="1mo",
            interval="15m",
            min_acceptable_winrate=70.0
        )

        date_info = res.get("date_range", {})
        perf = res.get("performance", {})
        date_range_str = date_info.get("formatted_summary", "N/A")

        print(f"   📅 Tested Dates: {date_range_str}")
        print(f"   🕯️ Candles Analyzed: {res.get('candles_analyzed')} (15m timeframe)")
        print(f"   🏆 Win Rate: {perf.get('win_rate_pct', 0.0)}%")
        print(f"   💰 Net Points Gained: {perf.get('total_points_gained', 0):+,.2f} pts")
        print(f"   🔢 Executed Trades: {perf.get('total_trades', 0)} (Wins: {perf.get('winning_trades', 0)}, Losses: {perf.get('losing_trades', 0)}, Breakeven: {perf.get('breakeven_trades', 0)})")
        print(f"   ⚙️ Auto-Corrections: {res.get('auto_corrections_applied')}")

        # Format Telegram message
        msg = telegram_service.format_backtest_report_message(res)
        print(f"\n📲 [2/2] Dispatching to Telegram ({target_chat})...")
        print("-" * 45)
        # Clean text preview (stripping HTML tags for terminal print)
        clean_preview = msg.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "").replace("<code>", "").replace("</code>", "")
        print(clean_preview)
        print("-" * 45)

        # Dispatch to Telegram
        send_result = telegram_service.send_backtest_report(res, chat_id=target_chat)
        print(f"   ✅ Telegram Dispatch Status: {send_result.get('status')} | Success: {send_result.get('success')}")

        all_reports.append({
            "symbol": symbol,
            "date_range": date_info,
            "performance": perf,
            "telegram_status": send_result.get("status"),
            "telegram_success": send_result.get("success"),
        })

    print("\n" + "=" * 65)
    print("🎉 ALL 5 HISTORICAL BACKTESTS COMPLETED & DISPATCHED TO TELEGRAM")
    print("=" * 65)
    return all_reports

if __name__ == "__main__":
    main()
