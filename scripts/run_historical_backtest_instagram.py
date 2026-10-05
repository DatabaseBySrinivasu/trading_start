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
from app.services.instagram_service import instagram_service

def main():
    symbols = ["^NSEI", "^NSEBANK", "^BSESN", "CRUDEOIL", "NATURALGAS"]
    recipient = "9100040008"

    print("=" * 60)
    print(f"🚀 RUNNING HISTORICAL BACKTESTING & INSTAGRAM DISPATCH")
    print(f"📱 Target Instagram Recipient: {recipient}")
    print(f"⏰ Execution Time: {datetime.now().strftime('%d %b %Y, %I:%M:%S %p')}")
    print("=" * 60)

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

        print(f"   📅 Tested Dates: {date_info.get('formatted_summary')}")
        print(f"   🕯️ Candles Analyzed: {res.get('candles_analyzed')} (15m)")
        print(f"   🏆 Win Rate: {perf.get('win_rate_pct', 0.0)}%")
        print(f"   💰 Net Points: {perf.get('total_points_gained', 0):,.2f} pts")
        print(f"   🔢 Total Trades: {perf.get('total_trades', 0)} (Wins: {perf.get('winning_trades', 0)}, Losses: {perf.get('losing_trades', 0)}, Breakeven: {perf.get('breakeven_trades', 0)})")
        print(f"   ⚙️ Auto-Corrections: {res.get('auto_corrections_applied')}")

        # Format message
        msg = instagram_service.format_backtest_report_message(res)
        print(f"\n📲 [2/2] Dispatching to Instagram ({recipient})...")
        print("-" * 40)
        print(msg)
        print("-" * 40)

        # Dispatch to Instagram
        send_result = instagram_service.send_backtest_report(res, recipient_id=recipient)
        print(f"   ✅ Instagram Dispatch Status: {send_result.get('status')} | Success: {send_result.get('success')}")

        all_reports.append({
            "symbol": symbol,
            "date_range": date_info,
            "performance": perf,
            "instagram_status": send_result.get("status"),
        })

    print("\n" + "=" * 60)
    print("🎉 ALL 5 HISTORICAL BACKTESTS COMPLETED & LOGGED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()
