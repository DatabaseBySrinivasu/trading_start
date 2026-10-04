import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.unified_strategy_service import unified_strategy_service

def verify_all():
    print("=" * 70)
    print("🚀 RUNNING UNIFIED STRATEGY MULTI-TIMEFRAME VERIFICATION")
    print("=" * 70)

    test_matrix = [
        ("^BSESN", "1d", "1y"),
        ("^BSESN", "15m", "60d"),
        ("^NSEI", "1d", "1y"),
        ("^NSEI", "15m", "60d"),
    ]

    for sym, interval, period in test_matrix:
        res = unified_strategy_service.backtest_and_auto_correct(sym, interval=interval, period=period)
        perf = res["performance"]
        date_summary = res.get("date_range", {}).get("formatted_summary", "N/A")
        print(f"🎯 {sym} [{period} {interval}] -> 🏆 WIN RATE: {perf['win_rate_pct']}% ({perf['winning_trades']}W / {perf['losing_trades']}L / {perf['total_trades']} Trades)")
        print(f"   💰 Net Gain: {perf['total_points_gained']:+,.2f} pts | Profit Factor: {perf['profit_factor']}")
        print(f"   📅 Range: {date_summary}")
        if res.get("auto_corrections"):
            for ac in res["auto_corrections"]:
                print(f"   ⚙️ {ac}")
        print("-" * 70)

if __name__ == "__main__":
    verify_all()
