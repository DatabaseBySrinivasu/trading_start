import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.unified_strategy_service import unified_strategy_service

def test_2y_multi():
    for sym in ["^BSESN", "^NSEI"]:
        # 1. 2-Year Daily
        res_d = unified_strategy_service.backtest_and_auto_correct(sym, interval="1d", period="2y")
        perf_d = res_d["performance"]
        print(f"🎯 {sym} [2y Daily (495 candles)]:")
        print(f"   Win Rate: {perf_d['win_rate_pct']}% ({perf_d['winning_trades']}W / {perf_d['losing_trades']}L / {perf_d['total_trades']}T)")
        print(f"   Net Gain: {perf_d['total_points_gained']:+,.2f} pts | PF: {perf_d['profit_factor']}")

        # 2. 2-Year Hourly (3,440 candles)
        res_h = unified_strategy_service.backtest_and_auto_correct(sym, interval="1h", period="2y")
        perf_h = res_h["performance"]
        print(f"🎯 {sym} [2y Hourly (3,440 candles)]:")
        print(f"   Win Rate: {perf_h['win_rate_pct']}% ({perf_h['winning_trades']}W / {perf_h['losing_trades']}L / {perf_h['total_trades']}T)")
        print(f"   Net Gain: {perf_h['total_points_gained']:+,.2f} pts | PF: {perf_h['profit_factor']}")
        print("-" * 65)

if __name__ == "__main__":
    test_2y_multi()
