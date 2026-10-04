import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.market_data import market_data_service
from app.services.unified_strategy_service import unified_strategy_service

def test_2y():
    for sym in ["^BSESN", "^NSEI"]:
        df = market_data_service.get_historical_candles(sym, period="2y", interval="1d")
        print(f"Symbol: {sym} | 2y Data shape: {df.shape if df is not None else 'None'}")
        if df is not None:
            print(f"Start: {df.index[0]} | End: {df.index[-1]}")
            res = unified_strategy_service.backtest_and_auto_correct(sym, interval="1d", period="2y")
            perf = res["performance"]
            print(f"2Y Daily Results for {sym}:")
            print(f"  Win Rate: {perf['win_rate_pct']}% ({perf['winning_trades']}W / {perf['losing_trades']}L / {perf['total_trades']} Trades)")
            print(f"  Net Gain: {perf['total_points_gained']:+,.2f} pts | Profit Factor: {perf['profit_factor']}")
            print(f"  Auto Corrections: {res.get('auto_corrections_applied')}")
            print("-" * 60)

if __name__ == "__main__":
    test_2y()
