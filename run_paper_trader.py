"""
Interactive Paper Trading, MCX Commodities & Strategy CLI
No API Keys or Login Required. Runs completely free using real-time NSE & MCX data.
"""
from app.services.market_data import market_data_service
from app.services.paper_trader import paper_trader
from app.services.strategy_engine import strategy_engine
from app.services.news_service import news_service
from app.services.pattern_service import pattern_service
from app.services.volatility_service import volatility_service
from app.services.telegram_service import telegram_service


def print_banner():
    print("=" * 65)
    print("   QuantPulse India: Algo, MCX & Paper Trading Terminal (100% Free)")
    print("=" * 65)


def print_portfolio():
    summary = paper_trader.get_portfolio_summary()
    print("\n" + "-" * 50)
    print("💼 PORTFOLIO SUMMARY")
    print("-" * 50)
    print(f"Available Cash    : ₹{summary['available_cash']:,.2f}")
    print(f"Holdings Value    : ₹{summary['holdings_value']:,.2f}")
    print(f"Total Portfolio   : ₹{summary['total_portfolio_value']:,.2f}")
    print(f"Realized P&L      : ₹{summary['realized_pnl']:+,.2f}")
    print(f"Total Return      : {summary['total_return_pct']:+.2f}%")
    print("-" * 50)

    if summary["positions"]:
        print("📊 OPEN POSITIONS:")
        for p in summary["positions"]:
            print(
                f"  • {p['symbol']}: {p['quantity']} shares @ Avg ₹{p['average_price']} "
                f"| LTP: ₹{p['current_price']} | Unrealized P&L: ₹{p['unrealized_pnl']:+,.2f} ({p['pnl_percent']:+.2f}%)"
            )
    else:
        print("No open positions.")
    print("-" * 50 + "\n")


def check_live_price():
    sym = input("Enter Symbol (e.g. SBIN, RELIANCE, CRUDEOIL, GOLD, SILVER): ").strip().upper()
    print(f"Fetching live price for {sym}...")
    data = market_data_service.get_live_price(sym)
    if data:
        print("\n" + "=" * 40)
        print(f"Symbol     : {data['symbol']} ({data.get('name', '')})")
        print(f"Live Price : ₹{data['price']:,.2f} {data.get('unit', '')}")
        print(f"Day Change : ₹{data['change']:+,.2f} ({data['change_percent']:+.2f}%)")
        print("=" * 40 + "\n")
    else:
        print(f"❌ Could not find symbol: {sym}")


def run_commodities_scan():
    print("\nScanning MCX Commodities (Crude Oil, Gold, Silver, Natural Gas, Copper)...")
    from app.api.v1.endpoints.market import POPULAR_COMMODITIES
    print("\n" + "=" * 70)
    print("🥇 MCX COMMODITIES RADAR & STRIKE SIGNALS")
    print("=" * 70)
    for item in POPULAR_COMMODITIES:
        sig = strategy_engine.generate_options_call_put_signal(item["symbol"])
        unit = item.get("unit", "₹")
        print(f"\n[{sig.get('recommendation')}] {item['name']} ({item['symbol']})")
        print(f"  • Spot Price       : ₹{sig.get('spot_price', 0):,.2f} {unit}")
        print(f"  • Suggested Strike : {sig.get('suggested_strike')}")
        print(f"  • Option Premium Buy: ₹{sig.get('option_entry_price', 0):,.2f} | T1: ₹{sig.get('option_target_1', 0):,.2f} | SL: ₹{sig.get('option_stop_loss', 0):,.2f}")
        print(f"  • Lot Specification : {sig.get('lot_size')} qty | Capital/Lot: ₹{sig.get('capital_required_per_lot', 0):,.2f}")
        if sig.get("price_action_momentum"):
            pam = sig["price_action_momentum"]
            print(f"  • Momentum Regime   : {pam.get('momentum_regime')} (Score: {pam.get('momentum_score', 0):+0.1f}) | {pam.get('bos_status')}")
    print("=" * 70 + "\n")


def run_options_pcr_scan():
    sym = input("Enter Index, Stock or Commodity Symbol (e.g. ^NSEI, ^BSESN, CRUDEOIL, GOLD, SBIN): ").strip().upper()
    print(f"\nScanning {sym} with Full 12-Factor Confluence (PCR, Fibonacci, Momentum, SMC, VIX)...")

    res = strategy_engine.generate_options_call_put_signal(sym)
    pcr = res.get("pcr", {})
    day_levels = res.get("day_levels", {})
    vol = res.get("volume_analysis", {})
    obs = res.get("order_blocks", {})
    wm = res.get("wm_patterns", {})
    pam = res.get("price_action_momentum", {})
    candles = res.get("candle_patterns", [])

    print("\n" + "=" * 65)
    print(f"🎯 QUANT SIGNAL & PRICE ACTION SCAN FOR {res.get('instrument', sym)}")
    print("=" * 65)
    print(f"Spot LTP Price     : ₹{res.get('spot_price', 0):,.2f}")
    print(f"Recommendation     : {res.get('recommendation')}")
    print(f"Suggested Strike   : {res.get('suggested_strike')} (Delta: {res.get('option_delta', 0.5)})")
    print(f"Confidence Score   : {res.get('confidence_score')}%")
    print(f"Risk-Reward Ratio  : {res.get('risk_reward_ratio')}")
    print("-" * 65)
    print("💰 OPTION CONTRACT STRIKE PREMIUM PRICING (BUY/SELL EXECUTION):")
    print(f"  • Option Entry Price (Buy): ₹{res.get('option_entry_price', res.get('entry_price', 0)):,.2f}")
    print(f"  • Option Target 1         : ₹{res.get('option_target_1', res.get('target_1', 0)):,.2f}")
    print(f"  • Option Target 2         : ₹{res.get('option_target_2', res.get('target_2', 0)):,.2f}")
    print(f"  • Option Stop Loss        : ₹{res.get('option_stop_loss', res.get('stop_loss', 0)):,.2f}")
    print(f"  • Lot Sizing & Capital    : {res.get('lot_size', 75)} Qty / Lot | Capital: ₹{res.get('capital_required_per_lot', 0):,.2f}")
    print(f"  • Projected Profit (T1)   : +₹{res.get('est_profit_per_lot_t1', 0):,.2f} / lot")
    print(f"  • Projected Loss (SL)     : -₹{res.get('est_risk_per_lot', 0):,.2f} / lot")
    print("-" * 65)
    print("📍 UNDERLYING SPOT REFERENCE LEVELS:")
    print(f"  • Spot Trigger Price      : ₹{res.get('spot_entry_price', res.get('spot_price', 0)):,.2f}")
    print(f"  • Spot Target 1           : ₹{res.get('spot_target_1', 0):,.2f}")
    print(f"  • Spot Target 2           : ₹{res.get('spot_target_2', 0):,.2f}")
    print(f"  • Spot Stop Loss          : ₹{res.get('spot_stop_loss', 0):,.2f}")
    print("-" * 65)
    if pam:
        print(f"⚡ PRICE ACTION MOMENTUM & IMPULSE:")
        print(f"  • Momentum Regime: {pam.get('momentum_regime')} (Score: {pam.get('momentum_score', 0):+0.1f})")
        print(f"  • Trend Structure: {pam.get('trend_structure')} | Expansion: {pam.get('body_expansion_ratio')}x")
        print(f"  • BOS Status     : {pam.get('bos_status')} | Wicks: {pam.get('wick_rejection')}")
        print("-" * 65)
    if pcr:
        print(f"📊 PUT-CALL RATIO (PCR) & OI DYNAMICS:")
        print(f"  • PCR (OI)       : {pcr.get('pcr_oi')} ({pcr.get('sentiment')})")
        print(f"  • Max Pain Strike: ₹{pcr.get('max_pain_strike', 'N/A')}")
        print("-" * 65)
    if day_levels:
        print(f"📐 DAY LEVELS & CENTRAL PIVOT RANGE (CPR):")
        print(f"  • Status         : {day_levels.get('status')} ({day_levels.get('description')})")
        print(f"  • PDH / PDL      : ₹{day_levels.get('pdh', 0):,.2f} / ₹{day_levels.get('pdl', 0):,.2f}")
        print(f"  • CPR Expectation: {day_levels.get('cpr', {}).get('expectation')}")
        print("-" * 65)
    if obs:
        print(f"🏛️ SMART MONEY CONCEPTS (SMC) & ORDER BLOCKS:")
        print(f"  • Bias           : {obs.get('smc_bias')} -> {obs.get('description')}")
        print("-" * 65)

    print("Confluence Reasons:")
    for reason in res.get("confluence_reasons", []):
        print(f"  ✓ {reason}")
    print("=" * 65 + "\n")

    tg_choice = input(f"📱 Send this signal alert to Telegram (Target: 9100040008)? (y/n): ").strip().lower()
    if tg_choice == 'y':
        print("Dispatching signal alert to Telegram...")
        tg_res = telegram_service.send_signal_alert(res)
        if tg_res.get("success"):
            print(f"✅ {tg_res.get('message')}")
        else:
            print(f"⚠️ {tg_res.get('message')}")
            if "preview_text" in tg_res:
                print("\n[Message Preview Formatted for Telegram]:\n" + "-"*40)
                print(tg_res["preview_text"])
                print("-" * 40 + "\n")


def send_telegram_radar_broadcast():
    print("\n" + "=" * 65)
    print("📱 BROADCAST SIGNALS TO TELEGRAM (Target: 9100040008)")
    print("=" * 65)
    status = telegram_service.get_status()
    print(f"Telegram Config Status : {status['bot_token_status']}")
    print(f"Destination Chat ID    : {status['target_chat_id']}")
    print("-" * 65)
    print("1. Send Test Verification Ping to Telegram (9100040008)")
    print("2. Send Signal for Specific Symbol to Telegram")
    print("3. Broadcast Major Indices (NIFTY 50, BANK NIFTY, SENSEX) to Telegram")
    print("4. Broadcast MCX Commodities (Crude, Gold, Silver, Natural Gas, Copper)")
    print("5. Return to Main Menu")

    sub_choice = input("\nSelect Telegram action (1-5): ").strip()
    if sub_choice == "1":
        print("\nSending test verification ping to 9100040008...")
        res = telegram_service.send_test_message()
        if res.get("success"):
            print(f"✅ {res.get('message')}")
        else:
            print(f"⚠️ {res.get('message')}")
    elif sub_choice == "2":
        sym = input("Enter Symbol to Scan & Alert (e.g. ^NSEI, CRUDEOIL, RELIANCE): ").strip().upper()
        print(f"Scanning {sym} and sending to Telegram...")
        sig = strategy_engine.generate_options_call_put_signal(sym)
        res = telegram_service.send_signal_alert(sig)
        if res.get("success"):
            print(f"✅ {res.get('message')}")
        else:
            print(f"⚠️ {res.get('message')}")
    elif sub_choice == "3":
        print("Broadcasting NIFTY 50, BANK NIFTY, and SENSEX signals to Telegram...")
        for sym in ["^NSEI", "^NSEBANK", "^BSESN"]:
            sig = strategy_engine.generate_options_call_put_signal(sym)
            res = telegram_service.send_signal_alert(sig)
            print(f"  • {sym}: {sig.get('recommendation')} -> {res.get('status')}")
    elif sub_choice == "4":
        print("Broadcasting MCX Commodities to Telegram...")
        for sym in ["CRUDEOIL", "GOLD", "SILVER", "NATURALGAS", "COPPER"]:
            sig = strategy_engine.generate_options_call_put_signal(sym)
            res = telegram_service.send_signal_alert(sig)
            print(f"  • {sym}: {sig.get('recommendation')} -> {res.get('status')}")


def view_vix_gift_nifty():
    print("\nFetching Live India VIX, Volatility Regime, GIFT NIFTY & Global Market Cues...")
    vix = volatility_service.get_india_vix()
    gift = volatility_service.get_gift_nifty_and_global_cues()

    print("\n" + "=" * 65)
    print("⚡ INDIA VIX (VOLATILITY REGIME) & GIFT NIFTY OPENING BIAS")
    print("=" * 65)
    print(f"India VIX Current : {vix.get('current_vix')} ({vix.get('change'):+,.2f} / {vix.get('change_percent'):+.2f}%)")
    print(f"Day Range (H / L) : High {vix.get('day_high')} | Low {vix.get('day_low')}")
    print(f"Volatility Regime : {vix.get('regime_label')}")
    print(f"Market Sentiment  : {vix.get('sentiment')} ({vix.get('description')})")
    print(f"Options Suggestion: {vix.get('options_strategy')}")
    print("-" * 65)
    print(f"GIFT NIFTY Price  : ₹{gift.get('gift_nifty_price', 0):,.2f}")
    print(f"NIFTY 50 Spot     : ₹{gift.get('nifty_spot_price', 0):,.2f}")
    print(f"Projected Gap     : {gift.get('projected_gap_pts', 0):+,.2f} pts ({gift.get('projected_gap_pct', 0):+.2f}%) -> {gift.get('bias_label')}")
    print(f"Opening Bias Tip  : {gift.get('strategy_tip')}")
    print("-" * 65)
    print("GLOBAL MARKET CUES:")
    for gm in gift.get("global_markets", []):
        print(f"  • {gm['name']:<12}: {gm['price']:>10,.2f} | {gm['change_percent']:+,.2f}%")
    print("=" * 65 + "\n")


def place_paper_buy():
    sym = input("Enter Symbol to Buy (e.g. SBIN, RELIANCE, CRUDEOIL): ").strip().upper()
    try:
        qty = int(input(f"Enter Quantity for {sym}: ").strip())
        if qty <= 0:
            print("Quantity must be greater than 0.")
            return
    except ValueError:
        print("Invalid number.")
        return

    print(f"Placing simulated BUY order for {qty} shares/units of {sym}...")
    res = paper_trader.buy(sym, qty)
    if res["success"]:
        print(f"✅ {res['message']}")
    else:
        print(f"❌ {res['message']}")


def place_paper_sell():
    sym = input("Enter Symbol to Sell: ").strip().upper()
    try:
        qty = int(input(f"Enter Quantity to Sell: ").strip())
        if qty <= 0:
            print("Quantity must be greater than 0.")
            return
    except ValueError:
        print("Invalid number.")
        return

    print(f"Placing simulated SELL order for {qty} units of {sym}...")
    res = paper_trader.sell(sym, qty)
    if res["success"]:
        print(f"✅ {res['message']}")
    else:
        print(f"❌ {res['message']}")


def main():
    print_banner()
    while True:
        print("MAIN MENU:")
        print("1. View Portfolio & Balance")
        print("2. Check Live Price (NSE / MCX)")
        print("3. Scan MCX Commodities Radar (Crude Oil, Gold, Silver, Natural Gas, Copper)")
        print("4. Scan Multi-Strategy Options Signal, Confluence & Momentum (NIFTY/SENSEX/Commodities/Stocks)")
        print("5. View Live India VIX, Volatility Regime & GIFT NIFTY Opening Bias")
        print("6. Buy Stock/Commodity (Paper Trade)")
        print("7. Sell Position (Paper Trade)")
        print("8. Broadcast Live Signals to Telegram (Target: 9100040008)")
        print("9. Exit")

        choice = input("\nSelect an option (1-9): ").strip()

        if choice == "1":
            print_portfolio()
        elif choice == "2":
            check_live_price()
        elif choice == "3":
            run_commodities_scan()
        elif choice == "4":
            run_options_pcr_scan()
        elif choice == "5":
            view_vix_gift_nifty()
        elif choice == "6":
            place_paper_buy()
        elif choice == "7":
            place_paper_sell()
        elif choice == "8":
            send_telegram_radar_broadcast()
        elif choice == "9":
            print("\nExiting QuantPulse. Happy Trading!\n")
            break
        else:
            print("Invalid choice. Please enter 1-9.")


if __name__ == "__main__":
    main()
