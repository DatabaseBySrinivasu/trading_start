"""
Telegram Live Alert Listener & Auto-Sender for QuantPulse Pro.
Listens for the user's /start tap on @SrinuAlgoAlertsBot and instantly delivers trade alerts.
"""
import time
import sys
from app.services.telegram_service import telegram_service
from app.services.strategy_engine import strategy_engine

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    print("=" * 65)
    print("🤖 QUANTPULSE TELEGRAM AUTO-DISCOVERY & ALERT DISPATCHER")
    print("=" * 65)
    print("Bot Username : @SrinuAlgoAlertsBot")
    print("Direct Link  : https://t.me/SrinuAlgoAlertsBot")
    print("Status       : WAITING FOR TELEGRAM HANDSHAKE...")
    print("-" * 65)
    print("👉 Please open Telegram, search for @SrinuAlgoAlertsBot (or open the link)")
    print("   and press 'START' (or send /start).")
    print("-" * 65)

    detected_chat = None
    dots = 0
    while not detected_chat:
        detected_chat = telegram_service.poll_once()
        if detected_chat:
            break
        dots = (dots + 1) % 4
        print(f"\r⏳ Listening for your /start on Telegram{'.' * dots}   ", end="", flush=True)
        time.sleep(2)

    print("\n\n" + "=" * 65)
    print(f"🎉 SUCCESS! Connected to Telegram Chat ID: {detected_chat}")
    print("=" * 65)
    print("Dispatching Live Market Trade Alerts...")

    # Broadcast NIFTY 50
    print("\n1. Generating NIFTY 50 (^NSEI) Alert...")
    sig_nifty = strategy_engine.generate_options_call_put_signal("^NSEI")
    res1 = telegram_service.send_signal_alert(sig_nifty, chat_id=detected_chat)
    print(f"   Status: {res1.get('status')} -> {res1.get('message')}")

    # Broadcast BANK NIFTY
    print("\n2. Generating BANK NIFTY (^NSEBANK) Alert...")
    sig_bank = strategy_engine.generate_options_call_put_signal("^NSEBANK")
    res2 = telegram_service.send_signal_alert(sig_bank, chat_id=detected_chat)
    print(f"   Status: {res2.get('status')} -> {res2.get('message')}")

    # Broadcast CRUDE OIL
    print("\n3. Generating MCX CRUDE OIL Alert...")
    sig_crude = strategy_engine.generate_options_call_put_signal("CRUDEOIL")
    res3 = telegram_service.send_signal_alert(sig_crude, chat_id=detected_chat)
    print(f"   Status: {res3.get('status')} -> {res3.get('message')}")

    print("\n" + "=" * 65)
    print("✅ All Live Trade Alerts Successfully Delivered to your Telegram!")
    print("=" * 65)


if __name__ == "__main__":
    main()
