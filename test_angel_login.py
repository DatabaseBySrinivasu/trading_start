"""
Script to test Angel One SmartAPI Login and Connection.
Make sure your .env file is updated with your credentials before running.
"""
from app.services.angel_service import angel_client

if __name__ == "__main__":
    print("Connecting to Angel One SmartAPI...")
    try:
        session = angel_client.login()
        user_info = session.get("data", {})
        print("\n Login Successful!")
        print(f"Client Name : {user_info.get('name')}")
        print(f"Client Code : {user_info.get('clientcode')}")
        print(f"Email       : {user_info.get('email')}")

        # Test LTP for SBIN-EQ (Token: 3045)
        ltp = angel_client.get_ltp(exchange="NSE", tradingsymbol="SBIN-EQ", symboltoken="3045")
        if ltp:
            print(f"\nLive LTP for SBIN-EQ: ₹{ltp}")

    except Exception as e:
        print(f"\n[!] Login Failed: {e}")
        print("\nPlease check that your .env file has valid credentials:")
        print("- ANGEL_API_KEY")
        print("- ANGEL_CLIENT_CODE")
        print("- ANGEL_MPIN")
        print("- ANGEL_TOTP_SECRET (must be your Base32 TOTP secret from Angel One, not placeholder text)")
