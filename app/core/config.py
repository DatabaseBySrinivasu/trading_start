from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Trading Start"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    # Angel One SmartAPI Settings
    ANGEL_API_KEY: Optional[str] = None
    ANGEL_CLIENT_CODE: Optional[str] = None
    ANGEL_MPIN: Optional[str] = None
    ANGEL_TOTP_SECRET: Optional[str] = None

    # Global Alert Quality & Timing Filters
    ALERT_MIN_MOVE_POINTS: float = 5.0
    ALERT_DURING_MARKET_HOURS_ONLY: bool = True
    ALERT_ONLY_GOOD_MOVES: bool = True
    ALERT_MIN_CONFIDENCE: float = 60.0

    # Telegram Bot Alerts Settings
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: str = "6817447645"
    TELEGRAM_AUTO_ALERT: bool = True
    TELEGRAM_MIN_CONFIDENCE: float = 60.0

    # Instagram & Meta Direct Alerts Settings
    INSTAGRAM_ACCESS_TOKEN: Optional[str] = None
    INSTAGRAM_ACCOUNT_ID: Optional[str] = None
    INSTAGRAM_RECIPIENT_ID: str = "9100040008"
    INSTAGRAM_WEBHOOK_URL: Optional[str] = None
    INSTAGRAM_AUTO_ALERT: bool = False
    INSTAGRAM_MIN_CONFIDENCE: float = 60.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()

