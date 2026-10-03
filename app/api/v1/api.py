from fastapi import APIRouter
from app.api.v1.endpoints import trades, market, paper_trading, telegram, instagram, audit

api_router = APIRouter()
api_router.include_router(trades.router, prefix="/trades", tags=["trades"])
api_router.include_router(market.router, prefix="/market", tags=["market"])
api_router.include_router(paper_trading.router, prefix="/paper", tags=["paper-trading"])
api_router.include_router(telegram.router, prefix="/telegram", tags=["telegram-alerts"])
api_router.include_router(instagram.router, prefix="/instagram", tags=["instagram-alerts"])
api_router.include_router(audit.router, prefix="/audit", tags=["trade-audit-self-correction"])



