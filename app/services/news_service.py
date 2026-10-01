import logging
import re
import time
from typing import Dict, Any, List, Optional
import xml.etree.ElementTree as ET
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

# Financial sentiment keywords with weights
BULLISH_KEYWORDS = {
    "surge": 2.0, "surges": 2.0, "surging": 2.0, "jump": 1.8, "jumps": 1.8, "jumping": 1.8,
    "rally": 2.0, "rallies": 2.0, "rallying": 2.0, "gain": 1.5, "gains": 1.5, "gaining": 1.5,
    "soar": 2.2, "soars": 2.2, "soaring": 2.2, "boost": 1.5, "boosts": 1.5, "breakout": 2.2,
    "bull": 1.8, "bullish": 2.0, "high": 1.2, "all-time high": 2.5, "record high": 2.5,
    "profit": 1.8, "profit jump": 2.5, "revenue growth": 2.0, "growth": 1.5, "dividend": 1.5,
    "buyback": 2.0, "upgrade": 2.2, "upgrades": 2.2, "upgraded": 2.2, "buy": 1.5,
    "outperform": 2.0, "outperformed": 2.0, "target raised": 2.0, "deal": 1.4,
    "acquisition": 1.6, "contract win": 2.2, "order win": 2.0, "expansion": 1.5,
    "inflows": 1.8, "fii buying": 2.2, "dii buying": 1.8, "rebound": 1.8, "rebounds": 1.8,
    "green": 1.2, "positive": 1.5, "strong": 1.4, "beat": 1.8, "beats": 1.8, "beating": 1.8
}

BEARISH_KEYWORDS = {
    "fall": 1.5, "falls": 1.5, "falling": 1.5, "drop": 1.5, "drops": 1.5, "dropping": 1.5,
    "plunge": 2.2, "plunges": 2.2, "plunging": 2.2, "tumble": 2.0, "tumbles": 2.0, "tumbling": 2.0,
    "crash": 2.5, "crashes": 2.5, "crashing": 2.5, "slump": 2.0, "slumps": 2.0, "slumping": 2.0,
    "sink": 1.8, "sinks": 1.8, "sinking": 1.8, "decline": 1.6, "declines": 1.6, "declining": 1.6,
    "loss": 1.8, "losses": 1.8, "downgrade": 2.2, "downgrades": 2.2, "downgraded": 2.2,
    "bear": 1.8, "bearish": 2.0, "sell-off": 2.2, "selloff": 2.2, "sell": 1.5,
    "fii selling": 2.2, "outflows": 1.8, "inflation": 1.5, "rate hike": 1.8, "penalty": 2.2,
    "probe": 2.0, "fraud": 2.5, "investigation": 2.0, "default": 2.5, "debt": 1.4,
    "weak": 1.5, "drag": 1.5, "drags": 1.5, "red": 1.2, "negative": 1.5, "miss": 1.8, "misses": 1.8,
    "tariff": 1.8, "tariffs": 1.8, "warning": 1.6, "cut": 1.5, "cuts": 1.5, "margin pressure": 2.0
}


class NewsService:
    """
    Free Indian Stock Market & Company News Ingestion & Sentiment Intelligence Service.
    Powered by Google News RSS & Yahoo Finance news feeds.
    Includes automated Natural Language Sentiment scoring and actionable trading suggestions.
    """

    def __init__(self, cache_ttl_seconds: int = 300):
        self.cache_ttl = cache_ttl_seconds
        self._market_news_cache: Optional[Dict[str, Any]] = None
        self._market_news_time: float = 0
        self._stock_news_cache: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove HTML tags and clean up whitespace."""
        if not text:
            return ""
        clean = re.sub(r"<[^>]+>", "", text)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze financial sentiment from headline and description text.
        Returns sentiment score, label, confidence, and detected keywords.
        """
        lower_text = text.lower()
        bull_score = 0.0
        bear_score = 0.0
        bull_matches = []
        bear_matches = []

        for kw, weight in BULLISH_KEYWORDS.items():
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, lower_text):
                bull_score += weight
                bull_matches.append(kw)

        for kw, weight in BEARISH_KEYWORDS.items():
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, lower_text):
                bear_score += weight
                bear_matches.append(kw)

        total = bull_score + bear_score
        if total == 0:
            net_sentiment = 0.0
            sentiment_label = "NEUTRAL"
            tag_color = "gray"
            confidence = 50.0
        else:
            net_sentiment = round((bull_score - bear_score) / (total + 1e-9), 2)
            if net_sentiment >= 0.25:
                sentiment_label = "BULLISH"
                tag_color = "green"
            elif net_sentiment <= -0.25:
                sentiment_label = "BEARISH"
                tag_color = "red"
            else:
                sentiment_label = "NEUTRAL"
                tag_color = "gray"
            confidence = round(min(100.0, 50.0 + (abs(net_sentiment) * 50.0)), 1)

        return {
            "score": net_sentiment,
            "label": sentiment_label,
            "tag_color": tag_color,
            "confidence": confidence,
            "bullish_keywords": bull_matches[:4],
            "bearish_keywords": bear_matches[:4],
        }

    def _generate_suggestion(
        self,
        sentiment_label: str,
        score: float,
        bull_count: int,
        bear_count: int,
        symbol: Optional[str] = None
    ) -> str:
        """Generate human-readable, actionable trading guidance based on news sentiment."""
        asset_name = symbol if symbol else "Indian Markets (NIFTY/SENSEX)"
        if sentiment_label == "BULLISH":
            if score >= 0.5:
                return (
                    f"Strong Bullish News Flow for {asset_name} ({bull_count} positive catalysts detected). "
                    f"Positive momentum supports intraday CALL (CE) / buying on dips."
                )
            return (
                f"Moderately Bullish Sentiment for {asset_name}. "
                f"Favor long / CALL setups with strict trailing stop-losses."
            )
        elif sentiment_label == "BEARISH":
            if score <= -0.5:
                return (
                    f"High Bearish News Headwinds for {asset_name} ({bear_count} negative catalysts detected). "
                    f"Downside pressure favors PUT (PE) / hedging and cautious positioning."
                )
            return (
                f"Moderately Bearish Bias for {asset_name}. "
                f"Exercise caution on long positions; look for PUT opportunities on breakdown."
            )
        else:
            return (
                f"Balanced / Mixed News Flow for {asset_name}. "
                f"Market is digesting conflicting cues. Await technical breakout before taking aggressive directional bets."
            )

    def fetch_google_news_rss(self, query: str = "NIFTY 50 OR SENSEX OR Indian stock market", limit: int = 15) -> List[Dict[str, Any]]:
        """Fetch real-time news headlines from Google News RSS feed for Indian market topics."""
        encoded_query = requests.utils.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }

        articles = []
        try:
            resp = requests.get(url, headers=headers, timeout=8)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall("./channel/item")[:limit]:
                    raw_title = item.find("title").text if item.find("title") is not None else ""
                    link = item.find("link").text if item.find("link") is not None else "#"
                    pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                    source_elem = item.find("source")
                    source_name = source_elem.text if source_elem is not None else "Financial News"

                    # Parse title and separate source if appended with hyphen
                    title = raw_title
                    if " - " in raw_title:
                        parts = raw_title.rsplit(" - ", 1)
                        title = parts[0]
                        if source_name == "Financial News":
                            source_name = parts[1]

                    sent = self.analyze_sentiment(title)

                    articles.append({
                        "title": self._clean_text(title),
                        "source": self._clean_text(source_name),
                        "published_at": pub_date,
                        "url": link,
                        "summary": self._clean_text(title),
                        "sentiment": sent["label"],
                        "sentiment_score": sent["score"],
                        "sentiment_tag": sent["tag_color"],
                        "confidence": sent["confidence"],
                        "bullish_keywords": sent["bullish_keywords"],
                        "bearish_keywords": sent["bearish_keywords"],
                    })
        except Exception as e:
            logger.error(f"Error fetching Google News RSS for query '{query}': {e}")

        return articles

    def fetch_yfinance_news(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch ticker news from Yahoo Finance."""
        formatted_symbol = symbol.strip().upper()
        if not formatted_symbol.startswith("^") and not formatted_symbol.endswith(".NS") and not formatted_symbol.endswith(".BO"):
            formatted_symbol = f"{formatted_symbol}.NS"

        articles = []
        try:
            ticker = yf.Ticker(formatted_symbol)
            news_items = ticker.news or []
            for item in news_items[:limit]:
                content = item.get("content", {}) if "content" in item else item
                title = content.get("title", "")
                summary = content.get("summary", "") or content.get("description", "")
                provider = content.get("provider", {}).get("displayName", "Yahoo Finance") if isinstance(content.get("provider"), dict) else "Yahoo Finance"
                pub_date = content.get("pubDate", "") or content.get("displayTime", "")

                # Url extraction
                canonical = content.get("canonicalUrl", {})
                url = canonical.get("url", "#") if isinstance(canonical, dict) else content.get("clickThroughUrl", {}).get("url", "#")
                if not url or url == "#":
                    url = item.get("link", "#")

                combined_text = f"{title}. {summary}"
                sent = self.analyze_sentiment(combined_text)

                articles.append({
                    "title": self._clean_text(title),
                    "source": provider,
                    "published_at": pub_date,
                    "url": url,
                    "summary": self._clean_text(summary) if summary else self._clean_text(title),
                    "sentiment": sent["label"],
                    "sentiment_score": sent["score"],
                    "sentiment_tag": sent["tag_color"],
                    "confidence": sent["confidence"],
                    "bullish_keywords": sent["bullish_keywords"],
                    "bearish_keywords": sent["bearish_keywords"],
                })
        except Exception as e:
            logger.error(f"Error fetching yfinance news for {symbol}: {e}")

        return articles

    def get_market_news(self, force_refresh: bool = False, limit: int = 15) -> Dict[str, Any]:
        """
        Get aggregated macro market news with overall sentiment and AI trading suggestions.
        Cached in memory for high-speed responsiveness.
        """
        now = time.time()
        if not force_refresh and self._market_news_cache and (now - self._market_news_time < self.cache_ttl):
            return self._market_news_cache

        # Fetch Google News RSS for NIFTY / SENSEX & Macro Indian economy
        articles = self.fetch_google_news_rss("NIFTY 50 OR SENSEX OR Indian stock market", limit=limit)

        # Also get NIFTY ticker news from yfinance
        yf_articles = self.fetch_yfinance_news("^NSEI", limit=5)

        # Merge and deduplicate by title
        seen_titles = set()
        merged_articles = []
        for art in articles + yf_articles:
            t = art["title"].lower()
            if t not in seen_titles and len(t) > 10:
                seen_titles.add(t)
                merged_articles.append(art)

        # Aggregate sentiment
        bull_count = sum(1 for a in merged_articles if a["sentiment"] == "BULLISH")
        bear_count = sum(1 for a in merged_articles if a["sentiment"] == "BEARISH")
        neutral_count = len(merged_articles) - bull_count - bear_count

        if merged_articles:
            avg_score = round(sum(a["sentiment_score"] for a in merged_articles) / len(merged_articles), 2)
        else:
            avg_score = 0.0

        if avg_score >= 0.15:
            overall_sentiment = "BULLISH"
            tag_color = "green"
        elif avg_score <= -0.15:
            overall_sentiment = "BEARISH"
            tag_color = "red"
        else:
            overall_sentiment = "NEUTRAL / MIXED"
            tag_color = "gray"

        suggestion = self._generate_suggestion(
            sentiment_label=overall_sentiment,
            score=avg_score,
            bull_count=bull_count,
            bear_count=bear_count,
        )

        result = {
            "overall_sentiment": overall_sentiment,
            "average_score": avg_score,
            "sentiment_tag": tag_color,
            "bullish_articles_count": bull_count,
            "bearish_articles_count": bear_count,
            "neutral_articles_count": neutral_count,
            "total_articles": len(merged_articles),
            "suggestion": suggestion,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S IST"),
            "articles": merged_articles[:limit],
        }

        self._market_news_cache = result
        self._market_news_time = now
        return result

    def get_stock_news(self, symbol: str, limit: int = 10, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get company-specific news and sentiment for a given stock or index symbol.
        """
        clean_sym = symbol.strip().upper()
        now = time.time()
        cached = self._stock_news_cache.get(clean_sym)
        if not force_refresh and cached and (now - cached["_cached_at"] < self.cache_ttl):
            return cached["data"]

        # 1. Fetch yfinance news
        articles = self.fetch_yfinance_news(clean_sym, limit=limit)

        # 2. Fetch Google News RSS for stock name / symbol
        company_query = f"{clean_sym} stock OR {clean_sym} share price NSE"
        google_articles = self.fetch_google_news_rss(company_query, limit=5)

        # Merge & deduplicate
        seen_titles = set()
        merged_articles = []
        for art in articles + google_articles:
            t = art["title"].lower()
            if t not in seen_titles and len(t) > 10:
                seen_titles.add(t)
                merged_articles.append(art)

        bull_count = sum(1 for a in merged_articles if a["sentiment"] == "BULLISH")
        bear_count = sum(1 for a in merged_articles if a["sentiment"] == "BEARISH")
        neutral_count = len(merged_articles) - bull_count - bear_count

        avg_score = round(sum(a["sentiment_score"] for a in merged_articles) / len(merged_articles), 2) if merged_articles else 0.0

        if avg_score >= 0.15:
            overall_sentiment = "BULLISH"
            tag_color = "green"
        elif avg_score <= -0.15:
            overall_sentiment = "BEARISH"
            tag_color = "red"
        else:
            overall_sentiment = "NEUTRAL / MIXED"
            tag_color = "gray"

        suggestion = self._generate_suggestion(
            sentiment_label=overall_sentiment,
            score=avg_score,
            bull_count=bull_count,
            bear_count=bear_count,
            symbol=clean_sym
        )

        data = {
            "symbol": clean_sym,
            "overall_sentiment": overall_sentiment,
            "average_score": avg_score,
            "sentiment_tag": tag_color,
            "bullish_articles_count": bull_count,
            "bearish_articles_count": bear_count,
            "neutral_articles_count": neutral_count,
            "total_articles": len(merged_articles),
            "suggestion": suggestion,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S IST"),
            "articles": merged_articles[:limit],
        }

        self._stock_news_cache[clean_sym] = {"_cached_at": now, "data": data}
        return data


news_service = NewsService()
