"""Market data sources. Each returns bars oldest-first."""
from __future__ import annotations

import csv
import math
import os
import random
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import Bar


class DataSource(ABC):
    @abstractmethod
    def get_bars(self, symbol: str, limit: int | None = None) -> list[Bar]:
        """Return up to ``limit`` most recent daily bars for ``symbol``."""


class SyntheticDataSource(DataSource):
    """Geometric Brownian motion prices; deterministic per (seed, symbol).

    Handy for trying the platform offline and for tests.
    """

    def __init__(self, days: int = 750, start_price: float = 100.0, drift: float = 0.08,
                 volatility: float = 0.25, seed: int = 42,
                 start: datetime = datetime(2022, 1, 3, tzinfo=timezone.utc)):
        self.days, self.start_price = days, start_price
        self.drift, self.volatility, self.seed, self.start = drift, volatility, seed, start

    def get_bars(self, symbol, limit=None):
        rng = random.Random(f"{self.seed}:{symbol}")
        dt = 1 / 252
        price = self.start_price * (0.5 + rng.random())
        bars, day = [], self.start
        while len(bars) < self.days:
            if day.weekday() < 5:
                shock = rng.gauss(0, 1)
                ret = (self.drift - 0.5 * self.volatility ** 2) * dt + self.volatility * math.sqrt(dt) * shock
                open_ = price * (1 + rng.gauss(0, 0.003))
                close = price * math.exp(ret)
                hi = max(open_, close) * (1 + abs(rng.gauss(0, 0.005)))
                lo = min(open_, close) * (1 - abs(rng.gauss(0, 0.005)))
                bars.append(Bar(symbol, day, round(open_, 4), round(hi, 4), round(lo, 4),
                                round(close, 4), float(rng.randint(100_000, 5_000_000))))
                price = close
            day += timedelta(days=1)
        return bars[-limit:] if limit else bars


class CSVDataSource(DataSource):
    """Reads ``<directory>/<SYMBOL>.csv`` with columns
    ``date,open,high,low,close[,volume]`` (header names are case-insensitive;
    ``timestamp`` is accepted for ``date`` and ``adj close`` is ignored)."""

    def __init__(self, directory: str | os.PathLike):
        self.directory = Path(directory)

    def get_bars(self, symbol, limit=None):
        path = self.directory / f"{symbol}.csv"
        bars = []
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                row = {k.strip().lower(): v for k, v in row.items() if k}
                raw_ts = row.get("date") or row.get("timestamp")
                try:
                    bars.append(Bar(symbol, _parse_date(raw_ts), float(row["open"]),
                                    float(row["high"]), float(row["low"]), float(row["close"]),
                                    float(row.get("volume") or 0)))
                except (TypeError, ValueError):
                    continue  # skip malformed / null rows
        bars.sort(key=lambda b: b.timestamp)
        return bars[-limit:] if limit else bars


class YahooDataSource(DataSource):
    """Daily bars from Yahoo Finance's public chart endpoint (no API key)."""

    URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

    def __init__(self, range_: str = "2y", session=None):
        import requests

        self.range = range_
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "Mozilla/5.0 autotrader")

    def get_bars(self, symbol, limit=None):
        resp = self.session.get(self.URL.format(symbol=symbol),
                                params={"range": self.range, "interval": "1d"}, timeout=15)
        resp.raise_for_status()
        result = resp.json()["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]
        bars = []
        for i, ts in enumerate(result.get("timestamp", [])):
            o, h, l, c = (quote[k][i] for k in ("open", "high", "low", "close"))
            if None in (o, h, l, c):
                continue
            bars.append(Bar(symbol, datetime.fromtimestamp(ts, tz=timezone.utc),
                            o, h, l, c, float(quote["volume"][i] or 0)))
        return bars[-limit:] if limit else bars


class AlpacaDataSource(DataSource):
    """Daily bars from Alpaca's market-data API (uses the same API keys as the broker)."""

    URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"

    def __init__(self, key_id=None, secret_key=None, lookback_days: int = 400,
                 feed: str = "iex", session=None):
        import requests

        key_id = key_id or os.environ.get("APCA_API_KEY_ID")
        secret_key = secret_key or os.environ.get("APCA_API_SECRET_KEY")
        if not key_id or not secret_key:
            raise RuntimeError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY to use Alpaca data")
        self.lookback_days, self.feed = lookback_days, feed
        self.session = session or requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key})

    def get_bars(self, symbol, limit=None):
        start = (datetime.now(timezone.utc) - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        params = {"timeframe": "1Day", "start": start, "limit": 10_000,
                  "adjustment": "all", "feed": self.feed}
        bars, token = [], None
        while True:
            if token:
                params["page_token"] = token
            resp = self.session.get(self.URL.format(symbol=symbol), params=params, timeout=15)
            resp.raise_for_status()
            body = resp.json()
            for b in body.get("bars") or []:
                bars.append(Bar(symbol, _parse_date(b["t"]), b["o"], b["h"], b["l"], b["c"], b.get("v", 0)))
            token = body.get("next_page_token")
            if not token:
                break
        return bars[-limit:] if limit else bars


def _parse_date(raw: str) -> datetime:
    raw = raw.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    ts = datetime.fromisoformat(raw)
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
