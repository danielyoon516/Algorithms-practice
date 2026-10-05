"""Brokers execute orders. ``PaperBroker`` simulates fills locally;
``AlpacaBroker`` routes orders to Alpaca's REST API (paper account by default)."""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from .models import Bar, Fill, Order, OrderStatus, Position, Side

log = logging.getLogger(__name__)


class Broker(ABC):
    @abstractmethod
    def submit_order(self, order: Order) -> Order: ...

    @abstractmethod
    def get_cash(self) -> float: ...

    @abstractmethod
    def get_positions(self) -> dict[str, Position]: ...

    @abstractmethod
    def get_equity(self) -> float: ...

    def get_position(self, symbol: str) -> Position:
        return self.get_positions().get(symbol, Position(symbol))


class PaperBroker(Broker):
    """Simulated broker.

    In backtests (``fill_on_next_bar=True``) market orders submitted on one bar
    are filled at the *next* bar's open, which avoids look-ahead bias. In live
    paper trading (``fill_on_next_bar=False``) orders fill immediately at the
    last known price.
    """

    def __init__(self, cash: float = 100_000.0, commission_per_share: float = 0.0,
                 min_commission: float = 0.0, slippage_bps: float = 5.0,
                 fill_on_next_bar: bool = True):
        self.cash = float(cash)
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.slippage_bps = slippage_bps
        self.fill_on_next_bar = fill_on_next_bar
        self.positions: dict[str, Position] = {}
        self.last_prices: dict[str, float] = {}
        self.pending: list[Order] = []
        self.fills: list[Fill] = []
        self.now: datetime | None = None

    # -- Broker interface ---------------------------------------------------
    def submit_order(self, order: Order) -> Order:
        if order.qty <= 0:
            order.status = OrderStatus.REJECTED
            return order
        order.created_at = self.now
        if self.fill_on_next_bar:
            self.pending.append(order)
        else:
            price = self.last_prices.get(order.symbol)
            if price is None:
                order.status = OrderStatus.REJECTED
                log.warning("no price for %s; order rejected", order.symbol)
            else:
                self._fill(order, price)
        return order

    def get_cash(self) -> float:
        return self.cash

    def get_positions(self) -> dict[str, Position]:
        return {s: p for s, p in self.positions.items() if p.qty != 0}

    def get_equity(self) -> float:
        value = sum(p.qty * self.last_prices.get(s, p.avg_price) for s, p in self.positions.items())
        return self.cash + value

    # -- Simulation hooks ---------------------------------------------------
    def on_bar(self, bar: Bar) -> list[Fill]:
        """Fill pending orders for ``bar.symbol`` at its open, then mark to close."""
        self.now = bar.timestamp
        fills = []
        remaining = []
        for order in self.pending:
            if order.symbol == bar.symbol:
                fill = self._fill(order, bar.open)
                if fill:
                    fills.append(fill)
            else:
                remaining.append(order)
        self.pending = remaining
        self.last_prices[bar.symbol] = bar.close
        return fills

    def update_price(self, symbol: str, price: float, timestamp: datetime | None = None) -> None:
        self.last_prices[symbol] = price
        if timestamp:
            self.now = timestamp

    def has_pending(self, symbol: str) -> bool:
        return any(o.symbol == symbol for o in self.pending)

    # -- Internals ----------------------------------------------------------
    def _commission(self, qty: int) -> float:
        if self.commission_per_share == 0 and self.min_commission == 0:
            return 0.0
        return max(qty * self.commission_per_share, self.min_commission)

    def _fill(self, order: Order, ref_price: float) -> Fill | None:
        slip = ref_price * self.slippage_bps / 10_000
        price = ref_price + slip if order.side == Side.BUY else ref_price - slip
        pos = self.positions.setdefault(order.symbol, Position(order.symbol))
        qty = order.qty

        if order.side == Side.BUY:
            commission = self._commission(qty)
            affordable = int((self.cash - commission) // price) if price > 0 else 0
            qty = min(qty, affordable)
            if qty <= 0:
                order.status = OrderStatus.REJECTED
                return None
            commission = self._commission(qty)
            cost = qty * price + commission
            pos.avg_price = (pos.avg_price * pos.qty + price * qty) / (pos.qty + qty)
            pos.qty += qty
            self.cash -= cost
        else:
            qty = min(qty, pos.qty)  # long-only: never sell more than we hold
            if qty <= 0:
                order.status = OrderStatus.REJECTED
                return None
            commission = self._commission(qty)
            pos.qty -= qty
            self.cash += qty * price - commission
            if pos.qty == 0:
                pos.avg_price = 0.0

        order.qty = qty
        order.status = OrderStatus.FILLED
        fill = Fill(order.id, order.symbol, order.side, qty, price, commission,
                    self.now or datetime.now(timezone.utc), order.reason)
        self.fills.append(fill)
        return fill


class AlpacaBroker(Broker):
    """Thin wrapper over the Alpaca Trading API v2.

    Credentials come from ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY``.
    Uses the paper-trading endpoint unless ``live=True`` is passed explicitly.
    """

    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"

    def __init__(self, key_id: str | None = None, secret_key: str | None = None,
                 live: bool = False, session=None):
        import requests  # imported lazily so the rest of the platform has no deps

        self.key_id = key_id or os.environ.get("APCA_API_KEY_ID")
        self.secret_key = secret_key or os.environ.get("APCA_API_SECRET_KEY")
        if not self.key_id or not self.secret_key:
            raise RuntimeError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY to use Alpaca")
        self.base_url = self.LIVE_URL if live else self.PAPER_URL
        self.session = session or requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret_key,
        })

    def _request(self, method: str, path: str, **kwargs):
        resp = self.session.request(method, self.base_url + path, timeout=15, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def submit_order(self, order: Order) -> Order:
        payload = {"symbol": order.symbol, "qty": str(order.qty), "side": order.side.value,
                   "type": "market", "time_in_force": "day"}
        try:
            self._request("POST", "/v2/orders", json=payload)
            order.status = OrderStatus.PENDING  # accepted; Alpaca fills asynchronously
        except Exception as exc:  # noqa: BLE001 - surface any API failure as a rejection
            log.error("Alpaca rejected %s %s %s: %s", order.side.value, order.qty, order.symbol, exc)
            order.status = OrderStatus.REJECTED
        return order

    def get_cash(self) -> float:
        return float(self._request("GET", "/v2/account")["cash"])

    def get_equity(self) -> float:
        return float(self._request("GET", "/v2/account")["equity"])

    def get_positions(self) -> dict[str, Position]:
        return {
            p["symbol"]: Position(p["symbol"], int(float(p["qty"])), float(p["avg_entry_price"]))
            for p in self._request("GET", "/v2/positions")
        }

    def is_market_open(self) -> bool:
        return bool(self._request("GET", "/v2/clock")["is_open"])
