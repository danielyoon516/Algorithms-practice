"""Core data types shared across the platform."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from itertools import count

_order_ids = count(1)


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class Signal(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class Order:
    symbol: str
    side: Side
    qty: int
    reason: str = ""
    id: int = field(default_factory=lambda: next(_order_ids))
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime | None = None


@dataclass(frozen=True)
class Fill:
    order_id: int
    symbol: str
    side: Side
    qty: int
    price: float
    commission: float
    timestamp: datetime
    reason: str = ""


@dataclass
class Position:
    symbol: str
    qty: int = 0
    avg_price: float = 0.0

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.avg_price) * self.qty
