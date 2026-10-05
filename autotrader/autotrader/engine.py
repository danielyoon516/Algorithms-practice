"""The trading engine: turns bars into orders using a strategy + risk manager.

The same engine drives backtests and live trading; only the broker and the
source of bars differ.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Iterable

from .broker import Broker, PaperBroker
from .models import Bar, Order, Side, Signal
from .risk import RiskManager
from .strategies import Strategy

log = logging.getLogger(__name__)


class TradingEngine:
    def __init__(self, broker: Broker, strategy: Strategy, risk: RiskManager,
                 max_history: int = 1000):
        self.broker = broker
        self.strategy = strategy
        self.risk = risk
        self.max_history = max(max_history, strategy.warmup + 1)
        self.history: dict[str, deque[Bar]] = {}
        self.orders: list[Order] = []

    def seed_history(self, bars: Iterable[Bar]) -> None:
        """Load past bars (e.g. at live startup) without trading on them."""
        for bar in bars:
            self._remember(bar)

    def step(self, bars: list[Bar]) -> list[Order]:
        """Process one bar per symbol for a single point in time.

        Returns the orders submitted. Brokers should already have been told
        about these bars' prices (``PaperBroker.on_bar``/``update_price``).
        """
        for bar in bars:
            self._remember(bar)

        equity = self.broker.get_equity()
        self.risk.update_equity(equity)
        cash = self.broker.get_cash()
        positions = self.broker.get_positions()
        open_count = len(positions)
        submitted: list[Order] = []

        for bar in bars:
            symbol = bar.symbol
            pos = positions.get(symbol)
            held = pos is not None and pos.qty > 0
            if isinstance(self.broker, PaperBroker) and self.broker.has_pending(symbol):
                continue

            if self.risk.halted:
                if held:
                    submitted.append(self._submit(symbol, Side.SELL, pos.qty,
                                                  f"kill switch: {self.risk.halt_reason}"))
                continue

            if held:
                reason = self.risk.exit_reason(pos, bar.close)
                if reason:
                    submitted.append(self._submit(symbol, Side.SELL, pos.qty, reason))
                    continue

            signal = self.strategy.generate_signal(self.history[symbol], held)
            if signal == Signal.BUY and not held:
                qty = self.risk.entry_quantity(bar.close, equity, cash, open_count)
                if qty > 0:
                    submitted.append(self._submit(symbol, Side.BUY, qty, f"{self.strategy.name} buy"))
                    cash -= qty * bar.close  # reserve cash for other entries this step
                    open_count += 1
            elif signal == Signal.SELL and held:
                submitted.append(self._submit(symbol, Side.SELL, pos.qty, f"{self.strategy.name} sell"))

        return [o for o in submitted if o is not None]

    def _remember(self, bar: Bar) -> None:
        hist = self.history.setdefault(bar.symbol, deque(maxlen=self.max_history))
        if hist and hist[-1].timestamp >= bar.timestamp:
            if hist[-1].timestamp == bar.timestamp:
                hist[-1] = bar  # updated intraday bar for the same day
            return
        hist.append(bar)

    def _submit(self, symbol: str, side: Side, qty: int, reason: str) -> Order:
        order = self.broker.submit_order(Order(symbol, side, qty, reason))
        self.orders.append(order)
        log.info("%s %s %d %s (%s) -> %s", bar_time(self.broker), side.value.upper(), qty,
                 symbol, reason, order.status.value)
        return order


def bar_time(broker: Broker) -> str:
    now = getattr(broker, "now", None)
    return now.date().isoformat() if now else ""
