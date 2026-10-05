"""Live / paper trading loop.

Polls the data source on an interval, feeds the newest bar for each symbol to
the engine and lets it submit orders to the broker. Daily strategies only act
once per new bar, so polling more often than the bar interval is harmless.
"""
from __future__ import annotations

import logging
import signal
import time
from datetime import datetime, timezone

from .broker import AlpacaBroker, Broker, PaperBroker
from .data import DataSource
from .engine import TradingEngine
from .risk import RiskManager
from .strategies import Strategy

log = logging.getLogger(__name__)


class LiveTrader:
    def __init__(self, broker: Broker, data: DataSource, strategy: Strategy,
                 risk: RiskManager, symbols: list[str], interval_seconds: int = 60):
        self.broker, self.data, self.symbols = broker, data, symbols
        self.interval = interval_seconds
        self.engine = TradingEngine(broker, strategy, risk)
        self.last_seen: dict[str, datetime] = {}
        self._running = False

    def warm_up(self) -> None:
        """Load history so the strategy is ready on the first live bar."""
        need = self.engine.strategy.warmup + 5
        for symbol in self.symbols:
            bars = self.data.get_bars(symbol, limit=need)
            if not bars:
                log.warning("no history for %s", symbol)
                continue
            self.engine.seed_history(bars[:-1])  # newest bar is traded on in run_once
            log.info("loaded %d bars for %s (last close %.2f)", len(bars), symbol, bars[-1].close)

    def run_once(self) -> list:
        new_bars = []
        for symbol in self.symbols:
            try:
                bars = self.data.get_bars(symbol, limit=2)
            except Exception as exc:  # noqa: BLE001 - keep the loop alive on data hiccups
                log.error("data error for %s: %s", symbol, exc)
                continue
            if not bars:
                continue
            bar = bars[-1]
            if isinstance(self.broker, PaperBroker):
                self.broker.update_price(symbol, bar.close, bar.timestamp)
            if self.last_seen.get(symbol) == bar.timestamp:
                continue  # already acted on this bar
            self.last_seen[symbol] = bar.timestamp
            new_bars.append(bar)
        if not new_bars:
            return []
        orders = self.engine.step(new_bars)
        positions = {s: p.qty for s, p in self.broker.get_positions().items()}
        log.info("equity $%s | cash $%s | positions %s",
                 f"{self.broker.get_equity():,.2f}", f"{self.broker.get_cash():,.2f}",
                 positions or "none")
        return orders

    def run(self, max_iterations: int | None = None) -> None:
        self._running = True
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)
        self.warm_up()
        i = 0
        while self._running and (max_iterations is None or i < max_iterations):
            i += 1
            if isinstance(self.broker, AlpacaBroker):
                try:
                    if not self.broker.is_market_open():
                        log.info("market closed; waiting")
                        self._sleep()
                        continue
                except Exception as exc:  # noqa: BLE001
                    log.error("clock check failed: %s", exc)
            self.run_once()
            if self.engine.risk.halted:
                log.warning("trading halted: %s", self.engine.risk.halt_reason)
            if max_iterations is None or i < max_iterations:
                self._sleep()
        log.info("live trader stopped at %s", datetime.now(timezone.utc).isoformat())

    def _sleep(self) -> None:
        end = time.monotonic() + self.interval
        while self._running and time.monotonic() < end:
            time.sleep(min(1.0, end - time.monotonic()))

    def _stop(self, *_):
        log.info("shutdown requested")
        self._running = False
