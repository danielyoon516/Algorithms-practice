"""Risk management: position sizing, protective exits and a drawdown kill switch."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .models import Position


@dataclass
class RiskConfig:
    position_size_pct: float = 0.10   # fraction of equity committed to each new position
    max_positions: int = 5            # max simultaneous open positions
    stop_loss_pct: float | None = 0.05     # exit when price falls this far below entry
    take_profit_pct: float | None = 0.15   # exit when price rises this far above entry
    max_drawdown_pct: float | None = 0.20  # halt trading and flatten beyond this drawdown
    min_cash_reserve_pct: float = 0.0      # keep this fraction of equity in cash

    def __post_init__(self):
        if not 0 < self.position_size_pct <= 1:
            raise ValueError("position_size_pct must be in (0, 1]")
        if self.max_positions < 1:
            raise ValueError("max_positions must be >= 1")


class RiskManager:
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        self.peak_equity: float | None = None
        self.halted = False
        self.halt_reason = ""

    def update_equity(self, equity: float) -> None:
        """Track peak equity and trip the kill switch on excessive drawdown."""
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        limit = self.config.max_drawdown_pct
        if limit is not None and not self.halted and self.peak_equity > 0:
            drawdown = 1 - equity / self.peak_equity
            if drawdown >= limit:
                self.halted = True
                self.halt_reason = f"max drawdown {drawdown:.1%} >= {limit:.1%}"

    def entry_quantity(self, price: float, equity: float, cash: float, open_positions: int) -> int:
        """Shares to buy for a new position, or 0 if the trade is not allowed."""
        if self.halted or price <= 0 or open_positions >= self.config.max_positions:
            return 0
        budget = equity * self.config.position_size_pct
        spendable = cash - equity * self.config.min_cash_reserve_pct
        budget = min(budget, spendable)
        return max(int(math.floor(budget / price)), 0)

    def exit_reason(self, position: Position, price: float) -> str | None:
        """Return a reason string if a protective exit should fire."""
        if position.qty <= 0 or position.avg_price <= 0:
            return None
        change = price / position.avg_price - 1
        sl, tp = self.config.stop_loss_pct, self.config.take_profit_pct
        if sl is not None and change <= -sl:
            return f"stop-loss ({change:+.1%})"
        if tp is not None and change >= tp:
            return f"take-profit ({change:+.1%})"
        return None
