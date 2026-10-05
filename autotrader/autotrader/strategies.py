"""Trading strategies.

A strategy looks at the bar history for one symbol and returns a ``Signal``.
It never sizes positions or touches the broker; that is the risk manager's
and engine's job. Add your own by subclassing ``Strategy`` and registering it
in ``STRATEGIES``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from . import indicators as ind
from .models import Bar, Signal


class Strategy(ABC):
    name = "base"

    @property
    def warmup(self) -> int:
        """Minimum number of bars needed before the strategy can act."""
        return 1

    @abstractmethod
    def generate_signal(self, history: Sequence[Bar], has_position: bool) -> Signal:
        ...

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{type(self).__name__}({params})"


class SMACrossover(Strategy):
    """Buy when the fast SMA crosses above the slow SMA, sell on the reverse."""

    name = "sma"

    def __init__(self, fast: int = 10, slow: int = 30):
        if fast >= slow:
            raise ValueError("fast period must be shorter than slow period")
        self.fast, self.slow = fast, slow

    @property
    def warmup(self) -> int:
        return self.slow + 1

    def generate_signal(self, history, has_position):
        closes = [b.close for b in history]
        if len(closes) < self.warmup:
            return Signal.HOLD
        fast_now, slow_now = ind.sma(closes, self.fast), ind.sma(closes, self.slow)
        fast_prev, slow_prev = ind.sma(closes[:-1], self.fast), ind.sma(closes[:-1], self.slow)
        if fast_prev <= slow_prev and fast_now > slow_now and not has_position:
            return Signal.BUY
        if fast_prev >= slow_prev and fast_now < slow_now and has_position:
            return Signal.SELL
        return Signal.HOLD


class RSIMeanReversion(Strategy):
    """Buy oversold (RSI < lower), sell when overbought (RSI > upper)."""

    name = "rsi"

    def __init__(self, period: int = 14, lower: float = 30.0, upper: float = 70.0):
        self.period, self.lower, self.upper = period, lower, upper

    @property
    def warmup(self) -> int:
        return self.period + 1

    def generate_signal(self, history, has_position):
        value = ind.rsi([b.close for b in history], self.period)
        if value is None:
            return Signal.HOLD
        if value < self.lower and not has_position:
            return Signal.BUY
        if value > self.upper and has_position:
            return Signal.SELL
        return Signal.HOLD


class BollingerBreakout(Strategy):
    """Buy a close below the lower band, exit at the middle band or above."""

    name = "bollinger"

    def __init__(self, period: int = 20, num_std: float = 2.0):
        self.period, self.num_std = period, num_std

    @property
    def warmup(self) -> int:
        return self.period

    def generate_signal(self, history, has_position):
        bands = ind.bollinger([b.close for b in history], self.period, self.num_std)
        if bands is None:
            return Signal.HOLD
        lower, mid, _ = bands
        price = history[-1].close
        if price < lower and not has_position:
            return Signal.BUY
        if price >= mid and has_position:
            return Signal.SELL
        return Signal.HOLD


class Momentum(Strategy):
    """Hold while trailing return over ``lookback`` bars exceeds ``threshold``
    and price is above its long-term trend SMA."""

    name = "momentum"

    def __init__(self, lookback: int = 20, threshold: float = 0.02, trend: int = 50):
        self.lookback, self.threshold, self.trend = lookback, threshold, trend

    @property
    def warmup(self) -> int:
        return max(self.lookback + 1, self.trend)

    def generate_signal(self, history, has_position):
        closes = [b.close for b in history]
        roc = ind.rate_of_change(closes, self.lookback)
        trend = ind.sma(closes, self.trend)
        if roc is None or trend is None:
            return Signal.HOLD
        bullish = roc > self.threshold and closes[-1] > trend
        if bullish and not has_position:
            return Signal.BUY
        if not bullish and roc < 0 and has_position:
            return Signal.SELL
        return Signal.HOLD


STRATEGIES: dict[str, type[Strategy]] = {
    cls.name: cls for cls in (SMACrossover, RSIMeanReversion, BollingerBreakout, Momentum)
}


def build_strategy(name: str, **params) -> Strategy:
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}") from None
    return cls(**params)
