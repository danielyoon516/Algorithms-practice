"""Technical indicators implemented on plain Python lists.

Every function takes a sequence of floats (oldest first) and returns the value
for the most recent point, or ``None`` when there is not enough history.
"""
from __future__ import annotations

import math
from typing import Sequence


def sma(values: Sequence[float], period: int) -> float | None:
    if period <= 0 or len(values) < period:
        return None
    return sum(values[-period:]) / period


def ema(values: Sequence[float], period: int) -> float | None:
    if period <= 0 or len(values) < period:
        return None
    k = 2.0 / (period + 1)
    result = sum(values[:period]) / period  # seed with SMA
    for v in values[period:]:
        result = v * k + result * (1 - k)
    return result


def rsi(values: Sequence[float], period: int = 14) -> float | None:
    """Wilder's Relative Strength Index (0-100)."""
    if period <= 0 or len(values) < period + 1:
        return None
    gains = losses = 0.0
    for i in range(1, period + 1):
        change = values[i] - values[i - 1]
        gains += max(change, 0.0)
        losses += max(-change, 0.0)
    avg_gain, avg_loss = gains / period, losses / period
    for i in range(period + 1, len(values)):
        change = values[i] - values[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(change, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-change, 0.0)) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def stddev(values: Sequence[float], period: int) -> float | None:
    mean = sma(values, period)
    if mean is None:
        return None
    window = values[-period:]
    return math.sqrt(sum((v - mean) ** 2 for v in window) / period)


def bollinger(values: Sequence[float], period: int = 20, num_std: float = 2.0):
    """Return ``(lower, middle, upper)`` bands or ``None``."""
    mid = sma(values, period)
    sd = stddev(values, period)
    if mid is None or sd is None:
        return None
    return mid - num_std * sd, mid, mid + num_std * sd


def rate_of_change(values: Sequence[float], period: int) -> float | None:
    if period <= 0 or len(values) < period + 1 or values[-period - 1] == 0:
        return None
    return values[-1] / values[-period - 1] - 1.0
