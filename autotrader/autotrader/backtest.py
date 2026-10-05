"""Event-driven backtester and performance metrics."""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from .broker import PaperBroker
from .engine import TradingEngine
from .models import Bar, Fill, Side
from .risk import RiskConfig, RiskManager
from .strategies import Strategy


@dataclass
class BacktestResult:
    initial_cash: float
    equity_curve: list[tuple[datetime, float]]
    fills: list[Fill]
    benchmark_return: float
    halted_reason: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"Start equity      : ${self.initial_cash:,.2f}",
            f"Final equity      : ${m['final_equity']:,.2f}",
            f"Total return      : {m['total_return']:+.2%}",
            f"Buy & hold return : {self.benchmark_return:+.2%}",
            f"CAGR              : {m['cagr']:+.2%}",
            f"Sharpe ratio      : {m['sharpe']:.2f}",
            f"Max drawdown      : {m['max_drawdown']:.2%}",
            f"Closed trades     : {int(m['trades'])}",
            f"Win rate          : {m['win_rate']:.1%}",
            f"Realized P&L      : {_money(m['realized_pnl'])}",
            f"Commissions paid  : ${m['commissions']:,.2f}",
        ]
        if self.halted_reason:
            lines.append(f"Kill switch       : {self.halted_reason}")
        return "\n".join(lines)


def _money(value: float) -> str:
    return f"{'-' if value < 0 else ''}${abs(value):,.2f}"


class Backtester:
    def __init__(self, strategy: Strategy, risk_config: RiskConfig | None = None,
                 initial_cash: float = 100_000.0, commission_per_share: float = 0.0,
                 min_commission: float = 0.0, slippage_bps: float = 5.0):
        self.strategy = strategy
        self.risk_config = risk_config or RiskConfig()
        self.initial_cash = initial_cash
        self.broker_kwargs = dict(commission_per_share=commission_per_share,
                                  min_commission=min_commission, slippage_bps=slippage_bps)

    def run(self, data: dict[str, list[Bar]]) -> BacktestResult:
        broker = PaperBroker(self.initial_cash, fill_on_next_bar=True, **self.broker_kwargs)
        risk = RiskManager(self.risk_config)
        engine = TradingEngine(broker, self.strategy, risk)

        by_time: dict[datetime, list[Bar]] = defaultdict(list)
        for bars in data.values():
            for bar in bars:
                by_time[bar.timestamp].append(bar)

        curve = []
        for ts in sorted(by_time):
            bars = by_time[ts]
            for bar in bars:
                broker.on_bar(bar)  # fill yesterday's orders at today's open
            engine.step(bars)
            curve.append((ts, broker.get_equity()))

        result = BacktestResult(self.initial_cash, curve, list(broker.fills),
                                _benchmark(data), risk.halt_reason)
        result.metrics = compute_metrics(self.initial_cash, curve, broker.fills)
        return result


def _benchmark(data: dict[str, list[Bar]]) -> float:
    """Equal-weight buy-and-hold return across all symbols."""
    rets = [bars[-1].close / bars[0].open - 1 for bars in data.values() if bars and bars[0].open]
    return sum(rets) / len(rets) if rets else 0.0


def compute_metrics(initial_cash: float, curve: list[tuple[datetime, float]],
                    fills: list[Fill], periods_per_year: int = 252) -> dict[str, float]:
    equities = [e for _, e in curve] or [initial_cash]
    final = equities[-1]
    total_return = final / initial_cash - 1

    years = 0.0
    if len(curve) > 1:
        years = (curve[-1][0] - curve[0][0]).days / 365.25
    cagr = (final / initial_cash) ** (1 / years) - 1 if years > 0 and final > 0 else total_return

    rets = [b / a - 1 for a, b in zip(equities, equities[1:]) if a > 0]
    sharpe = 0.0
    if len(rets) > 1:
        sd = statistics.stdev(rets)
        if sd > 0:
            sharpe = statistics.mean(rets) / sd * math.sqrt(periods_per_year)

    peak, max_dd = equities[0], 0.0
    for e in equities:
        peak = max(peak, e)
        if peak > 0:
            max_dd = max(max_dd, 1 - e / peak)

    # Realized P&L per closing (sell) fill, using average cost per symbol.
    cost: dict[str, tuple[int, float]] = {}
    trade_pnls = []
    for f in fills:
        qty, avg = cost.get(f.symbol, (0, 0.0))
        if f.side == Side.BUY:
            new_qty = qty + f.qty
            cost[f.symbol] = (new_qty, (avg * qty + f.price * f.qty + f.commission) / new_qty)
        else:
            trade_pnls.append((f.price - avg) * f.qty - f.commission)
            cost[f.symbol] = (qty - f.qty, avg if qty - f.qty else 0.0)
    wins = sum(1 for p in trade_pnls if p > 0)

    return {
        "final_equity": final,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "trades": float(len(trade_pnls)),
        "win_rate": wins / len(trade_pnls) if trade_pnls else 0.0,
        "realized_pnl": sum(trade_pnls),
        "commissions": sum(f.commission for f in fills),
    }
