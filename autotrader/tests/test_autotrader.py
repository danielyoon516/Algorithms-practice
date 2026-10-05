import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from autotrader import indicators as ind
from autotrader.backtest import Backtester, compute_metrics
from autotrader.broker import AlpacaBroker, PaperBroker
from autotrader.cli import main
from autotrader.data import CSVDataSource, SyntheticDataSource, YahooDataSource
from autotrader.engine import TradingEngine
from autotrader.live import LiveTrader
from autotrader.models import Bar, Order, OrderStatus, Position, Side, Signal
from autotrader.risk import RiskConfig, RiskManager
from autotrader.strategies import RSIMeanReversion, SMACrossover, Strategy, build_strategy

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


def bars_from_closes(closes, symbol="TEST"):
    return [Bar(symbol, T0 + timedelta(days=i), c, c, c, c, 1000) for i, c in enumerate(closes)]


class ScriptedStrategy(Strategy):
    """Emits signals from a dict keyed by bar index."""
    name = "scripted"

    def __init__(self, script):
        self.script = script

    def generate_signal(self, history, has_position):
        return self.script.get(len(history) - 1, Signal.HOLD)


class IndicatorTests(unittest.TestCase):
    def test_sma_and_ema(self):
        self.assertEqual(ind.sma([1, 2, 3, 4], 2), 3.5)
        self.assertIsNone(ind.sma([1], 2))
        self.assertAlmostEqual(ind.ema([1, 2, 3], 3), 2.0)
        self.assertAlmostEqual(ind.ema([1, 2, 3, 4], 3), 3.0)

    def test_rsi_bounds(self):
        self.assertEqual(ind.rsi(list(range(1, 30)), 14), 100.0)
        self.assertAlmostEqual(ind.rsi(list(range(30, 1, -1)), 14), 0.0)
        self.assertIsNone(ind.rsi([1, 2, 3], 14))

    def test_bollinger(self):
        lower, mid, upper = ind.bollinger([2, 4, 4, 4, 5, 5, 7, 9], 8, 2)
        self.assertAlmostEqual(mid, 5.0)
        self.assertAlmostEqual(upper - mid, 4.0)
        self.assertAlmostEqual(mid - lower, 4.0)


class StrategyTests(unittest.TestCase):
    def test_sma_crossover_signals(self):
        strat = SMACrossover(fast=2, slow=4)
        down_then_up = bars_from_closes([10, 9, 8, 7, 6, 12])
        self.assertEqual(strat.generate_signal(down_then_up, False), Signal.BUY)
        self.assertEqual(strat.generate_signal(down_then_up, True), Signal.HOLD)
        up_then_down = bars_from_closes([5, 6, 7, 8, 9, 3])
        self.assertEqual(strat.generate_signal(up_then_down, True), Signal.SELL)

    def test_rsi_strategy(self):
        strat = RSIMeanReversion(period=5)
        self.assertEqual(strat.generate_signal(bars_from_closes(range(20, 10, -1)), False), Signal.BUY)
        self.assertEqual(strat.generate_signal(bars_from_closes(range(10, 20)), True), Signal.SELL)

    def test_build_strategy(self):
        self.assertIsInstance(build_strategy("sma", fast=3, slow=5), SMACrossover)
        with self.assertRaises(ValueError):
            build_strategy("nope")
        with self.assertRaises(ValueError):
            SMACrossover(fast=10, slow=5)


class RiskTests(unittest.TestCase):
    def test_sizing_and_limits(self):
        rm = RiskManager(RiskConfig(position_size_pct=0.1, max_positions=2))
        self.assertEqual(rm.entry_quantity(50, 100_000, 100_000, 0), 200)
        self.assertEqual(rm.entry_quantity(50, 100_000, 1_000, 0), 20)  # cash-limited
        self.assertEqual(rm.entry_quantity(50, 100_000, 100_000, 2), 0)  # too many positions

    def test_stop_loss_take_profit(self):
        rm = RiskManager(RiskConfig(stop_loss_pct=0.05, take_profit_pct=0.10))
        pos = Position("X", 10, 100.0)
        self.assertIn("stop-loss", rm.exit_reason(pos, 94))
        self.assertIn("take-profit", rm.exit_reason(pos, 111))
        self.assertIsNone(rm.exit_reason(pos, 101))

    def test_drawdown_kill_switch(self):
        rm = RiskManager(RiskConfig(max_drawdown_pct=0.2))
        rm.update_equity(100)
        rm.update_equity(150)
        rm.update_equity(125)
        self.assertFalse(rm.halted)
        rm.update_equity(119)
        self.assertTrue(rm.halted)
        self.assertEqual(rm.entry_quantity(10, 1000, 1000, 0), 0)


class PaperBrokerTests(unittest.TestCase):
    def test_next_bar_fill_with_slippage_and_commission(self):
        broker = PaperBroker(10_000, commission_per_share=0.01, min_commission=1.0, slippage_bps=10)
        broker.submit_order(Order("X", Side.BUY, 10))
        self.assertEqual(broker.cash, 10_000)  # not filled until the next bar
        fills = broker.on_bar(Bar("X", T0, 100, 101, 99, 100.5))
        self.assertEqual(len(fills), 1)
        self.assertAlmostEqual(fills[0].price, 100.1)
        self.assertAlmostEqual(broker.cash, 10_000 - 1001 - 1.0)
        self.assertAlmostEqual(broker.get_equity(), broker.cash + 10 * 100.5)

        broker.submit_order(Order("X", Side.SELL, 50))  # more than held -> clipped
        fills = broker.on_bar(Bar("X", T0 + timedelta(days=1), 110, 110, 110, 110))
        self.assertEqual(fills[0].qty, 10)
        self.assertEqual(broker.get_positions(), {})

    def test_rejects_unaffordable_and_unheld(self):
        broker = PaperBroker(50, slippage_bps=0, fill_on_next_bar=False)
        broker.update_price("X", 100)
        self.assertEqual(broker.submit_order(Order("X", Side.BUY, 1)).status, OrderStatus.REJECTED)
        self.assertEqual(broker.submit_order(Order("X", Side.SELL, 1)).status, OrderStatus.REJECTED)
        self.assertEqual(broker.submit_order(Order("Y", Side.BUY, 1)).status, OrderStatus.REJECTED)


class BacktestTests(unittest.TestCase):
    def test_no_lookahead_and_pnl(self):
        closes = [100, 100, 100, 110, 120, 120]
        bars = [Bar("X", T0 + timedelta(days=i), c - 1, c, c - 1, c) for i, c in enumerate(closes)]
        strat = ScriptedStrategy({1: Signal.BUY, 4: Signal.SELL})
        risk = RiskConfig(position_size_pct=1.0, stop_loss_pct=None, take_profit_pct=None,
                          max_drawdown_pct=None)
        result = Backtester(strat, risk, initial_cash=1000, slippage_bps=0).run({"X": bars})
        buy, sell = result.fills
        self.assertEqual(buy.timestamp, bars[2].timestamp)  # signal on bar 1 -> fill at bar 2 open
        self.assertEqual(buy.price, 99)
        self.assertEqual(buy.qty, 10)
        self.assertEqual(sell.price, 119)
        self.assertAlmostEqual(result.metrics["realized_pnl"], 200)
        self.assertAlmostEqual(result.metrics["final_equity"], 1200)
        self.assertEqual(result.metrics["win_rate"], 1.0)

    def test_stop_loss_exits(self):
        closes = [100, 100, 100, 90, 90, 90]
        bars = bars_from_closes(closes)
        strat = ScriptedStrategy({1: Signal.BUY})
        risk = RiskConfig(position_size_pct=0.5, stop_loss_pct=0.05, max_drawdown_pct=None)
        result = Backtester(strat, risk, initial_cash=1000, slippage_bps=0).run({"X": bars})
        self.assertEqual([f.side for f in result.fills], [Side.BUY, Side.SELL])
        self.assertIn("stop-loss", result.fills[1].reason)

    def test_multi_symbol_synthetic_run(self):
        src = SyntheticDataSource(days=300)
        data = {s: src.get_bars(s) for s in ("A", "B", "C")}
        result = Backtester(SMACrossover(5, 20)).run(data)
        self.assertEqual(len(result.equity_curve), 300)
        self.assertGreater(len(result.fills), 0)
        self.assertGreaterEqual(result.metrics["max_drawdown"], 0)
        self.assertLessEqual(len(result.fills), 300 * 3)

    def test_metrics_empty(self):
        m = compute_metrics(100, [], [])
        self.assertEqual(m["total_return"], 0)
        self.assertEqual(m["trades"], 0)


class DataTests(unittest.TestCase):
    def test_csv_source(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "ABC.csv").write_text(
                "Date,Open,High,Low,Close,Adj Close,Volume\n"
                "2024-01-03,2,3,1,2.5,2.5,100\n"
                "2024-01-02,1,2,0.5,1.5,1.5,100\n"
                "2024-01-04,null,null,null,null,null,null\n")
            bars = CSVDataSource(d).get_bars("ABC")
        self.assertEqual([b.close for b in bars], [1.5, 2.5])

    def test_synthetic_is_deterministic_weekdays_only(self):
        a = SyntheticDataSource(days=50).get_bars("Z")
        b = SyntheticDataSource(days=50).get_bars("Z")
        self.assertEqual(a, b)
        self.assertTrue(all(bar.timestamp.weekday() < 5 for bar in a))
        self.assertEqual(len(SyntheticDataSource(days=50).get_bars("Z", limit=10)), 10)

    def test_yahoo_parsing(self):
        payload = {"chart": {"result": [{"timestamp": [1704205800, 1704292200],
                   "indicators": {"quote": [{"open": [1, None], "high": [2, None], "low": [0.5, None],
                                             "close": [1.5, None], "volume": [10, None]}]}}]}}
        bars = YahooDataSource(session=FakeSession(payload)).get_bars("AAPL")
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].close, 1.5)


class AlpacaBrokerTests(unittest.TestCase):
    def test_defaults_to_paper_and_parses(self):
        session = FakeSession([{"symbol": "AAPL", "qty": "3", "avg_entry_price": "150.5"}])
        broker = AlpacaBroker("k", "s", session=session)
        self.assertEqual(broker.base_url, AlpacaBroker.PAPER_URL)
        self.assertEqual(broker.get_positions()["AAPL"].qty, 3)
        broker.submit_order(Order("AAPL", Side.BUY, 2))
        method, url, kwargs = session.calls[-1]
        self.assertEqual((method, url), ("POST", AlpacaBroker.PAPER_URL + "/v2/orders"))
        self.assertEqual(kwargs["json"]["side"], "buy")


class LiveTraderTests(unittest.TestCase):
    def test_acts_once_per_new_bar(self):
        src = SyntheticDataSource(days=120)
        broker = PaperBroker(100_000, fill_on_next_bar=False)
        strat = ScriptedStrategy({})
        strat.generate_signal = lambda history, held: Signal.BUY
        trader = LiveTrader(broker, src, strat, RiskManager(), ["A"], interval_seconds=0)
        trader.warm_up()
        self.assertEqual(len(trader.run_once()), 1)
        self.assertEqual(trader.run_once(), [])  # same bar again: no duplicate order
        self.assertEqual(len(broker.get_positions()), 1)


class CLITests(unittest.TestCase):
    def test_backtest_command(self):
        with tempfile.TemporaryDirectory() as d:
            trades = Path(d, "t.csv")
            rc = main(["backtest", "-s", "A", "B", "--strategy", "rsi", "-p", "period=10",
                       "--trades-csv", str(trades)])
            self.assertEqual(rc, 0)
            self.assertTrue(trades.read_text().startswith("timestamp,symbol"))


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload, self.headers, self.calls = payload, {}, []

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return FakeResponse(self.payload)


if __name__ == "__main__":
    unittest.main()
