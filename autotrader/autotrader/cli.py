"""Command-line interface: ``python -m autotrader {backtest,live,strategies}``."""
from __future__ import annotations

import argparse
import csv
import logging
import sys

from .backtest import Backtester
from .broker import AlpacaBroker, PaperBroker
from .data import AlpacaDataSource, CSVDataSource, SyntheticDataSource, YahooDataSource
from .live import LiveTrader
from .risk import RiskConfig, RiskManager
from .strategies import STRATEGIES, build_strategy


def _parse_params(items: list[str]) -> dict:
    params = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        if not _:
            raise SystemExit(f"--param must look like key=value, got {item!r}")
        try:
            value = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                value = raw
        params[key.strip()] = value
    return params


def _pct_or_none(raw: str) -> float | None:
    value = float(raw)
    return None if value <= 0 else value


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--symbols", "-s", nargs="+", default=["AAPL", "MSFT", "NVDA"])
    p.add_argument("--strategy", choices=sorted(STRATEGIES), default="sma")
    p.add_argument("--param", "-p", action="append", metavar="KEY=VALUE",
                   help="strategy parameter, e.g. -p fast=10 -p slow=30")
    p.add_argument("--data", choices=["synthetic", "csv", "yahoo", "alpaca"], default="synthetic")
    p.add_argument("--csv-dir", default="data", help="directory of <SYMBOL>.csv files")
    risk = p.add_argument_group("risk")
    risk.add_argument("--position-size", type=float, default=0.10, help="fraction of equity per position")
    risk.add_argument("--max-positions", type=int, default=5)
    risk.add_argument("--stop-loss", type=_pct_or_none, default=0.05, help="0 disables")
    risk.add_argument("--take-profit", type=_pct_or_none, default=0.15, help="0 disables")
    risk.add_argument("--max-drawdown", type=_pct_or_none, default=0.20, help="0 disables")
    p.add_argument("-v", "--verbose", action="store_true")


def _data_source(args):
    if args.data == "csv":
        return CSVDataSource(args.csv_dir)
    if args.data == "yahoo":
        return YahooDataSource()
    if args.data == "alpaca":
        return AlpacaDataSource()
    return SyntheticDataSource()


def _risk_config(args) -> RiskConfig:
    return RiskConfig(position_size_pct=args.position_size, max_positions=args.max_positions,
                      stop_loss_pct=args.stop_loss, take_profit_pct=args.take_profit,
                      max_drawdown_pct=args.max_drawdown)


def cmd_backtest(args) -> int:
    strategy = build_strategy(args.strategy, **_parse_params(args.param))
    source = _data_source(args)
    try:
        data = {s: source.get_bars(s) for s in args.symbols}
    except Exception as exc:  # noqa: BLE001 - report data problems without a traceback
        print(f"error: could not load {args.data} data: {exc}", file=sys.stderr)
        return 1
    for sym, bars in data.items():
        if not bars:
            print(f"warning: no data for {sym}", file=sys.stderr)
    bt = Backtester(strategy, _risk_config(args), initial_cash=args.cash,
                    commission_per_share=args.commission, slippage_bps=args.slippage_bps)
    result = bt.run(data)
    print(f"Strategy: {strategy!r}   Symbols: {', '.join(args.symbols)}   Data: {args.data}")
    print("-" * 60)
    print(result.summary())
    if args.trades_csv:
        with open(args.trades_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["timestamp", "symbol", "side", "qty", "price", "commission", "reason"])
            for f in result.fills:
                w.writerow([f.timestamp.isoformat(), f.symbol, f.side.value, f.qty,
                            f"{f.price:.4f}", f"{f.commission:.2f}", f.reason])
        print(f"\nTrades written to {args.trades_csv}")
    if args.equity_csv:
        with open(args.equity_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["timestamp", "equity"])
            w.writerows((ts.isoformat(), f"{eq:.2f}") for ts, eq in result.equity_curve)
        print(f"Equity curve written to {args.equity_csv}")
    return 0


def cmd_live(args) -> int:
    strategy = build_strategy(args.strategy, **_parse_params(args.param))
    if args.broker == "alpaca":
        if args.real_money and input("Trade REAL money on Alpaca? type 'yes': ").strip() != "yes":
            print("aborted")
            return 1
        broker = AlpacaBroker(live=args.real_money)
    else:
        broker = PaperBroker(args.cash, slippage_bps=args.slippage_bps, fill_on_next_bar=False)
    trader = LiveTrader(broker, _data_source(args), strategy, RiskManager(_risk_config(args)),
                        args.symbols, interval_seconds=args.interval)
    trader.run(max_iterations=args.iterations)
    return 0


def cmd_strategies(_args) -> int:
    for name, cls in sorted(STRATEGIES.items()):
        doc = (cls.__doc__ or "").strip().splitlines()[0]
        print(f"{name:10s} {doc}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autotrader", description="Automatic stock trading platform")
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="simulate a strategy on historical data")
    _add_common(bt)
    bt.add_argument("--cash", type=float, default=100_000.0)
    bt.add_argument("--commission", type=float, default=0.0, help="per-share commission")
    bt.add_argument("--slippage-bps", type=float, default=5.0)
    bt.add_argument("--trades-csv", help="write fills to this CSV file")
    bt.add_argument("--equity-csv", help="write the equity curve to this CSV file")
    bt.set_defaults(func=cmd_backtest)

    lv = sub.add_parser("live", help="run the strategy continuously (paper by default)")
    _add_common(lv)
    lv.add_argument("--broker", choices=["paper", "alpaca"], default="paper")
    lv.add_argument("--real-money", action="store_true",
                    help="use Alpaca's LIVE endpoint instead of its paper account")
    lv.add_argument("--cash", type=float, default=100_000.0, help="starting cash for local paper broker")
    lv.add_argument("--slippage-bps", type=float, default=5.0)
    lv.add_argument("--interval", type=int, default=60, help="seconds between polls")
    lv.add_argument("--iterations", type=int, help="stop after N polls (default: run forever)")
    lv.set_defaults(func=cmd_live)

    st = sub.add_parser("strategies", help="list available strategies")
    st.set_defaults(func=cmd_strategies)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if getattr(args, "verbose", False) or args.command == "live"
                        else logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
