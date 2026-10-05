# autotrader: an automatic stock trading platform

`autotrader` is a small, readable Python platform for building, backtesting and running
automated stock trading strategies. The core needs **only the Python standard library** (3.10+).
`requests` is needed only for the Yahoo and Alpaca integrations.

> ⚠️ **Disclaimer:** this is educational software, not financial advice. Automated trading can
> lose money quickly. Backtest results do not predict future returns. Run it in paper mode
> for a long time before you risk real capital, and never trade money you can't afford to lose.

## Features

- **Strategies:** SMA crossover, RSI mean reversion, Bollinger band reversion and momentum.
  Each one is a small class, so you can add your own.
- **Risk management:** per-position sizing as a percentage of equity, a cap on open positions,
  stop-loss, take-profit, a cash reserve, and a **max-drawdown kill switch** that sells every
  position and stops new entries.
- **Backtester:** an event-driven engine. An order placed on one bar fills at the **next bar's
  open**, so a strategy can't see the future. It models slippage and commissions and reports
  return, CAGR, Sharpe, max drawdown, win rate and a buy-and-hold benchmark.
- **Brokers:**
  - `PaperBroker`: local simulated fills.
  - `AlpacaBroker`: real order routing through the Alpaca API. It uses the **paper account by
    default**. Real money needs both `--real-money` and typing `yes` at a prompt.
- **Data:** synthetic data (offline, deterministic), CSV files, Yahoo Finance, or Alpaca market data.
- **One engine** drives both backtests and live trading, so the code you test is the code that trades.

## Quick start

```bash
cd autotrader
python -m autotrader strategies                       # list strategies
python -m autotrader backtest                         # SMA crossover on synthetic data
python -m autotrader backtest --strategy rsi -s AAPL MSFT -p period=10 -p lower=25
python -m autotrader backtest --data yahoo -s AAPL SPY --trades-csv trades.csv --equity-csv equity.csv
python -m autotrader backtest --data csv --csv-dir ./data -s AAPL   # reads data/AAPL.csv
```

You can also run `pip install -e ".[live]"` and then use the `autotrader` command.

### Paper / live trading

```bash
# Local paper trading: prices come from Yahoo, fills are simulated
python -m autotrader live --data yahoo -s AAPL MSFT --strategy momentum --interval 300

# Alpaca paper account (free at alpaca.markets)
export APCA_API_KEY_ID=...  APCA_API_SECRET_KEY=...
python -m autotrader live --broker alpaca --data alpaca -s AAPL MSFT NVDA

# Real money (asks you to confirm)
python -m autotrader live --broker alpaca --data alpaca --real-money -s AAPL
```

At startup the live loop loads enough history to warm up the strategy. After that it polls every
`--interval` seconds and acts **once per new daily bar**. With Alpaca it waits while the market
is closed. Stop it with Ctrl-C.

### Risk flags (backtest and live)

| Flag | Default | Meaning |
|---|---|---|
| `--position-size` | `0.10` | fraction of equity put into each new position |
| `--max-positions` | `5` | maximum number of positions open at once |
| `--stop-loss` | `0.05` | exit when price is 5% below entry (`0` disables) |
| `--take-profit` | `0.15` | exit when price is 15% above entry (`0` disables) |
| `--max-drawdown` | `0.20` | kill switch: sell everything and stop trading at a 20% drawdown (`0` disables) |

## Writing a strategy

```python
from autotrader.strategies import Strategy, STRATEGIES
from autotrader.models import Signal
from autotrader import indicators as ind

class EMACross(Strategy):
    """Buy when price closes above its EMA, sell when it closes below."""
    name = "ema"

    def __init__(self, period: int = 20):
        self.period = period

    @property
    def warmup(self):
        return self.period

    def generate_signal(self, history, has_position):
        closes = [b.close for b in history]
        e = ind.ema(closes, self.period)
        if e is None:
            return Signal.HOLD
        if closes[-1] > e and not has_position:
            return Signal.BUY
        if closes[-1] < e and has_position:
            return Signal.SELL
        return Signal.HOLD

STRATEGIES["ema"] = EMACross
```

Strategies only produce signals. Position sizing, protective exits and execution happen in
`RiskManager`, `TradingEngine` and the broker.

## Layout

```
autotrader/
  models.py      Bar, Order, Fill, Position, Signal
  indicators.py  SMA, EMA, RSI, Bollinger bands, rate of change
  strategies.py  strategy base class and built-in strategies
  risk.py        position sizing, stop-loss/take-profit, drawdown kill switch
  broker.py      PaperBroker (simulated) and AlpacaBroker (REST)
  data.py        synthetic, CSV, Yahoo and Alpaca data sources
  engine.py      turns bars into orders (shared by backtest and live)
  backtest.py    backtester and performance metrics
  live.py        polling live-trading loop
  cli.py         command-line interface
tests/           unittest suite (python -m unittest discover -s tests -t .)
```

## Limitations

- Long-only, market orders, daily bars.
- No shorting, options, intraday bars, or bracket and limit orders yet.
- With Alpaca, a fill happens asynchronously after the order is accepted. Positions are read
  back from the account on the next poll.
