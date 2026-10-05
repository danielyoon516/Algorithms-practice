"""autotrader: a small, dependency-free automatic stock trading platform."""
from .backtest import Backtester, BacktestResult
from .broker import AlpacaBroker, PaperBroker
from .engine import TradingEngine
from .risk import RiskConfig, RiskManager
from .strategies import STRATEGIES, Strategy, build_strategy

__all__ = ["Backtester", "BacktestResult", "AlpacaBroker", "PaperBroker", "TradingEngine",
           "RiskConfig", "RiskManager", "STRATEGIES", "Strategy", "build_strategy"]
__version__ = "0.1.0"
