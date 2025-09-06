"""
量化交易系统包
币安新币动态Delta管理策略回测系统
"""

__version__ = "1.0.0"
__author__ = "Quant Trading System"
__description__ = "币安新币动态Delta管理量化交易系统"

# 导入主要模块
from .data.data_manager import BinanceDataManager
from .strategy.delta_strategy import DeltaStrategy
from .risk.risk_manager import RiskManager
from .backtest.backtest_engine import BacktestEngine
from .analysis.result_analyzer import ResultAnalyzer

__all__ = [
    'BinanceDataManager',
    'DeltaStrategy', 
    'RiskManager',
    'BacktestEngine',
    'ResultAnalyzer'
]
