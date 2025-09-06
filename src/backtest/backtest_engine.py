"""
回测引擎 - 向量化计算与并行处理
支持多币种并行回测和性能分析
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import pickle
import yaml
from loguru import logger
import asyncio

from ..strategy.delta_strategy import DeltaStrategy, Trade, Position
from ..risk.risk_manager import RiskManager
from ..data.data_manager import BinanceDataManager


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 2000.0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    symbols: List[str] = None
    parallel_workers: int = 4
    vectorized: bool = True
    save_trades: bool = True
    save_metrics: bool = True


@dataclass
class BacktestResult:
    """回测结果"""
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    max_drawdown: float
    sharpe_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_loss_ratio: float
    trades: List[Trade]
    equity_curve: pd.DataFrame
    risk_metrics: Dict


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化回测引擎"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.backtest_config = BacktestConfig(**self.config['backtest'])
        self.strategy_config = self.config['strategy']
        self.risk_config = self.config['risk_control']
        
        # 组件初始化
        self.data_manager = BinanceDataManager(config_path)
        self.results: List[BacktestResult] = []
        
        # 性能统计
        self.total_symbols = 0
        self.successful_backtests = 0
        self.failed_backtests = 0
        
    async def run_backtest(self, symbols: List[str] = None) -> List[BacktestResult]:
        """
        运行回测
        
        Args:
            symbols: 交易对列表，None则自动获取新币
            
        Returns:
            回测结果列表
        """
        try:
            # 获取交易对
            if symbols is None:
                new_listings = await self.data_manager.get_new_listings(20)
                symbols = [listing['symbol'] for listing in new_listings]
            
            self.total_symbols = len(symbols)
            logger.info(f"开始回测 {self.total_symbols} 个交易对")
            
            # 并行回测
            if self.backtest_config.parallel_workers > 1:
                results = await self._run_parallel_backtest(symbols)
            else:
                results = await self._run_sequential_backtest(symbols)
            
            self.results = results
            self.successful_backtests = len([r for r in results if r is not None])
            self.failed_backtests = self.total_symbols - self.successful_backtests
            
            logger.info(f"回测完成: 成功 {self.successful_backtests}, 失败 {self.failed_backtests}")
            
            return results
            
        except Exception as e:
            logger.error(f"回测运行失败: {e}")
            return []
    
    async def _run_parallel_backtest(self, symbols: List[str]) -> List[BacktestResult]:
        """并行回测"""
        try:
            # 使用线程池进行并行处理
            with ThreadPoolExecutor(max_workers=self.backtest_config.parallel_workers) as executor:
                # 创建任务
                tasks = []
                for symbol in symbols:
                    task = executor.submit(self._backtest_single_symbol, symbol)
                    tasks.append((symbol, task))
                
                # 等待结果
                results = []
                for symbol, task in tasks:
                    try:
                        result = task.result(timeout=300)  # 5分钟超时
                        results.append(result)
                    except Exception as e:
                        logger.error(f"回测失败 {symbol}: {e}")
                        results.append(None)
                
                return results
                
        except Exception as e:
            logger.error(f"并行回测失败: {e}")
            return []
    
    async def _run_sequential_backtest(self, symbols: List[str]) -> List[BacktestResult]:
        """顺序回测"""
        results = []
        for symbol in symbols:
            try:
                result = self._backtest_single_symbol(symbol)
                results.append(result)
            except Exception as e:
                logger.error(f"回测失败 {symbol}: {e}")
                results.append(None)
        return results
    
    def _backtest_single_symbol(self, symbol: str) -> Optional[BacktestResult]:
        """
        单个交易对回测
        
        Args:
            symbol: 交易对符号
            
        Returns:
            回测结果
        """
        try:
            logger.info(f"开始回测 {symbol}")
            
            # 获取数据
            data = asyncio.run(self.data_manager.fetch_historical_data(symbol, 14))
            if data is None or data.empty:
                logger.warning(f"数据获取失败 {symbol}")
                return None
            
            # 检查流动性
            liquidity_ok, _ = asyncio.run(self.data_manager.check_liquidity(symbol))
            if not liquidity_ok:
                logger.warning(f"流动性不足 {symbol}")
                return None
            
            # 初始化策略和风控
            strategy = DeltaStrategy()
            risk_manager = RiskManager()
            
            # 运行回测
            result = self._run_strategy_backtest(strategy, risk_manager, data, symbol)
            
            if result:
                logger.info(f"回测完成 {symbol}: 收益率 {result.total_return:.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"单币种回测失败 {symbol}: {e}")
            return None
    
    def _run_strategy_backtest(self, strategy: DeltaStrategy, risk_manager: RiskManager,
                              data: pd.DataFrame, symbol: str) -> Optional[BacktestResult]:
        """
        运行策略回测
        
        Args:
            strategy: 策略实例
            risk_manager: 风控实例
            data: 历史数据
            symbol: 交易对符号
            
        Returns:
            回测结果
        """
        try:
            initial_capital = self.backtest_config.initial_capital
            current_capital = initial_capital
            
            # 初始化权益曲线
            equity_curve = []
            trades = []
            
            # 按时间顺序处理数据
            timestamps = sorted(data.index.unique())
            start_time = timestamps[0]
            end_time = timestamps[-1]
            
            for i, timestamp in enumerate(timestamps):
                try:
                    current_data = data[data.index <= timestamp]
                    
                    # 检查是否建仓
                    if strategy.should_enter_position(current_data, timestamp):
                        new_trades = strategy.enter_position(current_data, timestamp, current_capital)
                        trades.extend(new_trades)
                        
                        # 记录交易成本
                        for trade in new_trades:
                            risk_manager.add_trade_cost(trade.__dict__)
                    
                    # 检查是否调仓
                    if strategy.should_adjust_position(current_data, timestamp):
                        new_trades = strategy.adjust_position(current_data, timestamp)
                        trades.extend(new_trades)
                        
                        # 记录交易成本
                        for trade in new_trades:
                            risk_manager.add_trade_cost(trade.__dict__)
                    
                    # 风控检查
                    if strategy.position.spot_amount > 0 or strategy.position.short_amount > 0:
                        current_price = self._get_current_price(current_data, timestamp)
                        
                        # 检查是否需要强制平仓
                        if risk_manager.should_force_close(
                            strategy.position.__dict__, current_price, 
                            strategy.position.entry_time or timestamp, timestamp
                        ):
                            # 强制平仓
                            force_trades = self._force_close_position(strategy, current_data, timestamp)
                            trades.extend(force_trades)
                    
                    # 计算当前权益
                    current_price = self._get_current_price(current_data, timestamp)
                    current_equity = self._calculate_current_equity(strategy, current_price, current_capital)
                    
                    # 记录权益曲线
                    equity_curve.append({
                        'timestamp': timestamp,
                        'equity': current_equity,
                        'price': current_price,
                        'spot_amount': strategy.position.spot_amount,
                        'short_amount': strategy.position.short_amount,
                        'position_type': strategy.position.position_type.value
                    })
                    
                except Exception as e:
                    logger.error(f"处理时间点失败 {timestamp}: {e}")
                    continue
            
            # 计算最终结果
            if not equity_curve:
                return None
            
            equity_df = pd.DataFrame(equity_curve)
            final_equity = equity_df['equity'].iloc[-1]
            
            # 计算性能指标
            metrics = self._calculate_performance_metrics(equity_df, trades, initial_capital)
            
            # 创建回测结果
            result = BacktestResult(
                symbol=symbol,
                start_date=start_time,
                end_date=end_time,
                initial_capital=initial_capital,
                final_capital=final_equity,
                total_return=(final_equity - initial_capital) / initial_capital,
                total_trades=len(trades),
                winning_trades=metrics['winning_trades'],
                losing_trades=metrics['losing_trades'],
                max_drawdown=metrics['max_drawdown'],
                sharpe_ratio=metrics['sharpe_ratio'],
                calmar_ratio=metrics['calmar_ratio'],
                win_rate=metrics['win_rate'],
                profit_loss_ratio=metrics['profit_loss_ratio'],
                trades=trades,
                equity_curve=equity_df,
                risk_metrics=risk_manager.get_risk_summary()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"策略回测失败: {e}")
            return None
    
    def _get_current_price(self, data: pd.DataFrame, timestamp: datetime) -> float:
        """获取当前价格"""
        try:
            current_data = data[data.index <= timestamp].tail(1)
            spot_data = current_data[current_data['type'] == 'spot']
            if not spot_data.empty:
                return spot_data.iloc[0]['close']
            return 0.0
        except:
            return 0.0
    
    def _calculate_current_equity(self, strategy: DeltaStrategy, current_price: float, 
                                 initial_capital: float) -> float:
        """计算当前权益"""
        try:
            if current_price == 0:
                return initial_capital
            
            # 计算现货价值
            spot_value = strategy.position.spot_amount * current_price
            
            # 计算空单价值（简化处理，假设价格相同）
            short_value = strategy.position.short_amount * current_price
            
            # 计算盈亏
            spot_pnl = (current_price - strategy.position.entry_price_spot) * strategy.position.spot_amount
            short_pnl = (strategy.position.entry_price_short - current_price) * strategy.position.short_amount
            
            # 总权益 = 初始资金 + 盈亏
            total_equity = initial_capital + spot_pnl + short_pnl
            
            return max(0, total_equity)  # 权益不能为负
            
        except Exception as e:
            logger.error(f"权益计算失败: {e}")
            return initial_capital
    
    def _force_close_position(self, strategy: DeltaStrategy, data: pd.DataFrame, 
                            timestamp: datetime) -> List[Trade]:
        """强制平仓"""
        trades = []
        
        try:
            current_data = data[data.index <= timestamp].tail(1)
            
            # 平仓现货
            if strategy.position.spot_amount > 0:
                spot_data = current_data[current_data['type'] == 'spot']
                if not spot_data.empty:
                    spot_price = spot_data.iloc[0]['close']
                    spot_amount = strategy.position.spot_amount
                    
                    trade = Trade(
                        timestamp=timestamp,
                        symbol=spot_data.iloc[0].get('symbol', 'UNKNOWN'),
                        side='sell',
                        amount=spot_amount,
                        price=spot_price,
                        trade_type='spot',
                        reason='force_close_spot',
                        position_after=Position()
                    )
                    
                    trades.append(trade)
                    strategy.position.spot_amount = 0
            
            # 平仓空单
            if strategy.position.short_amount > 0:
                future_data = current_data[current_data['type'] == 'future']
                if not future_data.empty:
                    future_price = future_data.iloc[0]['close']
                    short_amount = strategy.position.short_amount
                    
                    trade = Trade(
                        timestamp=timestamp,
                        symbol=future_data.iloc[0].get('symbol', 'UNKNOWN'),
                        side='buy',
                        amount=short_amount,
                        price=future_price,
                        trade_type='future',
                        reason='force_close_short',
                        position_after=Position()
                    )
                    
                    trades.append(trade)
                    strategy.position.short_amount = 0
            
            strategy.trades.extend(trades)
            
        except Exception as e:
            logger.error(f"强制平仓失败: {e}")
        
        return trades
    
    def _calculate_performance_metrics(self, equity_df: pd.DataFrame, trades: List[Trade], 
                                     initial_capital: float) -> Dict:
        """计算性能指标"""
        try:
            if equity_df.empty:
                return {}
            
            # 基础指标
            final_equity = equity_df['equity'].iloc[-1]
            total_return = (final_equity - initial_capital) / initial_capital
            
            # 计算回撤
            equity_series = equity_df['equity']
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak
            max_drawdown = drawdown.min()
            
            # 计算夏普比率
            returns = equity_series.pct_change().dropna()
            if len(returns) > 1 and returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24 * 4)  # 15分钟数据
            else:
                sharpe_ratio = 0.0
            
            # 卡玛比率
            calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
            
            # 交易统计
            winning_trades = 0
            losing_trades = 0
            total_profit = 0.0
            total_loss = 0.0
            
            # 简化交易盈亏计算
            for trade in trades:
                if trade.trade_type == 'spot':
                    if trade.side == 'buy':
                        total_profit += trade.amount * trade.price
                    else:
                        total_loss += trade.amount * trade.price
                elif trade.trade_type == 'future':
                    if trade.side == 'sell':
                        total_profit += trade.amount * trade.price
                    else:
                        total_loss += trade.amount * trade.price
            
            if total_profit > 0:
                winning_trades = 1
            if total_loss > 0:
                losing_trades = 1
            
            win_rate = winning_trades / max(1, winning_trades + losing_trades)
            profit_loss_ratio = total_profit / max(1, total_loss) if total_loss > 0 else 0.0
            
            return {
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'calmar_ratio': calmar_ratio,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_loss_ratio': profit_loss_ratio
            }
            
        except Exception as e:
            logger.error(f"性能指标计算失败: {e}")
            return {}
    
    def save_results(self, output_dir: str = "results") -> Dict[str, str]:
        """
        保存回测结果
        
        Args:
            output_dir: 输出目录
            
        Returns:
            保存的文件路径
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            saved_files = {}
            
            # 保存回测结果
            if self.results:
                results_file = output_path / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                with open(results_file, 'wb') as f:
                    pickle.dump(self.results, f)
                saved_files['results'] = str(results_file)
                
                # 保存汇总报告
                summary_file = output_path / f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                summary_df = self._create_summary_dataframe()
                summary_df.to_csv(summary_file, index=False)
                saved_files['summary'] = str(summary_file)
                
                # 保存详细交易记录
                if self.backtest_config.save_trades:
                    trades_file = output_path / f"trades_detail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    trades_df = self._create_trades_dataframe()
                    trades_df.to_csv(trades_file, index=False)
                    saved_files['trades'] = str(trades_file)
            
            logger.info(f"回测结果已保存到: {output_dir}")
            return saved_files
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            return {}
    
    def _create_summary_dataframe(self) -> pd.DataFrame:
        """创建汇总数据框"""
        try:
            summary_data = []
            
            for result in self.results:
                if result is None:
                    continue
                
                summary_data.append({
                    'symbol': result.symbol,
                    'start_date': result.start_date,
                    'end_date': result.end_date,
                    'initial_capital': result.initial_capital,
                    'final_capital': result.final_capital,
                    'total_return': result.total_return,
                    'total_trades': result.total_trades,
                    'winning_trades': result.winning_trades,
                    'losing_trades': result.losing_trades,
                    'max_drawdown': result.max_drawdown,
                    'sharpe_ratio': result.sharpe_ratio,
                    'calmar_ratio': result.calmar_ratio,
                    'win_rate': result.win_rate,
                    'profit_loss_ratio': result.profit_loss_ratio
                })
            
            return pd.DataFrame(summary_data)
            
        except Exception as e:
            logger.error(f"创建汇总数据框失败: {e}")
            return pd.DataFrame()
    
    def _create_trades_dataframe(self) -> pd.DataFrame:
        """创建交易记录数据框"""
        try:
            trades_data = []
            
            for result in self.results:
                if result is None or not result.trades:
                    continue
                
                for trade in result.trades:
                    trades_data.append({
                        'symbol': result.symbol,
                        'timestamp': trade.timestamp,
                        'side': trade.side,
                        'amount': trade.amount,
                        'price': trade.price,
                        'trade_type': trade.trade_type,
                        'reason': trade.reason
                    })
            
            return pd.DataFrame(trades_data)
            
        except Exception as e:
            logger.error(f"创建交易数据框失败: {e}")
            return pd.DataFrame()
    
    def get_overall_performance(self) -> Dict:
        """获取整体性能统计"""
        try:
            if not self.results:
                return {}
            
            valid_results = [r for r in self.results if r is not None]
            if not valid_results:
                return {}
            
            # 计算整体指标
            total_return = np.mean([r.total_return for r in valid_results])
            avg_max_drawdown = np.mean([r.max_drawdown for r in valid_results])
            avg_sharpe_ratio = np.mean([r.sharpe_ratio for r in valid_results])
            avg_win_rate = np.mean([r.win_rate for r in valid_results])
            
            # 成功率统计
            positive_returns = len([r for r in valid_results if r.total_return > 0])
            success_rate = positive_returns / len(valid_results)
            
            return {
                'total_symbols': self.total_symbols,
                'successful_backtests': self.successful_backtests,
                'failed_backtests': self.failed_backtests,
                'success_rate': success_rate,
                'avg_total_return': total_return,
                'avg_max_drawdown': avg_max_drawdown,
                'avg_sharpe_ratio': avg_sharpe_ratio,
                'avg_win_rate': avg_win_rate,
                'best_performer': max(valid_results, key=lambda x: x.total_return).symbol,
                'worst_performer': min(valid_results, key=lambda x: x.total_return).symbol
            }
            
        except Exception as e:
            logger.error(f"获取整体性能失败: {e}")
            return {}


# 使用示例
async def main():
    """回测引擎使用示例"""
    engine = BacktestEngine()
    
    # 运行回测
    results = await engine.run_backtest(['BTC/USDT', 'ETH/USDT'])
    
    # 保存结果
    saved_files = engine.save_results()
    print(f"结果已保存: {saved_files}")
    
    # 获取整体性能
    performance = engine.get_overall_performance()
    print(f"整体性能: {performance}")


if __name__ == "__main__":
    asyncio.run(main())

