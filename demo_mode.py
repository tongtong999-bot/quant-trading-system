#!/usr/bin/env python3
"""
演示模式 - 使用模拟数据展示系统功能
当网络连接有问题时，可以使用此模式查看系统功能
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from src.strategy.delta_strategy import DeltaStrategy
from src.risk.risk_manager import RiskManager
from src.analysis.result_analyzer import ResultAnalyzer
from src.utils.helpers import calculate_technical_indicators


def generate_mock_data(symbol: str, days: int = 14) -> pd.DataFrame:
    """生成模拟K线数据"""
    print(f"生成 {symbol} 的模拟数据...")
    
    # 生成时间序列
    timeframe = '15T'  # 15分钟
    periods = days * 24 * 4  # 每天96个15分钟K线
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        periods=periods,
        freq=timeframe
    )
    
    # 生成价格数据（带趋势和波动）
    np.random.seed(42)  # 固定随机种子
    
    # 基础价格
    base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
    
    # 生成价格变化
    trend = np.linspace(0, 0.1, periods)  # 10%的上升趋势
    noise = np.random.normal(0, 0.02, periods)  # 2%的随机波动
    price_changes = trend + noise
    
    # 计算价格
    prices = base_price * (1 + price_changes).cumprod()
    
    # 生成OHLC数据
    data = []
    for i, (timestamp, price) in enumerate(zip(dates, prices)):
        # 生成开盘价、最高价、最低价、收盘价
        open_price = price * (1 + np.random.normal(0, 0.001))
        close_price = price * (1 + np.random.normal(0, 0.001))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
        
        # 生成成交量
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'type': 'spot'
        })
    
    # 创建现货数据
    spot_df = pd.DataFrame(data)
    spot_df.set_index('timestamp', inplace=True)
    
    # 创建期货数据（价格略有差异）
    future_data = []
    for i, (timestamp, price) in enumerate(zip(dates, prices)):
        # 期货价格与现货略有差异
        future_price = price * (1 + np.random.normal(0, 0.0005))
        
        open_price = future_price * (1 + np.random.normal(0, 0.001))
        close_price = future_price * (1 + np.random.normal(0, 0.001))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
        
        volume = np.random.randint(1000, 10000)
        
        future_data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'type': 'future'
        })
    
    future_df = pd.DataFrame(future_data)
    future_df.set_index('timestamp', inplace=True)
    
    # 合并数据
    combined_df = pd.concat([spot_df, future_df])
    
    # 添加技术指标
    combined_df = calculate_technical_indicators(combined_df)
    
    print(f"生成了 {len(combined_df)} 行数据")
    return combined_df


def run_demo_backtest(symbols: list, days: int = 14):
    """运行演示回测"""
    print("="*60)
    print("量化交易系统演示模式")
    print("="*60)
    
    results = []
    
    for symbol in symbols:
        print(f"\n开始回测 {symbol}...")
        
        # 生成模拟数据
        data = generate_mock_data(symbol, days)
        
        # 初始化策略和风控
        strategy = DeltaStrategy()
        risk_manager = RiskManager()
        
        # 运行回测
        initial_capital = 2000
        current_capital = initial_capital
        trades = []
        equity_curve = []
        
        # 按时间顺序处理数据
        timestamps = sorted(data.index.unique())
        
        for i, timestamp in enumerate(timestamps):
            if i < 20:  # 跳过前20个数据点
                continue
            
            current_data = data[data.index <= timestamp]
            
            # 检查是否建仓
            if strategy.should_enter_position(current_data, timestamp):
                new_trades = strategy.enter_position(current_data, timestamp, current_capital)
                trades.extend(new_trades)
                print(f"  建仓: {timestamp}, 交易数: {len(new_trades)}")
            
            # 检查是否调仓
            if strategy.should_adjust_position(current_data, timestamp):
                new_trades = strategy.adjust_position(current_data, timestamp)
                trades.extend(new_trades)
                if new_trades:
                    print(f"  调仓: {timestamp}, 交易数: {len(new_trades)}")
            
            # 计算当前权益
            if not current_data.empty:
                spot_data = current_data[current_data['type'] == 'spot']
                if not spot_data.empty:
                    current_price = spot_data.iloc[-1]['close']
                    current_equity = strategy.calculate_pnl(current_price) + initial_capital
                    
                    equity_curve.append({
                        'timestamp': timestamp,
                        'equity': current_equity,
                        'price': current_price,
                        'spot_amount': strategy.position.spot_amount,
                        'short_amount': strategy.position.short_amount,
                        'position_type': strategy.position.position_type.value
                    })
        
        # 计算最终结果
        if equity_curve:
            equity_df = pd.DataFrame(equity_curve)
            final_equity = equity_df['equity'].iloc[-1]
            total_return = (final_equity - initial_capital) / initial_capital
            
            print(f"  {symbol} 回测完成:")
            print(f"    初始资金: {initial_capital:.2f} USDT")
            print(f"    最终资金: {final_equity:.2f} USDT")
            print(f"    总收益率: {total_return:.2%}")
            print(f"    交易次数: {len(trades)}")
            
            results.append({
                'symbol': symbol,
                'initial_capital': initial_capital,
                'final_capital': final_equity,
                'total_return': total_return,
                'total_trades': len(trades),
                'trades': trades,
                'equity_curve': equity_df
            })
        else:
            print(f"  {symbol} 回测失败: 没有有效数据")
    
    # 生成分析报告
    if results:
        print("\n" + "="*60)
        print("回测结果汇总")
        print("="*60)
        
        total_return = np.mean([r['total_return'] for r in results])
        avg_trades = np.mean([r['total_trades'] for r in results])
        
        print(f"测试交易对: {len(results)} 个")
        print(f"平均收益率: {total_return:.2%}")
        print(f"平均交易次数: {avg_trades:.1f}")
        
        # 最佳表现者
        best = max(results, key=lambda x: x['total_return'])
        print(f"最佳表现: {best['symbol']} ({best['total_return']:.2%})")
        
        # 生成详细报告
        analyzer = ResultAnalyzer()
        
        # 创建BacktestResult对象
        from src.backtest.backtest_engine import BacktestResult
        
        backtest_results = []
        for result in results:
            backtest_result = BacktestResult(
                symbol=result['symbol'],
                start_date=result['equity_curve']['timestamp'].iloc[0],
                end_date=result['equity_curve']['timestamp'].iloc[-1],
                initial_capital=result['initial_capital'],
                final_capital=result['final_capital'],
                total_return=result['total_return'],
                total_trades=result['total_trades'],
                winning_trades=len([t for t in result['trades'] if 'buy' in t.reason]),
                losing_trades=len([t for t in result['trades'] if 'sell' in t.reason]),
                max_drawdown=-0.05,  # 模拟值
                sharpe_ratio=1.5,    # 模拟值
                calmar_ratio=2.0,    # 模拟值
                win_rate=0.6,        # 模拟值
                profit_loss_ratio=1.2,  # 模拟值
                trades=result['trades'],
                equity_curve=result['equity_curve'],
                risk_metrics={}
            )
            backtest_results.append(backtest_result)
        
        # 生成HTML报告
        report_path = analyzer.generate_html_report(backtest_results, "results")
        if report_path:
            print(f"\n详细报告已生成: {report_path}")
        
        print("\n演示完成！")
    else:
        print("没有有效的回测结果")


def main():
    """主函数"""
    print("启动量化交易系统演示模式...")
    print("注意: 此模式使用模拟数据，仅用于演示系统功能")
    
    # 测试交易对
    symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
    
    # 运行演示
    run_demo_backtest(symbols, days=7)


if __name__ == "__main__":
    main()
