#!/usr/bin/env python3
"""
量化交易系统主程序入口
币安新币动态Delta管理策略回测系统
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import yaml
from loguru import logger

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.data_manager import BinanceDataManager
from src.strategy.delta_strategy import DeltaStrategy
from src.risk.risk_manager import RiskManager
from src.backtest.backtest_engine import BacktestEngine
from src.analysis.result_analyzer import ResultAnalyzer
from src.utils.helpers import format_currency, format_percentage, validate_config


class QuantTradingSystem:
    """量化交易系统主类"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化系统"""
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()
        
        # 初始化组件
        self.data_manager = BinanceDataManager(config_path)
        self.strategy = DeltaStrategy(config_path)
        self.risk_manager = RiskManager(config_path)
        self.backtest_engine = BacktestEngine(config_path)
        self.result_analyzer = ResultAnalyzer(config_path)
        
        logger.info("量化交易系统初始化完成")
    
    def _load_config(self) -> dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 验证配置完整性
            required_keys = ['data', 'strategy', 'risk_control', 'backtest', 'output']
            if not validate_config(config, required_keys):
                raise ValueError("配置文件缺少必需的配置项")
            
            return config
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            sys.exit(1)
    
    def _setup_logging(self):
        """设置日志系统"""
        try:
            # 创建日志目录
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            # 配置日志格式
            log_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )
            
            # 添加文件日志
            logger.add(
                log_dir / "system_{time:YYYY-MM-DD}.log",
                format=log_format,
                level="INFO",
                rotation="1 day",
                retention="30 days",
                compression="zip"
            )
            
            # 添加错误日志
            logger.add(
                log_dir / "error_{time:YYYY-MM-DD}.log",
                format=log_format,
                level="ERROR",
                rotation="1 day",
                retention="30 days",
                compression="zip"
            )
            
            logger.info("日志系统初始化完成")
            
        except Exception as e:
            print(f"日志系统设置失败: {e}")
    
    async def run_backtest(self, symbols: Optional[List[str]] = None, 
                          output_dir: str = "results") -> dict:
        """
        运行回测
        
        Args:
            symbols: 交易对列表，None则自动获取新币
            output_dir: 输出目录
            
        Returns:
            回测结果摘要
        """
        try:
            logger.info("开始运行回测")
            
            # 创建输出目录
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # 运行回测
            results = await self.backtest_engine.run_backtest(symbols)
            
            if not results:
                logger.warning("没有有效的回测结果")
                return {}
            
            # 保存回测结果
            saved_files = self.backtest_engine.save_results(output_dir)
            logger.info(f"回测结果已保存: {saved_files}")
            
            # 生成分析报告
            report_path = self.result_analyzer.generate_html_report(results, output_dir)
            csv_path = self.result_analyzer.export_results_to_csv(results, output_dir)
            
            # 获取整体性能统计
            performance = self.backtest_engine.get_overall_performance()
            
            # 创建结果摘要
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_symbols': performance.get('total_symbols', 0),
                'successful_backtests': performance.get('successful_backtests', 0),
                'failed_backtests': performance.get('failed_backtests', 0),
                'success_rate': performance.get('success_rate', 0),
                'avg_total_return': performance.get('avg_total_return', 0),
                'avg_max_drawdown': performance.get('avg_max_drawdown', 0),
                'avg_sharpe_ratio': performance.get('avg_sharpe_ratio', 0),
                'best_performer': performance.get('best_performer', ''),
                'worst_performer': performance.get('worst_performer', ''),
                'saved_files': saved_files,
                'report_path': report_path,
                'csv_path': csv_path
            }
            
            logger.info("回测完成")
            return summary
            
        except Exception as e:
            logger.error(f"回测运行失败: {e}")
            return {}
    
    async def run_data_collection(self, symbols: Optional[List[str]] = None, 
                                 days: int = 14) -> dict:
        """
        运行数据收集
        
        Args:
            symbols: 交易对列表，None则自动获取新币
            days: 获取天数
            
        Returns:
            数据收集结果
        """
        try:
            logger.info("开始数据收集")
            
            # 获取交易对列表
            if symbols is None:
                new_listings = await self.data_manager.get_new_listings(20)
                symbols = [listing['symbol'] for listing in new_listings]
            
            collected_data = {}
            
            for symbol in symbols:
                try:
                    logger.info(f"收集数据: {symbol}")
                    
                    # 检查流动性
                    liquidity_ok, liquidity_info = await self.data_manager.check_liquidity(symbol)
                    if not liquidity_ok:
                        logger.warning(f"{symbol} 流动性不足，跳过")
                        continue
                    
                    # 获取历史数据
                    data = await self.data_manager.fetch_historical_data(symbol, days)
                    if data is not None and not data.empty:
                        # 保存数据
                        filepath = self.data_manager.save_data(data, symbol, "historical")
                        collected_data[symbol] = {
                            'filepath': filepath,
                            'rows': len(data),
                            'liquidity_info': liquidity_info
                        }
                        logger.info(f"{symbol} 数据收集完成: {len(data)} 行")
                    else:
                        logger.warning(f"{symbol} 数据获取失败")
                
                except Exception as e:
                    logger.error(f"收集 {symbol} 数据失败: {e}")
                    continue
            
            logger.info(f"数据收集完成: {len(collected_data)} 个交易对")
            return collected_data
            
        except Exception as e:
            logger.error(f"数据收集失败: {e}")
            return {}
    
    def print_summary(self, summary: dict):
        """打印结果摘要"""
        try:
            print("\n" + "="*60)
            print("量化交易系统回测结果摘要")
            print("="*60)
            
            print(f"运行时间: {summary.get('timestamp', 'N/A')}")
            print(f"总交易对: {summary.get('total_symbols', 0)}")
            print(f"成功回测: {summary.get('successful_backtests', 0)}")
            print(f"失败回测: {summary.get('failed_backtests', 0)}")
            print(f"成功率: {format_percentage(summary.get('success_rate', 0))}")
            
            print(f"\n性能指标:")
            print(f"平均收益率: {format_percentage(summary.get('avg_total_return', 0))}")
            print(f"平均最大回撤: {format_percentage(summary.get('avg_max_drawdown', 0))}")
            print(f"平均夏普比率: {summary.get('avg_sharpe_ratio', 0):.2f}")
            
            print(f"\n最佳表现: {summary.get('best_performer', 'N/A')}")
            print(f"最差表现: {summary.get('worst_performer', 'N/A')}")
            
            if summary.get('report_path'):
                print(f"\n详细报告: {summary['report_path']}")
            if summary.get('csv_path'):
                print(f"CSV数据: {summary['csv_path']}")
            
            print("="*60)
            
        except Exception as e:
            logger.error(f"打印摘要失败: {e}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="币安新币动态Delta管理量化交易系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py --mode backtest                    # 运行回测
  python main.py --mode data                        # 收集数据
  python main.py --mode backtest --symbols BTC/USDT ETH/USDT  # 指定交易对
  python main.py --mode backtest --config custom.yaml         # 使用自定义配置
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['backtest', 'data', 'analysis'],
        default='backtest',
        help='运行模式 (默认: backtest)'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='指定交易对列表，如: BTC/USDT ETH/USDT'
    )
    
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='配置文件路径 (默认: config.yaml)'
    )
    
    parser.add_argument(
        '--output',
        default='results',
        help='输出目录 (默认: results)'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=14,
        help='数据获取天数 (默认: 14)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='并行工作进程数 (默认: 4)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出模式'
    )
    
    return parser.parse_args()


async def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 检查配置文件是否存在
        if not Path(args.config).exists():
            print(f"错误: 配置文件 {args.config} 不存在")
            sys.exit(1)
        
        # 创建系统实例
        system = QuantTradingSystem(args.config)
        
        # 根据模式运行
        if args.mode == 'backtest':
            # 更新并行工作进程数
            if 'backtest' in system.config:
                system.config['backtest']['parallel_workers'] = args.workers
            
            # 运行回测
            summary = await system.run_backtest(args.symbols, args.output)
            
            if summary:
                system.print_summary(summary)
            else:
                print("回测失败，请检查日志")
                sys.exit(1)
        
        elif args.mode == 'data':
            # 运行数据收集
            collected_data = await system.run_data_collection(args.symbols, args.days)
            
            print(f"\n数据收集完成:")
            print(f"成功收集: {len(collected_data)} 个交易对")
            
            for symbol, info in collected_data.items():
                print(f"  {symbol}: {info['rows']} 行数据")
        
        elif args.mode == 'analysis':
            # 分析现有结果
            results_dir = Path(args.output)
            if not results_dir.exists():
                print(f"错误: 结果目录 {args.output} 不存在")
                sys.exit(1)
            
            # 这里可以添加分析现有结果的逻辑
            print("分析模式功能待实现")
        
        else:
            print(f"未知模式: {args.mode}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n用户中断，程序退出")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序运行失败: {e}")
        print(f"程序运行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # 设置事件循环策略（Windows兼容性）
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # 运行主程序
    asyncio.run(main())
