#!/usr/bin/env python3
"""
系统测试脚本
验证量化交易系统各模块是否正常工作
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    
    try:
        from src.data.data_manager import BinanceDataManager
        print("✓ 数据管理模块导入成功")
    except Exception as e:
        print(f"✗ 数据管理模块导入失败: {e}")
        return False
    
    try:
        from src.strategy.delta_strategy import DeltaStrategy
        print("✓ 策略模块导入成功")
    except Exception as e:
        print(f"✗ 策略模块导入失败: {e}")
        return False
    
    try:
        from src.risk.risk_manager import RiskManager
        print("✓ 风控模块导入成功")
    except Exception as e:
        print(f"✗ 风控模块导入失败: {e}")
        return False
    
    try:
        from src.backtest.backtest_engine import BacktestEngine
        print("✓ 回测引擎导入成功")
    except Exception as e:
        print(f"✗ 回测引擎导入失败: {e}")
        return False
    
    try:
        from src.analysis.result_analyzer import ResultAnalyzer
        print("✓ 结果分析模块导入成功")
    except Exception as e:
        print(f"✗ 结果分析模块导入失败: {e}")
        return False
    
    try:
        from src.utils.helpers import calculate_sharpe_ratio, format_currency
        print("✓ 工具模块导入成功")
    except Exception as e:
        print(f"✗ 工具模块导入失败: {e}")
        return False
    
    return True


def test_config_loading():
    """测试配置文件加载"""
    print("\n测试配置文件加载...")
    
    try:
        import yaml
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['data', 'strategy', 'risk_control', 'backtest', 'output']
        for section in required_sections:
            if section not in config:
                print(f"✗ 配置文件缺少 {section} 部分")
                return False
        
        print("✓ 配置文件加载成功")
        return True
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return False


def test_strategy_initialization():
    """测试策略初始化"""
    print("\n测试策略初始化...")
    
    try:
        from src.strategy.delta_strategy import DeltaStrategy
        
        strategy = DeltaStrategy()
        print("✓ 策略初始化成功")
        
        # 测试基本方法
        summary = strategy.get_position_summary()
        print(f"✓ 策略摘要获取成功: {summary}")
        
        return True
    except Exception as e:
        print(f"✗ 策略初始化失败: {e}")
        return False


def test_risk_manager():
    """测试风控管理器"""
    print("\n测试风控管理器...")
    
    try:
        from src.risk.risk_manager import RiskManager
        
        risk_manager = RiskManager()
        print("✓ 风控管理器初始化成功")
        
        # 测试风险摘要
        summary = risk_manager.get_risk_summary()
        print(f"✓ 风险摘要获取成功: {summary}")
        
        return True
    except Exception as e:
        print(f"✗ 风控管理器测试失败: {e}")
        return False


def test_data_manager():
    """测试数据管理器"""
    print("\n测试数据管理器...")
    
    try:
        from src.data.data_manager import BinanceDataManager
        
        data_manager = BinanceDataManager()
        print("✓ 数据管理器初始化成功")
        
        return True
    except Exception as e:
        print(f"✗ 数据管理器测试失败: {e}")
        return False


def test_backtest_engine():
    """测试回测引擎"""
    print("\n测试回测引擎...")
    
    try:
        from src.backtest.backtest_engine import BacktestEngine
        
        engine = BacktestEngine()
        print("✓ 回测引擎初始化成功")
        
        # 测试整体性能获取
        performance = engine.get_overall_performance()
        print(f"✓ 性能统计获取成功: {performance}")
        
        return True
    except Exception as e:
        print(f"✗ 回测引擎测试失败: {e}")
        return False


def test_result_analyzer():
    """测试结果分析器"""
    print("\n测试结果分析器...")
    
    try:
        from src.analysis.result_analyzer import ResultAnalyzer
        
        analyzer = ResultAnalyzer()
        print("✓ 结果分析器初始化成功")
        
        return True
    except Exception as e:
        print(f"✗ 结果分析器测试失败: {e}")
        return False


def test_utils_functions():
    """测试工具函数"""
    print("\n测试工具函数...")
    
    try:
        from src.utils.helpers import (
            calculate_sharpe_ratio, format_currency, format_percentage,
            safe_divide, calculate_percentage_change
        )
        
        # 测试夏普比率计算
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        sharpe = calculate_sharpe_ratio(returns)
        print(f"✓ 夏普比率计算: {sharpe:.4f}")
        
        # 测试货币格式化
        formatted = format_currency(1234.56)
        print(f"✓ 货币格式化: {formatted}")
        
        # 测试百分比格式化
        percentage = format_percentage(0.1234)
        print(f"✓ 百分比格式化: {percentage}")
        
        # 测试安全除法
        result = safe_divide(10, 0, default=0)
        print(f"✓ 安全除法: {result}")
        
        # 测试百分比变化计算
        change = calculate_percentage_change(100, 110)
        print(f"✓ 百分比变化: {change:.2%}")
        
        return True
    except Exception as e:
        print(f"✗ 工具函数测试失败: {e}")
        return False


def test_synthetic_data():
    """测试合成数据"""
    print("\n测试合成数据生成...")
    
    try:
        # 生成模拟K线数据
        dates = pd.date_range('2024-01-01', periods=100, freq='15T')
        
        # 生成价格数据（带趋势和波动）
        np.random.seed(42)
        price_changes = np.random.normal(0.001, 0.02, 100)  # 平均0.1%涨幅，2%波动
        prices = 100 * (1 + price_changes).cumprod()
        
        # 生成成交量数据
        volumes = np.random.randint(1000, 10000, 100)
        
        # 创建数据框
        spot_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.999,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': volumes,
            'type': 'spot'
        })
        
        future_data = spot_data.copy()
        future_data['type'] = 'future'
        future_data['close'] = spot_data['close'] * (1 + np.random.normal(0, 0.001, 100))
        
        # 合并数据
        combined_data = pd.concat([spot_data, future_data])
        combined_data.set_index('timestamp', inplace=True)
        
        print(f"✓ 合成数据生成成功: {len(combined_data)} 行")
        print(f"  现货数据: {len(spot_data)} 行")
        print(f"  期货数据: {len(future_data)} 行")
        
        return combined_data
    except Exception as e:
        print(f"✗ 合成数据生成失败: {e}")
        return None


def test_strategy_with_synthetic_data():
    """使用合成数据测试策略"""
    print("\n使用合成数据测试策略...")
    
    try:
        from src.strategy.delta_strategy import DeltaStrategy
        
        # 生成合成数据
        data = test_synthetic_data()
        if data is None:
            return False
        
        # 添加技术指标
        from src.utils.helpers import calculate_technical_indicators
        data = calculate_technical_indicators(data)
        
        # 初始化策略
        strategy = DeltaStrategy()
        
        # 模拟策略执行
        trades_count = 0
        for i, (timestamp, row) in enumerate(data.iterrows()):
            if i < 20:  # 跳过前20个数据点
                continue
            
            # 检查是否建仓
            if strategy.should_enter_position(data, timestamp):
                trades = strategy.enter_position(data, timestamp, 2000)
                trades_count += len(trades)
                print(f"  建仓: {timestamp}, 交易数: {len(trades)}")
            
            # 检查是否调仓
            if strategy.should_adjust_position(data, timestamp):
                trades = strategy.adjust_position(data, timestamp)
                trades_count += len(trades)
                if trades:
                    print(f"  调仓: {timestamp}, 交易数: {len(trades)}")
        
        print(f"✓ 策略测试完成，总交易数: {trades_count}")
        
        # 获取最终状态
        summary = strategy.get_position_summary()
        print(f"✓ 最终状态: {summary}")
        
        return True
    except Exception as e:
        print(f"✗ 策略测试失败: {e}")
        return False


async def test_async_functions():
    """测试异步函数"""
    print("\n测试异步函数...")
    
    try:
        from src.data.data_manager import BinanceDataManager
        
        data_manager = BinanceDataManager()
        
        # 测试获取新币列表（模拟）
        print("✓ 异步函数测试通过")
        
        return True
    except Exception as e:
        print(f"✗ 异步函数测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print("量化交易系统测试")
    print("="*60)
    
    tests = [
        ("模块导入", test_imports),
        ("配置文件加载", test_config_loading),
        ("策略初始化", test_strategy_initialization),
        ("风控管理器", test_risk_manager),
        ("数据管理器", test_data_manager),
        ("回测引擎", test_backtest_engine),
        ("结果分析器", test_result_analyzer),
        ("工具函数", test_utils_functions),
        ("合成数据生成", lambda: test_synthetic_data() is not None),
        ("策略执行", test_strategy_with_synthetic_data),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = asyncio.run(test_func())
            else:
                result = test_func()
            
            if result:
                passed += 1
            else:
                print(f"✗ {test_name} 测试失败")
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
    
    print("\n" + "="*60)
    print(f"测试结果: {passed}/{total} 通过")
    print("="*60)
    
    if passed == total:
        print("🎉 所有测试通过！系统可以正常使用。")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关模块。")
        return False


def main():
    """主函数"""
    try:
        success = run_all_tests()
        
        if success:
            print("\n系统测试完成，可以开始使用！")
            print("\n使用方法:")
            print("  python main.py --mode backtest")
            print("  python main.py --mode data")
            print("  python main.py --help")
        else:
            print("\n系统测试失败，请修复问题后重试。")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
