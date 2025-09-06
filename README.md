# 币安新币动态Delta管理量化交易系统

## 系统概述

这是一个基于趋势判断能力的主动对冲策略量化交易系统，专门针对币安新币上线进行回测验证。系统通过建立Delta中性对冲仓位，然后根据市场趋势状态进行主动Delta管理，将高风险的方向性投机转化为风险可控的主动管理策略。

## 核心策略

### 策略逻辑
1. **建仓阶段**: 在新币现货与永续合约均上线且流动性充足时，以等名义价值建立Delta中性对冲仓位
2. **趋势判断**: 根据技术指标确认市场趋势状态（上升/下跌/震荡）
3. **动态调仓**: 根据趋势变化主动调整仓位，释放盈利潜力或控制风险
4. **风险控制**: 多层风控机制确保策略安全执行

### 关键特性
- 基于15分钟K线的趋势识别
- 分批减仓机制（20%/35%/50%触发点）
- 强势趋势转为纯多头策略
- 下跌趋势卖出现货保留空单
- 震荡市场维持中性对冲

## 系统架构

```
quant_trading_system/
├── src/                    # 源代码目录
│   ├── data/              # 数据管理模块
│   │   └── data_manager.py
│   ├── strategy/          # 策略模块
│   │   └── delta_strategy.py
│   ├── risk/              # 风控模块
│   │   └── risk_manager.py
│   ├── backtest/          # 回测引擎
│   │   └── backtest_engine.py
│   ├── analysis/          # 结果分析
│   │   └── result_analyzer.py
│   └── utils/             # 工具模块
│       └── helpers.py
├── config.yaml            # 配置文件
├── requirements.txt       # 依赖包
├── main.py               # 主程序入口
└── README.md             # 说明文档
```

## 安装和使用

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd quant_trading_system

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置设置

编辑 `config.yaml` 文件，调整策略参数：

```yaml
# 策略参数
strategy:
  initial_capital: 2000    # 初始资金
  spot_ratio: 0.5          # 现货比例
  leverage: 4              # 杠杆倍数
  
  # 趋势判断参数
  volume_threshold: 1.3    # 成交量放大倍数
  adx_threshold: 25        # ADX阈值
  
  # 调仓触发点
  reduce_short_levels: [0.20, 0.35, 0.50]  # 相对建仓价涨幅
  reduce_short_ratios: [0.30, 0.30, 0.40]  # 对应平仓比例

# 风控参数
risk_control:
  max_exposure_ratio: 0.30    # 最大敞口比例
  max_loss_ratio: 0.15        # 最大亏损比例
  time_stop_hours: 12         # 时间止损
```

### 3. 运行回测

```bash
# 基本回测（自动获取新币）
python main.py --mode backtest

# 指定交易对回测
python main.py --mode backtest --symbols BTC/USDT ETH/USDT

# 使用自定义配置
python main.py --mode backtest --config custom_config.yaml

# 调整并行工作进程数
python main.py --mode backtest --workers 8

# 详细输出模式
python main.py --mode backtest --verbose
```

### 4. 数据收集

```bash
# 收集新币数据
python main.py --mode data

# 指定交易对收集数据
python main.py --mode data --symbols BTC/USDT ETH/USDT

# 指定收集天数
python main.py --mode data --days 30
```

## 输出结果

### 1. 回测结果文件
- `backtest_results_YYYYMMDD_HHMMSS.pkl`: 完整回测结果
- `backtest_summary_YYYYMMDD_HHMMSS.csv`: 汇总统计
- `trades_detail_YYYYMMDD_HHMMSS.csv`: 详细交易记录

### 2. 分析报告
- `backtest_report_YYYYMMDD_HHMMSS.html`: 完整HTML报告
- `equity_curves_YYYYMMDD_HHMMSS.html`: 权益曲线图
- `performance_comparison_YYYYMMDD_HHMMSS.html`: 性能对比图
- `risk_analysis_YYYYMMDD_HHMMSS.html`: 风险分析图

### 3. 性能指标
- 总收益率、年化收益率
- 最大回撤、夏普比率、卡玛比率
- 胜率、盈亏比、交易次数
- VaR、CVaR风险指标

## 策略参数说明

### 趋势判断参数
- `volume_threshold`: 成交量放大倍数阈值
- `adx_threshold`: ADX趋势强度阈值
- `trend_reversal_adx`: 趋势反转ADX阈值

### 调仓参数
- `reduce_short_levels`: 减仓触发价格涨幅
- `reduce_short_ratios`: 对应减仓比例
- `strong_trend_threshold`: 强势趋势阈值
- `callback_threshold`: 回调接回阈值

### 风控参数
- `max_exposure_ratio`: 最大净敞口比例
- `max_loss_ratio`: 最大亏损比例
- `time_stop_hours`: 时间止损小时数
- `price_stop_ratio`: 价格止损比例

## 注意事项

1. **数据依赖**: 系统依赖币安API获取数据，需要稳定的网络连接
2. **流动性要求**: 只有满足流动性要求的交易对才会被纳入回测
3. **参数调优**: 建议根据历史数据调整策略参数以获得最佳表现
4. **风险控制**: 严格执行风控规则，避免过度风险暴露
5. **回测限制**: 回测结果仅供参考，实盘交易需谨慎

## 系统要求

- Python 3.8+
- 内存: 建议4GB以上
- 存储: 建议10GB以上可用空间
- 网络: 稳定的互联网连接

## 故障排除

### 常见问题

1. **数据获取失败**
   - 检查网络连接
   - 确认币安API可访问
   - 检查交易对是否有效

2. **回测结果为空**
   - 检查数据质量
   - 确认流动性要求
   - 调整策略参数

3. **内存不足**
   - 减少并行工作进程数
   - 减少回测时间范围
   - 增加系统内存

### 日志查看

系统日志保存在 `logs/` 目录：
- `system_YYYY-MM-DD.log`: 系统运行日志
- `error_YYYY-MM-DD.log`: 错误日志

## 贡献指南

欢迎提交Issue和Pull Request来改进系统。

## 免责声明

本系统仅供学习和研究使用，不构成投资建议。使用本系统进行实盘交易的风险由用户自行承担。
