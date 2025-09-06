# 快速开始指南

## 系统概述

这是一个基于趋势判断的主动对冲策略量化交易系统，专门针对币安新币上线进行回测验证。系统通过建立Delta中性对冲仓位，然后根据市场趋势状态进行主动Delta管理。

## 快速安装

### 1. 环境准备
```bash
# 确保Python 3.8+已安装
python3 --version

# 克隆或下载项目
cd quant_trading_system
```

### 2. 安装依赖
```bash
# 安装所有依赖包
pip3 install -r requirements.txt
```

### 3. 验证安装
```bash
# 运行系统测试
python3 test_system.py
```

如果看到 "🎉 所有测试通过！" 说明安装成功。

## 快速使用

### 1. 基本回测
```bash
# 运行默认回测（自动获取新币数据）
python3 main.py --mode backtest
```

### 2. 指定交易对回测
```bash
# 回测特定交易对
python3 main.py --mode backtest --symbols BTC/USDT ETH/USDT
```

### 3. 数据收集
```bash
# 收集新币数据
python3 main.py --mode data
```

### 4. 查看帮助
```bash
# 查看所有可用选项
python3 main.py --help
```

## 配置调整

编辑 `config.yaml` 文件来调整策略参数：

### 关键参数说明

#### 策略参数
```yaml
strategy:
  initial_capital: 2000    # 初始资金（USDT）
  spot_ratio: 0.5          # 现货比例
  leverage: 4              # 杠杆倍数
  
  # 趋势判断
  volume_threshold: 1.3    # 成交量放大倍数
  adx_threshold: 25        # ADX趋势强度阈值
  
  # 调仓触发点
  reduce_short_levels: [0.20, 0.35, 0.50]  # 价格涨幅触发点
  reduce_short_ratios: [0.30, 0.30, 0.40]  # 对应减仓比例
```

#### 风控参数
```yaml
risk_control:
  max_exposure_ratio: 0.30    # 最大净敞口比例
  max_loss_ratio: 0.15        # 最大亏损比例
  time_stop_hours: 12         # 时间止损（小时）
  price_stop_ratio: 0.10      # 价格止损比例
```

## 输出结果

### 1. 控制台输出
运行回测后，控制台会显示：
- 整体性能统计
- 成功率
- 平均收益率
- 最佳/最差表现者

### 2. 生成文件
在 `results/` 目录下会生成：
- `backtest_report_*.html` - 完整HTML报告
- `backtest_results_*.pkl` - 完整回测数据
- `backtest_summary_*.csv` - 汇总统计
- `trades_detail_*.csv` - 详细交易记录

### 3. 图表分析
- 权益曲线图
- 性能对比图
- 风险分析图

## 常见使用场景

### 场景1：新币策略验证
```bash
# 回测最近20个新币
python3 main.py --mode backtest
```

### 场景2：特定币种分析
```bash
# 分析特定币种
python3 main.py --mode backtest --symbols BTC/USDT ETH/USDT ADA/USDT
```

### 场景3：参数优化测试
```bash
# 修改config.yaml中的参数后重新测试
python3 main.py --mode backtest --config custom_config.yaml
```

### 场景4：数据收集
```bash
# 收集30天数据
python3 main.py --mode data --days 30
```

## 性能调优

### 1. 并行处理
```bash
# 使用8个并行进程
python3 main.py --mode backtest --workers 8
```

### 2. 内存优化
- 减少并行工作进程数
- 缩短回测时间范围
- 增加系统内存

### 3. 网络优化
- 确保稳定的网络连接
- 使用VPN（如需要）

## 故障排除

### 问题1：依赖安装失败
```bash
# 升级pip
python3 -m pip install --upgrade pip

# 重新安装依赖
pip3 install -r requirements.txt
```

### 问题2：数据获取失败
- 检查网络连接
- 确认币安API可访问
- 检查交易对是否有效

### 问题3：回测结果为空
- 检查数据质量
- 确认流动性要求
- 调整策略参数

### 问题4：内存不足
```bash
# 减少并行进程
python3 main.py --mode backtest --workers 2

# 减少回测天数
python3 main.py --mode data --days 7
```

## 日志查看

系统日志保存在 `logs/` 目录：
- `system_YYYY-MM-DD.log` - 系统运行日志
- `error_YYYY-MM-DD.log` - 错误日志

## 下一步

1. **参数调优**: 根据回测结果调整策略参数
2. **风险控制**: 根据实际需求调整风控参数
3. **实盘准备**: 在模拟环境中验证策略
4. **监控系统**: 建立实时监控和报警机制

## 获取帮助

- 查看完整文档：`README.md`
- 运行测试：`python3 test_system.py`
- 查看帮助：`python3 main.py --help`

## 注意事项

⚠️ **重要提醒**：
- 本系统仅供学习和研究使用
- 不构成投资建议
- 实盘交易风险自担
- 建议先在模拟环境中充分测试

---

**开始您的量化交易之旅！** 🚀
