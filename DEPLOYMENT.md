# 项目部署指南

## 系统要求

### 硬件要求
- **CPU**: 2核心以上（推荐4核心）
- **内存**: 4GB以上（推荐8GB）
- **存储**: 10GB以上可用空间
- **网络**: 稳定的互联网连接

### 软件要求
- **操作系统**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Python**: 3.8或更高版本
- **包管理器**: pip 20.0+

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/quant-trading-system.git
cd quant-trading-system
```

### 2. 创建虚拟环境

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置系统

```bash
# 复制配置文件
cp config.yaml config_local.yaml

# 编辑配置文件
nano config_local.yaml
```

### 5. 运行测试

```bash
python test_system.py
```

## 配置说明

### 基础配置

编辑 `config.yaml` 文件：

```yaml
# 数据配置
data:
  exchange: "binance"
  timeframe: "15m"
  lookback_days: 14
  min_liquidity: 50000
  max_spread_bps: 30

# 策略参数
strategy:
  initial_capital: 2000
  spot_ratio: 0.5
  leverage: 4
  volume_threshold: 1.3
  adx_threshold: 25

# 风控参数
risk_control:
  max_exposure_ratio: 0.30
  max_loss_ratio: 0.15
  time_stop_hours: 12
  price_stop_ratio: 0.10
```

### API配置

如果需要使用币安API，请配置：

```yaml
data:
  api_key: "your_api_key"
  secret_key: "your_secret_key"
  sandbox: false
```

## 运行方式

### 1. 基本回测

```bash
python main.py --mode backtest
```

### 2. 指定交易对

```bash
python main.py --mode backtest --symbols BTC/USDT ETH/USDT
```

### 3. 数据收集

```bash
python main.py --mode data --days 30
```

### 4. 演示模式

```bash
python demo_mode.py
```

## Docker部署

### 1. 创建Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py", "--mode", "backtest"]
```

### 2. 构建镜像

```bash
docker build -t quant-trading-system .
```

### 3. 运行容器

```bash
docker run -v $(pwd)/results:/app/results quant-trading-system
```

## 云部署

### AWS EC2

1. 启动EC2实例（推荐t3.medium）
2. 安装Python 3.8+
3. 克隆项目
4. 安装依赖
5. 运行系统

### Google Cloud Platform

1. 创建Compute Engine实例
2. 安装必要软件
3. 部署项目
4. 配置防火墙规则

### Azure

1. 创建虚拟机
2. 安装Python环境
3. 部署应用
4. 配置网络安全组

## 监控和维护

### 1. 日志监控

```bash
# 查看系统日志
tail -f logs/system_$(date +%Y-%m-%d).log

# 查看错误日志
tail -f logs/error_$(date +%Y-%m-%d).log
```

### 2. 性能监控

```bash
# 检查系统资源使用
htop
# 或
top

# 检查磁盘使用
df -h
```

### 3. 定期维护

```bash
# 清理旧日志
find logs/ -name "*.log" -mtime +30 -delete

# 清理结果文件
find results/ -name "*.pkl" -mtime +7 -delete
```

## 故障排除

### 常见问题

1. **依赖安装失败**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **网络连接问题**
   - 检查防火墙设置
   - 使用代理（如需要）
   - 检查DNS设置

3. **权限问题**
   ```bash
   chmod +x main.py
   chmod 755 results/
   ```

4. **内存不足**
   - 减少并行进程数
   - 增加系统内存
   - 优化数据量

### 日志分析

```bash
# 搜索错误信息
grep "ERROR" logs/system_*.log

# 统计交易次数
grep "交易数" logs/system_*.log | wc -l
```

## 安全考虑

### 1. API密钥安全

- 不要将API密钥提交到版本控制
- 使用环境变量存储敏感信息
- 定期轮换API密钥

### 2. 网络安全

- 使用HTTPS连接
- 配置防火墙规则
- 定期更新系统

### 3. 数据安全

- 定期备份重要数据
- 加密敏感信息
- 限制访问权限

## 性能优化

### 1. 系统优化

- 使用SSD存储
- 增加内存
- 优化网络连接

### 2. 代码优化

- 使用并行处理
- 优化算法
- 减少内存使用

### 3. 配置优化

- 调整并行进程数
- 优化缓存设置
- 调整超时参数

## 扩展功能

### 1. 添加新策略

1. 继承 `DeltaStrategy` 类
2. 重写相关方法
3. 更新配置文件
4. 测试新策略

### 2. 添加新数据源

1. 修改 `data_manager.py`
2. 添加新的交易所支持
3. 更新配置选项
4. 测试数据获取

### 3. 添加新指标

1. 在 `helpers.py` 中添加计算函数
2. 更新策略逻辑
3. 测试新指标
4. 更新文档

## 联系支持

如有问题，请：

1. 查看日志文件
2. 运行系统测试
3. 查看文档
4. 提交Issue

---

**部署完成！** 🚀
