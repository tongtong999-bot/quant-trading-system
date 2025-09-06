"""
工具模块 - 通用工具函数
提供常用的辅助函数、数据处理工具、时间处理工具等
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import yaml
from pathlib import Path
import hashlib
import time
from loguru import logger
import asyncio
import aiohttp
import requests
from functools import wraps
import pickle
import gzip


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff: 延迟时间倍数
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}, {current_delay}秒后重试")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"函数 {func.__name__} 在 {max_retries + 1} 次尝试后仍然失败")
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}, {current_delay}秒后重试")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"函数 {func.__name__} 在 {max_retries + 1} 次尝试后仍然失败")
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全除法，避免除零错误
    
    Args:
        numerator: 分子
        denominator: 分母
        default: 默认值
        
    Returns:
        除法结果或默认值
    """
    try:
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            return default
        result = numerator / denominator
        return result if not (np.isnan(result) or np.isinf(result)) else default
    except:
        return default


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    计算百分比变化
    
    Args:
        old_value: 旧值
        new_value: 新值
        
    Returns:
        百分比变化
    """
    if old_value == 0:
        return 0.0
    return (new_value - old_value) / old_value


def calculate_annualized_return(total_return: float, days: int) -> float:
    """
    计算年化收益率
    
    Args:
        total_return: 总收益率
        days: 天数
        
    Returns:
        年化收益率
    """
    if days <= 0:
        return 0.0
    return (1 + total_return) ** (365 / days) - 1


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    计算夏普比率
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率
        
    Returns:
        夏普比率
    """
    try:
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    except:
        return 0.0


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, datetime, datetime]:
    """
    计算最大回撤
    
    Args:
        equity_curve: 权益曲线
        
    Returns:
        (最大回撤, 回撤开始时间, 回撤结束时间)
    """
    try:
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_dd = drawdown.min()
        
        # 找到最大回撤的起止时间
        max_dd_idx = drawdown.idxmin()
        peak_before_dd = peak.loc[:max_dd_idx].idxmax()
        
        return max_dd, peak_before_dd, max_dd_idx
    except:
        return 0.0, None, None


def calculate_volatility(returns: pd.Series, annualized: bool = True) -> float:
    """
    计算波动率
    
    Args:
        returns: 收益率序列
        annualized: 是否年化
        
    Returns:
        波动率
    """
    try:
        if len(returns) == 0:
            return 0.0
        
        vol = returns.std()
        if annualized:
            vol *= np.sqrt(252)  # 假设日数据
        return vol
    except:
        return 0.0


def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    计算风险价值(VaR)
    
    Args:
        returns: 收益率序列
        confidence_level: 置信水平
        
    Returns:
        VaR值
    """
    try:
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, confidence_level * 100)
    except:
        return 0.0


def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    计算条件风险价值(CVaR)
    
    Args:
        returns: 收益率序列
        confidence_level: 置信水平
        
    Returns:
        CVaR值
    """
    try:
        if len(returns) == 0:
            return 0.0
        
        var = calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    except:
        return 0.0


def normalize_data(data: pd.Series, method: str = 'minmax') -> pd.Series:
    """
    数据标准化
    
    Args:
        data: 数据序列
        method: 标准化方法 ('minmax', 'zscore', 'robust')
        
    Returns:
        标准化后的数据
    """
    try:
        if method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'zscore':
            return (data - data.mean()) / data.std()
        elif method == 'robust':
            median = data.median()
            mad = np.median(np.abs(data - median))
            return (data - median) / (1.4826 * mad)
        else:
            return data
    except:
        return data


def detect_outliers(data: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
    """
    异常值检测
    
    Args:
        data: 数据序列
        method: 检测方法 ('iqr', 'zscore', 'modified_zscore')
        threshold: 阈值
        
    Returns:
        布尔序列，True表示异常值
    """
    try:
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (data < lower_bound) | (data > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            return z_scores > threshold
        
        elif method == 'modified_zscore':
            median = data.median()
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            return np.abs(modified_z_scores) > threshold
        
        else:
            return pd.Series([False] * len(data), index=data.index)
    except:
        return pd.Series([False] * len(data), index=data.index)


def smooth_data(data: pd.Series, window: int = 5, method: str = 'moving_average') -> pd.Series:
    """
    数据平滑
    
    Args:
        data: 数据序列
        window: 窗口大小
        method: 平滑方法 ('moving_average', 'exponential', 'savgol')
        
    Returns:
        平滑后的数据
    """
    try:
        if method == 'moving_average':
            return data.rolling(window=window).mean()
        elif method == 'exponential':
            return data.ewm(span=window).mean()
        elif method == 'savgol':
            from scipy.signal import savgol_filter
            return pd.Series(savgol_filter(data, window, 3), index=data.index)
        else:
            return data
    except:
        return data


def resample_data(data: pd.DataFrame, target_freq: str, method: str = 'last') -> pd.DataFrame:
    """
    数据重采样
    
    Args:
        data: 数据框
        target_freq: 目标频率
        method: 重采样方法
        
    Returns:
        重采样后的数据
    """
    try:
        if method == 'last':
            return data.resample(target_freq).last()
        elif method == 'first':
            return data.resample(target_freq).first()
        elif method == 'mean':
            return data.resample(target_freq).mean()
        elif method == 'ohlc':
            return data.resample(target_freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        else:
            return data
    except:
        return data


def create_time_features(data: pd.DataFrame, datetime_col: str = 'timestamp') -> pd.DataFrame:
    """
    创建时间特征
    
    Args:
        data: 数据框
        datetime_col: 时间列名
        
    Returns:
        添加时间特征后的数据框
    """
    try:
        df = data.copy()
        if datetime_col in df.columns:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            
            # 基础时间特征
            df['year'] = df[datetime_col].dt.year
            df['month'] = df[datetime_col].dt.month
            df['day'] = df[datetime_col].dt.day
            df['hour'] = df[datetime_col].dt.hour
            df['minute'] = df[datetime_col].dt.minute
            df['dayofweek'] = df[datetime_col].dt.dayofweek
            df['dayofyear'] = df[datetime_col].dt.dayofyear
            df['weekofyear'] = df[datetime_col].dt.isocalendar().week
            
            # 周期性特征
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    except Exception as e:
        logger.error(f"创建时间特征失败: {e}")
        return data


def calculate_technical_indicators(data: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    计算技术指标
    
    Args:
        data: 数据框
        price_col: 价格列名
        
    Returns:
        添加技术指标后的数据框
    """
    try:
        df = data.copy()
        
        # 移动平均线
        df['sma_5'] = df[price_col].rolling(window=5).mean()
        df['sma_10'] = df[price_col].rolling(window=10).mean()
        df['sma_20'] = df[price_col].rolling(window=20).mean()
        df['sma_50'] = df[price_col].rolling(window=50).mean()
        
        # 指数移动平均线
        df['ema_5'] = df[price_col].ewm(span=5).mean()
        df['ema_10'] = df[price_col].ewm(span=10).mean()
        df['ema_20'] = df[price_col].ewm(span=20).mean()
        df['ema_50'] = df[price_col].ewm(span=50).mean()
        
        # 布林带
        sma_20 = df[price_col].rolling(window=20).mean()
        std_20 = df[price_col].rolling(window=20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_middle'] = sma_20
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df[price_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df[price_col].ewm(span=12).mean()
        ema_26 = df[price_col].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 成交量指标
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['obv'] = (df['volume'] * np.where(df[price_col].diff() > 0, 1, -1)).cumsum()
        
        return df
    except Exception as e:
        logger.error(f"计算技术指标失败: {e}")
        return data


def save_data_compressed(data: Any, filepath: str) -> bool:
    """
    压缩保存数据
    
    Args:
        data: 要保存的数据
        filepath: 文件路径
        
    Returns:
        是否保存成功
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"数据已压缩保存: {filepath}")
        return True
    except Exception as e:
        logger.error(f"压缩保存数据失败: {e}")
        return False


def load_data_compressed(filepath: str) -> Any:
    """
    加载压缩数据
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载的数据
    """
    try:
        with gzip.open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"数据已加载: {filepath}")
        return data
    except Exception as e:
        logger.error(f"加载压缩数据失败: {e}")
        return None


def generate_hash(data: str) -> str:
    """
    生成数据哈希值
    
    Args:
        data: 数据字符串
        
    Returns:
        哈希值
    """
    try:
        return hashlib.md5(data.encode()).hexdigest()
    except:
        return ""


def format_currency(amount: float, currency: str = "USDT", decimals: int = 2) -> str:
    """
    格式化货币显示
    
    Args:
        amount: 金额
        currency: 货币符号
        decimals: 小数位数
        
    Returns:
        格式化后的字符串
    """
    try:
        if abs(amount) >= 1e9:
            return f"{amount/1e9:.{decimals}f}B {currency}"
        elif abs(amount) >= 1e6:
            return f"{amount/1e6:.{decimals}f}M {currency}"
        elif abs(amount) >= 1e3:
            return f"{amount/1e3:.{decimals}f}K {currency}"
        else:
            return f"{amount:.{decimals}f} {currency}"
    except:
        return f"{amount} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    格式化百分比显示
    
    Args:
        value: 数值
        decimals: 小数位数
        
    Returns:
        格式化后的字符串
    """
    try:
        return f"{value*100:.{decimals}f}%"
    except:
        return f"{value}%"


def validate_config(config: Dict, required_keys: List[str]) -> bool:
    """
    验证配置完整性
    
    Args:
        config: 配置字典
        required_keys: 必需的键列表
        
    Returns:
        是否验证通过
    """
    try:
        for key in required_keys:
            if key not in config:
                logger.error(f"配置缺少必需的键: {key}")
                return False
        return True
    except:
        return False


def merge_dicts(*dicts: Dict) -> Dict:
    """
    合并多个字典
    
    Args:
        *dicts: 要合并的字典
        
    Returns:
        合并后的字典
    """
    try:
        result = {}
        for d in dicts:
            if isinstance(d, dict):
                result.update(d)
        return result
    except:
        return {}


def deep_merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    深度合并字典
    
    Args:
        dict1: 字典1
        dict2: 字典2
        
    Returns:
        深度合并后的字典
    """
    try:
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
    except:
        return dict1


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    将列表分块
    
    Args:
        lst: 要分块的列表
        chunk_size: 块大小
        
    Returns:
        分块后的列表
    """
    try:
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    except:
        return [lst]


def flatten_list(nested_list: List) -> List:
    """
    展平嵌套列表
    
    Args:
        nested_list: 嵌套列表
        
    Returns:
        展平后的列表
    """
    try:
        result = []
        for item in nested_list:
            if isinstance(item, list):
                result.extend(flatten_list(item))
            else:
                result.append(item)
        return result
    except:
        return nested_list


def remove_duplicates_preserve_order(lst: List) -> List:
    """
    去重并保持顺序
    
    Args:
        lst: 要去重的列表
        
    Returns:
        去重后的列表
    """
    try:
        seen = set()
        result = []
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    except:
        return lst


def get_file_size(filepath: str) -> int:
    """
    获取文件大小
    
    Args:
        filepath: 文件路径
        
    Returns:
        文件大小（字节）
    """
    try:
        return Path(filepath).stat().st_size
    except:
        return 0


def get_directory_size(directory: str) -> int:
    """
    获取目录大小
    
    Args:
        directory: 目录路径
        
    Returns:
        目录大小（字节）
    """
    try:
        total_size = 0
        for file_path in Path(directory).rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    except:
        return 0


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小显示
    
    Args:
        size_bytes: 字节数
        
    Returns:
        格式化后的字符串
    """
    try:
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.2f} {size_names[i]}"
    except:
        return f"{size_bytes} B"


# 使用示例
def main():
    """工具函数使用示例"""
    # 创建示例数据
    dates = pd.date_range('2024-01-01', periods=100, freq='15T')
    data = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # 计算技术指标
    data_with_indicators = calculate_technical_indicators(data)
    print("技术指标计算完成")
    
    # 创建时间特征
    data_with_features = create_time_features(data_with_indicators)
    print("时间特征创建完成")
    
    # 计算性能指标
    returns = data_with_indicators['close'].pct_change().dropna()
    sharpe = calculate_sharpe_ratio(returns)
    volatility = calculate_volatility(returns)
    print(f"夏普比率: {sharpe:.2f}")
    print(f"波动率: {volatility:.2f}")


if __name__ == "__main__":
    main()
