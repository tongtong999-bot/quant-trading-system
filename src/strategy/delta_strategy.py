"""
Delta管理策略核心逻辑
实现基于趋势判断的主动对冲策略
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from loguru import logger
import yaml


class PositionType(Enum):
    """仓位类型"""
    NEUTRAL = "neutral"      # 中性对冲
    LONG_BIAS = "long_bias"  # 偏多
    SHORT_BIAS = "short_bias"  # 偏空
    PURE_LONG = "pure_long"  # 纯多头
    PURE_SHORT = "pure_short"  # 纯空头


class TrendState(Enum):
    """趋势状态"""
    UPTREND = "uptrend"      # 上升趋势
    DOWNTREND = "downtrend"  # 下跌趋势
    SIDEWAYS = "sideways"    # 震荡
    UNCERTAIN = "uncertain"  # 不确定


@dataclass
class Position:
    """仓位信息"""
    spot_amount: float = 0.0      # 现货数量
    short_amount: float = 0.0     # 空单数量
    entry_price_spot: float = 0.0  # 现货入场价
    entry_price_short: float = 0.0  # 空单入场价
    position_type: PositionType = PositionType.NEUTRAL
    entry_time: datetime = None
    last_adjust_time: datetime = None


@dataclass
class Trade:
    """交易记录"""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: float
    trade_type: str  # 'spot' or 'future'
    reason: str
    position_after: Position


class DeltaStrategy:
    """Delta管理策略"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化策略"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.strategy_config = self.config['strategy']
        self.risk_config = self.config['risk_control']
        
        # 策略状态
        self.position = Position()
        self.trades: List[Trade] = []
        self.current_trend = TrendState.UNCERTAIN
        self.last_trend_check = None
        
        # 性能统计
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        
    def check_trend_state(self, data: pd.DataFrame, current_time: datetime) -> TrendState:
        """
        检查当前趋势状态
        
        Args:
            data: K线数据
            current_time: 当前时间
            
        Returns:
            趋势状态
        """
        try:
            # 获取当前数据点
            current_data = data[data.index <= current_time].tail(1)
            if current_data.empty:
                return TrendState.UNCERTAIN
            
            spot_data = current_data[current_data['type'] == 'spot']
            if spot_data.empty:
                return TrendState.UNCERTAIN
            
            row = spot_data.iloc[0]
            
            # 趋势判断条件
            ema_20 = row.get('ema_20', 0)
            ema_50 = row.get('ema_50', 0)
            close_price = row['close']
            volume_ratio = row.get('volume_ratio', 1.0)
            adx = row.get('adx', 0)
            
            # 上升趋势确认
            uptrend_conditions = [
                close_price > ema_20 > ema_50,
                volume_ratio >= self.strategy_config['volume_threshold'],
                adx >= self.strategy_config['adx_threshold']
            ]
            
            # 下跌趋势确认
            downtrend_conditions = [
                close_price < ema_20 < ema_50,
                volume_ratio >= self.strategy_config['volume_threshold'],
                adx >= self.strategy_config['adx_threshold']
            ]
            
            # 震荡判断
            sideways_conditions = [
                adx < self.strategy_config['trend_reversal_adx'],
                abs(close_price - ema_20) / ema_20 < 0.05  # 价格在EMA20附近5%范围内
            ]
            
            if sum(uptrend_conditions) >= 2:
                return TrendState.UPTREND
            elif sum(downtrend_conditions) >= 2:
                return TrendState.DOWNTREND
            elif sum(sideways_conditions) >= 1:
                return TrendState.SIDEWAYS
            else:
                return TrendState.UNCERTAIN
                
        except Exception as e:
            logger.error(f"趋势判断失败: {e}")
            return TrendState.UNCERTAIN
    
    def should_enter_position(self, data: pd.DataFrame, current_time: datetime) -> bool:
        """
        判断是否应该建仓
        
        Args:
            data: K线数据
            current_time: 当前时间
            
        Returns:
            是否建仓
        """
        try:
            # 检查是否已有仓位
            if self.position.spot_amount > 0 or self.position.short_amount > 0:
                return False
            
            # 检查流动性
            current_data = data[data.index <= current_time].tail(1)
            if current_data.empty:
                return False
            
            # 检查现货和期货数据都存在
            spot_data = current_data[current_data['type'] == 'spot']
            future_data = current_data[current_data['type'] == 'future']
            
            if spot_data.empty or future_data.empty:
                return False
            
            # 检查价格合理性
            spot_price = spot_data.iloc[0]['close']
            future_price = future_data.iloc[0]['close']
            
            price_diff = abs(spot_price - future_price) / spot_price
            if price_diff > 0.01:  # 价差超过1%
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"建仓判断失败: {e}")
            return False
    
    def enter_position(self, data: pd.DataFrame, current_time: datetime, 
                      capital: float) -> List[Trade]:
        """
        建立初始对冲仓位
        
        Args:
            data: K线数据
            current_time: 当前时间
            capital: 可用资金
            
        Returns:
            交易记录列表
        """
        trades = []
        
        try:
            current_data = data[data.index <= current_time].tail(1)
            spot_data = current_data[current_data['type'] == 'spot']
            future_data = current_data[current_data['type'] == 'future']
            
            if spot_data.empty or future_data.empty:
                return trades
            
            spot_price = spot_data.iloc[0]['close']
            future_price = future_data.iloc[0]['close']
            
            # 计算仓位大小
            spot_ratio = self.strategy_config['spot_ratio']
            spot_capital = capital * spot_ratio
            short_capital = capital * (1 - spot_ratio)
            
            # 考虑杠杆
            leverage = self.strategy_config['leverage']
            short_capital *= leverage
            
            spot_amount = spot_capital / spot_price
            short_amount = short_capital / future_price
            
            # 更新仓位
            self.position.spot_amount = spot_amount
            self.position.short_amount = short_amount
            self.position.entry_price_spot = spot_price
            self.position.entry_price_short = future_price
            self.position.position_type = PositionType.NEUTRAL
            self.position.entry_time = current_time
            self.position.last_adjust_time = current_time
            
            # 记录交易
            spot_trade = Trade(
                timestamp=current_time,
                symbol=spot_data.iloc[0].get('symbol', 'UNKNOWN'),
                side='buy',
                amount=spot_amount,
                price=spot_price,
                trade_type='spot',
                reason='initial_hedge_spot',
                position_after=self.position
            )
            
            short_trade = Trade(
                timestamp=current_time,
                symbol=future_data.iloc[0].get('symbol', 'UNKNOWN'),
                side='sell',
                amount=short_amount,
                price=future_price,
                trade_type='future',
                reason='initial_hedge_short',
                position_after=self.position
            )
            
            trades.extend([spot_trade, short_trade])
            self.trades.extend(trades)
            
            logger.info(f"建立初始对冲仓位: 现货 {spot_amount:.4f} @ {spot_price:.4f}, "
                       f"空单 {short_amount:.4f} @ {future_price:.4f}")
            
        except Exception as e:
            logger.error(f"建仓失败: {e}")
        
        return trades
    
    def should_adjust_position(self, data: pd.DataFrame, current_time: datetime) -> bool:
        """
        判断是否应该调整仓位
        
        Args:
            data: K线数据
            current_time: 当前时间
            
        Returns:
            是否调整仓位
        """
        try:
            # 检查是否有仓位
            if self.position.spot_amount == 0 and self.position.short_amount == 0:
                return False
            
            # 检查调整间隔
            if self.position.last_adjust_time:
                time_diff = current_time - self.position.last_adjust_time
                if time_diff.total_seconds() < 300:  # 5分钟内不重复调整
                    return False
            
            # 检查趋势变化
            current_trend = self.check_trend_state(data, current_time)
            if current_trend != self.current_trend:
                self.current_trend = current_trend
                self.last_trend_check = current_time
                return True
            
            # 检查价格触发点
            current_data = data[data.index <= current_time].tail(1)
            if current_data.empty:
                return False
            
            spot_data = current_data[current_data['type'] == 'spot']
            if spot_data.empty:
                return False
            
            current_price = spot_data.iloc[0]['close']
            entry_price = self.position.entry_price_spot
            
            # 计算涨幅
            price_change = (current_price - entry_price) / entry_price
            
            # 检查是否触发减仓点
            reduce_levels = self.strategy_config['reduce_short_levels']
            for level in reduce_levels:
                if price_change >= level:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"调仓判断失败: {e}")
            return False
    
    def adjust_position(self, data: pd.DataFrame, current_time: datetime) -> List[Trade]:
        """
        调整仓位
        
        Args:
            data: K线数据
            current_time: 当前时间
            
        Returns:
            交易记录列表
        """
        trades = []
        
        try:
            current_data = data[data.index <= current_time].tail(1)
            spot_data = current_data[current_data['type'] == 'spot']
            future_data = current_data[current_data['type'] == 'future']
            
            if spot_data.empty or future_data.empty:
                return trades
            
            current_price = spot_data.iloc[0]['close']
            entry_price = self.position.entry_price_spot
            price_change = (current_price - entry_price) / entry_price
            
            # 根据趋势和价格变化调整仓位
            if self.current_trend == TrendState.UPTREND:
                trades.extend(self._handle_uptrend_adjustment(
                    data, current_time, price_change, current_price
                ))
            elif self.current_trend == TrendState.DOWNTREND:
                trades.extend(self._handle_downtrend_adjustment(
                    data, current_time, current_price
                ))
            elif self.current_trend == TrendState.SIDEWAYS:
                trades.extend(self._handle_sideways_adjustment(
                    data, current_time, current_price
                ))
            
            # 更新调整时间
            self.position.last_adjust_time = current_time
            
        except Exception as e:
            logger.error(f"调仓失败: {e}")
        
        return trades
    
    def _handle_uptrend_adjustment(self, data: pd.DataFrame, current_time: datetime,
                                 price_change: float, current_price: float) -> List[Trade]:
        """处理上升趋势的仓位调整"""
        trades = []
        
        try:
            # 检查是否触发减仓点
            reduce_levels = self.strategy_config['reduce_short_levels']
            reduce_ratios = self.strategy_config['reduce_short_ratios']
            
            for i, level in enumerate(reduce_levels):
                if price_change >= level and self.position.short_amount > 0:
                    # 计算减仓数量
                    reduce_ratio = reduce_ratios[i] if i < len(reduce_ratios) else 0.3
                    reduce_amount = self.position.short_amount * reduce_ratio
                    
                    # 平仓空单
                    if reduce_amount > 0:
                        future_data = data[data.index <= current_time].tail(1)
                        future_data = future_data[future_data['type'] == 'future']
                        
                        if not future_data.empty:
                            future_price = future_data.iloc[0]['close']
                            
                            # 更新仓位
                            self.position.short_amount -= reduce_amount
                            
                            # 记录交易
                            trade = Trade(
                                timestamp=current_time,
                                symbol=future_data.iloc[0].get('symbol', 'UNKNOWN'),
                                side='buy',
                                amount=reduce_amount,
                                price=future_price,
                                trade_type='future',
                                reason=f'uptrend_reduce_short_{level:.0%}',
                                position_after=self.position
                            )
                            
                            trades.append(trade)
                            self.trades.append(trade)
                            
                            logger.info(f"上升趋势减仓: 平空 {reduce_amount:.4f} @ {future_price:.4f}")
            
            # 检查是否转为纯多头
            strong_threshold = self.strategy_config['strong_trend_threshold']
            if (price_change >= strong_threshold and 
                self.position.short_amount > 0):
                
                # 平掉所有空单
                future_data = data[data.index <= current_time].tail(1)
                future_data = future_data[future_data['type'] == 'future']
                
                if not future_data.empty:
                    future_price = future_data.iloc[0]['close']
                    remaining_short = self.position.short_amount
                    
                    # 更新仓位
                    self.position.short_amount = 0
                    self.position.position_type = PositionType.PURE_LONG
                    
                    # 记录交易
                    trade = Trade(
                        timestamp=current_time,
                        symbol=future_data.iloc[0].get('symbol', 'UNKNOWN'),
                        side='buy',
                        amount=remaining_short,
                        price=future_price,
                        trade_type='future',
                        reason='strong_uptrend_pure_long',
                        position_after=self.position
                    )
                    
                    trades.append(trade)
                    self.trades.append(trade)
                    
                    logger.info(f"强势上升趋势: 转为纯多头，平空 {remaining_short:.4f}")
            
        except Exception as e:
            logger.error(f"上升趋势调整失败: {e}")
        
        return trades
    
    def _handle_downtrend_adjustment(self, data: pd.DataFrame, current_time: datetime,
                                   current_price: float) -> List[Trade]:
        """处理下跌趋势的仓位调整"""
        trades = []
        
        try:
            # 卖出现货，保留空单
            if self.position.spot_amount > 0:
                spot_data = data[data.index <= current_time].tail(1)
                spot_data = spot_data[spot_data['type'] == 'spot']
                
                if not spot_data.empty:
                    spot_price = spot_data.iloc[0]['close']
                    spot_amount = self.position.spot_amount
                    
                    # 更新仓位
                    self.position.spot_amount = 0
                    self.position.position_type = PositionType.PURE_SHORT
                    
                    # 记录交易
                    trade = Trade(
                        timestamp=current_time,
                        symbol=spot_data.iloc[0].get('symbol', 'UNKNOWN'),
                        side='sell',
                        amount=spot_amount,
                        price=spot_price,
                        trade_type='spot',
                        reason='downtrend_sell_spot',
                        position_after=self.position
                    )
                    
                    trades.append(trade)
                    self.trades.append(trade)
                    
                    logger.info(f"下跌趋势: 卖出现货 {spot_amount:.4f} @ {spot_price:.4f}")
            
        except Exception as e:
            logger.error(f"下跌趋势调整失败: {e}")
        
        return trades
    
    def _handle_sideways_adjustment(self, data: pd.DataFrame, current_time: datetime,
                                  current_price: float) -> List[Trade]:
        """处理震荡趋势的仓位调整"""
        trades = []
        
        try:
            # 震荡时维持中性对冲
            if (self.position.spot_amount == 0 and 
                self.position.short_amount > 0):
                
                # 接回现货
                spot_data = data[data.index <= current_time].tail(1)
                spot_data = spot_data[spot_data['type'] == 'spot']
                
                if not spot_data.empty:
                    spot_price = spot_data.iloc[0]['close']
                    entry_price = self.position.entry_price_short
                    
                    # 检查回调幅度
                    callback_threshold = self.strategy_config['callback_threshold']
                    callback_ratio = (entry_price - current_price) / entry_price
                    
                    if callback_ratio >= callback_threshold:
                        # 计算接回数量（与空单等值）
                        spot_amount = self.position.short_amount * spot_price / spot_price
                        
                        # 更新仓位
                        self.position.spot_amount = spot_amount
                        self.position.position_type = PositionType.NEUTRAL
                        
                        # 记录交易
                        trade = Trade(
                            timestamp=current_time,
                            symbol=spot_data.iloc[0].get('symbol', 'UNKNOWN'),
                            side='buy',
                            amount=spot_amount,
                            price=spot_price,
                            trade_type='spot',
                            reason='sideways_rebalance',
                            position_after=self.position
                        )
                        
                        trades.append(trade)
                        self.trades.append(trade)
                        
                        logger.info(f"震荡调整: 接回现货 {spot_amount:.4f} @ {spot_price:.4f}")
            
        except Exception as e:
            logger.error(f"震荡调整失败: {e}")
        
        return trades
    
    def calculate_pnl(self, current_price: float) -> float:
        """
        计算当前盈亏
        
        Args:
            current_price: 当前价格
            
        Returns:
            盈亏金额
        """
        try:
            spot_pnl = 0.0
            short_pnl = 0.0
            
            # 现货盈亏
            if self.position.spot_amount > 0:
                spot_pnl = (current_price - self.position.entry_price_spot) * self.position.spot_amount
            
            # 空单盈亏
            if self.position.short_amount > 0:
                short_pnl = (self.position.entry_price_short - current_price) * self.position.short_amount
            
            total_pnl = spot_pnl + short_pnl
            return total_pnl
            
        except Exception as e:
            logger.error(f"盈亏计算失败: {e}")
            return 0.0
    
    def get_position_summary(self) -> Dict:
        """获取仓位摘要"""
        return {
            'spot_amount': self.position.spot_amount,
            'short_amount': self.position.short_amount,
            'position_type': self.position.position_type.value,
            'entry_time': self.position.entry_time,
            'last_adjust_time': self.position.last_adjust_time,
            'total_trades': len(self.trades),
            'current_trend': self.current_trend.value,
        }


# 使用示例
def main():
    """策略使用示例"""
    strategy = DeltaStrategy()
    
    # 模拟数据
    dates = pd.date_range('2024-01-01', periods=100, freq='15T')
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100),
        'type': ['spot'] * 50 + ['future'] * 50
    })
    data.set_index('timestamp', inplace=True)
    
    # 添加技术指标
    data['ema_20'] = data['close'].ewm(span=20).mean()
    data['ema_50'] = data['close'].ewm(span=50).mean()
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
    data['adx'] = np.random.uniform(15, 35, 100)
    
    # 测试策略
    for i, (timestamp, row) in enumerate(data.iterrows()):
        if i < 20:  # 跳过前20个数据点
            continue
        
        # 检查是否建仓
        if strategy.should_enter_position(data, timestamp):
            trades = strategy.enter_position(data, timestamp, 2000)
            print(f"建仓: {timestamp}, 交易数: {len(trades)}")
        
        # 检查是否调仓
        if strategy.should_adjust_position(data, timestamp):
            trades = strategy.adjust_position(data, timestamp)
            if trades:
                print(f"调仓: {timestamp}, 交易数: {len(trades)}")
    
    # 输出最终状态
    summary = strategy.get_position_summary()
    print(f"\n策略摘要: {summary}")


if __name__ == "__main__":
    main()
