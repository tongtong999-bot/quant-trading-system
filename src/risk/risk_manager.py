"""
风控系统 - 多层风险控制机制
实现硬性风控规则、成本控制和实时监控
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import yaml


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAction(Enum):
    """风控动作"""
    ALLOW = "allow"
    WARN = "warn"
    REDUCE = "reduce"
    STOP = "stop"
    FORCE_CLOSE = "force_close"


@dataclass
class RiskMetrics:
    """风险指标"""
    net_exposure_ratio: float = 0.0      # 净敞口比例
    max_loss_ratio: float = 0.0          # 最大亏损比例
    margin_usage_ratio: float = 0.0      # 保证金使用率
    time_exposure_hours: float = 0.0     # 时间敞口（小时）
    price_deviation_ratio: float = 0.0   # 价格偏离比例
    total_fees: float = 0.0              # 总手续费
    slippage_cost: float = 0.0           # 滑点成本
    funding_cost: float = 0.0            # 资金费率成本


@dataclass
class RiskAlert:
    """风险警报"""
    timestamp: datetime
    risk_type: str
    risk_level: RiskLevel
    current_value: float
    threshold_value: float
    action: RiskAction
    message: str
    position_info: Dict


class RiskManager:
    """风控管理器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化风控管理器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.risk_config = self.config['risk_control']
        self.strategy_config = self.config['strategy']
        
        # 风控状态
        self.risk_alerts: List[RiskAlert] = []
        self.risk_metrics = RiskMetrics()
        self.position_history: List[Dict] = []
        self.trade_costs: List[Dict] = []
        
        # 风控参数
        self.max_exposure_ratio = self.risk_config['max_exposure_ratio']
        self.max_loss_ratio = self.risk_config['max_loss_ratio']
        self.time_stop_hours = self.risk_config['time_stop_hours']
        self.price_stop_ratio = self.risk_config['price_stop_ratio']
        self.margin_usage_limit = self.risk_config['margin_usage_limit']
        
        # 成本参数
        self.fee_rate = self.risk_config['fee_rate']
        self.slippage_bps = self.risk_config['slippage_bps']
        
    def calculate_risk_metrics(self, position: Dict, current_price: float, 
                             entry_time: datetime, current_time: datetime) -> RiskMetrics:
        """
        计算风险指标
        
        Args:
            position: 仓位信息
            current_price: 当前价格
            entry_time: 入场时间
            current_time: 当前时间
            
        Returns:
            风险指标
        """
        try:
            metrics = RiskMetrics()
            
            # 计算净敞口比例
            spot_value = position.get('spot_amount', 0) * current_price
            short_value = position.get('short_amount', 0) * current_price
            total_value = spot_value + short_value
            
            if total_value > 0:
                net_exposure = abs(spot_value - short_value)
                metrics.net_exposure_ratio = net_exposure / total_value
            
            # 计算最大亏损比例
            entry_price_spot = position.get('entry_price_spot', current_price)
            entry_price_short = position.get('entry_price_short', current_price)
            
            spot_pnl = (current_price - entry_price_spot) * position.get('spot_amount', 0)
            short_pnl = (entry_price_short - current_price) * position.get('short_amount', 0)
            total_pnl = spot_pnl + short_pnl
            
            initial_capital = self.strategy_config['initial_capital']
            metrics.max_loss_ratio = abs(min(0, total_pnl)) / initial_capital
            
            # 计算保证金使用率
            leverage = self.strategy_config['leverage']
            margin_required = short_value / leverage
            metrics.margin_usage_ratio = margin_required / initial_capital
            
            # 计算时间敞口
            time_diff = current_time - entry_time
            metrics.time_exposure_hours = time_diff.total_seconds() / 3600
            
            # 计算价格偏离比例
            price_deviation = abs(current_price - entry_price_spot) / entry_price_spot
            metrics.price_deviation_ratio = price_deviation
            
            # 计算成本
            metrics.total_fees = self._calculate_total_fees()
            metrics.slippage_cost = self._calculate_slippage_cost()
            metrics.funding_cost = self._calculate_funding_cost(position, current_time)
            
            self.risk_metrics = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"风险指标计算失败: {e}")
            return RiskMetrics()
    
    def check_risk_rules(self, position: Dict, current_price: float, 
                        entry_time: datetime, current_time: datetime) -> List[RiskAlert]:
        """
        检查风控规则
        
        Args:
            position: 仓位信息
            current_price: 当前价格
            entry_time: 入场时间
            current_time: 当前时间
            
        Returns:
            风险警报列表
        """
        alerts = []
        
        try:
            # 计算风险指标
            metrics = self.calculate_risk_metrics(position, current_price, entry_time, current_time)
            
            # 检查净敞口限制
            if metrics.net_exposure_ratio > self.max_exposure_ratio:
                alert = RiskAlert(
                    timestamp=current_time,
                    risk_type="net_exposure",
                    risk_level=RiskLevel.HIGH,
                    current_value=metrics.net_exposure_ratio,
                    threshold_value=self.max_exposure_ratio,
                    action=RiskAction.REDUCE,
                    message=f"净敞口比例 {metrics.net_exposure_ratio:.2%} 超过限制 {self.max_exposure_ratio:.2%}",
                    position_info=position
                )
                alerts.append(alert)
            
            # 检查最大亏损限制
            if metrics.max_loss_ratio > self.max_loss_ratio:
                alert = RiskAlert(
                    timestamp=current_time,
                    risk_type="max_loss",
                    risk_level=RiskLevel.CRITICAL,
                    current_value=metrics.max_loss_ratio,
                    threshold_value=self.max_loss_ratio,
                    action=RiskAction.FORCE_CLOSE,
                    message=f"最大亏损比例 {metrics.max_loss_ratio:.2%} 超过限制 {self.max_loss_ratio:.2%}",
                    position_info=position
                )
                alerts.append(alert)
            
            # 检查保证金使用率
            if metrics.margin_usage_ratio > self.margin_usage_limit:
                alert = RiskAlert(
                    timestamp=current_time,
                    risk_type="margin_usage",
                    risk_level=RiskLevel.HIGH,
                    current_value=metrics.margin_usage_ratio,
                    threshold_value=self.margin_usage_limit,
                    action=RiskAction.REDUCE,
                    message=f"保证金使用率 {metrics.margin_usage_ratio:.2%} 超过限制 {self.margin_usage_limit:.2%}",
                    position_info=position
                )
                alerts.append(alert)
            
            # 检查时间止损
            if metrics.time_exposure_hours > self.time_stop_hours:
                alert = RiskAlert(
                    timestamp=current_time,
                    risk_type="time_stop",
                    risk_level=RiskLevel.MEDIUM,
                    current_value=metrics.time_exposure_hours,
                    threshold_value=self.time_stop_hours,
                    action=RiskAction.STOP,
                    message=f"持仓时间 {metrics.time_exposure_hours:.1f} 小时超过限制 {self.time_stop_hours} 小时",
                    position_info=position
                )
                alerts.append(alert)
            
            # 检查价格止损
            if metrics.price_deviation_ratio > self.price_stop_ratio:
                alert = RiskAlert(
                    timestamp=current_time,
                    risk_type="price_stop",
                    risk_level=RiskLevel.HIGH,
                    current_value=metrics.price_deviation_ratio,
                    threshold_value=self.price_stop_ratio,
                    action=RiskAction.STOP,
                    message=f"价格偏离 {metrics.price_deviation_ratio:.2%} 超过限制 {self.price_stop_ratio:.2%}",
                    position_info=position
                )
                alerts.append(alert)
            
            # 检查成本控制
            total_cost_ratio = (metrics.total_fees + metrics.slippage_cost + metrics.funding_cost) / self.strategy_config['initial_capital']
            if total_cost_ratio > 0.05:  # 总成本超过5%
                alert = RiskAlert(
                    timestamp=current_time,
                    risk_type="cost_control",
                    risk_level=RiskLevel.MEDIUM,
                    current_value=total_cost_ratio,
                    threshold_value=0.05,
                    action=RiskAction.WARN,
                    message=f"总成本比例 {total_cost_ratio:.2%} 较高",
                    position_info=position
                )
                alerts.append(alert)
            
            # 记录警报
            self.risk_alerts.extend(alerts)
            
        except Exception as e:
            logger.error(f"风控规则检查失败: {e}")
        
        return alerts
    
    def _calculate_total_fees(self) -> float:
        """计算总手续费"""
        try:
            total_fees = 0.0
            for trade_cost in self.trade_costs:
                trade_value = trade_cost.get('amount', 0) * trade_cost.get('price', 0)
                fees = trade_value * self.fee_rate
                total_fees += fees
            return total_fees
        except Exception as e:
            logger.error(f"手续费计算失败: {e}")
            return 0.0
    
    def _calculate_slippage_cost(self) -> float:
        """计算滑点成本"""
        try:
            total_slippage = 0.0
            for trade_cost in self.trade_costs:
                trade_value = trade_cost.get('amount', 0) * trade_cost.get('price', 0)
                slippage = trade_value * (self.slippage_bps / 10000)
                total_slippage += slippage
            return total_slippage
        except Exception as e:
            logger.error(f"滑点成本计算失败: {e}")
            return 0.0
    
    def _calculate_funding_cost(self, position: Dict, current_time: datetime) -> float:
        """计算资金费率成本"""
        try:
            # 这里需要根据实际资金费率数据计算
            # 简化处理，假设平均资金费率为0.01%
            short_value = position.get('short_amount', 0) * position.get('entry_price_short', 0)
            funding_rate = 0.0001  # 0.01%
            
            # 计算持仓时间（小时）
            entry_time = position.get('entry_time', current_time)
            hours_held = (current_time - entry_time).total_seconds() / 3600
            
            funding_cost = short_value * funding_rate * hours_held / 8  # 8小时收取一次
            return funding_cost
        except Exception as e:
            logger.error(f"资金费率成本计算失败: {e}")
            return 0.0
    
    def add_trade_cost(self, trade: Dict):
        """添加交易成本记录"""
        try:
            cost_record = {
                'timestamp': trade.get('timestamp', datetime.now()),
                'amount': trade.get('amount', 0),
                'price': trade.get('price', 0),
                'trade_type': trade.get('trade_type', 'unknown'),
                'side': trade.get('side', 'unknown')
            }
            self.trade_costs.append(cost_record)
        except Exception as e:
            logger.error(f"添加交易成本记录失败: {e}")
    
    def get_risk_summary(self) -> Dict:
        """获取风险摘要"""
        try:
            critical_alerts = [alert for alert in self.risk_alerts if alert.risk_level == RiskLevel.CRITICAL]
            high_alerts = [alert for alert in self.risk_alerts if alert.risk_level == RiskLevel.HIGH]
            medium_alerts = [alert for alert in self.risk_alerts if alert.risk_level == RiskLevel.MEDIUM]
            
            return {
                'total_alerts': len(self.risk_alerts),
                'critical_alerts': len(critical_alerts),
                'high_alerts': len(high_alerts),
                'medium_alerts': len(medium_alerts),
                'current_metrics': {
                    'net_exposure_ratio': self.risk_metrics.net_exposure_ratio,
                    'max_loss_ratio': self.risk_metrics.max_loss_ratio,
                    'margin_usage_ratio': self.risk_metrics.margin_usage_ratio,
                    'time_exposure_hours': self.risk_metrics.time_exposure_hours,
                    'price_deviation_ratio': self.risk_metrics.price_deviation_ratio,
                    'total_fees': self.risk_metrics.total_fees,
                    'slippage_cost': self.risk_metrics.slippage_cost,
                    'funding_cost': self.risk_metrics.funding_cost,
                },
                'recent_alerts': self.risk_alerts[-10:] if self.risk_alerts else []
            }
        except Exception as e:
            logger.error(f"获取风险摘要失败: {e}")
            return {}
    
    def should_force_close(self, position: Dict, current_price: float, 
                          entry_time: datetime, current_time: datetime) -> bool:
        """
        判断是否应该强制平仓
        
        Args:
            position: 仓位信息
            current_price: 当前价格
            entry_time: 入场时间
            current_time: 当前时间
            
        Returns:
            是否强制平仓
        """
        try:
            alerts = self.check_risk_rules(position, current_price, entry_time, current_time)
            
            # 检查是否有需要强制平仓的警报
            for alert in alerts:
                if alert.action == RiskAction.FORCE_CLOSE:
                    logger.warning(f"触发强制平仓: {alert.message}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"强制平仓判断失败: {e}")
            return False
    
    def get_position_adjustment(self, position: Dict, current_price: float, 
                               entry_time: datetime, current_time: datetime) -> Dict:
        """
        获取仓位调整建议
        
        Args:
            position: 仓位信息
            current_price: 当前价格
            entry_time: 入场时间
            current_time: 当前时间
            
        Returns:
            调整建议
        """
        try:
            alerts = self.check_risk_rules(position, current_price, entry_time, current_time)
            
            adjustment = {
                'action': 'hold',
                'reduce_spot_ratio': 0.0,
                'reduce_short_ratio': 0.0,
                'reason': '',
                'priority': 'low'
            }
            
            # 根据警报确定调整动作
            for alert in alerts:
                if alert.action == RiskAction.FORCE_CLOSE:
                    adjustment.update({
                        'action': 'force_close',
                        'reason': alert.message,
                        'priority': 'critical'
                    })
                    break
                elif alert.action == RiskAction.REDUCE:
                    if alert.risk_type == 'net_exposure':
                        # 减少净敞口
                        if position.get('spot_amount', 0) > position.get('short_amount', 0):
                            adjustment.update({
                                'action': 'reduce_spot',
                                'reduce_spot_ratio': 0.5,
                                'reason': alert.message,
                                'priority': 'high'
                            })
                        else:
                            adjustment.update({
                                'action': 'reduce_short',
                                'reduce_short_ratio': 0.5,
                                'reason': alert.message,
                                'priority': 'high'
                            })
                    elif alert.risk_type == 'margin_usage':
                        adjustment.update({
                            'action': 'reduce_short',
                            'reduce_short_ratio': 0.3,
                            'reason': alert.message,
                            'priority': 'high'
                        })
                elif alert.action == RiskAction.STOP:
                    adjustment.update({
                        'action': 'stop_trading',
                        'reason': alert.message,
                        'priority': 'medium'
                    })
                elif alert.action == RiskAction.WARN:
                    adjustment.update({
                        'action': 'monitor',
                        'reason': alert.message,
                        'priority': 'low'
                    })
            
            return adjustment
            
        except Exception as e:
            logger.error(f"获取调整建议失败: {e}")
            return {'action': 'hold', 'reason': f'风控系统错误: {e}', 'priority': 'low'}
    
    def reset_risk_state(self):
        """重置风控状态"""
        self.risk_alerts.clear()
        self.risk_metrics = RiskMetrics()
        self.position_history.clear()
        self.trade_costs.clear()
        logger.info("风控状态已重置")


# 使用示例
def main():
    """风控管理器使用示例"""
    risk_manager = RiskManager()
    
    # 模拟仓位
    position = {
        'spot_amount': 10.0,
        'short_amount': 10.0,
        'entry_price_spot': 100.0,
        'entry_price_short': 100.0,
        'entry_time': datetime.now() - timedelta(hours=2)
    }
    
    current_price = 105.0
    current_time = datetime.now()
    
    # 检查风险
    alerts = risk_manager.check_risk_rules(position, current_price, position['entry_time'], current_time)
    
    print(f"风险警报数量: {len(alerts)}")
    for alert in alerts:
        print(f"  {alert.risk_type}: {alert.message}")
    
    # 获取调整建议
    adjustment = risk_manager.get_position_adjustment(position, current_price, position['entry_time'], current_time)
    print(f"\n调整建议: {adjustment}")
    
    # 获取风险摘要
    summary = risk_manager.get_risk_summary()
    print(f"\n风险摘要: {summary}")


if __name__ == "__main__":
    main()

