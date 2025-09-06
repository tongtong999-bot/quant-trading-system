"""
结果分析模块 - 性能统计和可视化
提供详细的回测结果分析和报告生成
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json
from loguru import logger
import yaml

from ..backtest.backtest_engine import BacktestResult


@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    calmar_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_loss_ratio: float
    total_trades: int
    avg_trade_duration: float
    volatility: float
    var_95: float  # 95% VaR
    cvar_95: float  # 95% CVaR


class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化结果分析器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.output_config = self.config['output']
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def analyze_single_result(self, result: BacktestResult) -> PerformanceMetrics:
        """
        分析单个回测结果
        
        Args:
            result: 回测结果
            
        Returns:
            性能指标
        """
        try:
            if not result or result.equity_curve.empty:
                return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
            equity_curve = result.equity_curve
            initial_capital = result.initial_capital
            final_capital = result.final_capital
            
            # 基础指标
            total_return = (final_capital - initial_capital) / initial_capital
            
            # 年化收益率
            days = (result.end_date - result.start_date).days
            annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
            
            # 最大回撤
            equity_series = equity_curve['equity']
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak
            max_drawdown = drawdown.min()
            
            # 夏普比率
            returns = equity_series.pct_change().dropna()
            if len(returns) > 1 and returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24 * 4)  # 15分钟数据
            else:
                sharpe_ratio = 0.0
            
            # 卡玛比率
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
            
            # 索提诺比率
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0 and negative_returns.std() > 0:
                sortino_ratio = returns.mean() / negative_returns.std() * np.sqrt(252 * 24 * 4)
            else:
                sortino_ratio = 0.0
            
            # 交易统计
            win_rate = result.win_rate
            profit_loss_ratio = result.profit_loss_ratio
            
            # 平均持仓时间
            if result.trades:
                trade_durations = []
                for i in range(0, len(result.trades), 2):  # 假设每对交易是开仓和平仓
                    if i + 1 < len(result.trades):
                        duration = (result.trades[i+1].timestamp - result.trades[i].timestamp).total_seconds() / 3600
                        trade_durations.append(duration)
                avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
            else:
                avg_trade_duration = 0
            
            # 波动率
            volatility = returns.std() * np.sqrt(252 * 24 * 4) if len(returns) > 1 else 0
            
            # VaR和CVaR
            var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
            cvar_95 = returns[returns <= var_95].mean() if len(returns) > 0 else 0
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                win_rate=win_rate,
                profit_loss_ratio=profit_loss_ratio,
                total_trades=result.total_trades,
                avg_trade_duration=avg_trade_duration,
                volatility=volatility,
                var_95=var_95,
                cvar_95=cvar_95
            )
            
        except Exception as e:
            logger.error(f"单个结果分析失败: {e}")
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def analyze_multiple_results(self, results: List[BacktestResult]) -> Dict[str, Any]:
        """
        分析多个回测结果
        
        Args:
            results: 回测结果列表
            
        Returns:
            综合分析结果
        """
        try:
            valid_results = [r for r in results if r is not None]
            if not valid_results:
                return {}
            
            # 单个结果分析
            individual_metrics = []
            for result in valid_results:
                metrics = self.analyze_single_result(result)
                individual_metrics.append({
                    'symbol': result.symbol,
                    'metrics': metrics
                })
            
            # 整体统计
            total_returns = [m['metrics'].total_return for m in individual_metrics]
            max_drawdowns = [m['metrics'].max_drawdown for m in individual_metrics]
            sharpe_ratios = [m['metrics'].sharpe_ratio for m in individual_metrics]
            win_rates = [m['metrics'].win_rate for m in individual_metrics]
            
            # 成功率统计
            positive_returns = len([r for r in total_returns if r > 0])
            success_rate = positive_returns / len(total_returns)
            
            # 分位数统计
            return_percentiles = {
                'p25': np.percentile(total_returns, 25),
                'p50': np.percentile(total_returns, 50),
                'p75': np.percentile(total_returns, 75),
                'p90': np.percentile(total_returns, 90),
                'p95': np.percentile(total_returns, 95)
            }
            
            # 最佳和最差表现
            best_performer = max(individual_metrics, key=lambda x: x['metrics'].total_return)
            worst_performer = min(individual_metrics, key=lambda x: x['metrics'].total_return)
            
            # 风险调整收益排名
            risk_adjusted_returns = [
                (m['symbol'], m['metrics'].sharpe_ratio) 
                for m in individual_metrics
            ]
            risk_adjusted_returns.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'total_symbols': len(valid_results),
                'success_rate': success_rate,
                'avg_total_return': np.mean(total_returns),
                'median_total_return': np.median(total_returns),
                'std_total_return': np.std(total_returns),
                'avg_max_drawdown': np.mean(max_drawdowns),
                'avg_sharpe_ratio': np.mean(sharpe_ratios),
                'avg_win_rate': np.mean(win_rates),
                'return_percentiles': return_percentiles,
                'best_performer': best_performer,
                'worst_performer': worst_performer,
                'risk_adjusted_ranking': risk_adjusted_returns[:10],  # 前10名
                'individual_metrics': individual_metrics
            }
            
        except Exception as e:
            logger.error(f"多结果分析失败: {e}")
            return {}
    
    def create_equity_curve_plot(self, result: BacktestResult, save_path: str = None) -> go.Figure:
        """
        创建权益曲线图
        
        Args:
            result: 回测结果
            save_path: 保存路径
            
        Returns:
            Plotly图表对象
        """
        try:
            if result.equity_curve.empty:
                return go.Figure()
            
            equity_curve = result.equity_curve
            
            # 创建子图
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('权益曲线', '回撤曲线', '仓位变化'),
                vertical_spacing=0.08,
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # 权益曲线
            fig.add_trace(
                go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve['equity'],
                    mode='lines',
                    name='权益',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # 添加初始资金线
            initial_capital = result.initial_capital
            fig.add_hline(
                y=initial_capital,
                line_dash="dash",
                line_color="gray",
                annotation_text="初始资金",
                row=1, col=1
            )
            
            # 回撤曲线
            peak = equity_curve['equity'].expanding().max()
            drawdown = (equity_curve['equity'] - peak) / peak * 100
            
            fig.add_trace(
                go.Scatter(
                    x=equity_curve.index,
                    y=drawdown,
                    mode='lines',
                    name='回撤',
                    fill='tonexty',
                    line=dict(color='red', width=1)
                ),
                row=2, col=1
            )
            
            # 仓位变化
            fig.add_trace(
                go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve['spot_amount'],
                    mode='lines',
                    name='现货仓位',
                    line=dict(color='green', width=1)
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve['short_amount'],
                    mode='lines',
                    name='空单仓位',
                    line=dict(color='orange', width=1)
                ),
                row=3, col=1
            )
            
            # 更新布局
            fig.update_layout(
                title=f'{result.symbol} 回测结果分析',
                height=800,
                showlegend=True,
                hovermode='x unified'
            )
            
            # 更新x轴
            fig.update_xaxes(title_text="时间", row=3, col=1)
            
            # 更新y轴标签
            fig.update_yaxes(title_text="权益 (USDT)", row=1, col=1)
            fig.update_yaxes(title_text="回撤 (%)", row=2, col=1)
            fig.update_yaxes(title_text="仓位数量", row=3, col=1)
            
            # 保存图表
            if save_path:
                fig.write_html(save_path)
                logger.info(f"权益曲线图已保存: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"创建权益曲线图失败: {e}")
            return go.Figure()
    
    def create_performance_comparison_plot(self, results: List[BacktestResult], save_path: str = None) -> go.Figure:
        """
        创建性能对比图
        
        Args:
            results: 回测结果列表
            save_path: 保存路径
            
        Returns:
            Plotly图表对象
        """
        try:
            valid_results = [r for r in results if r is not None]
            if not valid_results:
                return go.Figure()
            
            # 准备数据
            symbols = [r.symbol for r in valid_results]
            total_returns = [(r.final_capital - r.initial_capital) / r.initial_capital for r in valid_results]
            max_drawdowns = [r.max_drawdown for r in valid_results]
            sharpe_ratios = [r.sharpe_ratio for r in valid_results]
            
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('总收益率', '最大回撤', '夏普比率', '收益率vs回撤散点图'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "scatter"}]]
            )
            
            # 总收益率柱状图
            fig.add_trace(
                go.Bar(
                    x=symbols,
                    y=total_returns,
                    name='总收益率',
                    marker_color=['green' if r > 0 else 'red' for r in total_returns]
                ),
                row=1, col=1
            )
            
            # 最大回撤柱状图
            fig.add_trace(
                go.Bar(
                    x=symbols,
                    y=max_drawdowns,
                    name='最大回撤',
                    marker_color='red'
                ),
                row=1, col=2
            )
            
            # 夏普比率柱状图
            fig.add_trace(
                go.Bar(
                    x=symbols,
                    y=sharpe_ratios,
                    name='夏普比率',
                    marker_color='blue'
                ),
                row=2, col=1
            )
            
            # 收益率vs回撤散点图
            fig.add_trace(
                go.Scatter(
                    x=max_drawdowns,
                    y=total_returns,
                    mode='markers+text',
                    text=symbols,
                    textposition='top center',
                    name='收益率vs回撤',
                    marker=dict(size=10, color=sharpe_ratios, colorscale='Viridis')
                ),
                row=2, col=2
            )
            
            # 更新布局
            fig.update_layout(
                title='多币种回测结果对比',
                height=800,
                showlegend=True
            )
            
            # 更新轴标签
            fig.update_xaxes(title_text="交易对", row=1, col=1)
            fig.update_xaxes(title_text="交易对", row=1, col=2)
            fig.update_xaxes(title_text="交易对", row=2, col=1)
            fig.update_xaxes(title_text="最大回撤", row=2, col=2)
            
            fig.update_yaxes(title_text="总收益率", row=1, col=1)
            fig.update_yaxes(title_text="最大回撤", row=1, col=2)
            fig.update_yaxes(title_text="夏普比率", row=2, col=1)
            fig.update_yaxes(title_text="总收益率", row=2, col=2)
            
            # 保存图表
            if save_path:
                fig.write_html(save_path)
                logger.info(f"性能对比图已保存: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"创建性能对比图失败: {e}")
            return go.Figure()
    
    def create_risk_analysis_plot(self, results: List[BacktestResult], save_path: str = None) -> go.Figure:
        """
        创建风险分析图
        
        Args:
            results: 回测结果列表
            save_path: 保存路径
            
        Returns:
            Plotly图表对象
        """
        try:
            valid_results = [r for r in results if r is not None]
            if not valid_results:
                return go.Figure()
            
            # 收集所有收益率数据
            all_returns = []
            for result in valid_results:
                if not result.equity_curve.empty:
                    returns = result.equity_curve['equity'].pct_change().dropna()
                    all_returns.extend(returns.tolist())
            
            if not all_returns:
                return go.Figure()
            
            returns_series = pd.Series(all_returns)
            
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('收益率分布', 'Q-Q图', '滚动波动率', 'VaR分析'),
                specs=[[{"type": "histogram"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # 收益率分布直方图
            fig.add_trace(
                go.Histogram(
                    x=returns_series,
                    nbinsx=50,
                    name='收益率分布',
                    marker_color='blue',
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Q-Q图
            from scipy import stats
            qq_data = stats.probplot(returns_series, dist="norm")
            fig.add_trace(
                go.Scatter(
                    x=qq_data[0][0],
                    y=qq_data[0][1],
                    mode='markers',
                    name='Q-Q图',
                    marker=dict(size=4, color='red')
                ),
                row=1, col=2
            )
            
            # 理论正态分布线
            theoretical_quantiles = qq_data[0][0]
            theoretical_values = qq_data[1][0] * theoretical_quantiles + qq_data[1][1]
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=theoretical_values,
                    mode='lines',
                    name='理论正态分布',
                    line=dict(color='blue', dash='dash')
                ),
                row=1, col=2
            )
            
            # 滚动波动率
            rolling_vol = returns_series.rolling(window=100).std() * np.sqrt(252 * 24 * 4)
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(rolling_vol))),
                    y=rolling_vol,
                    mode='lines',
                    name='滚动波动率',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
            
            # VaR分析
            var_levels = [90, 95, 99]
            var_values = [np.percentile(returns_series, 100-level) for level in var_levels]
            
            fig.add_trace(
                go.Bar(
                    x=[f'{level}%' for level in var_levels],
                    y=var_values,
                    name='VaR',
                    marker_color='red'
                ),
                row=2, col=2
            )
            
            # 更新布局
            fig.update_layout(
                title='风险分析',
                height=800,
                showlegend=True
            )
            
            # 更新轴标签
            fig.update_xaxes(title_text="收益率", row=1, col=1)
            fig.update_xaxes(title_text="理论分位数", row=1, col=2)
            fig.update_xaxes(title_text="时间", row=2, col=1)
            fig.update_xaxes(title_text="置信水平", row=2, col=2)
            
            fig.update_yaxes(title_text="频次", row=1, col=1)
            fig.update_yaxes(title_text="样本分位数", row=1, col=2)
            fig.update_yaxes(title_text="波动率", row=2, col=1)
            fig.update_yaxes(title_text="VaR值", row=2, col=2)
            
            # 保存图表
            if save_path:
                fig.write_html(save_path)
                logger.info(f"风险分析图已保存: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"创建风险分析图失败: {e}")
            return go.Figure()
    
    def generate_html_report(self, results: List[BacktestResult], output_dir: str = "results") -> str:
        """
        生成HTML报告
        
        Args:
            results: 回测结果列表
            output_dir: 输出目录
            
        Returns:
            报告文件路径
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # 分析结果
            analysis = self.analyze_multiple_results(results)
            if not analysis:
                logger.warning("没有有效的回测结果")
                return ""
            
            # 生成图表
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 权益曲线图
            equity_plot_path = output_path / f"equity_curves_{timestamp}.html"
            if results:
                self.create_equity_curve_plot(results[0], str(equity_plot_path))
            
            # 性能对比图
            comparison_plot_path = output_path / f"performance_comparison_{timestamp}.html"
            self.create_performance_comparison_plot(results, str(comparison_plot_path))
            
            # 风险分析图
            risk_plot_path = output_path / f"risk_analysis_{timestamp}.html"
            self.create_risk_analysis_plot(results, str(risk_plot_path))
            
            # 生成HTML报告
            report_path = output_path / f"backtest_report_{timestamp}.html"
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>量化交易系统回测报告</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>量化交易系统回测报告</h1>
                    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>整体表现摘要</h2>
                    <div class="metric">总交易对: {analysis['total_symbols']}</div>
                    <div class="metric">成功率: {analysis['success_rate']:.2%}</div>
                    <div class="metric">平均收益率: {analysis['avg_total_return']:.2%}</div>
                    <div class="metric">中位数收益率: {analysis['median_total_return']:.2%}</div>
                    <div class="metric">平均最大回撤: {analysis['avg_max_drawdown']:.2%}</div>
                    <div class="metric">平均夏普比率: {analysis['avg_sharpe_ratio']:.2f}</div>
                </div>
                
                <div class="section">
                    <h2>收益率分位数分析</h2>
                    <table>
                        <tr><th>分位数</th><th>收益率</th></tr>
                        <tr><td>25%</td><td>{analysis['return_percentiles']['p25']:.2%}</td></tr>
                        <tr><td>50%</td><td>{analysis['return_percentiles']['p50']:.2%}</td></tr>
                        <tr><td>75%</td><td>{analysis['return_percentiles']['p75']:.2%}</td></tr>
                        <tr><td>90%</td><td>{analysis['return_percentiles']['p90']:.2%}</td></tr>
                        <tr><td>95%</td><td>{analysis['return_percentiles']['p95']:.2%}</td></tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>最佳表现者</h2>
                    <p>交易对: {analysis['best_performer']['symbol']}</p>
                    <p>收益率: {analysis['best_performer']['metrics'].total_return:.2%}</p>
                    <p>夏普比率: {analysis['best_performer']['metrics'].sharpe_ratio:.2f}</p>
                </div>
                
                <div class="section">
                    <h2>最差表现者</h2>
                    <p>交易对: {analysis['worst_performer']['symbol']}</p>
                    <p>收益率: {analysis['worst_performer']['metrics'].total_return:.2%}</p>
                    <p>夏普比率: {analysis['worst_performer']['metrics'].sharpe_ratio:.2f}</p>
                </div>
                
                <div class="section">
                    <h2>风险调整收益排名 (前10名)</h2>
                    <table>
                        <tr><th>排名</th><th>交易对</th><th>夏普比率</th></tr>
                        {''.join([f'<tr><td>{i+1}</td><td>{item[0]}</td><td>{item[1]:.2f}</td></tr>' for i, item in enumerate(analysis['risk_adjusted_ranking'])])}
                    </table>
                </div>
                
                <div class="section">
                    <h2>详细图表</h2>
                    <p><a href="{equity_plot_path.name}" target="_blank">权益曲线图</a></p>
                    <p><a href="{comparison_plot_path.name}" target="_blank">性能对比图</a></p>
                    <p><a href="{risk_plot_path.name}" target="_blank">风险分析图</a></p>
                </div>
                
                <div class="section">
                    <h2>详细交易对表现</h2>
                    <table>
                        <tr>
                            <th>交易对</th>
                            <th>总收益率</th>
                            <th>年化收益率</th>
                            <th>最大回撤</th>
                            <th>夏普比率</th>
                            <th>胜率</th>
                            <th>交易次数</th>
                        </tr>
                        {''.join([f'''
                        <tr>
                            <td>{item['symbol']}</td>
                            <td class="{'positive' if item['metrics'].total_return > 0 else 'negative'}">{item['metrics'].total_return:.2%}</td>
                            <td class="{'positive' if item['metrics'].annualized_return > 0 else 'negative'}">{item['metrics'].annualized_return:.2%}</td>
                            <td class="negative">{item['metrics'].max_drawdown:.2%}</td>
                            <td>{item['metrics'].sharpe_ratio:.2f}</td>
                            <td>{item['metrics'].win_rate:.2%}</td>
                            <td>{item['metrics'].total_trades}</td>
                        </tr>
                        ''' for item in analysis['individual_metrics']])}
                    </table>
                </div>
            </body>
            </html>
            """
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML报告已生成: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"生成HTML报告失败: {e}")
            return ""
    
    def export_results_to_csv(self, results: List[BacktestResult], output_dir: str = "results") -> str:
        """
        导出结果到CSV
        
        Args:
            results: 回测结果列表
            output_dir: 输出目录
            
        Returns:
            CSV文件路径
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # 准备数据
            csv_data = []
            for result in results:
                if result is None:
                    continue
                
                metrics = self.analyze_single_result(result)
                csv_data.append({
                    'symbol': result.symbol,
                    'start_date': result.start_date,
                    'end_date': result.end_date,
                    'initial_capital': result.initial_capital,
                    'final_capital': result.final_capital,
                    'total_return': metrics.total_return,
                    'annualized_return': metrics.annualized_return,
                    'max_drawdown': metrics.max_drawdown,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'calmar_ratio': metrics.calmar_ratio,
                    'sortino_ratio': metrics.sortino_ratio,
                    'win_rate': metrics.win_rate,
                    'profit_loss_ratio': metrics.profit_loss_ratio,
                    'total_trades': metrics.total_trades,
                    'avg_trade_duration': metrics.avg_trade_duration,
                    'volatility': metrics.volatility,
                    'var_95': metrics.var_95,
                    'cvar_95': metrics.cvar_95
                })
            
            # 保存CSV
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = output_path / f"backtest_results_{timestamp}.csv"
            
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            logger.info(f"结果已导出到CSV: {csv_path}")
            return str(csv_path)
            
        except Exception as e:
            logger.error(f"导出CSV失败: {e}")
            return ""


# 使用示例
def main():
    """结果分析器使用示例"""
    analyzer = ResultAnalyzer()
    
    # 模拟回测结果
    from ..backtest.backtest_engine import BacktestResult
    
    # 这里应该从实际回测结果加载数据
    # results = load_backtest_results()
    
    # 生成报告
    # report_path = analyzer.generate_html_report(results)
    # print(f"报告已生成: {report_path}")


if __name__ == "__main__":
    main()
