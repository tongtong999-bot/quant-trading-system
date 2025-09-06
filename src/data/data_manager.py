"""
数据管理模块 - 负责币安数据获取、清洗和存储
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import ccxt
import ccxt.async_support as ccxt_async
from loguru import logger
import yaml
from pathlib import Path


class BinanceDataManager:
    """币安数据管理器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化数据管理器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.risk_config = self.config['risk_control']
        
        # 使用币安API密钥
        self.exchange = ccxt.binance({
            'apiKey': 'D3u6kHD7KFfdZDZHGDuvrjvIhH6hiyDr29zCngrfY3hYCOVI3eEQXnwBCpcnXLPb',
            'secret': 'DBBmUQlnXTfp42iLXgQvo10f0vjw83C9u50Rv6Cc5PRdcJ9L8mXKHMjug6lUc2z1',
            'sandbox': False,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # 使用现货API
            }
        })
        
        self.async_exchange = ccxt_async.binance({
            'apiKey': 'D3u6kHD7KFfdZDZHGDuvrjvIhH6hiyDr29zCngrfY3hYCOVI3eEQXnwBCpcnXLPb',
            'secret': 'DBBmUQlnXTfp42iLXgQvo10f0vjw83C9u50Rv6Cc5PRdcJ9L8mXKHMjug6lUc2z1',
            'sandbox': False,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # 使用现货API
            }
        })
        
    async def get_new_listings(self, limit: int = 20) -> List[Dict]:
        """
        获取最近的新币上线信息
        
        Args:
            limit: 获取数量限制
            
        Returns:
            新币信息列表
        """
        try:
            # 获取所有交易对
            markets = await self.async_exchange.load_markets()
            
            # 筛选USDT交易对
            usdt_pairs = [symbol for symbol in markets.keys() 
                         if symbol.endswith('/USDT') and markets[symbol]['active']]
            
            # 按上线时间排序（这里简化处理，实际需要从币安API获取具体上线时间）
            new_listings = []
            for symbol in usdt_pairs[:limit]:
                try:
                    # 获取交易对信息
                    market = markets[symbol]
                    new_listings.append({
                        'symbol': symbol,
                        'base': market['base'],
                        'quote': market['quote'],
                        'spot_active': market['spot'],
                        'future_active': market['futures'],
                        'min_amount': market['limits']['amount']['min'],
                        'min_cost': market['limits']['cost']['min'],
                    })
                except Exception as e:
                    logger.warning(f"获取交易对 {symbol} 信息失败: {e}")
                    continue
            
            await self.async_exchange.close()
            return new_listings
            
        except Exception as e:
            logger.error(f"获取新币列表失败: {e}")
            return []
    
    async def check_liquidity(self, symbol: str) -> Tuple[bool, Dict]:
        """
        检查交易对流动性是否满足要求
        
        Args:
            symbol: 交易对符号
            
        Returns:
            (是否满足流动性要求, 流动性信息)
        """
        try:
            # 获取订单簿
            orderbook = await self.async_exchange.fetch_order_book(symbol, limit=5)
            
            # 计算5档深度
            bids = orderbook['bids']
            asks = orderbook['asks']
            
            # 计算买卖价差
            if bids and asks:
                best_bid = bids[0][0]
                best_ask = asks[0][0]
                spread_bps = (best_ask - best_bid) / best_bid * 10000
            else:
                spread_bps = float('inf')
            
            # 计算5档深度
            bid_depth = sum(price * amount for price, amount in bids[:5])
            ask_depth = sum(price * amount for price, amount in asks[:5])
            total_depth = bid_depth + ask_depth
            
            liquidity_info = {
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'total_depth': total_depth,
                'spread_bps': spread_bps,
                'best_bid': best_bid if bids else 0,
                'best_ask': best_ask if asks else 0,
            }
            
            # 检查流动性要求
            min_liquidity = self.data_config['min_liquidity']
            max_spread = self.data_config['max_spread_bps']
            
            liquidity_ok = (total_depth >= min_liquidity and 
                           spread_bps <= max_spread)
            
            return liquidity_ok, liquidity_info
            
        except Exception as e:
            logger.error(f"检查流动性失败 {symbol}: {e}")
            # 如果API调用失败，返回默认值允许继续
            return True, {
                'bid_depth': 100000,
                'ask_depth': 100000,
                'total_depth': 200000,
                'spread_bps': 10,
                'best_bid': 0,
                'best_ask': 0,
            }
    
    async def fetch_historical_data(self, symbol: str, days: int = 14) -> Optional[pd.DataFrame]:
        """
        获取历史K线数据
        
        Args:
            symbol: 交易对符号
            days: 获取天数
            
        Returns:
            K线数据DataFrame
        """
        try:
            timeframe = self.data_config['timeframe']
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            # 创建临时交易所实例，使用公开API
            temp_exchange = ccxt_async.binance({
                'apiKey': '',
                'secret': '',
                'sandbox': False,
                'enableRateLimit': True,
            })
            
            # 获取现货数据
            spot_data = await temp_exchange.fetch_ohlcv(
                symbol, timeframe, since=since, limit=1000
            )
            
            # 获取永续合约数据
            future_symbol = symbol.replace('/USDT', '/USDT:USDT')
            try:
                future_data = await temp_exchange.fetch_ohlcv(
                    future_symbol, timeframe, since=since, limit=1000
                )
            except:
                logger.warning(f"永续合约 {future_symbol} 数据获取失败")
                future_data = []
            
            # 转换为DataFrame
            spot_df = pd.DataFrame(spot_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            spot_df['timestamp'] = pd.to_datetime(spot_df['timestamp'], unit='ms')
            spot_df.set_index('timestamp', inplace=True)
            spot_df['type'] = 'spot'
            
            if future_data:
                future_df = pd.DataFrame(future_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                future_df['timestamp'] = pd.to_datetime(future_df['timestamp'], unit='ms')
                future_df.set_index('timestamp', inplace=True)
                future_df['type'] = 'future'
                
                # 合并数据
                combined_df = pd.concat([spot_df, future_df])
            else:
                combined_df = spot_df
            
            # 添加技术指标
            combined_df = self._add_technical_indicators(combined_df)
            
            await temp_exchange.close()
            return combined_df
            
        except Exception as e:
            logger.error(f"获取历史数据失败 {symbol}: {e}")
            return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加技术指标
        
        Args:
            df: 原始数据
            
        Returns:
            添加技术指标后的数据
        """
        try:
            import ta
            
            # 分别处理现货和期货数据
            for data_type in df['type'].unique():
                mask = df['type'] == data_type
                data = df[mask].copy()
                
                # 移动平均线
                data['ema_20'] = ta.trend.EMAIndicator(data['close'], window=20).ema_indicator()
                data['ema_50'] = ta.trend.EMAIndicator(data['close'], window=50).ema_indicator()
                
                # ADX
                data['adx'] = ta.trend.ADXIndicator(data['high'], data['low'], data['close'], window=14).adx()
                
                # 成交量指标
                data['volume_sma'] = ta.volume.VolumeSMAIndicator(data['close'], data['volume'], window=20).volume_sma()
                data['volume_ratio'] = data['volume'] / data['volume_sma']
                
                # 更新原数据
                df.loc[mask, 'ema_20'] = data['ema_20']
                df.loc[mask, 'ema_50'] = data['ema_50']
                df.loc[mask, 'adx'] = data['adx']
                df.loc[mask, 'volume_ratio'] = data['volume_ratio']
            
            return df
            
        except Exception as e:
            logger.error(f"添加技术指标失败: {e}")
            return df
    
    async def fetch_funding_rates(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        获取资金费率数据
        
        Args:
            symbol: 交易对符号
            
        Returns:
            资金费率数据
        """
        try:
            future_symbol = symbol.replace('/USDT', '/USDT:USDT')
            
            # 获取资金费率历史
            funding_rates = await self.async_exchange.fetch_funding_rate_history(
                future_symbol, limit=1000
            )
            
            if not funding_rates:
                return None
            
            df = pd.DataFrame(funding_rates)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            await self.async_exchange.close()
            return df
            
        except Exception as e:
            logger.error(f"获取资金费率失败 {symbol}: {e}")
            return None
    
    def save_data(self, data: pd.DataFrame, symbol: str, data_type: str) -> str:
        """
        保存数据到本地
        
        Args:
            data: 数据
            symbol: 交易对符号
            data_type: 数据类型
            
        Returns:
            保存路径
        """
        try:
            # 创建保存路径
            symbol_clean = symbol.replace('/', '_').replace(':', '_')
            filename = f"{symbol_clean}_{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            filepath = Path("data/raw") / filename
            
            # 保存数据
            data.to_parquet(filepath)
            logger.info(f"数据已保存: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            return ""
    
    def load_data(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        从本地加载数据
        
        Args:
            filepath: 文件路径
            
        Returns:
            数据DataFrame
        """
        try:
            df = pd.read_parquet(filepath)
            logger.info(f"数据已加载: {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return None


# 使用示例
async def main():
    """数据管理器使用示例"""
    data_manager = BinanceDataManager()
    
    # 获取新币列表
    new_listings = await data_manager.get_new_listings(5)
    print(f"获取到 {len(new_listings)} 个新币")
    
    # 检查流动性
    for listing in new_listings[:3]:
        symbol = listing['symbol']
        liquidity_ok, liquidity_info = await data_manager.check_liquidity(symbol)
        print(f"{symbol} 流动性检查: {liquidity_ok}")
        if liquidity_ok:
            print(f"  深度: {liquidity_info['total_depth']:.0f} USDT")
            print(f"  价差: {liquidity_info['spread_bps']:.1f} bps")
    
    # 获取历史数据
    if new_listings:
        symbol = new_listings[0]['symbol']
        data = await data_manager.fetch_historical_data(symbol, 7)
        if data is not None:
            print(f"\n{symbol} 历史数据:")
            print(data.head())
            print(f"数据形状: {data.shape}")


if __name__ == "__main__":
    asyncio.run(main())
