from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import random
import math

class Candle:
    """Represents a single candlestick"""
    def __init__(self, timestamp: datetime, open: float, high: float, low: float, close: float, volume: float):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        
    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }

class MarketDataProvider(ABC):
    """Abstract base class for market data providers"""
    
    @abstractmethod
    async def get_candles(self, symbol: str, timeframe: str, limit: int = 100) -> List[Candle]:
        """Fetch candle data for a symbol and timeframe"""
        pass
    
    @abstractmethod
    async def get_latest_price(self, symbol: str) -> float:
        """Get the latest price for a symbol"""
        pass

class DemoDataProvider(MarketDataProvider):
    """Simulated market data for development and testing"""
    
    def __init__(self):
        # Base prices for each pair
        self.base_prices = {
            'EURUSD': 1.0850,
            'GBPUSD': 1.2650,
            'USDJPY': 148.50,
            'XAUUSD': 2025.00
        }
        # Current simulated prices (will evolve)
        self.current_prices = self.base_prices.copy()
        
    def _generate_realistic_candles(self, symbol: str, timeframe: str, limit: int) -> List[Candle]:
        """Generate realistic forex candles with trends, reversals, and consolidations"""
        candles = []
        
        # Timeframe to minutes mapping
        tf_minutes = {
            '5m': 5,
            '15m': 15,
            '1h': 60,
            '4h': 240
        }
        
        minutes = tf_minutes.get(timeframe, 5)
        base_price = self.current_prices[symbol]
        
        # Generate market regime patterns
        regime = random.choice(['trend', 'range', 'breakout', 'reversal'])
        
        current_time = datetime.utcnow()
        current_price = base_price
        
        for i in range(limit):
            timestamp = current_time - timedelta(minutes=minutes * (limit - i))
            
            # Calculate ATR-like volatility based on symbol
            if symbol == 'XAUUSD':
                atr = random.uniform(5, 15)
            elif symbol == 'USDJPY':
                atr = random.uniform(0.15, 0.40)
            else:
                atr = random.uniform(0.0008, 0.0025)
            
            # Trend component
            if regime == 'trend':
                trend_move = random.uniform(-atr * 0.3, atr * 0.8)  # Uptrend bias
            elif regime == 'reversal':
                # Switch direction midway
                if i < limit / 2:
                    trend_move = random.uniform(-atr * 0.5, atr * 0.3)
                else:
                    trend_move = random.uniform(-atr * 0.3, atr * 0.5)
            elif regime == 'range':
                trend_move = random.uniform(-atr * 0.2, atr * 0.2)  # Minimal drift
            else:  # breakout
                if i < limit * 0.7:
                    trend_move = random.uniform(-atr * 0.1, atr * 0.1)  # Consolidation
                else:
                    trend_move = random.uniform(atr * 0.5, atr * 1.2)  # Breakout
            
            # Random noise
            noise = random.uniform(-atr * 0.3, atr * 0.3)
            
            open_price = current_price
            close_price = open_price + trend_move + noise
            
            # High and low with realistic wicks
            wick_factor = random.uniform(0.3, 0.8)
            high_price = max(open_price, close_price) + atr * wick_factor
            low_price = min(open_price, close_price) - atr * wick_factor
            
            # Volume
            volume = random.uniform(1000, 5000) * (1 + abs(close_price - open_price) / open_price * 100)
            
            candle = Candle(
                timestamp=timestamp,
                open=round(open_price, 5 if symbol != 'XAUUSD' else 2),
                high=round(high_price, 5 if symbol != 'XAUUSD' else 2),
                low=round(low_price, 5 if symbol != 'XAUUSD' else 2),
                close=round(close_price, 5 if symbol != 'XAUUSD' else 2),
                volume=round(volume, 2)
            )
            
            candles.append(candle)
            current_price = close_price
        
        # Update current price for this symbol
        self.current_prices[symbol] = current_price
        
        return candles
    
    async def get_candles(self, symbol: str, timeframe: str, limit: int = 100) -> List[Candle]:
        """Get simulated candle data"""
        if symbol not in self.base_prices:
            raise ValueError(f"Symbol {symbol} not supported in demo mode")
        
        return self._generate_realistic_candles(symbol, timeframe, limit)
    
    async def get_latest_price(self, symbol: str) -> float:
        """Get current simulated price"""
        if symbol not in self.current_prices:
            raise ValueError(f"Symbol {symbol} not supported in demo mode")
        
        return self.current_prices[symbol]

class TwelveDataProvider(MarketDataProvider):
    """Live market data from Twelve Data API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
    
    async def get_candles(self, symbol: str, timeframe: str, limit: int = 100) -> List[Candle]:
        """Fetch real candle data from Twelve Data"""
        # TODO: Implement actual API call when API key is provided
        # This is a placeholder for when live data is enabled
        import httpx
        
        # Convert symbol format (EURUSD -> EUR/USD)
        if len(symbol) == 6:
            formatted_symbol = f"{symbol[:3]}/{symbol[3:]}"
        else:
            formatted_symbol = symbol
        
        # Convert timeframe format to Twelve Data format
        timeframe_map = {
            '5m': '5min',
            '15m': '15min',
            '1h': '1h',
            '4h': '4h'
        }
        interval = timeframe_map.get(timeframe, timeframe)
        
        params = {
            'symbol': formatted_symbol,
            'interval': interval,
            'outputsize': limit,
            'apikey': self.api_key
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/time_series", params=params)
            data = response.json()
            
            if 'values' not in data:
                raise Exception(f"API Error: {data.get('message', 'Unknown error')}")
            
            candles = []
            for item in data['values']:
                candle = Candle(
                    timestamp=datetime.fromisoformat(item['datetime']),
                    open=float(item['open']),
                    high=float(item['high']),
                    low=float(item['low']),
                    close=float(item['close']),
                    volume=float(item.get('volume', 0))
                )
                candles.append(candle)
            
            return candles
    
    async def get_latest_price(self, symbol: str) -> float:
        """Get latest real price from Twelve Data"""
        import httpx
        
        if len(symbol) == 6:
            formatted_symbol = f"{symbol[:3]}/{symbol[3:]}"
        else:
            formatted_symbol = symbol
        
        params = {
            'symbol': formatted_symbol,
            'apikey': self.api_key
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/price", params=params)
            data = response.json()
            
            if 'price' not in data:
                raise Exception(f"API Error: {data.get('message', 'Unknown error')}")
            
            return float(data['price'])