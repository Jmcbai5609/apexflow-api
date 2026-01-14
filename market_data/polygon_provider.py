"""
Polygon.io Market Data Provider
Primary data source for ApexFlow - Forex and Crypto
"""
import httpx
from datetime import datetime, timedelta
from typing import List, Optional
import logging

from .data_provider import MarketDataProvider, Candle

logger = logging.getLogger(__name__)


class PolygonDataProvider(MarketDataProvider):
    """
    Live market data from Polygon.io API
    
    Free Tier: 5 API calls/minute
    Paid ($29/mo): Unlimited calls
    
    Supports: Forex, Crypto, Stocks
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.name = "Polygon.io"
        
        # Symbol format mapping for Polygon
        # Polygon uses C:EURUSD for forex, X:BTCUSD for crypto
        self.forex_pairs = {
            'EURUSD': 'C:EURUSD',
            'GBPUSD': 'C:GBPUSD',
            'USDJPY': 'C:USDJPY',
            'XAUUSD': 'C:XAUUSD',  # Gold
            'AUDUSD': 'C:AUDUSD',
            'USDCHF': 'C:USDCHF',
            'NZDUSD': 'C:NZDUSD',
            'USDCAD': 'C:USDCAD',
            'EURGBP': 'C:EURGBP',
            'EURJPY': 'C:EURJPY',
        }
        
        # Timeframe mapping to Polygon multiplier/timespan
        self.timeframe_map = {
            '1m': (1, 'minute'),
            '5m': (5, 'minute'),
            '15m': (15, 'minute'),
            '30m': (30, 'minute'),
            '1h': (1, 'hour'),
            '4h': (4, 'hour'),
            '1d': (1, 'day'),
        }
    
    def _format_symbol(self, symbol: str) -> str:
        """Convert internal symbol format to Polygon format"""
        if symbol in self.forex_pairs:
            return self.forex_pairs[symbol]
        # Default: assume it's already in correct format or crypto
        if symbol.startswith('C:') or symbol.startswith('X:'):
            return symbol
        # Assume forex if 6 chars
        if len(symbol) == 6:
            return f"C:{symbol}"
        return symbol
    
    async def get_candles(self, symbol: str, timeframe: str, limit: int = 100) -> List[Candle]:
        """
        Fetch candle data from Polygon.io
        
        Endpoint: /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
        """
        polygon_symbol = self._format_symbol(symbol)
        
        # Get timeframe parameters
        multiplier, timespan = self.timeframe_map.get(timeframe, (5, 'minute'))
        
        # Calculate date range
        end_date = datetime.utcnow()
        
        # Calculate start date based on limit and timeframe
        if timespan == 'minute':
            start_date = end_date - timedelta(minutes=multiplier * limit * 2)
        elif timespan == 'hour':
            start_date = end_date - timedelta(hours=multiplier * limit * 2)
        else:  # day
            start_date = end_date - timedelta(days=multiplier * limit * 2)
        
        # Format dates for API (milliseconds timestamp)
        from_ts = int(start_date.timestamp() * 1000)
        to_ts = int(end_date.timestamp() * 1000)
        
        url = f"{self.base_url}/v2/aggs/ticker/{polygon_symbol}/range/{multiplier}/{timespan}/{from_ts}/{to_ts}"
        
        params = {
            'apiKey': self.api_key,
            'limit': limit,
            'sort': 'desc'  # Newest first
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                data = response.json()
                
                if data.get('status') == 'ERROR':
                    error_msg = data.get('error', 'Unknown error')
                    logger.error(f"Polygon API error: {error_msg}")
                    raise Exception(f"Polygon API Error: {error_msg}")
                
                if 'results' not in data or not data['results']:
                    logger.warning(f"No data returned from Polygon for {symbol}")
                    raise Exception(f"No data available for {symbol}")
                
                candles = []
                for item in data['results'][:limit]:
                    candle = Candle(
                        timestamp=datetime.fromtimestamp(item['t'] / 1000),
                        open=float(item['o']),
                        high=float(item['h']),
                        low=float(item['l']),
                        close=float(item['c']),
                        volume=float(item.get('v', 0))
                    )
                    candles.append(candle)
                
                # Reverse to get oldest first (chronological order)
                candles.reverse()
                
                logger.info(f"Polygon: Fetched {len(candles)} candles for {symbol}")
                return candles
                
        except httpx.TimeoutException:
            logger.error(f"Polygon API timeout for {symbol}")
            raise Exception("Polygon API timeout")
        except Exception as e:
            logger.error(f"Polygon API error: {str(e)}")
            raise
    
    async def get_latest_price(self, symbol: str) -> float:
        """
        Get latest price from Polygon.io
        
        Uses the forex snapshot endpoint for real-time price
        """
        polygon_symbol = self._format_symbol(symbol)
        
        # Use previous close endpoint for forex
        url = f"{self.base_url}/v2/aggs/ticker/{polygon_symbol}/prev"
        
        params = {
            'apiKey': self.api_key
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                data = response.json()
                
                if data.get('status') == 'ERROR':
                    error_msg = data.get('error', 'Unknown error')
                    raise Exception(f"Polygon API Error: {error_msg}")
                
                if 'results' not in data or not data['results']:
                    raise Exception(f"No price data for {symbol}")
                
                # Return the close price from the last bar
                return float(data['results'][0]['c'])
                
        except Exception as e:
            logger.error(f"Polygon price fetch error: {str(e)}")
            raise
    
    async def check_connectivity(self) -> dict:
        """Check API connectivity and quota status"""
        try:
            # Use a simple endpoint to check connectivity
            url = f"{self.base_url}/v2/aggs/ticker/C:EURUSD/prev"
            params = {'apiKey': self.api_key}
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                data = response.json()
                
                if data.get('status') == 'OK':
                    return {
                        'status': 'online',
                        'provider': self.name,
                        'message': 'Connected successfully'
                    }
                else:
                    return {
                        'status': 'error',
                        'provider': self.name,
                        'message': data.get('error', 'Unknown error')
                    }
        except Exception as e:
            return {
                'status': 'offline',
                'provider': self.name,
                'message': str(e)
            }
