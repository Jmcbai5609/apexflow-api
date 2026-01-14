"""
OANDA Market Data Provider
Backup/fallback data source for ApexFlow - Forex specialist
"""
import httpx
from datetime import datetime, timedelta
from typing import List, Optional
import logging

from .data_provider import MarketDataProvider, Candle

logger = logging.getLogger(__name__)


class OandaDataProvider(MarketDataProvider):
    """
    Live market data from OANDA API
    
    Best for: Forex reliability and accuracy
    Free Demo: Unlimited API calls with demo account
    Live: Requires funded OANDA account
    
    Note: OANDA uses practice/live environments
    - Practice: api-fxpractice.oanda.com
    - Live: api-fxtrade.oanda.com
    """
    
    def __init__(self, api_key: str, account_id: str = None, practice: bool = True):
        self.api_key = api_key
        self.account_id = account_id
        self.name = "OANDA"
        
        # Use practice or live environment
        if practice:
            self.base_url = "https://api-fxpractice.oanda.com"
        else:
            self.base_url = "https://api-fxtrade.oanda.com"
        
        # OANDA uses underscore format: EUR_USD
        self.symbol_map = {
            'EURUSD': 'EUR_USD',
            'GBPUSD': 'GBP_USD',
            'USDJPY': 'USD_JPY',
            'XAUUSD': 'XAU_USD',
            'AUDUSD': 'AUD_USD',
            'USDCHF': 'USD_CHF',
            'NZDUSD': 'NZD_USD',
            'USDCAD': 'USD_CAD',
            'EURGBP': 'EUR_GBP',
            'EURJPY': 'EUR_JPY',
        }
        
        # OANDA granularity mapping
        self.timeframe_map = {
            '1m': 'M1',
            '5m': 'M5',
            '15m': 'M15',
            '30m': 'M30',
            '1h': 'H1',
            '4h': 'H4',
            '1d': 'D',
        }
    
    def _format_symbol(self, symbol: str) -> str:
        """Convert internal symbol format to OANDA format"""
        if symbol in self.symbol_map:
            return self.symbol_map[symbol]
        # If already has underscore, return as-is
        if '_' in symbol:
            return symbol
        # Try to convert 6-char format
        if len(symbol) == 6:
            return f"{symbol[:3]}_{symbol[3:]}"
        return symbol
    
    def _get_headers(self) -> dict:
        """Get authorization headers for OANDA API"""
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept-Datetime-Format': 'RFC3339'
        }
    
    async def get_candles(self, symbol: str, timeframe: str, limit: int = 100) -> List[Candle]:
        """
        Fetch candle data from OANDA
        
        Endpoint: /v3/instruments/{instrument}/candles
        """
        oanda_symbol = self._format_symbol(symbol)
        granularity = self.timeframe_map.get(timeframe, 'M5')
        
        url = f"{self.base_url}/v3/instruments/{oanda_symbol}/candles"
        
        params = {
            'granularity': granularity,
            'count': limit,
            'price': 'M'  # Midpoint prices
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params, headers=self._get_headers())
                
                if response.status_code == 401:
                    raise Exception("OANDA API: Invalid API key or unauthorized")
                
                if response.status_code != 200:
                    data = response.json()
                    error_msg = data.get('errorMessage', f'HTTP {response.status_code}')
                    raise Exception(f"OANDA API Error: {error_msg}")
                
                data = response.json()
                
                if 'candles' not in data:
                    raise Exception(f"No candle data returned for {symbol}")
                
                candles = []
                for item in data['candles']:
                    if not item.get('complete', True):
                        continue  # Skip incomplete candles
                    
                    mid = item.get('mid', {})
                    candle = Candle(
                        timestamp=datetime.fromisoformat(item['time'].replace('Z', '+00:00')),
                        open=float(mid.get('o', 0)),
                        high=float(mid.get('h', 0)),
                        low=float(mid.get('l', 0)),
                        close=float(mid.get('c', 0)),
                        volume=float(item.get('volume', 0))
                    )
                    candles.append(candle)
                
                logger.info(f"OANDA: Fetched {len(candles)} candles for {symbol}")
                return candles
                
        except httpx.TimeoutException:
            logger.error(f"OANDA API timeout for {symbol}")
            raise Exception("OANDA API timeout")
        except Exception as e:
            logger.error(f"OANDA API error: {str(e)}")
            raise
    
    async def get_latest_price(self, symbol: str) -> float:
        """
        Get latest price from OANDA
        
        Uses the pricing endpoint for real-time bid/ask
        """
        oanda_symbol = self._format_symbol(symbol)
        
        url = f"{self.base_url}/v3/accounts/{self.account_id}/pricing"
        
        params = {
            'instruments': oanda_symbol
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params, headers=self._get_headers())
                
                if response.status_code != 200:
                    # Fallback: use candles endpoint to get latest close
                    return await self._get_price_from_candles(symbol)
                
                data = response.json()
                
                if 'prices' not in data or not data['prices']:
                    raise Exception(f"No price data for {symbol}")
                
                price_data = data['prices'][0]
                # Calculate mid price from bid/ask
                bid = float(price_data.get('bids', [{'price': 0}])[0]['price'])
                ask = float(price_data.get('asks', [{'price': 0}])[0]['price'])
                
                return (bid + ask) / 2
                
        except Exception as e:
            logger.error(f"OANDA price fetch error: {str(e)}")
            # Fallback to candles
            return await self._get_price_from_candles(symbol)
    
    async def _get_price_from_candles(self, symbol: str) -> float:
        """Fallback: get latest price from most recent candle"""
        candles = await self.get_candles(symbol, '1m', limit=1)
        if candles:
            return candles[-1].close
        raise Exception(f"Could not get price for {symbol}")
    
    async def check_connectivity(self) -> dict:
        """Check API connectivity"""
        try:
            oanda_symbol = 'EUR_USD'
            url = f"{self.base_url}/v3/instruments/{oanda_symbol}/candles"
            params = {
                'granularity': 'M1',
                'count': 1,
                'price': 'M'
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params, headers=self._get_headers())
                
                if response.status_code == 200:
                    return {
                        'status': 'online',
                        'provider': self.name,
                        'message': 'Connected successfully'
                    }
                elif response.status_code == 401:
                    return {
                        'status': 'error',
                        'provider': self.name,
                        'message': 'Invalid API key'
                    }
                else:
                    return {
                        'status': 'error',
                        'provider': self.name,
                        'message': f'HTTP {response.status_code}'
                    }
        except Exception as e:
            return {
                'status': 'offline',
                'provider': self.name,
                'message': str(e)
            }
