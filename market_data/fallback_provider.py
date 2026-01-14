"""
Fallback Data Provider
Orchestrates multiple data providers with automatic failover
Primary: Polygon.io
Backup: OANDA
"""
import logging
from typing import List, Optional
from datetime import datetime

from .data_provider import MarketDataProvider, Candle, DemoDataProvider
from .polygon_provider import PolygonDataProvider
from .oanda_provider import OandaDataProvider

logger = logging.getLogger(__name__)


class FallbackDataProvider(MarketDataProvider):
    """
    Multi-provider data source with automatic failover
    
    Priority order:
    1. Polygon.io (primary)
    2. OANDA (backup)
    3. Demo data (last resort fallback)
    
    Features:
    - Automatic failover on errors
    - Health tracking for each provider
    - Provider status reporting
    """
    
    def __init__(
        self,
        polygon_api_key: Optional[str] = None,
        oanda_api_key: Optional[str] = None,
        oanda_account_id: Optional[str] = None,
        oanda_practice: bool = True
    ):
        self.name = "ApexFlow Multi-Provider"
        self.providers: List[MarketDataProvider] = []
        self.provider_status = {}
        
        # Initialize Polygon (primary)
        if polygon_api_key:
            try:
                polygon = PolygonDataProvider(api_key=polygon_api_key)
                self.providers.append(polygon)
                self.provider_status['polygon'] = {
                    'name': 'Polygon.io',
                    'status': 'ready',
                    'priority': 1,
                    'last_success': None,
                    'last_error': None,
                    'error_count': 0
                }
                logger.info("Polygon.io provider initialized (primary)")
            except Exception as e:
                logger.error(f"Failed to initialize Polygon provider: {e}")
        
        # Initialize OANDA (backup)
        if oanda_api_key:
            try:
                oanda = OandaDataProvider(
                    api_key=oanda_api_key,
                    account_id=oanda_account_id,
                    practice=oanda_practice
                )
                self.providers.append(oanda)
                self.provider_status['oanda'] = {
                    'name': 'OANDA',
                    'status': 'ready',
                    'priority': 2,
                    'last_success': None,
                    'last_error': None,
                    'error_count': 0
                }
                logger.info("OANDA provider initialized (backup)")
            except Exception as e:
                logger.error(f"Failed to initialize OANDA provider: {e}")
        
        # Always add demo as last resort
        demo = DemoDataProvider()
        self.providers.append(demo)
        self.provider_status['demo'] = {
            'name': 'Demo Data',
            'status': 'ready',
            'priority': 99,  # Last resort
            'last_success': None,
            'last_error': None,
            'error_count': 0
        }
        logger.info("Demo provider initialized (fallback)")
        
        # Track active provider
        self.active_provider_index = 0
    
    def _get_provider_key(self, provider: MarketDataProvider) -> str:
        """Get the status key for a provider"""
        if isinstance(provider, PolygonDataProvider):
            return 'polygon'
        elif isinstance(provider, OandaDataProvider):
            return 'oanda'
        else:
            return 'demo'
    
    def _mark_provider_success(self, provider: MarketDataProvider):
        """Mark a provider as successful"""
        key = self._get_provider_key(provider)
        if key in self.provider_status:
            self.provider_status[key]['status'] = 'online'
            self.provider_status[key]['last_success'] = datetime.utcnow().isoformat()
            self.provider_status[key]['error_count'] = 0
    
    def _mark_provider_error(self, provider: MarketDataProvider, error: str):
        """Mark a provider as errored"""
        key = self._get_provider_key(provider)
        if key in self.provider_status:
            self.provider_status[key]['status'] = 'error'
            self.provider_status[key]['last_error'] = {
                'message': error,
                'timestamp': datetime.utcnow().isoformat()
            }
            self.provider_status[key]['error_count'] += 1
    
    async def get_candles(self, symbol: str, timeframe: str, limit: int = 100) -> List[Candle]:
        """
        Fetch candles with automatic failover
        
        Tries each provider in priority order until one succeeds
        """
        errors = []
        
        for provider in self.providers:
            try:
                candles = await provider.get_candles(symbol, timeframe, limit)
                self._mark_provider_success(provider)
                
                provider_name = getattr(provider, 'name', provider.__class__.__name__)
                logger.info(f"Got {len(candles)} candles from {provider_name} for {symbol}")
                
                return candles
                
            except Exception as e:
                provider_name = getattr(provider, 'name', provider.__class__.__name__)
                error_msg = str(e)
                errors.append(f"{provider_name}: {error_msg}")
                self._mark_provider_error(provider, error_msg)
                logger.warning(f"{provider_name} failed for {symbol}: {error_msg}")
                continue
        
        # All providers failed
        raise Exception(f"All data providers failed: {'; '.join(errors)}")
    
    async def get_latest_price(self, symbol: str) -> float:
        """
        Get latest price with automatic failover
        """
        errors = []
        
        for provider in self.providers:
            try:
                price = await provider.get_latest_price(symbol)
                self._mark_provider_success(provider)
                return price
                
            except Exception as e:
                provider_name = getattr(provider, 'name', provider.__class__.__name__)
                error_msg = str(e)
                errors.append(f"{provider_name}: {error_msg}")
                self._mark_provider_error(provider, error_msg)
                continue
        
        raise Exception(f"All data providers failed: {'; '.join(errors)}")
    
    async def check_all_providers(self) -> dict:
        """
        Check connectivity status of all configured providers
        
        Returns detailed status for each provider
        """
        results = {}
        
        for provider in self.providers:
            key = self._get_provider_key(provider)
            
            if hasattr(provider, 'check_connectivity'):
                try:
                    status = await provider.check_connectivity()
                    results[key] = {
                        **self.provider_status.get(key, {}),
                        **status
                    }
                except Exception as e:
                    results[key] = {
                        **self.provider_status.get(key, {}),
                        'status': 'error',
                        'message': str(e)
                    }
            else:
                # Demo provider doesn't have connectivity check
                results[key] = {
                    **self.provider_status.get(key, {}),
                    'status': 'always_available'
                }
        
        return results
    
    def get_active_provider_name(self) -> str:
        """Get the name of the currently active primary provider"""
        for key, status in self.provider_status.items():
            if status['status'] == 'online' and status['priority'] < 99:
                return status['name']
        
        # Check if any live provider is ready
        for key, status in self.provider_status.items():
            if status['status'] == 'ready' and status['priority'] < 99:
                return status['name']
        
        return 'Demo Data'
    
    def get_status_summary(self) -> dict:
        """Get a summary of all provider statuses"""
        return {
            'active_provider': self.get_active_provider_name(),
            'providers': self.provider_status,
            'total_configured': len(self.providers),
            'live_providers': sum(1 for k, v in self.provider_status.items() if v['priority'] < 99)
        }
