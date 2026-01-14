from typing import List, Dict, Optional
from datetime import datetime, timedelta
from market_data import Candle, MarketDataProvider
from .regime_controller import MarketRegimeController
from .playbooks import (
    TrendContinuationPlaybook,
    LiquiditySweepReversalPlaybook,
    BreakoutRetestPlaybook,
    RangeMeanReversionPlaybook
)
from .confluence_scorer import ConfluenceScorer
import uuid

class SignalTiming:
    """Manages signal timing: entry window, setup expiry"""
    def __init__(self, entry_window_minutes: int = 7, setup_expiry_minutes: int = 20):
        self.entry_window_minutes = entry_window_minutes
        self.setup_expiry_minutes = setup_expiry_minutes
    
    def is_entry_valid(self, signal_time: datetime) -> bool:
        """Check if entry window is still valid"""
        elapsed = (datetime.utcnow() - signal_time).total_seconds() / 60
        return elapsed <= self.entry_window_minutes
    
    def is_setup_expired(self, signal_time: datetime) -> bool:
        """Check if setup has expired"""
        elapsed = (datetime.utcnow() - signal_time).total_seconds() / 60
        return elapsed > self.setup_expiry_minutes

class ApexFlowEngine:
    """Main signal engine - the brain of ApexFlow"""
    
    def __init__(self, data_provider: MarketDataProvider, signal_mode: str = 'balanced'):
        self.data_provider = data_provider
        self.signal_mode = signal_mode
        
        # Initialize components
        self.regime_controller = MarketRegimeController()
        self.confluence_scorer = ConfluenceScorer()
        self.timing_manager = SignalTiming()
        
        # Initialize playbooks
        self.playbooks = {
            'trend_continuation': TrendContinuationPlaybook(),
            'liquidity_sweep_reversal': LiquiditySweepReversalPlaybook(),
            'breakout_retest': BreakoutRetestPlaybook(),
            'range_mean_reversion': RangeMeanReversionPlaybook()
        }
        
        # Active signals tracking
        self.active_signals = {}
        
        # Cooldown tracking
        self.last_signal_time = {}
        
        # Active sessions (from settings)
        self.active_sessions = ['london', 'new_york', 'overlap', 'asia']  # Default: all sessions
        
        # Signal mode configuration
        # Quiet (Conservative): Only A+ and A signals (88+), strict thresholds
        # Balanced: A, B, limited C signals (70+), normal thresholds  
        # Aggressive: All grades A-D (50+), loose thresholds
        if signal_mode == 'quiet':
            self.cooldown_minutes = 45
            self.min_confluence_score = 88
            self.min_regime_confidence = 80
        elif signal_mode == 'aggressive':
            self.cooldown_minutes = 5
            self.min_confluence_score = 50  # Allow D grades
            self.min_regime_confidence = 55
        else:  # balanced
            self.cooldown_minutes = 15
            self.min_confluence_score = 70
            self.min_regime_confidence = 65
    
    async def analyze_and_generate_signal(self, symbol: str, 
                                         active_playbooks: List[str]) -> Optional[Dict]:
        """Main method: analyze market and potentially generate a signal"""
        
        # Step 1: Fetch multi-timeframe data
        candles_4h = await self.data_provider.get_candles(symbol, '4h', 100)
        candles_1h = await self.data_provider.get_candles(symbol, '1h', 100)
        candles_15m = await self.data_provider.get_candles(symbol, '15m', 100)
        candles_5m = await self.data_provider.get_candles(symbol, '5m', 100)
        
        # Step 2: Classify market regime
        regime = self.regime_controller.classify_regime(candles_4h, candles_1h, candles_15m)
        
        # Step 3: Check if regime is clear enough (threshold varies by signal_mode)
        if regime['confidence'] < self.min_regime_confidence:
            return None  # Regime unclear - no signal
        
        # Step 4: Check cooldown
        if not self._check_cooldown(symbol):
            return None
        
        # Step 4.5: Check if current session is active
        current_session = self._get_current_session()
        if current_session not in self.active_sessions:
            return None  # Outside of active trading sessions
        
        # Step 5: Try each allowed playbook
        for playbook_key in regime['allowed_playbooks']:
            if playbook_key not in active_playbooks:
                continue  # Playbook disabled by user
            
            playbook = self.playbooks.get(playbook_key)
            if not playbook:
                continue
            
            # Evaluate playbook
            signal = playbook.evaluate(candles_4h, candles_1h, candles_15m, candles_5m)
            
            if signal:
                # Step 6: Calculate confluence score
                session_quality = self._get_session_quality()
                confluence = self.confluence_scorer.calculate_score(
                    signal, candles_4h, candles_1h, candles_15m, candles_5m,
                    regime, session_quality
                )
                
                # Step 7: Apply confluence filter (threshold varies by signal_mode)
                # Quiet mode: >= 88 (A+ and A only), Balanced mode: >= 70 (A, B allowed)
                if confluence['score'] < self.min_confluence_score:
                    continue  # Score too low for current mode - reject signal
                
                # Step 8: Generate final signal
                final_signal = self._create_final_signal(
                    symbol, signal, confluence, regime, session_quality
                )
                
                # Update tracking
                self._update_tracking(symbol, final_signal)
                
                return final_signal
        
        return None  # No valid signal found
    
    def _check_cooldown(self, symbol: str) -> bool:
        """Check if cooldown period has passed"""
        if symbol not in self.last_signal_time:
            return True
        
        elapsed = (datetime.utcnow() - self.last_signal_time[symbol]).total_seconds() / 60
        return elapsed >= self.cooldown_minutes
    
    def _get_session_quality(self) -> str:
        """Determine current trading session quality"""
        current_hour = datetime.utcnow().hour
        
        # London session: 8:00 - 17:00 UTC
        # NY session: 13:00 - 22:00 UTC
        # Overlap: 13:00 - 17:00 UTC
        # Asia session: 0:00 - 8:00 UTC
        
        if 13 <= current_hour < 17:
            return 'overlap'
        elif 8 <= current_hour < 13:
            return 'london'
        elif 17 <= current_hour < 22:
            return 'new_york'
        else:
            return 'asia'
    
    def _get_current_session(self) -> str:
        """Get current trading session name (same as _get_session_quality)"""
        return self._get_session_quality()
        
        if 13 <= current_hour < 17:
            return 'overlap'
        elif 8 <= current_hour < 17:
            return 'london'
        elif 13 <= current_hour < 22:
            return 'newyork'
        else:
            return 'asian'
    
    def _create_final_signal(self, symbol: str, signal: Dict, 
                            confluence: Dict, regime: Dict, session: str) -> Dict:
        """Create final formatted signal"""
        signal_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        return {
            'signal_id': signal_id,
            'symbol': symbol,
            'timestamp': timestamp.isoformat(),
            'direction': signal['direction'],
            'entry': round(signal['entry'], 5),
            'stop_loss': round(signal['stop_loss'], 5),
            'take_profit': round(signal['take_profit'], 5),
            'playbook': signal['playbook'],
            'reason': signal['reason'],
            'confluence_score': confluence['total_score'],
            'confidence_breakdown': confluence['breakdown'],
            'regime': regime['regime'],
            'session': session,
            'timeframe': '5m',
            'entry_valid_until': (timestamp + timedelta(minutes=7)).isoformat(),
            'setup_expires_at': (timestamp + timedelta(minutes=20)).isoformat(),
            'mode': self.signal_mode,
            'status': 'active'
        }
    
    def _update_tracking(self, symbol: str, signal: Dict):
        """Update signal tracking"""
        self.last_signal_time[symbol] = datetime.utcnow()
        self.active_signals[signal['signal_id']] = signal
    
    def get_active_signals(self) -> List[Dict]:
        """Get all active signals"""
        # Clean up expired signals
        current_time = datetime.utcnow()
        expired_ids = []
        
        for signal_id, signal in self.active_signals.items():
            expiry_time = datetime.fromisoformat(signal['setup_expires_at'])
            if current_time > expiry_time:
                expired_ids.append(signal_id)
        
        for signal_id in expired_ids:
            self.active_signals[signal_id]['status'] = 'expired'
        
        return list(self.active_signals.values())
    
    def update_signal_status(self, signal_id: str, status: str):
        """Update signal status (active/expired/triggered)"""
        if signal_id in self.active_signals:
            self.active_signals[signal_id]['status'] = status