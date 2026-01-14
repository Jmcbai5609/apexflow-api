from typing import Dict, List, Optional
from datetime import datetime
from market_data import Candle
from .market_analysis import (
    MarketStructureAnalyzer,
    LiquidityDetector,
    ZoneMapper,
    ConsolidationDetector
)

class SignalPlaybook:
    """Base class for signal playbooks"""
    
    def __init__(self, name: str):
        self.name = name
        self.structure_analyzer = MarketStructureAnalyzer()
        self.liquidity_detector = LiquidityDetector()
        self.zone_mapper = ZoneMapper()
        self.consolidation_detector = ConsolidationDetector()
    
    def evaluate(self, 
                candles_4h: List[Candle],
                candles_1h: List[Candle],
                candles_15m: List[Candle],
                candles_5m: List[Candle]) -> Optional[Dict]:
        """Evaluate if this playbook generates a signal. Returns signal dict or None."""
        raise NotImplementedError

class TrendContinuationPlaybook(SignalPlaybook):
    """Playbook A: Trend Continuation - Trades pullbacks in strong trends"""
    
    def __init__(self):
        super().__init__("Trend Continuation")
    
    def evaluate(self, candles_4h, candles_1h, candles_15m, candles_5m) -> Optional[Dict]:
        # Check HTF trend alignment
        trend_4h = self.structure_analyzer.calculate_trend(candles_4h)
        trend_1h = self.structure_analyzer.calculate_trend(candles_1h)
        
        if trend_4h['direction'] != trend_1h['direction'] or trend_4h['direction'] == 'neutral':
            return None  # No signal: trend not aligned
        
        direction = trend_4h['direction']
        
        # Check for pullback on 15m
        swings_15m = self.structure_analyzer.detect_swing_points(candles_15m)
        zones = self.zone_mapper.detect_supply_demand_zones(candles_1h)
        
        # Look for demand zone test in uptrend or supply zone test in downtrend
        current_price = candles_5m[-1].close
        
        if direction == 'bullish':
            # Find recent demand zone being tested
            for zone in zones['demand_zones']:
                if zone['bottom'] <= current_price <= zone['top']:
                    # Check for bullish confirmation candle on 5m
                    last_candle = candles_5m[-1]
                    if last_candle.close > last_candle.open:
                        return self._generate_signal(
                            'BUY',
                            current_price,
                            zone['bottom'],
                            zone['top'] + (zone['top'] - zone['bottom']) * 2,
                            "Bullish pullback to demand zone in uptrend"
                        )
        
        elif direction == 'bearish':
            # Find recent supply zone being tested
            for zone in zones['supply_zones']:
                if zone['bottom'] <= current_price <= zone['top']:
                    last_candle = candles_5m[-1]
                    if last_candle.close < last_candle.open:
                        return self._generate_signal(
                            'SELL',
                            current_price,
                            zone['top'],
                            zone['bottom'] - (zone['top'] - zone['bottom']) * 2,
                            "Bearish pullback to supply zone in downtrend"
                        )
        
        return None
    
    def _generate_signal(self, direction, entry, stop_loss, take_profit, reason):
        return {
            'direction': direction,
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reason': reason,
            'playbook': self.name
        }

class LiquiditySweepReversalPlaybook(SignalPlaybook):
    """Playbook B: Liquidity Sweep Reversal"""
    
    def __init__(self):
        super().__init__("Liquidity Sweep Reversal")
    
    def evaluate(self, candles_4h, candles_1h, candles_15m, candles_5m) -> Optional[Dict]:
        # Detect liquidity pools
        swings_1h = self.structure_analyzer.detect_swing_points(candles_1h)
        liquidity_pools = self.liquidity_detector.detect_liquidity_pools(candles_1h, swings_1h)
        
        # Check for recent sweep
        sweeps = self.liquidity_detector.detect_liquidity_sweep(candles_5m, liquidity_pools)
        
        if not sweeps['sweeps']:
            return None
        
        # Get most recent sweep
        latest_sweep = sweeps['sweeps'][-1]
        current_price = candles_5m[-1].close
        
        # Check for structure shift (BOS/CHOCH)
        bos_choch = self.structure_analyzer.detect_bos_choch(candles_15m, 
            self.structure_analyzer.detect_swing_points(candles_15m))
        
        if latest_sweep['type'] == 'bullish_sweep':
            # Price swept lows and closed above - bullish reversal
            return {
                'direction': 'BUY',
                'entry': current_price,
                'stop_loss': latest_sweep['sweep_low'] - (current_price - latest_sweep['sweep_low']) * 0.5,
                'take_profit': current_price + (current_price - latest_sweep['sweep_low']) * 2,
                'reason': f"Bullish liquidity sweep at {latest_sweep['pool_price']}",
                'playbook': self.name
            }
        
        elif latest_sweep['type'] == 'bearish_sweep':
            # Price swept highs and closed below - bearish reversal
            return {
                'direction': 'SELL',
                'entry': current_price,
                'stop_loss': latest_sweep['sweep_high'] + (latest_sweep['sweep_high'] - current_price) * 0.5,
                'take_profit': current_price - (latest_sweep['sweep_high'] - current_price) * 2,
                'reason': f"Bearish liquidity sweep at {latest_sweep['pool_price']}",
                'playbook': self.name
            }
        
        return None

class BreakoutRetestPlaybook(SignalPlaybook):
    """Playbook C: Breakout → Retest"""
    
    def __init__(self):
        super().__init__("Breakout → Retest")
    
    def evaluate(self, candles_4h, candles_1h, candles_15m, candles_5m) -> Optional[Dict]:
        # Detect if we had consolidation
        range_data = self.consolidation_detector.detect_range(candles_15m[:-5])  # Exclude recent
        
        if not range_data['is_ranging']:
            return None
        
        # Check for breakout
        current_price = candles_5m[-1].close
        range_high = range_data['range_high']
        range_low = range_data['range_low']
        
        # Bullish breakout and retest
        if current_price > range_high:
            # Check if price pulled back to retest the range high
            recent_lows = [c.low for c in candles_5m[-5:]]
            if any(low <= range_high * 1.002 for low in recent_lows):  # Within 0.2% of range high
                return {
                    'direction': 'BUY',
                    'entry': current_price,
                    'stop_loss': range_high - (range_high - range_low) * 0.3,
                    'take_profit': current_price + (range_high - range_low) * 1.5,
                    'reason': "Bullish breakout with successful retest",
                    'playbook': self.name
                }
        
        # Bearish breakout and retest
        if current_price < range_low:
            recent_highs = [c.high for c in candles_5m[-5:]]
            if any(high >= range_low * 0.998 for high in recent_highs):
                return {
                    'direction': 'SELL',
                    'entry': current_price,
                    'stop_loss': range_low + (range_high - range_low) * 0.3,
                    'take_profit': current_price - (range_high - range_low) * 1.5,
                    'reason': "Bearish breakout with successful retest",
                    'playbook': self.name
                }
        
        return None

class RangeMeanReversionPlaybook(SignalPlaybook):
    """Playbook D: Range Mean Reversion"""
    
    def __init__(self):
        super().__init__("Range Mean Reversion")
    
    def evaluate(self, candles_4h, candles_1h, candles_15m, candles_5m) -> Optional[Dict]:
        # Confirm we're in a range
        range_data = self.consolidation_detector.detect_range(candles_15m)
        
        if not range_data['is_ranging']:
            return None
        
        current_price = candles_5m[-1].close
        range_high = range_data['range_high']
        range_low = range_data['range_low']
        range_mid = (range_high + range_low) / 2
        
        # Trade from extremes back to mean
        touch_threshold = (range_high - range_low) * 0.15  # Within 15% of boundary
        
        # At range high - sell
        if abs(current_price - range_high) <= touch_threshold:
            return {
                'direction': 'SELL',
                'entry': current_price,
                'stop_loss': range_high + (range_high - range_low) * 0.2,
                'take_profit': range_mid,
                'reason': "Range high rejection - mean reversion",
                'playbook': self.name
            }
        
        # At range low - buy
        if abs(current_price - range_low) <= touch_threshold:
            return {
                'direction': 'BUY',
                'entry': current_price,
                'stop_loss': range_low - (range_high - range_low) * 0.2,
                'take_profit': range_mid,
                'reason': "Range low support - mean reversion",
                'playbook': self.name
            }
        
        return None