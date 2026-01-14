from typing import Dict, List
from market_data import Candle
from .market_analysis import (
    MarketStructureAnalyzer, 
    LiquidityDetector, 
    ZoneMapper, 
    ConsolidationDetector
)

class MarketRegimeController:
    """Classifies market into: Trend / Reversal Zone / Breakout / Range"""
    
    def __init__(self):
        self.structure_analyzer = MarketStructureAnalyzer()
        self.consolidation_detector = ConsolidationDetector()
    
    def classify_regime(self, 
                       candles_4h: List[Candle], 
                       candles_1h: List[Candle], 
                       candles_15m: List[Candle]) -> Dict:
        """Determine current market regime"""
        
        # Analyze trend on higher timeframes
        trend_4h = self.structure_analyzer.calculate_trend(candles_4h)
        trend_1h = self.structure_analyzer.calculate_trend(candles_1h)
        
        # Check for ranging
        range_detection = self.consolidation_detector.detect_range(candles_15m)
        
        # Swing analysis
        swings_1h = self.structure_analyzer.detect_swing_points(candles_1h)
        bos_choch = self.structure_analyzer.detect_bos_choch(candles_1h, swings_1h)
        
        # Regime classification logic
        regime = self._determine_regime(
            trend_4h, 
            trend_1h, 
            range_detection, 
            bos_choch
        )
        
        return {
            'regime': regime['name'],
            'confidence': regime['confidence'],
            'allowed_playbooks': regime['allowed_playbooks'],
            'context': {
                'trend_4h': trend_4h['direction'],
                'trend_1h': trend_1h['direction'],
                'is_ranging': range_detection['is_ranging']
            }
        }
    
    def _determine_regime(self, trend_4h: Dict, trend_1h: Dict, 
                         range_detection: Dict, bos_choch: Dict) -> Dict:
        """Internal logic to determine regime"""
        
        # RANGE: Clear ranging pattern on lower timeframe
        if range_detection['is_ranging'] and range_detection['upper_touches'] >= 2:
            return {
                'name': 'Range',
                'confidence': 85,
                'allowed_playbooks': ['range_mean_reversion']
            }
        
        # TREND: Strong alignment across timeframes
        if (trend_4h['direction'] == trend_1h['direction'] and 
            trend_4h['direction'] != 'neutral' and
            trend_4h['strength'] > 40):
            return {
                'name': 'Trend',
                'confidence': 90,
                'allowed_playbooks': ['trend_continuation']
            }
        
        # REVERSAL ZONE: Conflicting trends or recent CHOCH
        choch_events = [e for e in bos_choch['events'] if e['type'] == 'CHOCH']
        if (trend_4h['direction'] != trend_1h['direction'] or 
            len(choch_events) > 0):
            return {
                'name': 'Reversal Zone',
                'confidence': 75,
                'allowed_playbooks': ['liquidity_sweep_reversal']
            }
        
        # BREAKOUT: Recent consolidation with directional pressure
        if (not range_detection['is_ranging'] and 
            trend_1h['strength'] > 50 and
            len(bos_choch['events']) > 0):
            return {
                'name': 'Breakout Environment',
                'confidence': 80,
                'allowed_playbooks': ['breakout_retest']
            }
        
        # DEFAULT: Uncertain regime - no signals allowed
        return {
            'name': 'Uncertain',
            'confidence': 0,
            'allowed_playbooks': []
        }