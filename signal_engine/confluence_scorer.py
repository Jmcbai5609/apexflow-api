from typing import Dict, List
from market_data import Candle
from .market_analysis import MarketStructureAnalyzer, LiquidityDetector, ZoneMapper
from datetime import datetime

class ConfluenceScorer:
    """Calculates confluence score for signals (must be >= 85 to trigger)"""
    
    def __init__(self):
        self.min_score = 85
        self.structure_analyzer = MarketStructureAnalyzer()
    
    def calculate_score(self, 
                       signal: Dict,
                       candles_4h: List[Candle],
                       candles_1h: List[Candle],
                       candles_15m: List[Candle],
                       candles_5m: List[Candle],
                       regime: Dict,
                       session_quality: str) -> Dict:
        """Calculate comprehensive confluence score"""
        
        score_breakdown = {}
        total_score = 0
        
        # 1. HTF Alignment (25 points)
        htf_score = self._score_htf_alignment(candles_4h, candles_1h, signal['direction'])
        score_breakdown['htf_alignment'] = htf_score
        total_score += htf_score
        
        # 2. Market Regime Clarity (20 points)
        regime_score = self._score_regime_clarity(regime)
        score_breakdown['regime_clarity'] = regime_score
        total_score += regime_score
        
        # 3. Liquidity Location (15 points)
        liquidity_score = self._score_liquidity_location(candles_1h, candles_5m, signal)
        score_breakdown['liquidity_location'] = liquidity_score
        total_score += liquidity_score
        
        # 4. Structure Quality (15 points)
        structure_score = self._score_structure_quality(candles_1h, candles_15m, signal['direction'])
        score_breakdown['structure_quality'] = structure_score
        total_score += structure_score
        
        # 5. Playbook Validity (10 points)
        playbook_score = self._score_playbook_validity(signal['playbook'], regime)
        score_breakdown['playbook_validity'] = playbook_score
        total_score += playbook_score
        
        # 6. Confirmation Candle Strength (10 points)
        confirmation_score = self._score_confirmation_candle(candles_5m[-1], signal['direction'])
        score_breakdown['confirmation_strength'] = confirmation_score
        total_score += confirmation_score
        
        # 7. Session Quality (5 points)
        session_score = self._score_session(session_quality)
        score_breakdown['session_quality'] = session_score
        total_score += session_score
        
        return {
            'total_score': round(total_score, 1),
            'breakdown': score_breakdown,
            'passes_threshold': total_score >= self.min_score
        }
    
    def _score_htf_alignment(self, candles_4h: List[Candle], candles_1h: List[Candle], 
                            direction: str) -> float:
        """Score based on higher timeframe trend alignment"""
        trend_4h = self.structure_analyzer.calculate_trend(candles_4h)
        trend_1h = self.structure_analyzer.calculate_trend(candles_1h)
        
        signal_direction = 'bullish' if direction == 'BUY' else 'bearish'
        
        # Perfect alignment
        if trend_4h['direction'] == signal_direction and trend_1h['direction'] == signal_direction:
            return 25.0
        
        # 1H aligned, 4H neutral
        if trend_1h['direction'] == signal_direction and trend_4h['direction'] == 'neutral':
            return 18.0
        
        # 1H aligned only
        if trend_1h['direction'] == signal_direction:
            return 12.0
        
        return 0.0
    
    def _score_regime_clarity(self, regime: Dict) -> float:
        """Score based on how clear the market regime is"""
        confidence = regime.get('confidence', 0)
        
        if confidence >= 85:
            return 20.0
        elif confidence >= 75:
            return 15.0
        elif confidence >= 65:
            return 10.0
        
        return 0.0
    
    def _score_liquidity_location(self, candles_1h: List[Candle], 
                                 candles_5m: List[Candle], signal: Dict) -> float:
        """Score based on proximity to key liquidity zones"""
        from .market_analysis import LiquidityDetector
        
        liquidity_det = LiquidityDetector()
        swings_1h = self.structure_analyzer.detect_swing_points(candles_1h)
        pools = liquidity_det.detect_liquidity_pools(candles_1h, swings_1h)
        
        current_price = candles_5m[-1].close
        
        # Check if we're near a significant liquidity zone
        for pool in pools['pools']:
            price_diff = abs(current_price - pool['price']) / current_price
            
            if price_diff < 0.002:  # Within 0.2%
                return 15.0
            elif price_diff < 0.005:  # Within 0.5%
                return 10.0
        
        return 5.0
    
    def _score_structure_quality(self, candles_1h: List[Candle], 
                                candles_15m: List[Candle], direction: str) -> float:
        """Score based on quality of market structure"""
        swings_1h = self.structure_analyzer.detect_swing_points(candles_1h)
        bos_choch = self.structure_analyzer.detect_bos_choch(candles_1h, swings_1h)
        
        signal_direction = 'bullish' if direction == 'BUY' else 'bearish'
        
        # Check for recent BOS in our direction
        recent_bos = [e for e in bos_choch['events'] 
                     if e['type'] == 'BOS' and e['direction'] == signal_direction]
        
        if recent_bos:
            return 15.0
        
        # Check swing structure
        if len(swings_1h['swing_highs']) >= 3 and len(swings_1h['swing_lows']) >= 3:
            return 10.0
        
        return 5.0
    
    def _score_playbook_validity(self, playbook: str, regime: Dict) -> float:
        """Score based on playbook appropriateness for regime"""
        allowed_playbooks = regime.get('allowed_playbooks', [])
        
        # Map playbook names to regime playbook keys
        playbook_mapping = {
            'Trend Continuation': 'trend_continuation',
            'Liquidity Sweep Reversal': 'liquidity_sweep_reversal',
            'Breakout → Retest': 'breakout_retest',
            'Range Mean Reversion': 'range_mean_reversion'
        }
        
        playbook_key = playbook_mapping.get(playbook)
        
        if playbook_key in allowed_playbooks:
            return 10.0
        
        return 0.0
    
    def _score_confirmation_candle(self, last_candle: Candle, direction: str) -> float:
        """Score based on confirmation candle strength"""
        candle_body = abs(last_candle.close - last_candle.open)
        candle_range = last_candle.high - last_candle.low
        
        if candle_range == 0:
            return 0.0
        
        body_ratio = candle_body / candle_range
        
        # Check if candle agrees with signal direction
        is_bullish_candle = last_candle.close > last_candle.open
        signal_is_buy = direction == 'BUY'
        
        if is_bullish_candle == signal_is_buy:
            # Strong confirmation: large body
            if body_ratio > 0.7:
                return 10.0
            elif body_ratio > 0.5:
                return 7.0
            else:
                return 4.0
        
        return 0.0
    
    def _score_session(self, session_quality: str) -> float:
        """Score based on trading session"""
        if session_quality == 'overlap':  # London-NY overlap
            return 5.0
        elif session_quality in ['london', 'newyork']:
            return 4.0
        elif session_quality == 'asian':
            return 2.0
        
        return 1.0