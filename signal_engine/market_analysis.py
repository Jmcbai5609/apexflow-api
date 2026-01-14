from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
from market_data import Candle

class MarketStructureAnalyzer:
    """Analyzes market structure: trends, swings, BOS, CHOCH"""
    
    def __init__(self):
        self.swing_threshold = 0.0015  # 15 pips for major pairs
    
    def detect_swing_points(self, candles: List[Candle]) -> Dict:
        """Detect swing highs and swing lows"""
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(candles) - 2):
            # Swing High: higher than 2 candles on each side
            if (candles[i].high > candles[i-1].high and 
                candles[i].high > candles[i-2].high and
                candles[i].high > candles[i+1].high and 
                candles[i].high > candles[i+2].high):
                swing_highs.append({
                    'index': i,
                    'price': candles[i].high,
                    'timestamp': candles[i].timestamp
                })
            
            # Swing Low: lower than 2 candles on each side
            if (candles[i].low < candles[i-1].low and 
                candles[i].low < candles[i-2].low and
                candles[i].low < candles[i+1].low and 
                candles[i].low < candles[i+2].low):
                swing_lows.append({
                    'index': i,
                    'price': candles[i].low,
                    'timestamp': candles[i].timestamp
                })
        
        return {
            'swing_highs': swing_highs,
            'swing_lows': swing_lows
        }
    
    def detect_bos_choch(self, candles: List[Candle], swings: Dict) -> Dict:
        """Detect Break of Structure (BOS) and Change of Character (CHOCH)"""
        events = []
        
        swing_highs = swings['swing_highs']
        swing_lows = swings['swing_lows']
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {'events': []}
        
        # Check for BOS (continuation of trend)
        # Bullish BOS: price breaks above recent swing high
        for i, high in enumerate(swing_highs[:-1]):
            next_highs = [h for h in swing_highs[i+1:] if h['price'] > high['price']]
            if next_highs:
                events.append({
                    'type': 'BOS',
                    'direction': 'bullish',
                    'price': high['price'],
                    'timestamp': next_highs[0]['timestamp']
                })
        
        # Bearish BOS: price breaks below recent swing low
        for i, low in enumerate(swing_lows[:-1]):
            next_lows = [l for l in swing_lows[i+1:] if l['price'] < low['price']]
            if next_lows:
                events.append({
                    'type': 'BOS',
                    'direction': 'bearish',
                    'price': low['price'],
                    'timestamp': next_lows[0]['timestamp']
                })
        
        # Check for CHOCH (reversal pattern)
        # After bullish trend, failing to make new high = CHOCH
        if len(swing_highs) >= 3:
            recent_highs = swing_highs[-3:]
            if recent_highs[1]['price'] > recent_highs[0]['price'] and recent_highs[2]['price'] < recent_highs[1]['price']:
                events.append({
                    'type': 'CHOCH',
                    'direction': 'bearish',
                    'price': recent_highs[1]['price'],
                    'timestamp': recent_highs[2]['timestamp']
                })
        
        return {'events': events}
    
    def calculate_trend(self, candles: List[Candle]) -> Dict:
        """Calculate overall trend direction and strength"""
        if len(candles) < 20:
            return {'direction': 'neutral', 'strength': 0}
        
        # Use EMAs for trend detection
        closes = [c.close for c in candles]
        
        # Simple moving averages
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else sma_20
        
        current_price = closes[-1]
        
        # Trend determination
        if current_price > sma_20 > sma_50:
            direction = 'bullish'
            strength = min(100, ((current_price - sma_20) / sma_20 * 10000))  # Normalize
        elif current_price < sma_20 < sma_50:
            direction = 'bearish'
            strength = min(100, ((sma_20 - current_price) / sma_20 * 10000))
        else:
            direction = 'neutral'
            strength = 0
        
        return {
            'direction': direction,
            'strength': strength,
            'sma_20': sma_20,
            'sma_50': sma_50
        }

class LiquidityDetector:
    """Detects liquidity zones and sweep patterns"""
    
    def detect_liquidity_pools(self, candles: List[Candle], swings: Dict) -> Dict:
        """Identify areas where stops are likely accumulated"""
        pools = []
        
        # Liquidity typically sits above swing highs and below swing lows
        for high in swings['swing_highs'][-5:]:  # Recent 5 highs
            pools.append({
                'type': 'above',
                'price': high['price'],
                'timestamp': high['timestamp'],
                'strength': 'high'  # Recent highs = strong liquidity
            })
        
        for low in swings['swing_lows'][-5:]:  # Recent 5 lows
            pools.append({
                'type': 'below',
                'price': low['price'],
                'timestamp': low['timestamp'],
                'strength': 'high'
            })
        
        return {'pools': pools}
    
    def detect_liquidity_sweep(self, candles: List[Candle], liquidity_pools: Dict) -> Dict:
        """Detect when price sweeps liquidity then reverses"""
        sweeps = []
        
        if len(candles) < 3:
            return {'sweeps': []}
        
        for pool in liquidity_pools['pools']:
            # Check recent candles for sweep
            for i in range(len(candles) - 3, len(candles)):
                candle = candles[i]
                
                if pool['type'] == 'above':
                    # Bullish sweep: wick above high, close below
                    if candle.high > pool['price'] and candle.close < pool['price']:
                        sweeps.append({
                            'type': 'bearish_sweep',
                            'pool_price': pool['price'],
                            'sweep_high': candle.high,
                            'close': candle.close,
                            'timestamp': candle.timestamp
                        })
                
                elif pool['type'] == 'below':
                    # Bearish sweep: wick below low, close above
                    if candle.low < pool['price'] and candle.close > pool['price']:
                        sweeps.append({
                            'type': 'bullish_sweep',
                            'pool_price': pool['price'],
                            'sweep_low': candle.low,
                            'close': candle.close,
                            'timestamp': candle.timestamp
                        })
        
        return {'sweeps': sweeps}

class ZoneMapper:
    """Maps supply/demand zones and Fair Value Gaps"""
    
    def detect_supply_demand_zones(self, candles: List[Candle]) -> Dict:
        """Identify supply and demand zones"""
        demand_zones = []
        supply_zones = []
        
        for i in range(1, len(candles) - 1):
            prev_candle = candles[i-1]
            curr_candle = candles[i]
            next_candle = candles[i+1]
            
            # Demand zone: strong move up from a small base
            if (curr_candle.close > curr_candle.open and  # Bullish candle
                (curr_candle.high - curr_candle.low) > 2 * abs(prev_candle.close - prev_candle.open) and  # Strong move
                abs(prev_candle.close - prev_candle.open) < (curr_candle.high - curr_candle.low) * 0.3):  # Small base
                demand_zones.append({
                    'top': prev_candle.high,
                    'bottom': prev_candle.low,
                    'timestamp': curr_candle.timestamp,
                    'strength': 'strong' if (curr_candle.close - curr_candle.open) > (curr_candle.high - curr_candle.low) * 0.7 else 'medium'
                })
            
            # Supply zone: strong move down from a small top
            if (curr_candle.close < curr_candle.open and  # Bearish candle
                (curr_candle.high - curr_candle.low) > 2 * abs(prev_candle.close - prev_candle.open) and
                abs(prev_candle.close - prev_candle.open) < (curr_candle.high - curr_candle.low) * 0.3):
                supply_zones.append({
                    'top': prev_candle.high,
                    'bottom': prev_candle.low,
                    'timestamp': curr_candle.timestamp,
                    'strength': 'strong' if (curr_candle.open - curr_candle.close) > (curr_candle.high - curr_candle.low) * 0.7 else 'medium'
                })
        
        return {
            'demand_zones': demand_zones[-5:],  # Keep recent 5
            'supply_zones': supply_zones[-5:]
        }
    
    def detect_fair_value_gaps(self, candles: List[Candle]) -> Dict:
        """Detect Fair Value Gaps (FVG) - imbalances in price"""
        fvgs = []
        
        for i in range(1, len(candles) - 1):
            candle1 = candles[i-1]
            candle2 = candles[i]
            candle3 = candles[i+1]
            
            # Bullish FVG: gap between candle1 high and candle3 low
            if candle1.high < candle3.low:
                fvgs.append({
                    'type': 'bullish',
                    'top': candle3.low,
                    'bottom': candle1.high,
                    'timestamp': candle3.timestamp
                })
            
            # Bearish FVG: gap between candle3 high and candle1 low
            if candle3.high < candle1.low:
                fvgs.append({
                    'type': 'bearish',
                    'top': candle1.low,
                    'bottom': candle3.high,
                    'timestamp': candle3.timestamp
                })
        
        return {'fvgs': fvgs[-10:]}  # Keep recent 10

class ConsolidationDetector:
    """Detects consolidation and range patterns"""
    
    def detect_range(self, candles: List[Candle]) -> Dict:
        """Detect if price is ranging"""
        if len(candles) < 20:
            return {'is_ranging': False}
        
        recent_candles = candles[-20:]
        highs = [c.high for c in recent_candles]
        lows = [c.low for c in recent_candles]
        
        range_high = max(highs)
        range_low = min(lows)
        range_size = range_high - range_low
        
        # Calculate how many times price touched the boundaries
        touch_threshold = range_size * 0.1  # Within 10% of boundary
        
        upper_touches = sum(1 for h in highs if abs(h - range_high) <= touch_threshold)
        lower_touches = sum(1 for l in lows if abs(l - range_low) <= touch_threshold)
        
        # Range criteria: multiple touches on both sides, limited breakouts
        is_ranging = (upper_touches >= 2 and lower_touches >= 2 and 
                     range_size < np.mean([c.high - c.low for c in recent_candles]) * 15)
        
        return {
            'is_ranging': is_ranging,
            'range_high': range_high if is_ranging else None,
            'range_low': range_low if is_ranging else None,
            'upper_touches': upper_touches,
            'lower_touches': lower_touches
        }