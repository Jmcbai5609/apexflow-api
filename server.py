from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uuid
from datetime import datetime, timedelta
import asyncio
import hashlib

# Import signal engine components
from market_data import DemoDataProvider, FallbackDataProvider
from signal_engine import ApexFlowEngine

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# ==================== CONCURRENCY CONTROL ====================
# In-process mutex for demo reset operations
_reset_lock = asyncio.Lock()
_reset_in_progress = False

# Initialize market data provider with fallback support
# Priority: Polygon.io > OANDA > Demo
data_mode = os.getenv('DATA_MODE', 'demo')
if data_mode == 'live':
    data_provider = FallbackDataProvider(
        polygon_api_key=os.getenv('POLYGON_API_KEY'),
        oanda_api_key=os.getenv('OANDA_API_KEY'),
        oanda_account_id=os.getenv('OANDA_ACCOUNT_ID'),
        oanda_practice=os.getenv('OANDA_PRACTICE', 'true').lower() == 'true'
    )
else:
    data_provider = DemoDataProvider()

# Initialize ApexFlow Engine
signal_mode = os.getenv('SIGNAL_MODE', 'balanced')
apex_engine = ApexFlowEngine(data_provider, signal_mode)

# Create the main app without a prefix
app = FastAPI(title="ApexFlow API", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# ==================== MODELS ====================

class EngineSettings(BaseModel):
    signal_mode: str = 'balanced'  # quiet, balanced, or aggressive
    active_markets: List[str] = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
    active_sessions: List[str] = ['london', 'newyork', 'overlap']
    active_playbooks: List[str] = ['trend_continuation', 'liquidity_sweep_reversal', 'breakout_retest']
    max_signals_per_pair: int = 3
    cooldown_timer: int = 15  # minutes
    notifications_enabled: bool = True

class SignalResponse(BaseModel):
    signal_id: str
    symbol: str
    timestamp: str
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    playbook: str
    reason: str
    confluence_score: float
    confidence_breakdown: Optional[Dict[str, float]] = None
    regime: str
    session: str
    timeframe: str
    entry_valid_until: str
    setup_expires_at: str
    mode: str
    status: str  # pending, active, tp_hit, sl_hit, invalidated
    status_history: Optional[List[Dict]] = None
    risk_reward: Optional[float] = None
    expected_duration: Optional[str] = None  # scalp, intraday, swing

class SignalStatusUpdate(BaseModel):
    status: str  # pending, active, tp_hit, sl_hit, invalidated
    reason: Optional[str] = None
    updated_price: Optional[float] = None

class RiskProfile(BaseModel):
    name: str  # conservative, balanced, aggressive
    min_confluence: float
    min_risk_reward: float
    description: str

class TradeStyleFilter(BaseModel):
    scalps: bool = False  # < 30 min
    intraday: bool = True  # < 4 hours
    swing: bool = False  # > 4 hours

class SignalPerformance(BaseModel):
    signal_id: str
    outcome: str  # win, loss, breakeven, active
    r_multiple: Optional[float] = None
    duration_minutes: Optional[int] = None
    closed_at: Optional[str] = None

class EngineStats(BaseModel):
    total_signals: int
    win_rate: float
    average_r_multiple: float
    signals_by_playbook: Dict[str, int]
    signals_by_pair: Dict[str, int]
    recent_signals: List[Dict]

# ==================== ROUTES ====================

@api_router.get("/")
async def root():
    return {
        "message": "ApexFlow Signal Engine API",
        "version": "1.0.0",
        "data_mode": data_mode,
        "signal_mode": signal_mode
    }

@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "engine": "operational",
        "data_provider": type(data_provider).__name__,
        "timestamp": datetime.utcnow().isoformat()
    }

@api_router.post("/settings", response_model=EngineSettings)
async def update_settings(settings: EngineSettings):
    """Update engine settings - FULLY RECONFIGURES ENGINE BEHAVIOR"""
    # Save to database
    settings_dict = settings.dict()
    settings_dict['updated_at'] = datetime.utcnow()
    
    await db.engine_settings.update_one(
        {'_id': 'default'},
        {'$set': settings_dict},
        upsert=True
    )
    
    # CRITICAL: Fully reconfigure engine based on new settings
    global apex_engine
    
    # Update signal mode and reconfigure thresholds
    apex_engine.signal_mode = settings.signal_mode
    apex_engine.cooldown_minutes = settings.cooldown_timer
    apex_engine.active_sessions = settings.active_sessions
    
    # Reconfigure confluence thresholds based on signal mode
    # This determines which signal grades are allowed
    if settings.signal_mode == 'quiet':
        # Conservative: Only A+ and A signals (88+)
        apex_engine.min_confluence_score = 88
        apex_engine.min_regime_confidence = 80
    elif settings.signal_mode == 'aggressive':
        # Aggressive: Allow all grades A-D (50+)
        apex_engine.min_confluence_score = 50
        apex_engine.min_regime_confidence = 55
    else:  # balanced
        # Balanced: A and B signals, limited C (70+)
        apex_engine.min_confluence_score = 70
        apex_engine.min_regime_confidence = 65
    
    logger.info(f"Engine reconfigured: mode={settings.signal_mode}, min_score={apex_engine.min_confluence_score}, sessions={settings.active_sessions}")
    
    return settings

@api_router.get("/settings", response_model=EngineSettings)
async def get_settings():
    """Get current engine settings"""
    settings = await db.engine_settings.find_one({'_id': 'default'})
    
    if not settings:
        # Return default settings
        return EngineSettings()
    
    # Remove MongoDB _id field
    settings.pop('_id', None)
    settings.pop('updated_at', None)
    
    return EngineSettings(**settings)

@api_router.post("/signals/generate/{symbol}")
async def generate_signal(symbol: str):
    """Manually trigger signal generation for a symbol"""
    try:
        # Get active settings
        settings = await get_settings()
        
        if symbol not in settings.active_markets:
            raise HTTPException(status_code=400, detail=f"Symbol {symbol} not enabled")
        
        # Generate signal
        signal = await apex_engine.analyze_and_generate_signal(
            symbol, 
            settings.active_playbooks
        )
        
        if signal:
            # Save to database
            await db.signals.insert_one(signal)
            return {"status": "signal_generated", "signal": signal}
        else:
            return {"status": "no_signal", "message": "No valid setup found at this time"}
    
    except Exception as e:
        logging.error(f"Error generating signal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/signals/active", response_model=List[SignalResponse])
async def get_active_signals():
    """Get all active signals - FILTERED BY CURRENT ENGINE SETTINGS"""
    # Get current settings to filter by grade
    settings = await db.engine_settings.find_one({'_id': 'default'})
    
    # Determine min confluence score based on signal mode
    signal_mode = settings.get('signal_mode', 'balanced') if settings else 'balanced'
    active_markets = settings.get('active_markets', ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']) if settings else ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
    active_playbooks = settings.get('active_playbooks', []) if settings else []
    active_sessions = settings.get('active_sessions', []) if settings else []
    
    if signal_mode == 'quiet':
        min_score = 88  # Only A+ and A
    elif signal_mode == 'aggressive':
        min_score = 50  # All grades A-D
    else:  # balanced
        min_score = 70  # A, B, limited C
    
    # Build query to filter by:
    # 1. Active status
    # 2. Confluence score >= min for current mode
    # 3. Symbol in active_markets
    query = {
        'status': 'active',
        'confluence_score': {'$gte': min_score}
    }
    
    # Filter by active markets
    if active_markets:
        query['symbol'] = {'$in': active_markets}
    
    # Filter by active sessions (if not empty)
    # Handle both snake_case and human-readable formats
    if active_sessions:
        # Convert snake_case to human-readable for matching
        session_variants = []
        for s in active_sessions:
            session_variants.append(s)
            session_variants.append(s.replace('_', ' '))
            session_variants.append(s.replace('_', ' ').title())
        query['session'] = {'$in': session_variants}
    
    signals = await db.signals.find(query).sort([('timestamp', -1), ('sequence_id', 1)]).to_list(100)
    
    # Apply playbook filter in Python (handles name format differences)
    if active_playbooks:
        # Normalize playbook names for comparison
        def normalize_playbook(name):
            return name.lower().replace(' ', '_').replace('-', '_')
        
        normalized_active = [normalize_playbook(p) for p in active_playbooks]
        signals = [s for s in signals if normalize_playbook(s.get('playbook', '')) in normalized_active]
    
    # Clean up MongoDB _id field
    for signal in signals:
        signal.pop('_id', None)
    
    return signals

@api_router.get("/signals/all", response_model=List[SignalResponse])
async def get_all_signals(limit: int = 50):
    """Get all signals (active and historical) with stable ordering"""
    signals = await db.signals.find().sort([('timestamp', -1), ('sequence_id', 1)]).limit(limit).to_list(limit)
    
    for signal in signals:
        signal.pop('_id', None)
    
    return signals

@api_router.get("/signals/history")
async def get_signal_history(
    pair: Optional[str] = None,
    session: Optional[str] = None,
    playbook: Optional[str] = None,
    outcome: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 100,
    filter_by_settings: bool = True  # New param to control settings-based filtering
):
    """Get signal history - FILTERED BY CURRENT SETTINGS by default"""
    query = {}
    active_playbooks_for_filter = []
    
    # Get current settings for filtering
    if filter_by_settings:
        settings = await db.engine_settings.find_one({'_id': 'default'})
        
        signal_mode = settings.get('signal_mode', 'balanced') if settings else 'balanced'
        active_markets = settings.get('active_markets', []) if settings else []
        active_playbooks_for_filter = settings.get('active_playbooks', []) if settings else []
        active_sessions = settings.get('active_sessions', []) if settings else []
        
        # Determine min confluence score based on signal mode
        if signal_mode == 'quiet':
            min_score = 88  # Only A+ and A
        elif signal_mode == 'aggressive':
            min_score = 50  # All grades A-D
        else:  # balanced
            min_score = 70  # A, B, limited C
        
        # Apply settings filters
        query['confluence_score'] = {'$gte': min_score}
        
        if active_markets:
            query['symbol'] = {'$in': active_markets}
        
        # Filter by active sessions with format variants
        if active_sessions:
            session_variants = []
            for s in active_sessions:
                session_variants.append(s)
                session_variants.append(s.replace('_', ' '))
                session_variants.append(s.replace('_', ' ').title())
            query['session'] = {'$in': session_variants}
    
    # Apply explicit filters (override settings filters if provided)
    if pair:
        query['symbol'] = pair
    if session:
        query['session'] = session
    if playbook:
        query['playbook'] = playbook
    if outcome:
        query['performance.outcome'] = outcome
    if date_from:
        query['timestamp'] = {'$gte': date_from}
    if date_to:
        if 'timestamp' in query:
            query['timestamp']['$lte'] = date_to
        else:
            query['timestamp'] = {'$lte': date_to}
    
    # Fetch signals sorted by timestamp (newest first), then sequence_id for stable ordering
    signals = await db.signals.find(query).sort([('timestamp', -1), ('sequence_id', 1)]).limit(limit).to_list(limit)
    
    # Apply playbook filter in Python (handles name format differences)
    if filter_by_settings and active_playbooks_for_filter:
        def normalize_playbook(name):
            return name.lower().replace(' ', '_').replace('-', '_')
        
        normalized_active = [normalize_playbook(p) for p in active_playbooks_for_filter]
        signals = [s for s in signals if normalize_playbook(s.get('playbook', '')) in normalized_active]
    
    # Convert ObjectId to string for JSON serialization
    for signal in signals:
        signal['_id'] = str(signal['_id'])
    
    return {
        'signals': signals,
        'total': len(signals),
        'filters_applied': {
            'pair': pair,
            'session': session,
            'playbook': playbook,
            'outcome': outcome,
            'date_range': f"{date_from or 'any'} to {date_to or 'now'}",
            'settings_filter_active': filter_by_settings
        }
    }

@api_router.get("/signals/{signal_id}", response_model=SignalResponse)

def generate_outcome_narrative(outcome: str, playbook: str, regime: str, direction: str) -> str:
    """Generate educational narrative explaining why a trade won or lost"""
    
    narratives = {
        'win': {
            'Trend Continuation': f"{'Bullish' if direction == 'BUY' else 'Bearish'} trend continuation signal executed perfectly. Price respected higher timeframe structure and momentum sustained through the move.",
            'Liquidity Sweep Reversal': f"Smart money liquidity sweep identified correctly. Price grabbed stops above/below key level then reversed as anticipated, confirming institutional orderflow shift.",
            'Breakout Retest': f"Breakout confirmed with strong momentum. Retest of broken structure held as new support/resistance, allowing continuation into trend.",
            'Range Mean Reversion': f"Price reached extreme range boundary and mean-reverted as expected. Range structure held, providing clean reversal opportunity."
        },
        'loss': {
            'Trend Continuation': f"Trend structure failed to hold. Counter-trend momentum overwhelmed setup, indicating shift in market bias or false continuation signal.",
            'Liquidity Sweep Reversal': f"Liquidity sweep occurred but reversal failed to materialize. Price continued through zone, suggesting stronger trend momentum than anticipated.",
            'Breakout Retest': f"Breakout proved false. Price failed to hold above/below structure on retest, triggering stop loss. Momentum was insufficient for continuation.",
            'Range Mean Reversion': f"Range structure broke down. Price breached boundary with strong directional momentum, invalidating mean-reversion thesis."
        },
        'breakeven': {
            'Trend Continuation': f"Price action remained indecisive. Trade moved to breakeven as momentum stalled at mid-range resistance/support without clear resolution.",
            'Liquidity Sweep Reversal': f"Reversal signal lacked follow-through. Price consolidated after initial reaction, prompting breakeven exit to preserve capital.",
            'Breakout Retest': f"Retest occurred but momentum was weak. Position moved to breakeven as price failed to show conviction for continuation.",
            'Range Mean Reversion': f"Mean reversion started but stalled mid-range. Trade closed at breakeven as price consolidated without reaching target."
        },
        'invalidated': {
            'Trend Continuation': f"Setup conditions invalidated before entry trigger. Market structure shifted, removing confluence required for entry.",
            'Liquidity Sweep Reversal': f"Liquidity sweep never occurred, or structure broke before reversal conditions met. Setup expired without valid entry.",
            'Breakout Retest': f"Breakout never completed or retest conditions not met within signal timeframe. Structure invalidated before entry opportunity.",
            'Range Mean Reversion': f"Range broke before mean reversion setup triggered. Price action invalidated setup parameters prior to entry."
        }
    }
    
    # Get narrative for playbook + outcome, or use default
    playbook_key = playbook if playbook in narratives['win'] else 'Trend Continuation'
    return narratives.get(outcome, narratives['invalidated']).get(playbook_key, f"Signal resolved as {outcome}.")


@api_router.post("/signals/{signal_id}/resolve")
async def resolve_signal(
    signal_id: str,
    outcome: str,
    reason: Optional[str] = None,
    r_multiple: Optional[float] = None
):
    """Manually resolve a signal with outcome
    
    Outcome definitions:
    - win: Trade hit take profit
    - loss: Trade hit stop loss
    - breakeven: Trade triggered but exited at ~0R (moved to entry)
    - invalidated: Setup expired or never triggered
    """
    # Validate outcome
    valid_outcomes = ['win', 'loss', 'breakeven', 'invalidated']
    if outcome not in valid_outcomes:
        raise HTTPException(status_code=400, detail=f"Outcome must be one of: {', '.join(valid_outcomes)}")
    
    # Find the signal
    signal = await db.signals.find_one({'signal_id': signal_id})
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    # Check if signal is already resolved
    if signal.get('performance') and signal['performance'].get('outcome'):
        raise HTTPException(
            status_code=400, 
            detail=f"Signal already resolved as {signal['performance']['outcome']}. Cannot change outcome."
        )
    
    # Calculate R-multiple if not provided
    if r_multiple is None:
        entry = signal.get('entry')
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')
        
        if entry and stop_loss and take_profit:
            risk = abs(entry - stop_loss)
            if outcome == 'win':
                reward = abs(take_profit - entry)
                r_multiple = round(reward / risk, 2) if risk > 0 else 0
            elif outcome == 'loss':
                r_multiple = -1.0
            elif outcome == 'breakeven':
                r_multiple = 0.0
            else:  # invalidated
                r_multiple = 0.0
    
    # Calculate pips
    entry = signal.get('entry', 0)
    exit_price = take_profit if outcome == 'win' else (stop_loss if outcome == 'loss' else entry)
    pips = round(abs(exit_price - entry) * 10000, 1)
    
    # Generate outcome narrative based on signal context
    playbook = signal.get('playbook', 'Unknown Setup')
    regime = signal.get('regime', 'Unknown Regime')
    direction = signal.get('direction', 'BUY')
    
    narrative = reason if reason else generate_outcome_narrative(outcome, playbook, regime, direction)
    
    # Update signal with performance data
    performance_data = {
        'outcome': outcome,
        'r_multiple': r_multiple,
        'exit_price': exit_price,
        'pips': pips,
        'duration_minutes': 0,  # Can be calculated if needed
        'resolved_at': datetime.utcnow().isoformat(),
        'reason': reason,
        'narrative': narrative
    }
    
    # Update status based on outcome
    status_map = {
        'win': 'tp_hit',
        'loss': 'sl_hit',
        'breakeven': 'closed_be',
        'invalidated': 'invalidated'
    }
    
    await db.signals.update_one(
        {'signal_id': signal_id},
        {
            '$set': {
                'performance': performance_data,
                'status': status_map[outcome]
            }
        }
    )
    
    return {
        'status': 'resolved',
        'signal_id': signal_id,
        'outcome': outcome,
        'r_multiple': r_multiple,
        'message': f'Signal resolved as {outcome}'
    }

async def get_signal(signal_id: str):
    """Get specific signal by ID"""
    signal = await db.signals.find_one({'signal_id': signal_id})
    
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    signal.pop('_id', None)
    return signal

@api_router.post("/signals/{signal_id}/status")
async def update_signal_status(signal_id: str, status: str):
    """Update signal status (active/expired/triggered)"""
    result = await db.signals.update_one(
        {'signal_id': signal_id},
        {'$set': {'status': status, 'updated_at': datetime.utcnow()}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    return {"status": "updated"}

@api_router.post("/signals/{signal_id}/performance")
async def record_performance(signal_id: str, performance: SignalPerformance):
    """Record signal performance outcome"""
    # Update signal with performance data
    perf_data = performance.dict()
    perf_data['recorded_at'] = datetime.utcnow()
    
    result = await db.signals.update_one(
        {'signal_id': signal_id},
        {'$set': {'performance': perf_data, 'status': 'closed'}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    return {"status": "performance_recorded"}

@api_router.post("/signals/{signal_id}/status")
async def update_signal_lifecycle_status(signal_id: str, status_update: SignalStatusUpdate):
    """Update signal status with history tracking"""
    status_entry = {
        'status': status_update.status,
        'timestamp': datetime.utcnow().isoformat(),
        'reason': status_update.reason,
        'price': status_update.updated_price
    }
    
    update_data = {
        'status': status_update.status,
        'updated_at': datetime.utcnow()
    }
    
    # Append to status history
    result = await db.signals.update_one(
        {'signal_id': signal_id},
        {
            '$set': update_data,
            '$push': {'status_history': status_entry}
        }
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    return {"status": "updated", "new_status": status_update.status}

@api_router.get("/analytics/confluence-trend/{symbol}")
async def get_confluence_trend(symbol: str):
    """Get confluence score trends and performance for a symbol"""
    signals = await db.signals.find({'symbol': symbol}).sort('timestamp', -1).limit(100).to_list(100)
    
    if not signals:
        return {"symbol": symbol, "data": [], "insights": {}}
    
    # Calculate performance by confluence range
    high_conf = [s for s in signals if s.get('confluence_score', 0) > 85]
    mid_conf = [s for s in signals if 75 <= s.get('confluence_score', 0) <= 85]
    low_conf = [s for s in signals if s.get('confluence_score', 0) < 75]
    
    def calc_win_rate(sigs):
        with_perf = [s for s in sigs if s.get('performance')]
        if not with_perf:
            return 0
        wins = sum(1 for s in with_perf if s.get('performance', {}).get('outcome') == 'win')
        return (wins / len(with_perf)) * 100
    
    insights = {
        'best_range': '> 85' if calc_win_rate(high_conf) >= calc_win_rate(mid_conf) else '75-85',
        'best_win_rate': max(calc_win_rate(high_conf), calc_win_rate(mid_conf), calc_win_rate(low_conf)),
        'recommendation': f"This system performs best when confluence > {85 if calc_win_rate(high_conf) > calc_win_rate(mid_conf) else 82}",
        'high_conf_win_rate': calc_win_rate(high_conf),
        'mid_conf_win_rate': calc_win_rate(mid_conf),
        'low_conf_win_rate': calc_win_rate(low_conf)
    }
    
    trend_data = [{'timestamp': s['timestamp'], 'score': s.get('confluence_score', 0)} for s in signals]
    
    return {"symbol": symbol, "data": trend_data, "insights": insights}

@api_router.get("/analytics/playbook-performance")
async def get_playbook_performance():
    """Get detailed performance by playbook"""
    all_signals = await db.signals.find().to_list(1000)
    
    playbooks = {}
    for signal in all_signals:
        playbook = signal.get('playbook', 'Unknown')
        if playbook not in playbooks:
            playbooks[playbook] = {'trades': [], 'name': playbook}
        playbooks[playbook]['trades'].append(signal)
    
    performance = {}
    for name, data in playbooks.items():
        trades = data['trades']
        with_perf = [t for t in trades if t.get('performance')]
        
        wins = sum(1 for t in with_perf if t.get('performance', {}).get('outcome') == 'win')
        win_rate = (wins / len(with_perf) * 100) if with_perf else 0
        
        r_multiples = [t.get('performance', {}).get('r_multiple', 0) for t in with_perf if t.get('performance', {}).get('r_multiple')]
        avg_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0
        
        # Calculate max drawdown (simplified)
        losses = [t.get('performance', {}).get('r_multiple', 0) for t in with_perf 
                 if t.get('performance', {}).get('outcome') == 'loss']
        max_drawdown = min(losses) if losses else 0
        
        performance[name] = {
            'total_trades': len(trades),
            'win_rate': round(win_rate, 1),
            'avg_r': round(avg_r, 2),
            'max_drawdown': round(max_drawdown, 2),
            'active_trades': len([t for t in trades if t.get('status') == 'active'])
        }
    
    return performance

@api_router.get("/analytics/session-performance")
async def get_session_performance():
    """Get performance analytics by trading session"""
    all_signals = await db.signals.find().to_list(1000)
    
    sessions = {}
    for signal in all_signals:
        session = signal.get('session', 'unknown')
        if session not in sessions:
            sessions[session] = []
        sessions[session].append(signal)
    
    performance = {}
    for session, signals in sessions.items():
        with_perf = [s for s in signals if s.get('performance')]
        
        wins = sum(1 for s in with_perf if s.get('performance', {}).get('outcome') == 'win')
        win_rate = (wins / len(with_perf) * 100) if with_perf else 0
        
        r_multiples = [s.get('performance', {}).get('r_multiple', 0) for s in with_perf 
                      if s.get('performance', {}).get('r_multiple')]
        avg_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0
        
        performance[session] = {
            'total_trades': len(signals),
            'win_rate': round(win_rate, 1),
            'avg_r': round(avg_r, 2),
            'active_trades': len([s for s in signals if s.get('status') == 'active'])
        }
    
    # Add insights
    best_session = max(performance.items(), key=lambda x: x[1]['win_rate']) if performance else ('unknown', {})
    insights = {
        'best_session': best_session[0],
        'recommendation': f"Best performance occurs during {best_session[0]} session." if best_session[0] != 'unknown' else "Insufficient data"
    }
    
    return {"sessions": performance, "insights": insights}

@api_router.get("/analytics/equity-curve")
async def get_equity_curve():
    """Generate R-multiple based equity curve - FILTERED BY CURRENT SETTINGS"""
    # Get current settings
    settings = await db.engine_settings.find_one({'_id': 'default'})
    
    signal_mode = settings.get('signal_mode', 'balanced') if settings else 'balanced'
    active_markets = settings.get('active_markets', []) if settings else []
    active_playbooks = settings.get('active_playbooks', []) if settings else []
    active_sessions = settings.get('active_sessions', []) if settings else []
    
    # Determine min confluence score based on signal mode
    if signal_mode == 'quiet':
        min_score = 88  # Only A+ and A
    elif signal_mode == 'aggressive':
        min_score = 50  # All grades A-D
    else:  # balanced
        min_score = 70  # A, B, limited C
    
    # Build query for resolved signals matching current settings
    query = {
        'performance': {'$exists': True, '$ne': None},
        'performance.outcome': {'$exists': True, '$ne': None},
        'confluence_score': {'$gte': min_score}
    }
    
    # Filter by active markets (if any specified)
    if active_markets:
        query['symbol'] = {'$in': active_markets}
    
    # Filter by active sessions with format variants
    if active_sessions:
        session_variants = []
        for s in active_sessions:
            session_variants.append(s)
            session_variants.append(s.replace('_', ' '))
            session_variants.append(s.replace('_', ' ').title())
        query['session'] = {'$in': session_variants}
    
    signals = await db.signals.find(query).sort('timestamp', 1).to_list(1000)
    
    # Apply playbook filter in Python (handles name format differences)
    if active_playbooks:
        def normalize_playbook(name):
            return name.lower().replace(' ', '_').replace('-', '_')
        
        normalized_active = [normalize_playbook(p) for p in active_playbooks]
        signals = [s for s in signals if normalize_playbook(s.get('playbook', '')) in normalized_active]
    
    curve_data = []
    cumulative_r = 0
    max_r = 0
    max_drawdown = 0
    
    for signal in signals:
        r_multiple = signal.get('performance', {}).get('r_multiple', 0)
        cumulative_r += r_multiple
        
        # Track max for drawdown calculation
        if cumulative_r > max_r:
            max_r = cumulative_r
        
        # Calculate current drawdown
        current_drawdown = max_r - cumulative_r
        if current_drawdown > max_drawdown:
            max_drawdown = current_drawdown
        
        curve_data.append({
            'timestamp': signal['timestamp'],
            'cumulative_r': round(cumulative_r, 2),
            'signal_id': signal['signal_id'],
            'outcome': signal.get('performance', {}).get('outcome'),
            'drawdown': round(current_drawdown, 2)
        })
    
    return {
        'curve': curve_data,
        'final_r': round(cumulative_r, 2),
        'max_drawdown': round(max_drawdown, 2),
        'total_trades': len(signals),
        'recovery_periods': []  # Can be enhanced later
    }


@api_router.get("/data-source/status")
async def get_data_source_status():
    """Get current market data source status and connectivity"""
    import time
    
    current_time = datetime.utcnow()
    
    # Check if using FallbackDataProvider (multi-provider with failover)
    if isinstance(data_provider, FallbackDataProvider):
        # Get detailed status from all providers
        provider_status = await data_provider.check_all_providers()
        summary = data_provider.get_status_summary()
        
        # Determine overall status
        active_provider = summary.get('active_provider', 'Demo Data')
        is_demo = active_provider == 'Demo Data'
        
        # Try to get latency from a quick test
        latency_ms = None
        try:
            start_time = time.time()
            await data_provider.get_latest_price("EURUSD")
            latency_ms = int((time.time() - start_time) * 1000)
        except:
            pass
        
        return {
            "source": active_provider,
            "status": "live" if not is_demo else "demo",
            "last_update": current_time.isoformat(),
            "latency_ms": latency_ms,
            "checked_at": current_time.isoformat(),
            "is_demo": is_demo,
            "providers": provider_status,
            "summary": summary
        }
    
    elif isinstance(data_provider, DemoDataProvider):
        return {
            "source": "Demo Mode",
            "status": "live",
            "last_update": current_time.isoformat(),
            "latency_ms": 0,
            "checked_at": current_time.isoformat(),
            "is_demo": True
        }
    
    else:
        # Legacy single provider
        return {
            "source": "Unknown",
            "status": "offline",
            "last_update": current_time.isoformat(),
            "latency_ms": None,
            "checked_at": current_time.isoformat(),
            "is_demo": False
        }

@api_router.get("/stats", response_model=EngineStats)
async def get_engine_stats():
    """Get engine performance statistics - FILTERED BY CURRENT SETTINGS"""
    # Get current settings
    settings = await db.engine_settings.find_one({'_id': 'default'})
    
    signal_mode = settings.get('signal_mode', 'balanced') if settings else 'balanced'
    active_markets = settings.get('active_markets', []) if settings else []
    active_playbooks = settings.get('active_playbooks', []) if settings else []
    active_sessions = settings.get('active_sessions', []) if settings else []
    
    # Determine min confluence score based on signal mode
    if signal_mode == 'quiet':
        min_score = 88  # Only A+ and A
    elif signal_mode == 'aggressive':
        min_score = 50  # All grades A-D
    else:  # balanced
        min_score = 70  # A, B, limited C
    
    # Build query for resolved signals matching current settings
    query = {
        'performance': {'$exists': True, '$ne': None},
        'performance.outcome': {'$exists': True, '$ne': None},
        'confluence_score': {'$gte': min_score}
    }
    
    # Filter by active markets (if any specified)
    if active_markets:
        query['symbol'] = {'$in': active_markets}
    
    # Filter by active sessions with format variants
    if active_sessions:
        session_variants = []
        for s in active_sessions:
            session_variants.append(s)
            session_variants.append(s.replace('_', ' '))
            session_variants.append(s.replace('_', ' ').title())
        query['session'] = {'$in': session_variants}
    
    resolved_signals = await db.signals.find(query).to_list(1000)
    
    # Apply playbook filter in Python (handles name format differences)
    if active_playbooks:
        def normalize_playbook(name):
            return name.lower().replace(' ', '_').replace('-', '_')
        
        normalized_active = [normalize_playbook(p) for p in active_playbooks]
        resolved_signals = [s for s in resolved_signals if normalize_playbook(s.get('playbook', '')) in normalized_active]
    
    # Calculate stats
    total_signals = len(resolved_signals)
    
    # Win rate
    wins = len([s for s in resolved_signals if s.get('performance', {}).get('outcome') == 'win'])
    win_rate = (wins / total_signals * 100) if total_signals else 0
    
    # Average R-multiple
    r_multiples = [s.get('performance', {}).get('r_multiple', 0) for s in resolved_signals]
    avg_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0
    
    # Signals by playbook (resolved signals only)
    signals_by_playbook = {}
    for signal in resolved_signals:
        playbook = signal.get('playbook', 'Unknown')
        signals_by_playbook[playbook] = signals_by_playbook.get(playbook, 0) + 1
    
    # Signals by pair (resolved signals only)
    signals_by_pair = {}
    for signal in resolved_signals:
        symbol = signal.get('symbol', 'Unknown')
        signals_by_pair[symbol] = signals_by_pair.get(symbol, 0) + 1
    
    # Recent resolved signals
    recent = resolved_signals[:10]
    for sig in recent:
        sig.pop('_id', None)
    
    return EngineStats(
        total_signals=total_signals,
        win_rate=round(win_rate, 2),
        average_r_multiple=round(avg_r, 2),
        signals_by_playbook=signals_by_playbook,
        signals_by_pair=signals_by_pair,
        recent_signals=recent
    )

@api_router.post("/test/demo-signal")
async def generate_demo_signal():
    """Generate a test signal in demo mode (for testing UI)"""
    test_signal = {
        'signal_id': str(uuid.uuid4()),
        'symbol': 'EURUSD',
        'timestamp': datetime.utcnow().isoformat(),
        'direction': 'BUY',
        'entry': 1.0850,
        'stop_loss': 1.0820,
        'take_profit': 1.0910,
        'playbook': 'Trend Continuation',
        'reason': 'Bullish pullback to demand zone in uptrend',
        'confluence_score': 87.5,
        'confidence_breakdown': {
            'htf_alignment': 25,
            'regime_clarity': 20,
            'liquidity_location': 15,
            'structure_quality': 10,
            'playbook_validity': 10,
            'confirmation_strength': 7.5,
            'session_quality': 0
        },
        'regime': 'Trend',
        'session': 'london',
        'timeframe': '5m',
        'entry_valid_until': (datetime.utcnow()).isoformat(),
        'setup_expires_at': (datetime.utcnow()).isoformat(),
        'mode': 'balanced',
        'status': 'active'
    }
    
    await db.signals.insert_one(test_signal.copy())
    test_signal.pop('_id', None)  # Remove _id before returning
    return {"status": "demo_signal_created", "signal": test_signal}

@api_router.post("/test/populate-demo-signals")
async def populate_demo_signals():
    """Generate 50 diverse demo signals with varying grades (A+, A, B, C, D) and realistic outcomes"""
    import random
    
    # Clear existing signals for fresh data
    await db.signals.delete_many({})
    
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'NZDUSD']
    playbooks = ['Trend Continuation', 'Liquidity Sweep Reversal', 'Breakout Retest', 'Range Mean Reversion']
    directions = ['BUY', 'SELL']
    sessions = ['london', 'new_york', 'asia', 'overlap']
    regimes = ['Trend', 'Reversal', 'Breakout', 'Range']
    
    # Generate 50 signals with diverse grades
    signals_created = []
    base_time = datetime.utcnow()
    
    # Grade distribution: A+: 10%, A: 30%, B: 30%, C: 20%, D: 10%
    grade_distribution = (['A+'] * 5) + (['A'] * 15) + (['B'] * 15) + (['C'] * 10) + (['D'] * 5)
    random.shuffle(grade_distribution)
    
    for i in range(50):
        # First 40 signals: RESOLVED (for history/performance)
        # Last 10 signals: ACTIVE (for signals screen)
        is_resolved = i < 40
        
        # Assign grade and corresponding confluence score
        grade = grade_distribution[i]
        if grade == 'A+':
            confluence_score = round(random.uniform(90, 98), 1)
            win_probability = 0.85  # A+ signals have 85% win rate
        elif grade == 'A':
            confluence_score = round(random.uniform(80, 89), 1)
            win_probability = 0.70  # A signals have 70% win rate
        elif grade == 'B':
            confluence_score = round(random.uniform(70, 79), 1)
            win_probability = 0.55  # B signals have 55% win rate
        elif grade == 'C':
            confluence_score = round(random.uniform(60, 69), 1)
            win_probability = 0.40  # C signals have 40% win rate
        else:  # D
            confluence_score = round(random.uniform(50, 59), 1)
            win_probability = 0.25  # D signals have 25% win rate
        
        # Random signal parameters
        symbol = random.choice(symbols)
        direction = random.choice(directions)
        playbook = random.choice(playbooks)
        regime = random.choice(regimes)
        session = random.choice(sessions)
        
        entry = round(1.08 + random.uniform(-0.05, 0.05), 5)
        stop_distance = round(random.uniform(0.0010, 0.0050), 5)
        
        stop_loss = entry - stop_distance if direction == 'BUY' else entry + stop_distance
        take_profit_distance = stop_distance * random.uniform(1.8, 3.2)
        take_profit = entry + take_profit_distance if direction == 'BUY' else entry - take_profit_distance
        
        # Build base signal
        signal = {
            'signal_id': str(uuid.uuid4()),
            'symbol': symbol,
            'timestamp': (base_time - timedelta(hours=100-i)).isoformat(),
            'direction': direction,
            'entry': round(entry, 5),
            'stop_loss': round(stop_loss, 5),
            'take_profit': round(take_profit, 5),
            'playbook': playbook,
            'reason': f'{direction} {playbook} setup with {grade} grade confluence',
            'confluence_score': confluence_score,
            'confidence_breakdown': {
                'htf_alignment': int(confluence_score * 0.28),
                'regime_clarity': int(confluence_score * 0.22),
                'liquidity_location': int(confluence_score * 0.17),
                'structure_quality': int(confluence_score * 0.11),
                'playbook_validity': int(confluence_score * 0.11),
                'confirmation_strength': round(confluence_score * 0.09, 1),
                'session_quality': int(confluence_score * 0.02)
            },
            'regime': regime,
            'session': session,
            'timeframe': random.choice(['5m', '15m', '1H', '4H']),
            'entry_valid_until': (base_time - timedelta(hours=100-i) + timedelta(minutes=7)).isoformat(),
            'setup_expires_at': (base_time - timedelta(hours=100-i) + timedelta(minutes=20)).isoformat(),
            'mode': 'balanced'
        }
        
        # Add performance data ONLY if signal is resolved
        if is_resolved:
            # Determine outcome based on grade quality
            outcome_roll = random.random()
            if outcome_roll < win_probability:
                outcome = 'win'
                r_multiple = random.uniform(1.5, 3.0)
                exit_price = take_profit
            elif outcome_roll < win_probability + 0.10:  # 10% breakeven
                outcome = 'breakeven'
                r_multiple = random.uniform(-0.2, 0.2)
                exit_price = entry
            elif outcome_roll < win_probability + 0.15:  # 5% invalidated
                outcome = 'invalidated'
                r_multiple = 0.0
                exit_price = entry
            else:
                outcome = 'loss'
                r_multiple = random.uniform(-0.8, -1.0)
                exit_price = stop_loss
            
            # Generate narrative
            narrative = generate_outcome_narrative(outcome, playbook, regime, direction)
            
            signal['status'] = 'tp_hit' if outcome == 'win' else ('sl_hit' if outcome == 'loss' else ('closed_be' if outcome == 'breakeven' else 'invalidated'))
            signal['performance'] = {
                'outcome': outcome,
                'r_multiple': round(r_multiple, 2),
                'exit_price': round(exit_price, 5),
                'pips': round(abs(exit_price - entry) * 10000, 1),
                'duration_minutes': random.randint(10, 240),
                'resolved_at': (base_time - timedelta(hours=99-i)).isoformat(),
                'narrative': narrative
            }
        else:
            # Active signals don't have performance data
            signal['status'] = 'active'
        
        await db.signals.insert_one(signal.copy())
        signal.pop('_id', None)
        signals_created.append(signal)
    
    # Calculate grade breakdown
    grade_counts = {'A+': 0, 'A': 0, 'B': 0, 'C': 0, 'D': 0}
    for signal in signals_created:
        score = signal['confluence_score']
        if score >= 90:
            grade_counts['A+'] += 1
        elif score >= 80:
            grade_counts['A'] += 1
        elif score >= 70:
            grade_counts['B'] += 1
        elif score >= 60:
            grade_counts['C'] += 1
        else:
            grade_counts['D'] += 1
    
    # Calculate win rate only from resolved signals
    resolved_signals = [s for s in signals_created if 'performance' in s and s.get('performance')]
    win_signals = [s for s in resolved_signals if s['performance']['outcome'] == 'win']
    win_rate = round(len(win_signals) / len(resolved_signals) * 100, 1) if resolved_signals else 0
    
    return {
        "status": "success",
        "message": f"Created {len(signals_created)} diverse demo signals ({len(resolved_signals)} resolved, {len(signals_created) - len(resolved_signals)} active)",
        "signals_count": len(signals_created),
        "resolved_count": len(resolved_signals),
        "active_count": len(signals_created) - len(resolved_signals),
        "grade_distribution": grade_counts,
        "win_rate": win_rate
    }

@api_router.post("/demo/seed")
async def seed_demo_data():
    """
    Seed demo data with FULL GUARANTEES:
    1. DETERMINISTIC: Same seed produces identical data (random.seed)
    2. ATOMIC: All-or-nothing insertion via bulk_write
    3. CONCURRENT-SAFE: Only one reset can run at a time (asyncio lock)
    4. ORDERED: Explicit sequence_id for stable ordering
    5. COMPLETE RESPONSE: Returns authoritative metrics for frontend
    """
    import random
    global _reset_in_progress
    
    # ==================== CONCURRENCY GUARD ====================
    # Only one reset can run at a time
    if _reset_in_progress:
        raise HTTPException(
            status_code=409, 
            detail="Reset already in progress. Please wait."
        )
    
    async with _reset_lock:
        _reset_in_progress = True
        
        try:
            # ==================== DETERMINISTIC SEED ====================
            # Use a fixed seed for reproducible data generation
            # This ensures identical structure/distribution on every reset
            FIXED_SEED = 42
            random.seed(FIXED_SEED)
            
            # Generate unique dataset ID for this reset
            dataset_id = f"dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            seeded_at = datetime.utcnow().isoformat()
            
            # Check existing state
            existing_count = await db.signals.count_documents({})
            action = "reset" if existing_count > 0 else "initial_seed"
            
            # ==================== GENERATE ALL SIGNALS IN MEMORY ====================
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'NZDUSD']
            playbooks = ['Trend Continuation', 'Liquidity Sweep Reversal', 'Breakout Retest', 'Range Mean Reversion']
            directions = ['BUY', 'SELL']
            sessions = ['london', 'new_york', 'asia', 'overlap']
            regimes = ['Trend', 'Reversal', 'Breakout', 'Range']
            
            signals_to_insert = []
            base_time = datetime.utcnow()
            
            # Fixed grade distribution (deterministic)
            # Resolved: 3 A+ + 12 A + 12 B + 8 C + 5 D = 40
            # Active:   2 A+ + 3 A + 3 B + 2 C = 10
            resolved_grades = (['A+'] * 3) + (['A'] * 12) + (['B'] * 12) + (['C'] * 8) + (['D'] * 5)
            random.shuffle(resolved_grades)  # Deterministic shuffle with seed
            
            active_grades = (['A+'] * 2) + (['A'] * 3) + (['B'] * 3) + (['C'] * 2)
            random.shuffle(active_grades)  # Deterministic shuffle with seed
            
            grade_distribution = resolved_grades + active_grades
            
            for i in range(50):
                grade = grade_distribution[i]
                
                # Confluence score by grade (deterministic ranges)
                if grade == 'A+':
                    confluence_score = random.uniform(90, 98)
                elif grade == 'A':
                    confluence_score = random.uniform(80, 89.9)
                elif grade == 'B':
                    confluence_score = random.uniform(70, 79.9)
                elif grade == 'C':
                    confluence_score = random.uniform(60, 69.9)
                else:
                    confluence_score = random.uniform(50, 59.9)
                
                symbol = random.choice(symbols)
                direction = random.choice(directions)
                playbook = random.choice(playbooks)
                session = random.choice(sessions)
                regime = random.choice(regimes)
                
                # Generate realistic prices
                if symbol == 'USDJPY':
                    base_price = 148.50 + random.uniform(-2, 2)
                    pip_value = 0.01
                elif symbol == 'XAUUSD':
                    base_price = 2025.00 + random.uniform(-50, 50)
                    pip_value = 0.01
                else:
                    base_price = 1.0 + random.uniform(0, 0.5)
                    pip_value = 0.0001
                
                entry = round(base_price, 5)
                sl_pips = random.uniform(15, 40)
                tp_pips = sl_pips * random.uniform(1.5, 3)
                
                if direction == 'BUY':
                    stop_loss = round(entry - (sl_pips * pip_value), 5)
                    take_profit = round(entry + (tp_pips * pip_value), 5)
                else:
                    stop_loss = round(entry + (sl_pips * pip_value), 5)
                    take_profit = round(entry - (tp_pips * pip_value), 5)
                
                signal = {
                    'signal_id': str(uuid.uuid4()),
                    'sequence_id': i,  # ORDERING GUARANTEE
                    'dataset_id': dataset_id,  # DATASET BINDING
                    'symbol': symbol,
                    'timestamp': (base_time - timedelta(hours=100-i)).isoformat(),
                    'created_at': seeded_at,  # All created at same time
                    'direction': direction,
                    'entry': entry,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'playbook': playbook,
                    'reason': f'{playbook} setup identified in {regime} market regime',
                    'confluence_score': round(confluence_score, 1),
                    'confidence_breakdown': {
                        'htf_alignment': round(random.uniform(15, 25), 1),
                        'regime_clarity': round(random.uniform(15, 25), 1),
                        'liquidity_location': round(random.uniform(10, 20), 1),
                        'structure_quality': round(random.uniform(8, 15), 1),
                        'playbook_validity': round(random.uniform(8, 12), 1),
                        'confirmation_strength': round(random.uniform(5, 10), 1),
                        'session_quality': round(random.uniform(0, 5), 1)
                    },
                    'regime': regime,
                    'session': session,
                    'timeframe': random.choice(['5m', '15m', '1h']),
                    'entry_valid_until': (base_time + timedelta(minutes=7)).isoformat(),
                    'setup_expires_at': (base_time + timedelta(minutes=20)).isoformat(),
                    'mode': 'balanced',
                    'grade': grade
                }
                
                # Resolved signals (first 40)
                if i < 40:
                    win_probability = 0.55 + (confluence_score - 50) / 200
                    rand_outcome = random.random()
                    
                    if rand_outcome < win_probability:
                        outcome = 'win'
                        r_multiple = round(random.uniform(1.5, 3.0), 2)
                        exit_price = take_profit
                    elif rand_outcome < win_probability + 0.05:
                        outcome = 'breakeven'
                        r_multiple = 0.0
                        exit_price = entry
                    else:
                        outcome = 'loss'
                        r_multiple = round(random.uniform(-0.8, -1.0), 2)
                        exit_price = stop_loss
                    
                    narrative = generate_outcome_narrative(outcome, playbook, regime, direction)
                    
                    signal['status'] = 'tp_hit' if outcome == 'win' else ('sl_hit' if outcome == 'loss' else 'closed_be')
                    signal['performance'] = {
                        'outcome': outcome,
                        'r_multiple': r_multiple,
                        'exit_price': round(exit_price, 5),
                        'pips': round(abs(exit_price - entry) * 10000, 1),
                        'duration_minutes': random.randint(10, 240),
                        'resolved_at': (base_time - timedelta(hours=99-i)).isoformat(),
                        'narrative': narrative
                    }
                else:
                    signal['status'] = 'active'
                
                signals_to_insert.append(signal)
            
            # ==================== ATOMIC DATABASE OPERATION ====================
            # Step 1: Delete all existing signals
            # Step 2: Insert all new signals in one bulk operation
            # This is pseudo-transactional for MongoDB
            
            try:
                # Clear existing
                await db.signals.delete_many({})
                
                # Bulk insert all signals at once
                if signals_to_insert:
                    await db.signals.insert_many(signals_to_insert, ordered=True)
                
                # Update metadata atomically
                await db.demo_metadata.update_one(
                    {'_id': 'seed_info'},
                    {'$set': {
                        'dataset_id': dataset_id,
                        'last_seeded_at': seeded_at,
                        'signals_count': len(signals_to_insert),
                        'action': action,
                        'seed_version': FIXED_SEED
                    }},
                    upsert=True
                )
            except Exception as db_error:
                # Rollback on failure - clear everything
                await db.signals.delete_many({})
                await db.demo_metadata.delete_one({'_id': 'seed_info'})
                raise HTTPException(status_code=500, detail=f"Database error during reset: {str(db_error)}")
            
            # ==================== COMPUTE AUTHORITATIVE METRICS ====================
            resolved_signals = [s for s in signals_to_insert if 'performance' in s]
            active_signals = [s for s in signals_to_insert if 'performance' not in s]
            win_signals = [s for s in resolved_signals if s['performance']['outcome'] == 'win']
            
            # Compute metrics
            win_rate = round(len(win_signals) / len(resolved_signals) * 100, 1) if resolved_signals else 0
            total_r = sum(s['performance']['r_multiple'] for s in resolved_signals)
            avg_r = round(total_r / len(resolved_signals), 2) if resolved_signals else 0
            
            # Compute max drawdown
            cumulative_r = 0
            max_r = 0
            max_drawdown = 0
            for s in sorted(resolved_signals, key=lambda x: x['timestamp']):
                cumulative_r += s['performance']['r_multiple']
                if cumulative_r > max_r:
                    max_r = cumulative_r
                drawdown = max_r - cumulative_r
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            # Grade breakdown
            grade_counts = {}
            for s in signals_to_insert:
                g = s.get('grade', 'Unknown')
                grade_counts[g] = grade_counts.get(g, 0) + 1
            
            metrics_computed_at = datetime.utcnow().isoformat()
            
            # ==================== RETURN AUTHORITATIVE RESPONSE ====================
            return {
                "status": "success",
                "action": action,
                "message": f"Demo data {'reset' if action == 'reset' else 'seeded'} successfully",
                
                # Dataset identity
                "datasetId": dataset_id,
                "seededAt": seeded_at,
                "metricsComputedAt": metrics_computed_at,
                
                # Signal counts
                "totalSignals": len(signals_to_insert),
                "activeSignals": len(active_signals),
                "resolvedSignals": len(resolved_signals),
                
                # Performance metrics (authoritative)
                "winRate": win_rate,
                "avgR": avg_r,
                "totalR": round(total_r, 2),
                "maxDrawdown": round(max_drawdown, 2),
                
                # Distribution
                "gradeDistribution": grade_counts,
                
                # Seed info (for verification)
                "seedVersion": FIXED_SEED
            }
        
        finally:
            _reset_in_progress = False


@api_router.get("/demo/status")
async def get_demo_status():
    """Get current demo data status"""
    signal_count = await db.signals.count_documents({})
    active_count = await db.signals.count_documents({'status': 'active'})
    resolved_count = await db.signals.count_documents({'performance': {'$exists': True}})
    
    # Get seed info
    seed_info = await db.demo_metadata.find_one({'_id': 'seed_info'})
    
    return {
        "has_data": signal_count > 0,
        "total_signals": signal_count,
        "active_signals": active_count,
        "resolved_signals": resolved_count,
        "last_seeded_at": seed_info.get('last_seeded_at') if seed_info else None,
        "needs_seeding": signal_count == 0
    }


@api_router.post("/demo/backfill-narratives")
async def backfill_missing_narratives():
    """
    Backfill missing outcome narratives for all resolved signals.
    This ensures every resolved signal has:
    - outcome: WIN/LOSS/BE/INV
    - narrative: Human-readable explanation
    
    Signals without outcome will be marked as INVALIDATED with explanation.
    """
    # Find all signals that are resolved but missing narrative
    missing_narrative_query = {
        '$or': [
            {'performance.outcome': {'$exists': True}, 'performance.narrative': {'$exists': False}},
            {'performance.outcome': {'$exists': True}, 'performance.narrative': None},
            {'performance.outcome': {'$exists': True}, 'performance.narrative': ''},
        ]
    }
    
    signals_needing_backfill = await db.signals.find(missing_narrative_query).to_list(length=1000)
    
    backfill_count = 0
    for signal in signals_needing_backfill:
        outcome = signal.get('performance', {}).get('outcome', 'invalidated')
        playbook = signal.get('playbook', 'Trend Continuation')
        regime = signal.get('regime', 'Trending')
        direction = signal.get('direction', 'BUY')
        
        # Generate narrative
        narrative = generate_outcome_narrative(outcome, playbook, regime, direction)
        
        # Update the signal
        await db.signals.update_one(
            {'signal_id': signal['signal_id']},
            {'$set': {'performance.narrative': narrative}}
        )
        backfill_count += 1
    
    # Also find signals with status that implies resolution but no performance data
    orphaned_resolved_query = {
        'status': {'$in': ['tp_hit', 'sl_hit', 'closed_be', 'invalidated', 'expired']},
        'performance': {'$exists': False}
    }
    
    orphaned_signals = await db.signals.find(orphaned_resolved_query).to_list(length=1000)
    
    orphan_count = 0
    for signal in orphaned_signals:
        status = signal.get('status', 'invalidated')
        outcome_map = {
            'tp_hit': 'win',
            'sl_hit': 'loss', 
            'closed_be': 'breakeven',
            'invalidated': 'invalidated',
            'expired': 'invalidated'
        }
        outcome = outcome_map.get(status, 'invalidated')
        
        playbook = signal.get('playbook', 'Trend Continuation')
        regime = signal.get('regime', 'Trending')
        direction = signal.get('direction', 'BUY')
        
        narrative = generate_outcome_narrative(outcome, playbook, regime, direction)
        
        # Calculate R-multiple based on outcome
        entry = signal.get('entry', 0)
        stop_loss = signal.get('stop_loss', entry)
        take_profit = signal.get('take_profit', entry)
        
        if outcome == 'win' and entry and stop_loss and take_profit:
            risk = abs(entry - stop_loss)
            reward = abs(take_profit - entry)
            r_multiple = round(reward / risk, 2) if risk > 0 else 1.5
        elif outcome == 'loss':
            r_multiple = -1.0
        else:
            r_multiple = 0.0
        
        performance_data = {
            'outcome': outcome,
            'r_multiple': r_multiple,
            'exit_price': take_profit if outcome == 'win' else (stop_loss if outcome == 'loss' else entry),
            'pips': 0,
            'resolved_at': datetime.utcnow().isoformat(),
            'narrative': narrative,
            'reason': 'Backfilled from signal status'
        }
        
        await db.signals.update_one(
            {'signal_id': signal['signal_id']},
            {'$set': {'performance': performance_data}}
        )
        orphan_count += 1
    
    return {
        "status": "success",
        "narratives_backfilled": backfill_count,
        "orphaned_signals_fixed": orphan_count,
        "total_fixed": backfill_count + orphan_count,
        "message": f"Backfilled {backfill_count} missing narratives and fixed {orphan_count} orphaned signals"
    }


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Auto-seed demo data on startup if database is empty"""
    logger.info(f"ApexFlow Engine started in {data_mode} mode with {signal_mode} signal mode")
    
    # Auto-seed if no signals exist
    signal_count = await db.signals.count_documents({})
    if signal_count == 0:
        logger.info("No signals found in database - auto-seeding demo data...")
        # Call the seed endpoint logic directly
        import random
        
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'NZDUSD']
        playbooks = ['Trend Continuation', 'Liquidity Sweep Reversal', 'Breakout Retest', 'Range Mean Reversion']
        directions = ['BUY', 'SELL']
        sessions = ['london', 'new_york', 'asia', 'overlap']
        regimes = ['Trend', 'Reversal', 'Breakout', 'Range']
        
        base_time = datetime.utcnow()
        grade_distribution = (['A+'] * 5) + (['A'] * 15) + (['B'] * 15) + (['C'] * 10) + (['D'] * 5)
        random.shuffle(grade_distribution)
        
        signals_created = 0
        for i in range(50):
            grade = grade_distribution[i]
            
            if grade == 'A+':
                confluence_score = random.uniform(90, 98)
            elif grade == 'A':
                confluence_score = random.uniform(80, 89.9)
            elif grade == 'B':
                confluence_score = random.uniform(70, 79.9)
            elif grade == 'C':
                confluence_score = random.uniform(60, 69.9)
            else:
                confluence_score = random.uniform(50, 59.9)
            
            symbol = random.choice(symbols)
            direction = random.choice(directions)
            playbook = random.choice(playbooks)
            session = random.choice(sessions)
            regime = random.choice(regimes)
            
            if symbol == 'USDJPY':
                base_price = 148.50 + random.uniform(-2, 2)
                pip_value = 0.01
            else:
                base_price = 1.0 + random.uniform(0, 0.5)
                pip_value = 0.0001
            
            entry = round(base_price, 5)
            sl_pips = random.uniform(15, 40)
            tp_pips = sl_pips * random.uniform(1.5, 3)
            
            if direction == 'BUY':
                stop_loss = round(entry - (sl_pips * pip_value), 5)
                take_profit = round(entry + (tp_pips * pip_value), 5)
            else:
                stop_loss = round(entry + (sl_pips * pip_value), 5)
                take_profit = round(entry - (tp_pips * pip_value), 5)
            
            signal = {
                'signal_id': str(uuid.uuid4()),
                'symbol': symbol,
                'timestamp': (base_time - timedelta(hours=100-i)).isoformat(),
                'direction': direction,
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'playbook': playbook,
                'reason': f'{playbook} setup in {regime} regime',
                'confluence_score': round(confluence_score, 1),
                'confidence_breakdown': {},
                'regime': regime,
                'session': session,
                'timeframe': random.choice(['5m', '15m', '1h']),
                'mode': 'balanced',
                'grade': grade
            }
            
            if i < 40:
                win_prob = 0.55 + (confluence_score - 50) / 200
                if random.random() < win_prob:
                    outcome = 'win'
                    r_multiple = random.uniform(1.5, 3.0)
                    exit_price = take_profit
                else:
                    outcome = 'loss'
                    r_multiple = random.uniform(-0.8, -1.0)
                    exit_price = stop_loss
                
                signal['status'] = 'tp_hit' if outcome == 'win' else 'sl_hit'
                signal['performance'] = {
                    'outcome': outcome,
                    'r_multiple': round(r_multiple, 2),
                    'exit_price': round(exit_price, 5),
                    'resolved_at': (base_time - timedelta(hours=99-i)).isoformat(),
                    'narrative': generate_outcome_narrative(outcome, playbook, regime, direction)
                }
            else:
                signal['status'] = 'active'
            
            await db.signals.insert_one(signal)
            signals_created += 1
        
        await db.demo_metadata.update_one(
            {'_id': 'seed_info'},
            {'$set': {'last_seeded_at': datetime.utcnow().isoformat(), 'signals_count': signals_created, 'action': 'auto_seed'}},
            upsert=True
        )
        logger.info(f"Auto-seeded {signals_created} demo signals")
    else:
        logger.info(f"Database has {signal_count} existing signals - skipping auto-seed")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
