"""
Pydantic schemas for API request/response models.
"""
from datetime import datetime
from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class OHLCVRequest(BaseModel):
    """Request model for OHLCV data."""
    symbol: str = Field(default="XAUUSD", description="Trading symbol")
    timeframe: str = Field(default="1h", description="Timeframe (1m, 5m, 15m, 1h, 4h, 1d, 1w)")
    start_date: Optional[datetime] = Field(default=None, description="Start date filter")
    end_date: Optional[datetime] = Field(default=None, description="End date filter")


class IndicatorRequest(BaseModel):
    """Request model for indicator calculation."""
    symbol: str = Field(default="XAUUSD")
    timeframe: str = Field(default="1h")
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    swing_period: int = Field(default=5, ge=2, le=20)


class BacktestRequest(BaseModel):
    """Request model for running a single backtest."""
    symbol: str = Field(default="XAUUSD")
    timeframe: str = Field(default="1h")
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Entry conditions
    entry_conditions: dict = Field(
        default={},
        description="Entry condition toggles",
        examples=[{
            "fvg_bullish": True,
            "sweep_bearish": True,
            "session": "london"
        }]
    )
    
    # Exit parameters
    stop_loss_type: Literal["lowest_low_n", "atr_multiplier", "fixed_pips"] = "lowest_low_n"
    stop_loss_value: float = Field(default=3.0, description="Stop loss parameter value")
    take_profit_rrr: float = Field(default=2.0, ge=0.5, le=10.0)
    
    # Trade direction
    direction: Literal["long", "short", "both"] = "both"


class OptimizationRequest(BaseModel):
    """Request model for parameter optimization."""
    symbol: str = Field(default="XAUUSD")
    timeframe: str = Field(default="1h")
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Entry conditions
    entry_conditions: dict = Field(default={})
    
    # Optimization ranges
    stop_loss_type: Literal["lowest_low_n", "atr_multiplier", "fixed_pips"] = "lowest_low_n"
    sl_min: float = Field(default=1.0)
    sl_max: float = Field(default=5.0)
    sl_steps: int = Field(default=5)
    
    rrr_min: float = Field(default=1.0)
    rrr_max: float = Field(default=5.0)
    rrr_steps: int = Field(default=5)
    
    direction: Literal["long", "short", "both"] = "both"


class TradeResult(BaseModel):
    """Single trade result."""
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    direction: str
    pnl: float
    pnl_percent: float
    is_winner: bool


class BacktestResult(BaseModel):
    """Result of a single backtest."""
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_factor: float
    max_drawdown: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    trades: Optional[List[TradeResult]] = None


class OptimizationResult(BaseModel):
    """Single optimization result row."""
    rrr: float
    stop_loss: float
    win_rate: float
    total_trades: int
    profit_factor: float
    max_drawdown: float
    total_pnl: float


class OptimizationResponse(BaseModel):
    """Response for optimization endpoint."""
    results: List[OptimizationResult]
    best_by_profit_factor: OptimizationResult
    best_by_win_rate: OptimizationResult

