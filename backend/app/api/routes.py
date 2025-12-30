"""
API routes for the OHLC Analyzer.
"""
from fastapi import APIRouter, HTTPException
from typing import List

from app.api.schemas import (
    OHLCVRequest,
    IndicatorRequest,
    BacktestRequest,
    BacktestResult,
    OptimizationRequest,
    OptimizationResponse,
)
from app.core.data_loader import load_ohlcv_data
from app.indicators.indicators import add_all_indicators
from app.engine.engine import Backtester

router = APIRouter()


@router.get("/symbols")
async def get_available_symbols():
    """Get list of available trading symbols."""
    return {"symbols": ["XAUUSD", "XAGUSD"]}


@router.get("/timeframes")
async def get_available_timeframes():
    """Get list of available timeframes."""
    return {"timeframes": ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]}


@router.post("/data/ohlcv")
async def get_ohlcv_data(request: OHLCVRequest):
    """Get OHLCV data for charting."""
    try:
        df = load_ohlcv_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        
        # Convert to list of dicts for JSON response
        df_reset = df.reset_index()
        df_reset['time'] = df_reset['timestamp'].astype(int) // 10**9  # Unix timestamp
        
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "count": len(df),
            "data": df_reset[['time', 'open', 'high', 'low', 'close', 'volume']].to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/indicators/calculate")
async def calculate_indicators(request: IndicatorRequest):
    """Calculate all SMC indicators for the given data."""
    try:
        df = load_ohlcv_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        
        df = add_all_indicators(df, swing_period=request.swing_period)
        
        # Prepare response with indicator data
        df_reset = df.reset_index()
        df_reset['time'] = df_reset['timestamp'].astype(int) // 10**9
        
        # Convert boolean columns to int for JSON
        bool_cols = df_reset.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df_reset[col] = df_reset[col].astype(int)
        
        # Handle NaN values
        df_reset = df_reset.fillna(0)
        
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "count": len(df),
            "columns": list(df_reset.columns),
            "data": df_reset.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest/run", response_model=BacktestResult)
async def run_backtest(request: BacktestRequest):
    """Run a single backtest with specified parameters."""
    try:
        # Load and prepare data
        df = load_ohlcv_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        df = add_all_indicators(df)
        
        # Create backtester and run
        backtester = Backtester(df)
        result = backtester.backtest(
            entry_conditions=request.entry_conditions,
            stop_loss_type=request.stop_loss_type,
            stop_loss_value=request.stop_loss_value,
            take_profit_rrr=request.take_profit_rrr,
            direction=request.direction,
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest/optimize", response_model=OptimizationResponse)
async def optimize_backtest(request: OptimizationRequest):
    """Optimize backtest parameters over specified ranges."""
    try:
        # Load and prepare data
        df = load_ohlcv_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        df = add_all_indicators(df)
        
        # Create backtester and optimize
        backtester = Backtester(df)
        results = backtester.optimize(
            entry_conditions=request.entry_conditions,
            stop_loss_type=request.stop_loss_type,
            rrr_range=(request.rrr_min, request.rrr_max),
            sl_range=(request.sl_min, request.sl_max),
            rrr_steps=request.rrr_steps,
            sl_steps=request.sl_steps,
            direction=request.direction,
        )
        
        # Sort and find best results
        sorted_by_pf = sorted(results, key=lambda x: x['profit_factor'], reverse=True)
        sorted_by_wr = sorted(results, key=lambda x: x['win_rate'], reverse=True)
        
        return OptimizationResponse(
            results=results,
            best_by_profit_factor=sorted_by_pf[0] if sorted_by_pf else None,
            best_by_win_rate=sorted_by_wr[0] if sorted_by_wr else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

