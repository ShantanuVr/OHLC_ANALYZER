"""
Vectorized Backtesting Engine for OHLCV data with SMC indicators.

All calculations are vectorized using NumPy/Pandas for performance.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Literal, Optional


class Backtester:
    """
    Vectorized backtesting engine for SMC trading strategies.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with indicators already calculated
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize the backtester with indicator-enriched data."""
        self.df = df.copy()
        self._validate_data()
    
    def _validate_data(self):
        """Validate that required columns exist."""
        required_cols = ['open', 'high', 'low', 'close']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def _build_entry_signal(
        self,
        entry_conditions: Dict,
        direction: Literal["long", "short", "both"] = "both",
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Build entry signal Series from condition dictionary.
        
        Parameters
        ----------
        entry_conditions : dict
            Dictionary of condition toggles, e.g.:
            {
                "fvg_bullish": True,
                "sweep_bullish": True,
                "session": "london",
                "above_ema": True,
            }
        direction : str
            Trade direction: "long", "short", or "both"
        
        Returns
        -------
        Tuple[pd.Series, pd.Series]
            (long_entries, short_entries) boolean Series
        """
        df = self.df
        
        # Start with all True
        long_signal = pd.Series(True, index=df.index)
        short_signal = pd.Series(True, index=df.index)
        
        # Apply each condition
        for condition, value in entry_conditions.items():
            if not value:
                continue
                
            if condition == "fvg_bullish" and 'fvg_bullish' in df.columns:
                long_signal &= df['fvg_bullish']
            elif condition == "fvg_bearish" and 'fvg_bearish' in df.columns:
                short_signal &= df['fvg_bearish']
            
            elif condition == "ob_bullish" and 'ob_bullish' in df.columns:
                long_signal &= df['ob_bullish']
            elif condition == "ob_bearish" and 'ob_bearish' in df.columns:
                short_signal &= df['ob_bearish']
            
            elif condition == "sweep_bullish" and 'sweep_bullish' in df.columns:
                long_signal &= df['sweep_bullish']
            elif condition == "sweep_bearish" and 'sweep_bearish' in df.columns:
                short_signal &= df['sweep_bearish']
            
            elif condition == "mss_bullish" and 'mss_bullish' in df.columns:
                long_signal &= df['mss_bullish']
            elif condition == "mss_bearish" and 'mss_bearish' in df.columns:
                short_signal &= df['mss_bearish']
            
            elif condition == "bos_bullish" and 'bos_bullish' in df.columns:
                long_signal &= df['bos_bullish']
            elif condition == "bos_bearish" and 'bos_bearish' in df.columns:
                short_signal &= df['bos_bearish']
            
            elif condition == "displacement_bullish" and 'displacement_bullish' in df.columns:
                long_signal &= df['displacement_bullish']
            elif condition == "displacement_bearish" and 'displacement_bearish' in df.columns:
                short_signal &= df['displacement_bearish']
            
            elif condition == "orb_breakout_long" and 'orb_breakout_long' in df.columns:
                long_signal &= df['orb_breakout_long']
            elif condition == "orb_breakout_short" and 'orb_breakout_short' in df.columns:
                short_signal &= df['orb_breakout_short']
            
            # Session filters
            elif condition == "session" and 'session' in df.columns:
                if isinstance(value, str):
                    session_filter = df['session'] == value.lower()
                    long_signal &= session_filter
                    short_signal &= session_filter
            
            # EMA filter
            elif condition == "above_ema" and 'ema_200' in df.columns:
                long_signal &= df['close'] > df['ema_200']
            elif condition == "below_ema" and 'ema_200' in df.columns:
                short_signal &= df['close'] < df['ema_200']
        
        # Apply direction filter
        if direction == "long":
            short_signal = pd.Series(False, index=df.index)
        elif direction == "short":
            long_signal = pd.Series(False, index=df.index)
        
        return long_signal, short_signal
    
    def _calculate_stop_loss(
        self,
        entry_idx: np.ndarray,
        direction: np.ndarray,
        stop_loss_type: str,
        stop_loss_value: float,
    ) -> np.ndarray:
        """
        Calculate stop loss prices for each entry.
        
        Parameters
        ----------
        entry_idx : np.ndarray
            Array of entry indices
        direction : np.ndarray
            Array of directions (1 for long, -1 for short)
        stop_loss_type : str
            Type of stop loss calculation
        stop_loss_value : float
            Parameter for stop loss calculation
        
        Returns
        -------
        np.ndarray
            Array of stop loss prices
        """
        df = self.df
        n = int(stop_loss_value)
        
        stop_losses = np.zeros(len(entry_idx))
        
        for i, (idx, dir_) in enumerate(zip(entry_idx, direction)):
            if stop_loss_type == "lowest_low_n":
                # Stop loss at lowest low of last n candles
                start_idx = max(0, idx - n)
                if dir_ == 1:  # Long
                    stop_losses[i] = df['low'].iloc[start_idx:idx+1].min()
                else:  # Short
                    stop_losses[i] = df['high'].iloc[start_idx:idx+1].max()
            
            elif stop_loss_type == "atr_multiplier":
                # Stop loss at entry price +/- ATR * multiplier
                if 'atr' in df.columns:
                    atr = df['atr'].iloc[idx]
                    entry_price = df['close'].iloc[idx]
                    if dir_ == 1:  # Long
                        stop_losses[i] = entry_price - (atr * stop_loss_value)
                    else:  # Short
                        stop_losses[i] = entry_price + (atr * stop_loss_value)
                else:
                    # Fallback to fixed pips if ATR not available
                    entry_price = df['close'].iloc[idx]
                    if dir_ == 1:
                        stop_losses[i] = entry_price - stop_loss_value
                    else:
                        stop_losses[i] = entry_price + stop_loss_value
            
            elif stop_loss_type == "fixed_pips":
                # Fixed pip distance
                entry_price = df['close'].iloc[idx]
                if dir_ == 1:  # Long
                    stop_losses[i] = entry_price - stop_loss_value
                else:  # Short
                    stop_losses[i] = entry_price + stop_loss_value
        
        return stop_losses
    
    def backtest(
        self,
        entry_conditions: Dict,
        stop_loss_type: Literal["lowest_low_n", "atr_multiplier", "fixed_pips"] = "lowest_low_n",
        stop_loss_value: float = 3.0,
        take_profit_rrr: float = 2.0,
        direction: Literal["long", "short", "both"] = "both",
        include_trades: bool = False,
    ) -> Dict:
        """
        Run backtest with specified parameters.
        
        Parameters
        ----------
        entry_conditions : dict
            Dictionary of entry condition toggles
        stop_loss_type : str
            Stop loss calculation method
        stop_loss_value : float
            Stop loss parameter (n candles, ATR multiplier, or pips)
        take_profit_rrr : float
            Risk:Reward ratio for take profit
        direction : str
            Trade direction: "long", "short", or "both"
        include_trades : bool
            Whether to include individual trade details
        
        Returns
        -------
        dict
            Backtest results including win_rate, total_trades, profit_factor, etc.
        """
        df = self.df
        
        # Build entry signals
        long_entries, short_entries = self._build_entry_signal(entry_conditions, direction)
        
        # Combine entries with direction marker
        entries_df = pd.DataFrame({
            'long_entry': long_entries,
            'short_entry': short_entries,
        })
        
        # Get entry indices and directions
        long_idx = np.where(entries_df['long_entry'])[0]
        short_idx = np.where(entries_df['short_entry'])[0]
        
        # Combine and sort
        all_entries = np.concatenate([long_idx, short_idx])
        all_directions = np.concatenate([np.ones(len(long_idx)), -np.ones(len(short_idx))])
        
        if len(all_entries) == 0:
            return {
                "win_rate": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "total_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "trades": [] if include_trades else None,
            }
        
        # Sort by entry time
        sort_idx = np.argsort(all_entries)
        all_entries = all_entries[sort_idx]
        all_directions = all_directions[sort_idx]
        
        # Calculate stop losses
        stop_losses = self._calculate_stop_loss(
            all_entries, all_directions, stop_loss_type, stop_loss_value
        )
        
        # Calculate take profits
        entry_prices = df['close'].iloc[all_entries].values
        risks = np.abs(entry_prices - stop_losses)
        take_profits = np.where(
            all_directions == 1,
            entry_prices + (risks * take_profit_rrr),
            entry_prices - (risks * take_profit_rrr)
        )
        
        # Simulate trades
        trades = []
        wins = 0
        losses = 0
        gross_profit = 0.0
        gross_loss = 0.0
        equity_curve = [0.0]
        
        # Track when we're in a trade to avoid overlapping trades
        last_exit_idx = -1
        
        for i, (entry_idx, dir_, sl, tp) in enumerate(zip(all_entries, all_directions, stop_losses, take_profits)):
            # Skip if we're still in a trade
            if entry_idx <= last_exit_idx:
                continue
            
            entry_price = entry_prices[i]
            entry_time = df.index[entry_idx]
            
            # Look for exit
            exit_idx = None
            exit_price = None
            is_winner = None
            
            # Scan forward for exit
            for j in range(entry_idx + 1, len(df)):
                candle = df.iloc[j]
                
                if dir_ == 1:  # Long
                    # Check stop loss first (hit if low <= SL)
                    if candle['low'] <= sl:
                        exit_idx = j
                        exit_price = sl
                        is_winner = False
                        break
                    # Check take profit (hit if high >= TP)
                    if candle['high'] >= tp:
                        exit_idx = j
                        exit_price = tp
                        is_winner = True
                        break
                else:  # Short
                    # Check stop loss first (hit if high >= SL)
                    if candle['high'] >= sl:
                        exit_idx = j
                        exit_price = sl
                        is_winner = False
                        break
                    # Check take profit (hit if low <= TP)
                    if candle['low'] <= tp:
                        exit_idx = j
                        exit_price = tp
                        is_winner = True
                        break
            
            # If no exit found, skip this trade
            if exit_idx is None:
                continue
            
            last_exit_idx = exit_idx
            exit_time = df.index[exit_idx]
            
            # Calculate PnL
            if dir_ == 1:
                pnl = exit_price - entry_price
            else:
                pnl = entry_price - exit_price
            
            pnl_percent = (pnl / entry_price) * 100
            
            # Update stats
            if is_winner:
                wins += 1
                gross_profit += pnl
            else:
                losses += 1
                gross_loss += abs(pnl)
            
            equity_curve.append(equity_curve[-1] + pnl)
            
            if include_trades:
                trades.append({
                    "entry_time": entry_time.isoformat(),
                    "entry_price": float(entry_price),
                    "exit_time": exit_time.isoformat(),
                    "exit_price": float(exit_price),
                    "direction": "long" if dir_ == 1 else "short",
                    "pnl": float(pnl),
                    "pnl_percent": float(pnl_percent),
                    "is_winner": is_winner,
                })
        
        # Calculate metrics
        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        # Max drawdown
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdowns = running_max - equity_array
        max_drawdown = drawdowns.max() if len(drawdowns) > 0 else 0.0
        
        # Average win/loss
        avg_win = gross_profit / wins if wins > 0 else 0.0
        avg_loss = gross_loss / losses if losses > 0 else 0.0
        
        return {
            "win_rate": float(win_rate),
            "total_trades": int(total_trades),
            "winning_trades": int(wins),
            "losing_trades": int(losses),
            "profit_factor": float(min(profit_factor, 999.99)),  # Cap for JSON
            "max_drawdown": float(max_drawdown),
            "total_pnl": float(equity_curve[-1]) if equity_curve else 0.0,
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "trades": trades if include_trades else None,
        }
    
    def optimize(
        self,
        entry_conditions: Dict,
        stop_loss_type: Literal["lowest_low_n", "atr_multiplier", "fixed_pips"] = "lowest_low_n",
        rrr_range: Tuple[float, float] = (1.0, 5.0),
        sl_range: Tuple[float, float] = (1.0, 5.0),
        rrr_steps: int = 5,
        sl_steps: int = 5,
        direction: Literal["long", "short", "both"] = "both",
    ) -> List[Dict]:
        """
        Optimize backtest parameters over specified ranges.
        
        Parameters
        ----------
        entry_conditions : dict
            Dictionary of entry condition toggles
        stop_loss_type : str
            Stop loss calculation method
        rrr_range : tuple
            (min, max) for RRR optimization
        sl_range : tuple
            (min, max) for stop loss parameter optimization
        rrr_steps : int
            Number of RRR values to test
        sl_steps : int
            Number of stop loss values to test
        direction : str
            Trade direction: "long", "short", or "both"
        
        Returns
        -------
        list
            List of result dicts for each parameter combination
        """
        results = []
        
        # Generate parameter grids
        rrr_values = np.linspace(rrr_range[0], rrr_range[1], rrr_steps)
        sl_values = np.linspace(sl_range[0], sl_range[1], sl_steps)
        
        # Grid search
        for rrr in rrr_values:
            for sl in sl_values:
                result = self.backtest(
                    entry_conditions=entry_conditions,
                    stop_loss_type=stop_loss_type,
                    stop_loss_value=sl,
                    take_profit_rrr=rrr,
                    direction=direction,
                    include_trades=False,
                )
                
                results.append({
                    "rrr": float(rrr),
                    "stop_loss": float(sl),
                    "win_rate": result["win_rate"],
                    "total_trades": result["total_trades"],
                    "profit_factor": result["profit_factor"],
                    "max_drawdown": result["max_drawdown"],
                    "total_pnl": result["total_pnl"],
                })
        
        # Sort by profit factor (descending)
        results.sort(key=lambda x: x["profit_factor"], reverse=True)
        
        return results

