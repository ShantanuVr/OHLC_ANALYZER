"""
Smart Money Concepts (SMC) Indicators for OHLCV data.

All functions are fully vectorized using NumPy/Pandas operations.
No for-loops are used for iteration to ensure speed over 23 years of data.

Exception: SMCStructure class uses sequential processing for accurate
state machine tracking of MSS/BoS per SMC rules.
"""
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import Tuple, Optional, List


# =============================================================================
# MARKET STRUCTURE (SWINGS)
# =============================================================================

def add_swing_highs_lows(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Identify Swing Highs and Swing Lows using argrelextrema logic.
    
    A swing high is a local maximum with n candles on each side lower.
    A swing low is a local minimum with n candles on each side higher.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame
    n : int
        Number of candles to check on each side (default: 5)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - swing_high: boolean, True if this candle is a swing high
        - swing_low: boolean, True if this candle is a swing low
        - swing_high_price: float, price of swing high (NaN otherwise)
        - swing_low_price: float, price of swing low (NaN otherwise)
    """
    df = df.copy()
    
    # Find local maxima (swing highs) in the 'high' column
    swing_high_idx = argrelextrema(df['high'].values, np.greater_equal, order=n)[0]
    
    # Find local minima (swing lows) in the 'low' column
    swing_low_idx = argrelextrema(df['low'].values, np.less_equal, order=n)[0]
    
    # Create boolean columns
    df['swing_high'] = False
    df['swing_low'] = False
    
    # Use iloc for positional indexing
    df.iloc[swing_high_idx, df.columns.get_loc('swing_high')] = True
    df.iloc[swing_low_idx, df.columns.get_loc('swing_low')] = True
    
    # Store swing prices (NaN for non-swings)
    df['swing_high_price'] = np.where(df['swing_high'], df['high'], np.nan)
    df['swing_low_price'] = np.where(df['swing_low'], df['low'], np.nan)
    
    # Forward fill swing prices for later use
    df['last_swing_high'] = df['swing_high_price'].ffill()
    df['last_swing_low'] = df['swing_low_price'].ffill()
    
    return df


def add_market_structure_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label market structure: HH (Higher High), LH (Lower High), 
    LL (Lower Low), HL (Higher Low).
    
    Compares each swing with the previous swing of the same type.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with swing_high and swing_low columns
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - swing_high_label: 'HH' or 'LH' for swing highs
        - swing_low_label: 'HL' or 'LL' for swing lows
        - market_structure: combined label
    """
    df = df.copy()
    
    # Get only swing high prices, shift to compare with previous
    swing_highs = df.loc[df['swing_high'], 'high'].copy()
    prev_swing_high = swing_highs.shift(1)
    
    # Label swing highs
    high_labels = pd.Series(index=swing_highs.index, dtype='object')
    high_labels[swing_highs > prev_swing_high] = 'HH'
    high_labels[swing_highs <= prev_swing_high] = 'LH'
    
    # Get only swing low prices, shift to compare with previous
    swing_lows = df.loc[df['swing_low'], 'low'].copy()
    prev_swing_low = swing_lows.shift(1)
    
    # Label swing lows
    low_labels = pd.Series(index=swing_lows.index, dtype='object')
    low_labels[swing_lows < prev_swing_low] = 'LL'
    low_labels[swing_lows >= prev_swing_low] = 'HL'
    
    # Merge labels back to main DataFrame
    df['swing_high_label'] = ''
    df['swing_low_label'] = ''
    
    df.loc[high_labels.index, 'swing_high_label'] = high_labels
    df.loc[low_labels.index, 'swing_low_label'] = low_labels
    
    # Combined market structure column
    df['market_structure'] = df['swing_high_label'] + df['swing_low_label']
    df['market_structure'] = df['market_structure'].replace('', np.nan)
    
    return df


def add_mss_bos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Market Structure Shift (MSS) and Break of Structure (BoS).
    
    MSS: Close crosses the previous major swing high/low (trend reversal signal)
    BoS: Continuation pattern - breaking swing high in uptrend, swing low in downtrend
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with last_swing_high and last_swing_low columns
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - mss_bullish: boolean, bullish market structure shift
        - mss_bearish: boolean, bearish market structure shift  
        - bos_bullish: boolean, bullish break of structure
        - bos_bearish: boolean, bearish break of structure
    """
    df = df.copy()
    
    # Shift last swing values to get the "previous" swing level
    prev_swing_high = df['last_swing_high'].shift(1)
    prev_swing_low = df['last_swing_low'].shift(1)
    
    # MSS Bullish: Close crosses above previous swing high (potential reversal from downtrend)
    # This is a break of a significant high after making lower lows
    df['mss_bullish'] = (df['close'] > prev_swing_high) & (df['close'].shift(1) <= prev_swing_high.shift(1))
    
    # MSS Bearish: Close crosses below previous swing low (potential reversal from uptrend)
    df['mss_bearish'] = (df['close'] < prev_swing_low) & (df['close'].shift(1) >= prev_swing_low.shift(1))
    
    # Determine trend based on swing sequence
    # Uptrend: Higher Highs and Higher Lows
    # Downtrend: Lower Highs and Lower Lows
    df['in_uptrend'] = (df['swing_high_label'] == 'HH') | (df['swing_low_label'] == 'HL')
    df['in_downtrend'] = (df['swing_high_label'] == 'LH') | (df['swing_low_label'] == 'LL')
    
    # Forward fill trend
    df['in_uptrend'] = df['in_uptrend'].replace(False, np.nan).ffill().fillna(False).astype(bool)
    df['in_downtrend'] = df['in_downtrend'].replace(False, np.nan).ffill().fillna(False).astype(bool)
    
    # BoS Bullish: Breaking swing high while in uptrend (continuation)
    df['bos_bullish'] = (
        (df['high'] > prev_swing_high) & 
        (df['high'].shift(1) <= prev_swing_high.shift(1)) &
        df['in_uptrend'].shift(1)
    )
    
    # BoS Bearish: Breaking swing low while in downtrend (continuation)
    df['bos_bearish'] = (
        (df['low'] < prev_swing_low) & 
        (df['low'].shift(1) >= prev_swing_low.shift(1)) &
        df['in_downtrend'].shift(1)
    )
    
    return df


class SMCStructure:
    """
    Smart Money Concepts Market Structure Analyzer.
    
    Tracks market structure using pivot detection (fractals) to reduce noise.
    Only updates target swings when pivot points are detected, not on every candle.
    
    Features:
    - Pivot detection (fractals) using swing_window
    - Temporary target swings (only updated when pivots found)
    - Confirmed swings (protected high/low)
    - Break of Structure (BoS) - continuation signals
    - Market Structure Shift (MSS) - reversal signals
    - High-Quality BoS detection using Premium/Discount zones
    """
    def __init__(self, swing_window: int = 3):
        """
        Parameters
        ----------
        swing_window : int
            Number of candles to look left/right to define a swing point (fractal).
            Higher = fewer, more major signals (e.g., 5-10).
            Lower = more sensitive, minor structure (e.g., 3).
            Default: 3
        """
        self.swing_window = swing_window
        self.trend = 'uptrend'  # Default start
        
        # State Variables
        self.confirmed_swing_price = 0.0  # The "Protected" High/Low
        self.confirmed_swing_idx = 0     # Index of the Protected High/Low
        self.temp_swing_price = 0.0      # The Target High/Low we are trying to break
        self.temp_swing_idx = 0          # Index of the Target High/Low

    def _get_lowest_in_range(self, df: pd.DataFrame, start_idx, end_idx):
        """Finds the lowest low between two indices (can be integer or DataFrame index)."""
        if isinstance(start_idx, (int, np.integer)):
            start_pos = start_idx
        else:
            start_pos = df.index.get_loc(start_idx) if start_idx in df.index else None
            if start_pos is None:
                return None, None
        
        if isinstance(end_idx, (int, np.integer)):
            end_pos = end_idx
        else:
            end_pos = df.index.get_loc(end_idx) if end_idx in df.index else None
            if end_pos is None:
                return None, None
        
        if start_pos >= end_pos or start_pos < 0 or end_pos > len(df):
            return None, None
        
        subset = df['low'].iloc[start_pos:end_pos]
        if subset.empty:
            return None, None
        min_val = subset.min()
        min_pos = subset.idxmin()
        return min_val, min_pos

    def _get_highest_in_range(self, df: pd.DataFrame, start_idx, end_idx):
        """Finds the highest high between two indices (can be integer or DataFrame index)."""
        if isinstance(start_idx, (int, np.integer)):
            start_pos = start_idx
        else:
            start_pos = df.index.get_loc(start_idx) if start_idx in df.index else None
            if start_pos is None:
                return None, None
        
        if isinstance(end_idx, (int, np.integer)):
            end_pos = end_idx
        else:
            end_pos = df.index.get_loc(end_idx) if end_idx in df.index else None
            if end_pos is None:
                return None, None
        
        if start_pos >= end_pos or start_pos < 0 or end_pos > len(df):
            return None, None
        
        subset = df['high'].iloc[start_pos:end_pos]
        if subset.empty:
            return None, None
        max_val = subset.max()
        max_pos = subset.idxmax()
        return max_val, max_pos

    def _is_pivot_high(self, df: pd.DataFrame, idx: int) -> bool:
        """Checks if a candle is a local High (Fractal/Pivot High)."""
        if idx < self.swing_window or idx >= len(df) - self.swing_window:
            return False
        
        curr_high = df['high'].iloc[idx]
        # Check Left and Right neighbors
        for offset in range(1, self.swing_window + 1):
            if df['high'].iloc[idx - offset] > curr_high or \
               df['high'].iloc[idx + offset] >= curr_high:  # Strict > on left, >= on right
                return False
        return True

    def _is_pivot_low(self, df: pd.DataFrame, idx: int) -> bool:
        """Checks if a candle is a local Low (Fractal/Pivot Low)."""
        if idx < self.swing_window or idx >= len(df) - self.swing_window:
            return False
        
        curr_low = df['low'].iloc[idx]
        for offset in range(1, self.swing_window + 1):
            if df['low'].iloc[idx - offset] < curr_low or \
               df['low'].iloc[idx + offset] <= curr_low:
                return False
        return True

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main execution method.
        Requires DataFrame with columns: 'high', 'low', 'close' (lowercase).
        """
        df = df.copy()
        
        # 1. Identify all Pivot Highs and Lows (Fractals) first
        # This prevents "Floating High" from changing every single candle
        df['is_pivot_high'] = [self._is_pivot_high(df, i) for i in range(len(df))]
        df['is_pivot_low'] = [self._is_pivot_low(df, i) for i in range(len(df))]
        
        # Initialize Output Columns
        df['structure'] = None      # HH, HL, LH, LL
        df['bos'] = 0               # 1 if BoS detected
        df['mss'] = 0               # 1 if MSS detected
        df['is_high_quality'] = 0   # 1 if retracement was deep enough (Premium/Discount)
        df['trend'] = 'uptrend'     # Track trend state
        
        # Additional columns for compatibility with existing frontend
        df['bos_bullish'] = False
        df['bos_bearish'] = False
        df['bos_hq_bullish'] = False
        df['bos_hq_bearish'] = False
        df['mss_bullish'] = False
        df['mss_bearish'] = False
        df['confirmed_hh'] = False
        df['confirmed_hl_at_idx'] = False
        df['confirmed_lh_at_idx'] = False
        df['confirmed_ll'] = False
        df['current_trend'] = None

        if len(df) == 0:
            return df

        # Initialize State using first candle
        self.trend = 'uptrend'
        # Start with the first available pivot or global extrema
        self.confirmed_swing_price = df['low'].iloc[0] 
        self.confirmed_swing_idx = df.index[0]
        self.temp_swing_price = df['high'].iloc[0]
        self.temp_swing_idx = df.index[0]

        # Iterate through candles (starting from 2nd candle)
        for i in range(1, len(df)):
            curr_idx = df.index[i]
            curr_close = df['close'].iloc[i]
            curr_high = df['high'].iloc[i]
            curr_low = df['low'].iloc[i]
            
            # ------------------------------------------------------------------
            # UPTREND LOGIC
            # ------------------------------------------------------------------
            if self.trend == 'uptrend':
                
                # Store previous temp_swing BEFORE updating
                prev_temp_swing_price = self.temp_swing_price
                prev_temp_swing_idx = self.temp_swing_idx
                
                # A. Update Target: If we found a new Pivot High higher than current target, update it
                # (Only if we haven't broken structure yet)
                if df['is_pivot_high'].iloc[i]:
                    if curr_high > self.temp_swing_price:
                        self.temp_swing_price = curr_high
                        self.temp_swing_idx = curr_idx
                
                # B. Check for BoS (Break of Structure) -> Continuation
                # Rule: Close > PREVIOUS Temp Swing (before update)
                if curr_close > prev_temp_swing_price:
                    # 1. Mark the PREVIOUS temp_swing as confirmed HH (the one that was broken)
                    if prev_temp_swing_idx != curr_idx:
                        df.at[prev_temp_swing_idx, 'structure'] = 'HH'
                        prev_temp_swing_idx_int = df.index.get_loc(prev_temp_swing_idx) if prev_temp_swing_idx in df.index else None
                        if prev_temp_swing_idx_int is not None:
                            df.iloc[prev_temp_swing_idx_int, df.columns.get_loc('confirmed_hh')] = True
                    
                    # 2. Retroactive HL: Find lowest point between the HH that was broken and current candle
                    # This is the retracement between the previous HH and this BoS
                    hl_price, hl_idx = self._get_lowest_in_range(df, prev_temp_swing_idx, curr_idx)
                    
                    if hl_price is not None and hl_idx is not None and hl_idx != prev_temp_swing_idx:
                        # Mark the Confirmed HL on the chart
                        df.at[hl_idx, 'structure'] = 'HL'
                        # Convert to integer position for boolean array
                        hl_idx_int = df.index.get_loc(hl_idx) if hl_idx in df.index else None
                        if hl_idx_int is not None:
                            df.iloc[hl_idx_int, df.columns.get_loc('confirmed_hl_at_idx')] = True
                        
                        # High Quality Check (Discount Zone)
                        # Range from previous HH to the last confirmed HL (protected low)
                        midpoint = (prev_temp_swing_price + self.confirmed_swing_price) / 2
                        if hl_price < midpoint:
                            df.at[curr_idx, 'is_high_quality'] = 1
                            curr_idx_int = df.index.get_loc(curr_idx) if curr_idx in df.index else None
                            if curr_idx_int is not None:
                                df.iloc[curr_idx_int, df.columns.get_loc('bos_hq_bullish')] = True
                        
                        # Update Protected Low to this new HL
                        self.confirmed_swing_price = hl_price
                        self.confirmed_swing_idx = hl_idx
                    
                    # 3. Mark BoS
                    df.at[curr_idx, 'bos'] = 1
                    df.at[curr_idx, 'bos_bullish'] = True
                    df.at[curr_idx, 'structure'] = 'BoS'
                    
                    # 4. Reset State for next leg - wait for next pivot
                    self.temp_swing_price = curr_high 
                    self.temp_swing_idx = curr_idx
                
                # C. Check for MSS (Market Structure Shift) -> Reversal
                # Rule: Close < Last Confirmed HL
                elif curr_close < self.confirmed_swing_price:
                    df.at[curr_idx, 'mss'] = 1
                    df.at[curr_idx, 'mss_bearish'] = True
                    df.at[curr_idx, 'structure'] = 'MSS'
                    
                    # Flip Trend
                    self.trend = 'downtrend'
                    
                    # Mark HH (The highest point that failed to hold)
                    hh_price, hh_idx = self._get_highest_in_range(df, self.confirmed_swing_idx, curr_idx)
                    if hh_price is not None and hh_idx is not None:
                        df.at[hh_idx, 'structure'] = 'HH'
                        hh_idx_int = df.index.get_loc(hh_idx) if hh_idx in df.index else None
                        if hh_idx_int is not None:
                            df.iloc[hh_idx_int, df.columns.get_loc('confirmed_hh')] = True
                        self.confirmed_swing_price = hh_price  # Now Protected High (LH)
                        self.confirmed_swing_idx = hh_idx
                    
                    # Mark current low as LL
                    df.at[curr_idx, 'structure'] = 'LL'
                    curr_idx_int = df.index.get_loc(curr_idx) if curr_idx in df.index else None
                    if curr_idx_int is not None:
                        df.iloc[curr_idx_int, df.columns.get_loc('confirmed_ll')] = True
                    
                    # Start tracking lows
                    self.temp_swing_price = curr_low
                    self.temp_swing_idx = curr_idx

            # ------------------------------------------------------------------
            # DOWNTREND LOGIC
            # ------------------------------------------------------------------
            elif self.trend == 'downtrend':
                
                # Store previous temp_swing BEFORE updating
                prev_temp_swing_price = self.temp_swing_price
                prev_temp_swing_idx = self.temp_swing_idx
                
                # A. Update Target: Pivot Lows
                if df['is_pivot_low'].iloc[i]:
                    if curr_low < self.temp_swing_price:
                        self.temp_swing_price = curr_low
                        self.temp_swing_idx = curr_idx
                
                # B. Check for BoS (Continuation Down)
                # Rule: Close < PREVIOUS Temp Swing (before update)
                if curr_close < prev_temp_swing_price:
                    # 1. Mark the PREVIOUS temp_swing as confirmed LL (the one that was broken)
                    if prev_temp_swing_idx != curr_idx:
                        df.at[prev_temp_swing_idx, 'structure'] = 'LL'
                        prev_temp_swing_idx_int = df.index.get_loc(prev_temp_swing_idx) if prev_temp_swing_idx in df.index else None
                        if prev_temp_swing_idx_int is not None:
                            df.iloc[prev_temp_swing_idx_int, df.columns.get_loc('confirmed_ll')] = True
                    
                    # 2. Retroactive LH: Find highest point between the LL that was broken and current candle
                    # This is the retracement between the previous LL and this BoS
                    lh_price, lh_idx = self._get_highest_in_range(df, prev_temp_swing_idx, curr_idx)
                    
                    if lh_price is not None and lh_idx is not None and lh_idx != prev_temp_swing_idx:
                        df.at[lh_idx, 'structure'] = 'LH'
                        lh_idx_int = df.index.get_loc(lh_idx) if lh_idx in df.index else None
                        if lh_idx_int is not None:
                            df.iloc[lh_idx_int, df.columns.get_loc('confirmed_lh_at_idx')] = True
                        
                        # High Quality Check (Premium Zone)
                        # Range from previous LL to the last confirmed LH (protected high)
                        midpoint = (self.confirmed_swing_price + prev_temp_swing_price) / 2
                        if lh_price > midpoint:
                            df.at[curr_idx, 'is_high_quality'] = 1
                            curr_idx_int = df.index.get_loc(curr_idx) if curr_idx in df.index else None
                            if curr_idx_int is not None:
                                df.iloc[curr_idx_int, df.columns.get_loc('bos_hq_bearish')] = True
                        
                        # Update Protected High to this new LH
                        self.confirmed_swing_price = lh_price
                        self.confirmed_swing_idx = lh_idx
                    
                    # 3. Mark BoS
                    df.at[curr_idx, 'bos'] = 1
                    df.at[curr_idx, 'bos_bearish'] = True
                    df.at[curr_idx, 'structure'] = 'BoS'
                    
                    # 4. Reset State for next leg
                    self.temp_swing_price = curr_low
                    self.temp_swing_idx = curr_idx
                
                # C. Check for MSS (Reversal Up)
                # Rule: Close > Last Confirmed LH
                elif curr_close > self.confirmed_swing_price:
                    df.at[curr_idx, 'mss'] = 1
                    df.at[curr_idx, 'mss_bullish'] = True
                    df.at[curr_idx, 'structure'] = 'MSS'
                    
                    self.trend = 'uptrend'
                    
                    # Mark LL
                    ll_price, ll_idx = self._get_lowest_in_range(df, self.confirmed_swing_idx, curr_idx)
                    if ll_price is not None and ll_idx is not None:
                        df.at[ll_idx, 'structure'] = 'LL'
                        ll_idx_int = df.index.get_loc(ll_idx) if ll_idx in df.index else None
                        if ll_idx_int is not None:
                            df.iloc[ll_idx_int, df.columns.get_loc('confirmed_ll')] = True
                        self.confirmed_swing_price = ll_price  # Now Protected Low (HL)
                        self.confirmed_swing_idx = ll_idx
                    
                    # Mark current high as HH
                    df.at[curr_idx, 'structure'] = 'HH'
                    curr_idx_int = df.index.get_loc(curr_idx) if curr_idx in df.index else None
                    if curr_idx_int is not None:
                        df.iloc[curr_idx_int, df.columns.get_loc('confirmed_hh')] = True
                    
                    self.temp_swing_price = curr_high
                    self.temp_swing_idx = curr_idx
            
            # Log trend state for this candle
            df.at[curr_idx, 'trend'] = self.trend
            curr_idx_int = df.index.get_loc(curr_idx) if curr_idx in df.index else None
            if curr_idx_int is not None:
                df.iloc[curr_idx_int, df.columns.get_loc('current_trend')] = self.trend

        return df




def add_smc_market_structure(df: pd.DataFrame, warmup_period: int = 50) -> pd.DataFrame:
    """
    SMC Market Structure Analysis using class-based state machine approach.
    
    Wrapper function that uses SMCStructure class for cleaner implementation.
    
    This function delegates all market structure logic to the SMCStructure class,
    which implements strict SMC rules:
    - All breaks use BODY CLOSE only (not wicks)
    - HL/LH are confirmed RETROACTIVELY after BoS occurs
    - High-Quality BoS requires retracement into 50% equilibrium zone
    - Tracks floating Temporary HH/LL with indices for efficient lookback
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with datetime index
    warmup_period : int
        Number of candles to skip at start to avoid false signals (default: 50)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - current_trend: 'uptrend' | 'downtrend' | None
        - confirmed_hh / confirmed_hl_at_idx / confirmed_lh_at_idx / confirmed_ll: Pivot markers
        - bos_bullish / bos_bearish: Break of Structure signals
        - bos_hq_bullish / bos_hq_bearish: High-Quality BoS signals
        - mss_bullish / mss_bearish: Market Structure Shift signals
        - structure: String labels ('HH', 'HL', 'LH', 'LL', 'BoS', 'MSS')
    """
    df = df.copy()
    n = len(df)
    
    if n <= warmup_period:
        # Not enough data, return empty signals
        df['current_trend'] = None
        df['temp_hh'] = np.nan
        df['temp_ll'] = np.nan
        df['confirmed_hl'] = np.nan
        df['confirmed_lh'] = np.nan
        df['confirmed_hl_idx'] = np.nan
        df['confirmed_lh_idx'] = np.nan
        df['confirmed_hh'] = False
        df['confirmed_hl_at_idx'] = False
        df['confirmed_lh_at_idx'] = False
        df['confirmed_ll'] = False
        df['bos_bullish'] = False
        df['bos_bearish'] = False
        df['bos_hq_bullish'] = False
        df['bos_hq_bearish'] = False
        df['mss_bullish'] = False
        df['mss_bearish'] = False
        df['structure'] = None
        df['bos'] = 0
        df['mss'] = 0
        df['is_high_quality'] = 0
        return df
    
    # Use the class-based implementation with pivot detection to reduce noise
    # swing_window=3 means a pivot must be higher/lower than 3 candles on each side
    smc = SMCStructure(swing_window=3)
    df = smc.apply(df)
    
    # Add compatibility columns for existing code
    df['in_uptrend'] = df['current_trend'] == 'uptrend'
    df['in_downtrend'] = df['current_trend'] == 'downtrend'
    
    # Add temp_hh/temp_ll columns for compatibility (not used but expected)
    df['temp_hh'] = np.nan
    df['temp_ll'] = np.nan
    df['confirmed_hl'] = np.nan
    df['confirmed_lh'] = np.nan
    df['confirmed_hl_idx'] = np.nan
    df['confirmed_lh_idx'] = np.nan
    
    return df


# =============================================================================
# SMART MONEY CONCEPTS
# =============================================================================

def add_displacement(df: pd.DataFrame, multiplier: float = 2.0, period: int = 20) -> pd.DataFrame:
    """
    Identify displacement candles (large body candles showing institutional activity).
    
    A displacement candle has a body size > multiplier * average body size.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame
    multiplier : float
        How many times larger than average (default: 2.0)
    period : int
        Lookback period for average calculation (default: 20)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - body_size: absolute body size
        - avg_body_size: rolling average body size
        - displacement: boolean, True if displacement candle
        - displacement_bullish: boolean, bullish displacement
        - displacement_bearish: boolean, bearish displacement
    """
    df = df.copy()
    
    # Calculate body size
    df['body_size'] = np.abs(df['close'] - df['open'])
    
    # Calculate rolling average body size
    df['avg_body_size'] = df['body_size'].rolling(window=period, min_periods=1).mean()
    
    # Identify displacement candles
    df['displacement'] = df['body_size'] > (multiplier * df['avg_body_size'])
    
    # Classify bullish/bearish
    df['displacement_bullish'] = df['displacement'] & (df['close'] > df['open'])
    df['displacement_bearish'] = df['displacement'] & (df['close'] < df['open'])
    
    return df


def add_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify Fair Value Gaps (FVG) - 3-candle patterns with price imbalance.
    
    Bullish FVG: Low of Candle 1 > High of Candle 3 (gap up)
    Bearish FVG: High of Candle 1 < Low of Candle 3 (gap down)
    
    The middle candle (Candle 2) is where the FVG is marked.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - fvg_bullish: boolean, bullish FVG on this candle
        - fvg_bearish: boolean, bearish FVG on this candle
        - fvg_top: top of the FVG zone
        - fvg_bottom: bottom of the FVG zone
    """
    df = df.copy()
    
    # Candle 1 is shift(2), Candle 2 is shift(1), Candle 3 is current
    # But we mark the FVG on the middle candle (current when looking back)
    
    # For marking on the middle candle:
    # Candle before (shift 1) = Candle 1
    # Current candle = Candle 2 (where we mark)
    # Candle after (shift -1) = Candle 3
    
    candle1_low = df['low'].shift(1)
    candle1_high = df['high'].shift(1)
    candle3_low = df['low'].shift(-1)
    candle3_high = df['high'].shift(-1)
    
    # Bullish FVG: Low of Candle 1 > High of Candle 3
    # This means there's a gap between Candle 1's low and Candle 3's high
    df['fvg_bullish'] = candle1_low > candle3_high
    
    # Bearish FVG: High of Candle 1 < Low of Candle 3
    df['fvg_bearish'] = candle1_high < candle3_low
    
    # Calculate FVG zones
    # Bullish FVG zone: from Candle 3 high to Candle 1 low
    df['fvg_top'] = np.where(df['fvg_bullish'], candle1_low, np.nan)
    df['fvg_bottom'] = np.where(df['fvg_bullish'], candle3_high, np.nan)
    
    # Bearish FVG zone: from Candle 1 high to Candle 3 low
    df.loc[df['fvg_bearish'], 'fvg_top'] = candle3_low[df['fvg_bearish']]
    df.loc[df['fvg_bearish'], 'fvg_bottom'] = candle1_high[df['fvg_bearish']]
    
    return df


def add_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify Order Blocks (OB) - last opposite candle(s) before displacement.
    
    Bullish OB: Last bearish candle before a bullish displacement
    Bearish OB: Last bullish candle before a bearish displacement
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with displacement columns
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - ob_bullish: boolean, bullish order block
        - ob_bearish: boolean, bearish order block
        - ob_top: top of the order block zone
        - ob_bottom: bottom of the order block zone
    """
    df = df.copy()
    
    # Initialize columns
    df['ob_bullish'] = False
    df['ob_bearish'] = False
    df['ob_top'] = np.nan
    df['ob_bottom'] = np.nan
    
    # Identify bearish candles (close < open)
    is_bearish = df['close'] < df['open']
    is_bullish = df['close'] > df['open']
    
    # Bullish OB: bearish candle followed by bullish displacement
    # Look for displacement on next candle
    next_is_bullish_displacement = df['displacement_bullish'].shift(-1).fillna(False)
    df['ob_bullish'] = is_bearish & next_is_bullish_displacement
    
    # Bearish OB: bullish candle followed by bearish displacement
    next_is_bearish_displacement = df['displacement_bearish'].shift(-1).fillna(False)
    df['ob_bearish'] = is_bullish & next_is_bearish_displacement
    
    # Set OB zones
    # Bullish OB zone: use the candle's body
    df.loc[df['ob_bullish'], 'ob_top'] = df.loc[df['ob_bullish'], 'open']  # Open is higher for bearish candle
    df.loc[df['ob_bullish'], 'ob_bottom'] = df.loc[df['ob_bullish'], 'close']
    
    df.loc[df['ob_bearish'], 'ob_top'] = df.loc[df['ob_bearish'], 'close']  # Close is higher for bullish candle
    df.loc[df['ob_bearish'], 'ob_bottom'] = df.loc[df['ob_bearish'], 'open']
    
    return df


def add_liquidity_sweeps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Liquidity Sweeps (BSL - Buy Side Liquidity, SSL - Sell Side Liquidity).
    
    BSL Sweep: Price high breaks previous swing high but closes below it
    SSL Sweep: Price low breaks previous swing low but closes above it
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with last_swing_high and last_swing_low columns
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - sweep_bullish: boolean, bullish liquidity sweep (SSL sweep = bullish signal)
        - sweep_bearish: boolean, bearish liquidity sweep (BSL sweep = bearish signal)
    """
    df = df.copy()
    
    # Get previous swing levels (shift by 1 to not include current candle's swing)
    prev_swing_high = df['last_swing_high'].shift(1)
    prev_swing_low = df['last_swing_low'].shift(1)
    
    # BSL Sweep (Bearish signal): High breaks above swing high, but close is below it
    # This sweeps the buy-side liquidity (stop losses of shorts, buy orders)
    df['sweep_bearish'] = (df['high'] > prev_swing_high) & (df['close'] < prev_swing_high)
    
    # SSL Sweep (Bullish signal): Low breaks below swing low, but close is above it  
    # This sweeps the sell-side liquidity (stop losses of longs, sell orders)
    df['sweep_bullish'] = (df['low'] < prev_swing_low) & (df['close'] > prev_swing_low)
    
    return df


# =============================================================================
# TIME & SESSION
# =============================================================================

def add_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trading session information and session ranges.
    
    Sessions (UTC):
    - Asian: 00:00 - 08:00
    - London: 08:00 - 16:00
    - New York: 13:00 - 21:00
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with datetime index
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - session: current session name
        - session_date: date identifier for session grouping
        - asian_high, asian_low: Asian session range
        - london_high, london_low: London session range
        - ny_high, ny_low: New York session range
    """
    df = df.copy()
    
    # Extract hour from index
    hours = df.index.hour
    
    # Determine session for each candle
    df['session'] = 'asian'  # Default
    df.loc[(hours >= 8) & (hours < 13), 'session'] = 'london'
    df.loc[(hours >= 13) & (hours < 21), 'session'] = 'ny'
    df.loc[(hours >= 21) | (hours < 0), 'session'] = 'asian'  # Overlap handling
    
    # Create session date (for grouping)
    df['session_date'] = df.index.date
    
    # Calculate session ranges using groupby and transform
    # Asian session (00:00 - 08:00)
    asian_mask = (hours >= 0) & (hours < 8)
    df['asian_high'] = np.nan
    df['asian_low'] = np.nan
    
    if asian_mask.any():
        asian_highs = df.loc[asian_mask].groupby(df.loc[asian_mask].index.date)['high'].transform('max')
        asian_lows = df.loc[asian_mask].groupby(df.loc[asian_mask].index.date)['low'].transform('min')
        df.loc[asian_mask, 'asian_high'] = asian_highs
        df.loc[asian_mask, 'asian_low'] = asian_lows
    
    # London session (08:00 - 16:00)
    london_mask = (hours >= 8) & (hours < 16)
    df['london_high'] = np.nan
    df['london_low'] = np.nan
    
    if london_mask.any():
        london_highs = df.loc[london_mask].groupby(df.loc[london_mask].index.date)['high'].transform('max')
        london_lows = df.loc[london_mask].groupby(df.loc[london_mask].index.date)['low'].transform('min')
        df.loc[london_mask, 'london_high'] = london_highs
        df.loc[london_mask, 'london_low'] = london_lows
    
    # NY session (13:00 - 21:00)
    ny_mask = (hours >= 13) & (hours < 21)
    df['ny_high'] = np.nan
    df['ny_low'] = np.nan
    
    if ny_mask.any():
        ny_highs = df.loc[ny_mask].groupby(df.loc[ny_mask].index.date)['high'].transform('max')
        ny_lows = df.loc[ny_mask].groupby(df.loc[ny_mask].index.date)['low'].transform('min')
        df.loc[ny_mask, 'ny_high'] = ny_highs
        df.loc[ny_mask, 'ny_low'] = ny_lows
    
    # Forward fill session ranges within the day
    for col in ['asian_high', 'asian_low', 'london_high', 'london_low', 'ny_high', 'ny_low']:
        df[col] = df.groupby(df.index.date)[col].ffill()
    
    return df


def add_orb(df: pd.DataFrame, orb_minutes: int = 60) -> pd.DataFrame:
    """
    Add Opening Range Breakout (ORB) levels.
    
    Calculates the high and low of the first hour (or specified minutes) of each trading day.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with datetime index
    orb_minutes : int
        Minutes for opening range (default: 60 = first hour)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - orb_high: Opening range high
        - orb_low: Opening range low
        - orb_breakout_long: boolean, price breaks above ORB high
        - orb_breakout_short: boolean, price breaks below ORB low
    """
    df = df.copy()
    
    # Get the date and time components
    df['trade_date'] = df.index.date
    
    # Determine the start of the trading day
    # For forex, we'll use 00:00 UTC as the start
    day_start_hour = 0
    
    # Calculate minutes from day start
    minutes_from_start = df.index.hour * 60 + df.index.minute
    
    # Identify candles within the opening range
    orb_mask = minutes_from_start < orb_minutes
    
    # Initialize ORB columns
    df['orb_high'] = np.nan
    df['orb_low'] = np.nan
    
    # Calculate ORB for each day
    if orb_mask.any():
        orb_highs = df.loc[orb_mask].groupby('trade_date')['high'].transform('max')
        orb_lows = df.loc[orb_mask].groupby('trade_date')['low'].transform('min')
        df.loc[orb_mask, 'orb_high'] = orb_highs
        df.loc[orb_mask, 'orb_low'] = orb_lows
    
    # Forward fill ORB levels throughout the day
    df['orb_high'] = df.groupby('trade_date')['orb_high'].ffill()
    df['orb_low'] = df.groupby('trade_date')['orb_low'].ffill()
    
    # Detect breakouts (only after ORB period)
    after_orb = minutes_from_start >= orb_minutes
    df['orb_breakout_long'] = after_orb & (df['high'] > df['orb_high']) & (df['high'].shift(1) <= df['orb_high'].shift(1))
    df['orb_breakout_short'] = after_orb & (df['low'] < df['orb_low']) & (df['low'].shift(1) >= df['orb_low'].shift(1))
    
    # Clean up helper column
    df = df.drop(columns=['trade_date'])
    
    return df


# =============================================================================
# ADDITIONAL INDICATORS
# =============================================================================

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add Average True Range (ATR) indicator.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame
    period : int
        ATR period (default: 14)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'atr' column
    """
    df = df.copy()
    
    # Calculate True Range
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    
    # Calculate ATR using exponential moving average
    df['atr'] = true_range.ewm(span=period, adjust=False).mean()
    
    return df


def add_ema(df: pd.DataFrame, period: int = 200) -> pd.DataFrame:
    """
    Add Exponential Moving Average (EMA).
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame
    period : int
        EMA period (default: 200)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'ema_{period}' column
    """
    df = df.copy()
    df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    return df


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def add_all_indicators(
    df: pd.DataFrame,
    swing_period: int = 5,
    displacement_multiplier: float = 2.0,
    atr_period: int = 14,
    ema_period: int = 200,
    orb_minutes: int = 60,
    warmup_period: int = 50,
    use_legacy_market_structure: bool = False,
) -> pd.DataFrame:
    """
    Add all SMC indicators to the DataFrame in the correct order.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns: open, high, low, close, volume
    swing_period : int
        Period for swing detection (default: 5) - used only in legacy mode
    displacement_multiplier : float
        Multiplier for displacement detection (default: 2.0)
    atr_period : int
        ATR period (default: 14)
    ema_period : int
        EMA period (default: 200)
    orb_minutes : int
        Opening range minutes (default: 60)
    warmup_period : int
        Warmup period for SMC market structure (default: 50)
    use_legacy_market_structure : bool
        If True, use old swing-based market structure detection (default: False)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with all indicator columns added
    """
    # Start with a copy
    df = df.copy()
    
    # 1. Basic indicators (needed for some SMC calculations)
    df = add_atr(df, period=atr_period)
    df = add_ema(df, period=ema_period)
    
    # 2. Market Structure
    if use_legacy_market_structure:
        # Legacy: Fixed swing detection with argrelextrema
        df = add_swing_highs_lows(df, n=swing_period)
        df = add_market_structure_labels(df)
        df = add_mss_bos(df)
    else:
        # New: SMC state machine with strict rules
        # - Body close only for breaks
        # - Retroactive HL/LH confirmation
        # - High-Quality BoS with 50% equilibrium
        df = add_smc_market_structure(df, warmup_period=warmup_period)
        
        # Add legacy swing columns for compatibility with liquidity sweeps
        # These are needed for add_liquidity_sweeps which uses last_swing_high/low
        df = add_swing_highs_lows(df, n=swing_period)
    
    # 3. Smart Money Concepts
    df = add_displacement(df, multiplier=displacement_multiplier)
    df = add_fvg(df)
    df = add_order_blocks(df)
    df = add_liquidity_sweeps(df)
    
    # 4. Time & Session
    df = add_sessions(df)
    df = add_orb(df, orb_minutes=orb_minutes)
    
    return df

