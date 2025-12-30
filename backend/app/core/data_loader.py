"""
Data loader for partitioned parquet OHLCV files.
"""
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import pandas as pd
import pyarrow.parquet as pq

from app.core.config import settings


def get_parquet_files(
    symbol: str = "XAUUSD",
    timeframe: str = "1h",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[Path]:
    """
    Get list of parquet files matching the given filters.
    
    Data is partitioned as:
    normalized/tf={timeframe}/symbol={symbol}/year={year}/month={month}/part.parquet
    """
    base_path = settings.NORMALIZED_DATA_PATH / f"tf={timeframe}" / f"symbol={symbol}"
    
    if not base_path.exists():
        raise FileNotFoundError(f"No data found for {symbol} at {timeframe} timeframe")
    
    parquet_files = []
    
    # Iterate through year directories
    for year_dir in sorted(base_path.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.startswith("year="):
            continue
        
        year = int(year_dir.name.split("=")[1])
        
        # Apply year filter if dates provided
        if start_date and year < start_date.year:
            continue
        if end_date and year > end_date.year:
            continue
        
        # Iterate through month directories
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir() or not month_dir.name.startswith("month="):
                continue
            
            month = int(month_dir.name.split("=")[1])
            
            # Apply month filter for boundary years
            if start_date and year == start_date.year and month < start_date.month:
                continue
            if end_date and year == end_date.year and month > end_date.month:
                continue
            
            # Find parquet files in this month directory
            # Filter out macOS resource fork files (._*)
            for pq_file in month_dir.glob("*.parquet"):
                if not pq_file.name.startswith("._"):
                    parquet_files.append(pq_file)
    
    return sorted(parquet_files)


def load_ohlcv_data(
    symbol: str = "XAUUSD",
    timeframe: str = "1h",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Load OHLCV data from partitioned parquet files.
    
    Parameters
    ----------
    symbol : str
        Trading symbol (default: "XAUUSD")
    timeframe : str
        Timeframe string (e.g., "1m", "5m", "15m", "1h", "4h", "1d", "1w")
    start_date : datetime, optional
        Start date filter
    end_date : datetime, optional
        End date filter
    
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and columns: open, high, low, close, volume
    """
    parquet_files = get_parquet_files(symbol, timeframe, start_date, end_date)
    
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found for {symbol} at {timeframe} "
            f"between {start_date} and {end_date}"
        )
    
    # Read all parquet files and concatenate
    dfs = []
    for pq_file in parquet_files:
        df = pd.read_parquet(pq_file)
        dfs.append(df)
    
    # Concatenate preserving the index (which is already timestamp)
    df = pd.concat(dfs)
    
    # If index is already a DatetimeIndex, use it; otherwise look for timestamp column
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to find a timestamp column
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        else:
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                df = df.set_index(datetime_cols[0])
            elif 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'])
                df = df.set_index('timestamp')
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
                df = df.set_index('timestamp')
            else:
                raise ValueError("No timestamp column found in parquet files")
    
    # Ensure index name is 'timestamp'
    df.index.name = 'timestamp'
    
    # Sort by index
    df = df.sort_index()
    
    # Remove timezone info for consistency (convert to UTC then make naive)
    if df.index.tz is not None:
        df.index = df.index.tz_convert('UTC').tz_localize(None)
    
    # Standardize column names to lowercase
    df.columns = df.columns.str.lower()
    
    # Ensure we have the required columns
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Add volume column if missing
    if 'volume' not in df.columns:
        df['volume'] = 0
    
    # Apply precise date filtering
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    
    # Select only the required columns
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    
    # Ensure correct dtypes for performance
    df['open'] = df['open'].astype('float64')
    df['high'] = df['high'].astype('float64')
    df['low'] = df['low'].astype('float64')
    df['close'] = df['close'].astype('float64')
    df['volume'] = df['volume'].astype('float64')
    
    return df


def get_data_info(symbol: str = "XAUUSD", timeframe: str = "1h") -> dict:
    """
    Get information about available data for a symbol/timeframe.
    """
    try:
        files = get_parquet_files(symbol, timeframe)
        if not files:
            return {"available": False}
        
        # Load first and last files to get date range
        first_df = pd.read_parquet(files[0])
        last_df = pd.read_parquet(files[-1])
        
        # Find timestamp column
        ts_col = None
        for col in ['timestamp', 'time', 'date']:
            if col in first_df.columns:
                ts_col = col
                break
        
        if ts_col is None:
            datetime_cols = first_df.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                ts_col = datetime_cols[0]
        
        if ts_col:
            start_date = pd.to_datetime(first_df[ts_col].min())
            end_date = pd.to_datetime(last_df[ts_col].max())
        else:
            start_date = None
            end_date = None
        
        return {
            "available": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "file_count": len(files),
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}

