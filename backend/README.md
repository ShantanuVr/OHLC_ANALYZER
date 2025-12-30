# OHLC Analyzer Backend

Python FastAPI backend for the No-Code Backtesting Platform.

## Features

- **Smart Money Concepts (SMC) Indicators**: Swing highs/lows, FVG, Order Blocks, Liquidity Sweeps, MSS/BoS
- **Session Analysis**: Asian, London, NY session ranges and ORB detection
- **Vectorized Backtesting**: Fast backtesting engine using NumPy/Pandas
- **Parameter Optimization**: Grid search over RRR and Stop Loss ranges

## Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /health` - Health check
- `GET /api/symbols` - Available trading symbols
- `GET /api/timeframes` - Available timeframes
- `POST /api/data/ohlcv` - Get OHLCV data
- `POST /api/indicators/calculate` - Calculate all SMC indicators
- `POST /api/backtest/run` - Run single backtest
- `POST /api/backtest/optimize` - Optimize parameters

## Configuration

Data path is configured in `app/core/config.py`. Default: `/Volumes/Extreme SSD/ohlcv_data`

