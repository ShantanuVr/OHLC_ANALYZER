# OHLC Analyzer

A **No-Code Backtesting Platform** for XAUUSD trading with Smart Money Concepts (SMC) indicators.

## Features

### Smart Money Concepts Indicators
- **Swing Highs/Lows**: Identified using `argrelextrema` with configurable lookback
- **Market Structure**: HH, LH, LL, HL labeling with MSS and BoS detection
- **Fair Value Gaps (FVG)**: Bullish and bearish imbalances
- **Order Blocks (OB)**: Last opposite candle before displacement
- **Liquidity Sweeps**: BSL/SSL detection for stop hunts
- **Displacement**: Large body candles indicating institutional activity

### Session Analysis
- Asian Session (00:00-08:00 UTC)
- London Session (08:00-16:00 UTC)  
- New York Session (13:00-21:00 UTC)
- Opening Range Breakout (ORB) levels

### Backtesting Engine
- Vectorized calculations for 23 years of data
- Multiple stop loss types (Lowest Low, ATR Multiplier, Fixed Pips)
- Configurable Risk:Reward ratios
- Parameter optimization with grid search

## Tech Stack

- **Backend**: Python (FastAPI) with Pandas & NumPy
- **Frontend**: Next.js with TypeScript & Tailwind CSS
- **Charting**: TradingView Lightweight Charts
- **State Management**: Zustand

## Project Structure

```
OHLC_ANALYZER/
├── backend/                 # Python FastAPI backend
│   ├── app/
│   │   ├── api/            # API routes & schemas
│   │   ├── core/           # Config & data loading
│   │   ├── indicators/     # SMC indicator calculations
│   │   └── engine/         # Backtesting engine
│   └── requirements.txt
├── frontend/               # Next.js frontend
│   ├── src/
│   │   ├── app/           # Next.js pages
│   │   ├── components/    # React components
│   │   ├── store/         # Zustand store
│   │   └── lib/           # API client
│   └── package.json
└── README.md
```

## Quick Start

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Data Requirements

Place your OHLCV parquet files in the configured data path (default: `/Volumes/Extreme SSD/ohlcv_data`).

Expected structure:
```
ohlcv_data/
└── normalized/
    └── tf={timeframe}/
        └── symbol={symbol}/
            └── year={year}/
                └── month={month}/
                    └── part.parquet
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/symbols` | Available symbols |
| GET | `/api/timeframes` | Available timeframes |
| POST | `/api/data/ohlcv` | Get OHLCV data |
| POST | `/api/indicators/calculate` | Calculate SMC indicators |
| POST | `/api/backtest/run` | Run single backtest |
| POST | `/api/backtest/optimize` | Optimize parameters |

## License

MIT

