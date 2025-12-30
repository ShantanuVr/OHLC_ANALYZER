# OHLC Analyzer Frontend

Next.js frontend for the No-Code Backtesting Platform.

## Features

- **TradingView Charts**: Lightweight Charts integration with OHLCV display
- **Indicator Overlays**: FVG, Order Blocks, Swing points, Liquidity sweeps
- **Strategy Builder**: No-code strategy configuration with SMC concepts
- **Optimization**: Grid search over RRR and Stop Loss parameters
- **Results Dashboard**: Sortable tables with key performance metrics

## Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the application.

## Tech Stack

- **Next.js 14** with App Router
- **TypeScript**
- **Tailwind CSS** for styling
- **Zustand** for state management
- **TradingView Lightweight Charts** for charting
- **Axios** for API calls

## Environment Variables

Create a `.env.local` file:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

