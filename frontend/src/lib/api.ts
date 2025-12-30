import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export interface OHLCVData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface IndicatorData extends OHLCVData {
  swing_high: boolean;
  swing_low: boolean;
  fvg_bullish: boolean;
  fvg_bearish: boolean;
  fvg_top: number | null;
  fvg_bottom: number | null;
  ob_bullish: boolean;
  ob_bearish: boolean;
  ob_top: number | null;
  ob_bottom: number | null;
  sweep_bullish: boolean;
  sweep_bearish: boolean;
  mss_bullish: boolean;
  mss_bearish: boolean;
  bos_bullish: boolean;
  bos_bearish: boolean;
  bos_hq_bullish?: boolean;
  bos_hq_bearish?: boolean;
  confirmed_hh?: boolean | number;
  confirmed_hl_at_idx?: boolean | number;
  confirmed_lh_at_idx?: boolean | number;
  confirmed_ll?: boolean | number;
  session: string;
  [key: string]: unknown;
}

export interface BacktestRequest {
  symbol: string;
  timeframe: string;
  start_date?: string;
  end_date?: string;
  entry_conditions: Record<string, boolean | string>;
  stop_loss_type: 'lowest_low_n' | 'atr_multiplier' | 'fixed_pips';
  stop_loss_value: number;
  take_profit_rrr: number;
  direction: 'long' | 'short' | 'both';
}

export interface BacktestResult {
  win_rate: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  profit_factor: number;
  max_drawdown: number;
  total_pnl: number;
  avg_win: number;
  avg_loss: number;
  trades?: TradeResult[];
}

export interface TradeResult {
  entry_time: string;
  entry_price: number;
  exit_time: string;
  exit_price: number;
  direction: string;
  pnl: number;
  pnl_percent: number;
  is_winner: boolean;
}

export interface OptimizationRequest {
  symbol: string;
  timeframe: string;
  start_date?: string;
  end_date?: string;
  entry_conditions: Record<string, boolean | string>;
  stop_loss_type: 'lowest_low_n' | 'atr_multiplier' | 'fixed_pips';
  sl_min: number;
  sl_max: number;
  sl_steps: number;
  rrr_min: number;
  rrr_max: number;
  rrr_steps: number;
  direction: 'long' | 'short' | 'both';
}

export interface OptimizationResult {
  rrr: number;
  stop_loss: number;
  win_rate: number;
  total_trades: number;
  profit_factor: number;
  max_drawdown: number;
  total_pnl: number;
}

export interface OptimizationResponse {
  results: OptimizationResult[];
  best_by_profit_factor: OptimizationResult;
  best_by_win_rate: OptimizationResult;
}

// API Functions
export async function getOHLCVData(
  symbol: string = 'XAUUSD',
  timeframe: string = '1h',
  startDate?: string,
  endDate?: string
): Promise<OHLCVData[]> {
  const response = await api.post('/api/data/ohlcv', {
    symbol,
    timeframe,
    start_date: startDate,
    end_date: endDate,
  });
  return response.data.data;
}

export async function getIndicatorData(
  symbol: string = 'XAUUSD',
  timeframe: string = '1h',
  startDate?: string,
  endDate?: string,
  swingPeriod: number = 5
): Promise<IndicatorData[]> {
  const response = await api.post('/api/indicators/calculate', {
    symbol,
    timeframe,
    start_date: startDate,
    end_date: endDate,
    swing_period: swingPeriod,
  });
  return response.data.data;
}

export async function runBacktest(request: BacktestRequest): Promise<BacktestResult> {
  const response = await api.post('/api/backtest/run', request);
  return response.data;
}

export async function runOptimization(request: OptimizationRequest): Promise<OptimizationResponse> {
  const response = await api.post('/api/backtest/optimize', request);
  return response.data;
}

export async function getSymbols(): Promise<string[]> {
  const response = await api.get('/api/symbols');
  return response.data.symbols;
}

export async function getTimeframes(): Promise<string[]> {
  const response = await api.get('/api/timeframes');
  return response.data.timeframes;
}

