import { create } from 'zustand';
import { 
  OHLCVData, 
  IndicatorData, 
  BacktestResult, 
  OptimizationResult 
} from '@/lib/api';

export interface EntryConditions {
  fvg_bullish: boolean;
  fvg_bearish: boolean;
  ob_bullish: boolean;
  ob_bearish: boolean;
  sweep_bullish: boolean;
  sweep_bearish: boolean;
  mss_bullish: boolean;
  mss_bearish: boolean;
  bos_bullish: boolean;
  bos_bearish: boolean;
  displacement_bullish: boolean;
  displacement_bearish: boolean;
  orb_breakout_long: boolean;
  orb_breakout_short: boolean;
  above_ema: boolean;
  below_ema: boolean;
  session: string;
}

export interface StrategyParams {
  symbol: string;
  timeframe: string;
  startDate: string;
  endDate: string;
  direction: 'long' | 'short' | 'both';
  stopLossType: 'lowest_low_n' | 'atr_multiplier' | 'fixed_pips';
  stopLossValue: number;
  takeProfitRRR: number;
  entryConditions: EntryConditions;
}

export interface OptimizationParams {
  slMin: number;
  slMax: number;
  slSteps: number;
  rrrMin: number;
  rrrMax: number;
  rrrSteps: number;
}

interface BacktestState {
  // Data
  ohlcvData: OHLCVData[];
  indicatorData: IndicatorData[];
  
  // Strategy parameters
  params: StrategyParams;
  optimizationParams: OptimizationParams;
  
  // Results
  backtestResult: BacktestResult | null;
  optimizationResults: OptimizationResult[];
  
  // UI state
  isLoading: boolean;
  error: string | null;
  activeTab: 'backtest' | 'optimize';
  showIndicators: {
    swings: boolean;
    fvg: boolean;
    ob: boolean;
    sweeps: boolean;
    sessions: boolean;
    mss: boolean;
    bos: boolean;
  };
  
  // Lazy loading state
  loadedDateRange: { start: string; end: string } | null;
  isLoadingMore: boolean;
  
  // Actions
  setOHLCVData: (data: OHLCVData[]) => void;
  setIndicatorData: (data: IndicatorData[]) => void;
  appendIndicatorData: (data: IndicatorData[], prepend?: boolean) => void;
  setLoadedDateRange: (range: { start: string; end: string } | null) => void;
  setIsLoadingMore: (loading: boolean) => void;
  setParams: (params: Partial<StrategyParams>) => void;
  setOptimizationParams: (params: Partial<OptimizationParams>) => void;
  setEntryCondition: (key: keyof EntryConditions, value: boolean | string) => void;
  setBacktestResult: (result: BacktestResult | null) => void;
  setOptimizationResults: (results: OptimizationResult[]) => void;
  setIsLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setActiveTab: (tab: 'backtest' | 'optimize') => void;
  toggleIndicator: (indicator: keyof BacktestState['showIndicators']) => void;
  resetEntryConditions: () => void;
}

const defaultEntryConditions: EntryConditions = {
  fvg_bullish: false,
  fvg_bearish: false,
  ob_bullish: false,
  ob_bearish: false,
  sweep_bullish: false,
  sweep_bearish: false,
  mss_bullish: false,
  mss_bearish: false,
  bos_bullish: false,
  bos_bearish: false,
  displacement_bullish: false,
  displacement_bearish: false,
  orb_breakout_long: false,
  orb_breakout_short: false,
  above_ema: false,
  below_ema: false,
  session: '',
};

const defaultParams: StrategyParams = {
  symbol: 'XAUUSD',
  timeframe: '1h',
  startDate: '',
  endDate: '',
  direction: 'both',
  stopLossType: 'lowest_low_n',
  stopLossValue: 3,
  takeProfitRRR: 2,
  entryConditions: defaultEntryConditions,
};

const defaultOptimizationParams: OptimizationParams = {
  slMin: 1,
  slMax: 5,
  slSteps: 5,
  rrrMin: 1,
  rrrMax: 5,
  rrrSteps: 5,
};

export const useBacktestStore = create<BacktestState>((set) => ({
  // Initial state
  ohlcvData: [],
  indicatorData: [],
  params: defaultParams,
  optimizationParams: defaultOptimizationParams,
  backtestResult: null,
  optimizationResults: [],
  isLoading: false,
  error: null,
  activeTab: 'backtest',
  showIndicators: {
    swings: false,
    fvg: false,
    ob: false,
    sweeps: false,
    sessions: false,
    mss: false,
    bos: false,
  },
  loadedDateRange: null,
  isLoadingMore: false,
  
  // Actions
  setOHLCVData: (data) => set({ ohlcvData: data }),
  setIndicatorData: (data) => set({ indicatorData: data, loadedDateRange: null }),
  appendIndicatorData: (data, prepend = false) =>
    set((state) => {
      // Remove duplicates based on time
      const existingTimes = new Set(state.indicatorData.map(d => d.time));
      const newData = data.filter(d => !existingTimes.has(d.time));
      
      if (prepend) {
        // Prepend older data (for scrolling to start) - sort by time
        const combined = [...newData, ...state.indicatorData];
        combined.sort((a, b) => (a.time as number) - (b.time as number));
        return { indicatorData: combined };
      } else {
        // Append newer data (for scrolling to end) - sort by time
        const combined = [...state.indicatorData, ...newData];
        combined.sort((a, b) => (a.time as number) - (b.time as number));
        return { indicatorData: combined };
      }
    }),
  setLoadedDateRange: (range) => set({ loadedDateRange: range }),
  setIsLoadingMore: (loading) => set({ isLoadingMore: loading }),
  
  setParams: (newParams) =>
    set((state) => ({
      params: { ...state.params, ...newParams },
    })),
  
  setOptimizationParams: (newParams) =>
    set((state) => ({
      optimizationParams: { ...state.optimizationParams, ...newParams },
    })),
  
  setEntryCondition: (key, value) =>
    set((state) => ({
      params: {
        ...state.params,
        entryConditions: {
          ...state.params.entryConditions,
          [key]: value,
        },
      },
    })),
  
  setBacktestResult: (result) => set({ backtestResult: result }),
  setOptimizationResults: (results) => set({ optimizationResults: results }),
  setIsLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
  setActiveTab: (tab) => set({ activeTab: tab }),
  
  toggleIndicator: (indicator) =>
    set((state) => ({
      showIndicators: {
        ...state.showIndicators,
        [indicator]: !state.showIndicators[indicator],
      },
    })),
  
  resetEntryConditions: () =>
    set((state) => ({
      params: {
        ...state.params,
        entryConditions: defaultEntryConditions,
      },
    })),
}));

