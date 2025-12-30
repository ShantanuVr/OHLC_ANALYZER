'use client';

import { useState } from 'react';
import { useBacktestStore, EntryConditions } from '@/store/useBacktestStore';
import { 
  getIndicatorData, 
  runBacktest, 
  runOptimization,
  BacktestRequest,
  OptimizationRequest,
} from '@/lib/api';
import clsx from 'clsx';

interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  description?: string;
}

function Toggle({ checked, onChange, label, description }: ToggleProps) {
  return (
    <label className="flex items-center justify-between py-2 cursor-pointer group">
      <div>
        <span className="text-sm text-zinc-300 group-hover:text-white transition-colors">
          {label}
        </span>
        {description && (
          <p className="text-xs text-zinc-600">{description}</p>
        )}
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        data-state={checked ? 'checked' : 'unchecked'}
        className="toggle"
        onClick={() => onChange(!checked)}
      >
        <span className="toggle-thumb" />
      </button>
    </label>
  );
}

export default function StrategyBuilder() {
  const {
    params,
    optimizationParams,
    activeTab,
    isLoading,
    showIndicators,
    setParams,
    setOptimizationParams,
    setEntryCondition,
    setIndicatorData,
    setBacktestResult,
    setOptimizationResults,
    setIsLoading,
    setError,
    setActiveTab,
    toggleIndicator,
    resetEntryConditions,
  } = useBacktestStore();

  const [loadingData, setLoadingData] = useState(false);

  const handleLoadData = async () => {
    setLoadingData(true);
    setError(null);
    try {
      const data = await getIndicatorData(
        params.symbol,
        params.timeframe,
        params.startDate || undefined,
        params.endDate || undefined
      );
      setIndicatorData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setLoadingData(false);
    }
  };

  const handleRunBacktest = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const request: BacktestRequest = {
        symbol: params.symbol,
        timeframe: params.timeframe,
        start_date: params.startDate || undefined,
        end_date: params.endDate || undefined,
        entry_conditions: params.entryConditions as unknown as Record<string, boolean | string>,
        stop_loss_type: params.stopLossType,
        stop_loss_value: params.stopLossValue,
        take_profit_rrr: params.takeProfitRRR,
        direction: params.direction,
      };
      const result = await runBacktest(request);
      setBacktestResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Backtest failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunOptimization = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const request: OptimizationRequest = {
        symbol: params.symbol,
        timeframe: params.timeframe,
        start_date: params.startDate || undefined,
        end_date: params.endDate || undefined,
        entry_conditions: params.entryConditions as unknown as Record<string, boolean | string>,
        stop_loss_type: params.stopLossType,
        sl_min: optimizationParams.slMin,
        sl_max: optimizationParams.slMax,
        sl_steps: optimizationParams.slSteps,
        rrr_min: optimizationParams.rrrMin,
        rrr_max: optimizationParams.rrrMax,
        rrr_steps: optimizationParams.rrrSteps,
        direction: params.direction,
      };
      const result = await runOptimization(request);
      setOptimizationResults(result.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Optimization failed');
    } finally {
      setIsLoading(false);
    }
  };

  const entryConditionGroups = [
    {
      title: 'Fair Value Gap',
      conditions: [
        { key: 'fvg_bullish' as keyof EntryConditions, label: 'Bullish FVG' },
        { key: 'fvg_bearish' as keyof EntryConditions, label: 'Bearish FVG' },
      ],
    },
    {
      title: 'Order Blocks',
      conditions: [
        { key: 'ob_bullish' as keyof EntryConditions, label: 'Bullish OB' },
        { key: 'ob_bearish' as keyof EntryConditions, label: 'Bearish OB' },
      ],
    },
    {
      title: 'Liquidity Sweeps',
      conditions: [
        { key: 'sweep_bullish' as keyof EntryConditions, label: 'SSL Sweep (Bullish)' },
        { key: 'sweep_bearish' as keyof EntryConditions, label: 'BSL Sweep (Bearish)' },
      ],
    },
    {
      title: 'Market Structure',
      conditions: [
        { key: 'mss_bullish' as keyof EntryConditions, label: 'MSS Bullish' },
        { key: 'mss_bearish' as keyof EntryConditions, label: 'MSS Bearish' },
        { key: 'bos_bullish' as keyof EntryConditions, label: 'BoS Bullish' },
        { key: 'bos_bearish' as keyof EntryConditions, label: 'BoS Bearish' },
      ],
    },
    {
      title: 'Displacement',
      conditions: [
        { key: 'displacement_bullish' as keyof EntryConditions, label: 'Bullish Displacement' },
        { key: 'displacement_bearish' as keyof EntryConditions, label: 'Bearish Displacement' },
      ],
    },
    {
      title: 'ORB Breakout',
      conditions: [
        { key: 'orb_breakout_long' as keyof EntryConditions, label: 'ORB Long' },
        { key: 'orb_breakout_short' as keyof EntryConditions, label: 'ORB Short' },
      ],
    },
    {
      title: 'EMA Filter',
      conditions: [
        { key: 'above_ema' as keyof EntryConditions, label: 'Above EMA 200' },
        { key: 'below_ema' as keyof EntryConditions, label: 'Below EMA 200' },
      ],
    },
  ];

  return (
    <div className="h-full overflow-y-auto">
      <div className="p-4 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gold-400">Strategy Builder</h2>
          <button
            onClick={resetEntryConditions}
            className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            Reset All
          </button>
        </div>

        {/* Data Loading */}
        <div className="card p-4 space-y-4">
          <h3 className="text-sm font-medium text-zinc-400">Data Selection</h3>
          
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-zinc-500 mb-1 block">Symbol</label>
              <select
                value={params.symbol}
                onChange={(e) => setParams({ symbol: e.target.value })}
              >
                <option value="XAUUSD">XAUUSD</option>
                <option value="XAGUSD">XAGUSD</option>
              </select>
            </div>
            <div>
              <label className="text-xs text-zinc-500 mb-1 block">Timeframe</label>
              <select
                value={params.timeframe}
                onChange={(e) => setParams({ timeframe: e.target.value })}
              >
                <option value="1m">1 Minute</option>
                <option value="5m">5 Minutes</option>
                <option value="15m">15 Minutes</option>
                <option value="1h">1 Hour</option>
                <option value="4h">4 Hours</option>
                <option value="1d">1 Day</option>
                <option value="1w">1 Week</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-zinc-500 mb-1 block">Start Date</label>
              <input
                type="date"
                value={params.startDate}
                onChange={(e) => setParams({ startDate: e.target.value })}
                className="w-full px-3 py-2 rounded-md border bg-[#0a0a0f] border-zinc-800 text-zinc-300 text-sm"
              />
            </div>
            <div>
              <label className="text-xs text-zinc-500 mb-1 block">End Date</label>
              <input
                type="date"
                value={params.endDate}
                onChange={(e) => setParams({ endDate: e.target.value })}
                className="w-full px-3 py-2 rounded-md border bg-[#0a0a0f] border-zinc-800 text-zinc-300 text-sm"
              />
            </div>
          </div>

          <button
            onClick={handleLoadData}
            disabled={loadingData}
            className="btn btn-secondary w-full"
          >
            {loadingData ? 'Loading...' : 'Load Data & Indicators'}
          </button>
        </div>

        {/* Chart Indicators */}
        <div className="card p-4 space-y-3">
          <h3 className="text-sm font-medium text-zinc-400">Chart Display</h3>
          <div className="space-y-1">
            <Toggle
              checked={showIndicators.swings}
              onChange={() => toggleIndicator('swings')}
              label="Swing Points"
            />
            <Toggle
              checked={showIndicators.fvg}
              onChange={() => toggleIndicator('fvg')}
              label="Fair Value Gaps"
            />
            <Toggle
              checked={showIndicators.ob}
              onChange={() => toggleIndicator('ob')}
              label="Order Blocks"
            />
            <Toggle
              checked={showIndicators.sweeps}
              onChange={() => toggleIndicator('sweeps')}
              label="Liquidity Sweeps"
            />
            <Toggle
              checked={showIndicators.sessions}
              onChange={() => toggleIndicator('sessions')}
              label="Sessions"
            />
          </div>
        </div>

        {/* Tab Selection */}
        <div className="flex rounded-lg overflow-hidden border border-zinc-800">
          <button
            onClick={() => setActiveTab('backtest')}
            className={clsx(
              'flex-1 py-2 text-sm font-medium transition-colors',
              activeTab === 'backtest'
                ? 'bg-gold-500 text-black'
                : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200'
            )}
          >
            Backtest
          </button>
          <button
            onClick={() => setActiveTab('optimize')}
            className={clsx(
              'flex-1 py-2 text-sm font-medium transition-colors',
              activeTab === 'optimize'
                ? 'bg-gold-500 text-black'
                : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200'
            )}
          >
            Optimize
          </button>
        </div>

        {/* Entry Conditions */}
        <div className="card p-4 space-y-4">
          <h3 className="text-sm font-medium text-zinc-400">Entry Conditions</h3>
          
          {entryConditionGroups.map((group) => (
            <div key={group.title} className="space-y-1">
              <div className="text-xs text-zinc-600 uppercase tracking-wider">
                {group.title}
              </div>
              {group.conditions.map((condition) => (
                <Toggle
                  key={condition.key}
                  checked={params.entryConditions[condition.key] as boolean}
                  onChange={(checked) => setEntryCondition(condition.key, checked)}
                  label={condition.label}
                />
              ))}
            </div>
          ))}

          {/* Session Filter */}
          <div className="space-y-1">
            <div className="text-xs text-zinc-600 uppercase tracking-wider">Session Filter</div>
            <select
              value={params.entryConditions.session}
              onChange={(e) => setEntryCondition('session', e.target.value)}
            >
              <option value="">All Sessions</option>
              <option value="asian">Asian (00:00-08:00 UTC)</option>
              <option value="london">London (08:00-16:00 UTC)</option>
              <option value="ny">New York (13:00-21:00 UTC)</option>
            </select>
          </div>
        </div>

        {/* Exit Parameters */}
        <div className="card p-4 space-y-4">
          <h3 className="text-sm font-medium text-zinc-400">Exit Parameters</h3>
          
          <div>
            <label className="text-xs text-zinc-500 mb-1 block">Direction</label>
            <select
              value={params.direction}
              onChange={(e) => setParams({ direction: e.target.value as 'long' | 'short' | 'both' })}
            >
              <option value="both">Both Long & Short</option>
              <option value="long">Long Only</option>
              <option value="short">Short Only</option>
            </select>
          </div>

          <div>
            <label className="text-xs text-zinc-500 mb-1 block">Stop Loss Type</label>
            <select
              value={params.stopLossType}
              onChange={(e) => setParams({ stopLossType: e.target.value as 'lowest_low_n' | 'atr_multiplier' | 'fixed_pips' })}
            >
              <option value="lowest_low_n">Lowest Low (N Candles)</option>
              <option value="atr_multiplier">ATR Multiplier</option>
              <option value="fixed_pips">Fixed Pips</option>
            </select>
          </div>

          {activeTab === 'backtest' ? (
            <>
              <div>
                <label className="text-xs text-zinc-500 mb-1 block">
                  Stop Loss Value
                </label>
                <input
                  type="number"
                  value={params.stopLossValue}
                  onChange={(e) => setParams({ stopLossValue: parseFloat(e.target.value) })}
                  min={0.5}
                  step={0.5}
                />
              </div>
              <div>
                <label className="text-xs text-zinc-500 mb-1 block">
                  Take Profit RRR
                </label>
                <input
                  type="number"
                  value={params.takeProfitRRR}
                  onChange={(e) => setParams({ takeProfitRRR: parseFloat(e.target.value) })}
                  min={0.5}
                  max={10}
                  step={0.5}
                />
              </div>
            </>
          ) : (
            <>
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <label className="text-xs text-zinc-500 mb-1 block">SL Min</label>
                  <input
                    type="number"
                    value={optimizationParams.slMin}
                    onChange={(e) => setOptimizationParams({ slMin: parseFloat(e.target.value) })}
                    min={0.5}
                    step={0.5}
                  />
                </div>
                <div>
                  <label className="text-xs text-zinc-500 mb-1 block">SL Max</label>
                  <input
                    type="number"
                    value={optimizationParams.slMax}
                    onChange={(e) => setOptimizationParams({ slMax: parseFloat(e.target.value) })}
                    min={0.5}
                    step={0.5}
                  />
                </div>
                <div>
                  <label className="text-xs text-zinc-500 mb-1 block">Steps</label>
                  <input
                    type="number"
                    value={optimizationParams.slSteps}
                    onChange={(e) => setOptimizationParams({ slSteps: parseInt(e.target.value) })}
                    min={2}
                    max={20}
                  />
                </div>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <label className="text-xs text-zinc-500 mb-1 block">RRR Min</label>
                  <input
                    type="number"
                    value={optimizationParams.rrrMin}
                    onChange={(e) => setOptimizationParams({ rrrMin: parseFloat(e.target.value) })}
                    min={0.5}
                    step={0.5}
                  />
                </div>
                <div>
                  <label className="text-xs text-zinc-500 mb-1 block">RRR Max</label>
                  <input
                    type="number"
                    value={optimizationParams.rrrMax}
                    onChange={(e) => setOptimizationParams({ rrrMax: parseFloat(e.target.value) })}
                    min={0.5}
                    step={0.5}
                  />
                </div>
                <div>
                  <label className="text-xs text-zinc-500 mb-1 block">Steps</label>
                  <input
                    type="number"
                    value={optimizationParams.rrrSteps}
                    onChange={(e) => setOptimizationParams({ rrrSteps: parseInt(e.target.value) })}
                    min={2}
                    max={20}
                  />
                </div>
              </div>
            </>
          )}
        </div>

        {/* Run Button */}
        <button
          onClick={activeTab === 'backtest' ? handleRunBacktest : handleRunOptimization}
          disabled={isLoading}
          className={clsx(
            'btn w-full py-3 font-semibold',
            isLoading ? 'bg-zinc-700 text-zinc-400' : 'btn-primary'
          )}
        >
          {isLoading 
            ? 'Running...' 
            : activeTab === 'backtest' 
              ? 'Run Backtest' 
              : 'Run Optimization'
          }
        </button>
      </div>
    </div>
  );
}

