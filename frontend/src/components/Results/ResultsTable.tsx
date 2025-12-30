'use client';

import { useState, useMemo } from 'react';
import { useBacktestStore } from '@/store/useBacktestStore';
import { OptimizationResult } from '@/lib/api';
import clsx from 'clsx';

type SortKey = keyof OptimizationResult;
type SortDirection = 'asc' | 'desc';

function StatCard({ 
  label, 
  value, 
  subValue,
  positive 
}: { 
  label: string; 
  value: string | number; 
  subValue?: string;
  positive?: boolean;
}) {
  return (
    <div className="card stat-card p-4">
      <div className="text-xs text-zinc-500 uppercase tracking-wider mb-1">{label}</div>
      <div className={clsx(
        'text-2xl font-semibold',
        positive === true && 'text-green-400',
        positive === false && 'text-red-400',
        positive === undefined && 'text-gold-400'
      )}>
        {value}
      </div>
      {subValue && (
        <div className="text-xs text-zinc-600 mt-1">{subValue}</div>
      )}
    </div>
  );
}

export default function ResultsTable() {
  const { backtestResult, optimizationResults, activeTab, isLoading } = useBacktestStore();
  
  const [sortKey, setSortKey] = useState<SortKey>('profit_factor');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const sortedResults = useMemo(() => {
    if (!optimizationResults.length) return [];
    
    return [...optimizationResults].sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];
      const modifier = sortDirection === 'asc' ? 1 : -1;
      return (aVal - bVal) * modifier;
    });
  }, [optimizationResults, sortKey, sortDirection]);

  const handleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDirection('desc');
    }
  };

  const getSortIndicator = (key: SortKey) => {
    if (key !== sortKey) return null;
    return sortDirection === 'asc' ? ' ↑' : ' ↓';
  };

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-gold-400 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <div className="text-zinc-500">Running analysis...</div>
        </div>
      </div>
    );
  }

  // Backtest Results
  if (activeTab === 'backtest') {
    if (!backtestResult) {
      return (
        <div className="h-full flex items-center justify-center">
          <div className="text-center text-zinc-600">
            <div className="text-4xl mb-4">📊</div>
            <div>Run a backtest to see results</div>
          </div>
        </div>
      );
    }

    return (
      <div className="h-full overflow-y-auto p-4 space-y-4">
        <h2 className="text-lg font-semibold text-gold-400">Backtest Results</h2>
        
        {/* Stats Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard 
            label="Win Rate" 
            value={`${(backtestResult.win_rate * 100).toFixed(1)}%`}
            positive={backtestResult.win_rate >= 0.5}
          />
          <StatCard 
            label="Total Trades" 
            value={backtestResult.total_trades}
          />
          <StatCard 
            label="Profit Factor" 
            value={backtestResult.profit_factor.toFixed(2)}
            positive={backtestResult.profit_factor >= 1}
          />
          <StatCard 
            label="Max Drawdown" 
            value={backtestResult.max_drawdown.toFixed(2)}
            positive={false}
          />
        </div>

        {/* Additional Stats */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard 
            label="Winning Trades" 
            value={backtestResult.winning_trades}
            positive={true}
          />
          <StatCard 
            label="Losing Trades" 
            value={backtestResult.losing_trades}
            positive={false}
          />
          <StatCard 
            label="Avg Win" 
            value={backtestResult.avg_win.toFixed(2)}
            positive={true}
          />
          <StatCard 
            label="Avg Loss" 
            value={backtestResult.avg_loss.toFixed(2)}
            positive={false}
          />
        </div>

        {/* Total PnL */}
        <div className="card p-6">
          <div className="text-xs text-zinc-500 uppercase tracking-wider mb-2">Total P&L</div>
          <div className={clsx(
            'text-4xl font-bold',
            backtestResult.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'
          )}>
            {backtestResult.total_pnl >= 0 ? '+' : ''}{backtestResult.total_pnl.toFixed(2)}
          </div>
        </div>
      </div>
    );
  }

  // Optimization Results
  if (!optimizationResults.length) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center text-zinc-600">
          <div className="text-4xl mb-4">⚡</div>
          <div>Run optimization to see results</div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-hidden flex flex-col">
      <div className="p-4 border-b border-zinc-800">
        <h2 className="text-lg font-semibold text-gold-400">Optimization Results</h2>
        <div className="text-xs text-zinc-500 mt-1">
          {optimizationResults.length} combinations tested
        </div>
      </div>
      
      <div className="flex-1 overflow-auto">
        <table className="results-table">
          <thead className="sticky top-0 bg-[#0a0a0f]">
            <tr>
              <th onClick={() => handleSort('rrr')}>
                RRR{getSortIndicator('rrr')}
              </th>
              <th onClick={() => handleSort('stop_loss')}>
                Stop Loss{getSortIndicator('stop_loss')}
              </th>
              <th onClick={() => handleSort('win_rate')}>
                Win Rate{getSortIndicator('win_rate')}
              </th>
              <th onClick={() => handleSort('total_trades')}>
                Trades{getSortIndicator('total_trades')}
              </th>
              <th onClick={() => handleSort('profit_factor')}>
                Profit Factor{getSortIndicator('profit_factor')}
              </th>
              <th onClick={() => handleSort('max_drawdown')}>
                Max DD{getSortIndicator('max_drawdown')}
              </th>
              <th onClick={() => handleSort('total_pnl')}>
                Total P&L{getSortIndicator('total_pnl')}
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedResults.slice(0, 50).map((result, idx) => (
              <tr key={`${result.rrr}-${result.stop_loss}`} className={idx < 3 ? 'bg-gold-500/5' : ''}>
                <td className="font-medium">{result.rrr.toFixed(1)}</td>
                <td>{result.stop_loss.toFixed(1)}</td>
                <td className={result.win_rate >= 0.5 ? 'text-green-400' : 'text-red-400'}>
                  {(result.win_rate * 100).toFixed(1)}%
                </td>
                <td>{result.total_trades}</td>
                <td className={result.profit_factor >= 1 ? 'text-green-400' : 'text-red-400'}>
                  {result.profit_factor.toFixed(2)}
                </td>
                <td className="text-red-400">{result.max_drawdown.toFixed(2)}</td>
                <td className={result.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                  {result.total_pnl >= 0 ? '+' : ''}{result.total_pnl.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

