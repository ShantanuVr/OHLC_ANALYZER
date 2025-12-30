'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { useBacktestStore } from '@/store/useBacktestStore';
import TradingChart from '@/components/Chart/TradingChart';
import StrategyBuilder from '@/components/StrategyBuilder/StrategyBuilder';
import ResultsTable from '@/components/Results/ResultsTable';

export default function Home() {
  const { error, indicatorData } = useBacktestStore();
  const [chartHeight, setChartHeight] = useState(60); // Percentage of available height
  const [isResizing, setIsResizing] = useState(false);
  const mainRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = useCallback(() => {
    setIsResizing(true);
  }, []);

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isResizing || !mainRef.current) return;

      const rect = mainRef.current.getBoundingClientRect();
      const newHeight = ((e.clientY - rect.top) / rect.height) * 100;
      
      // Constrain between 20% and 80%
      const constrainedHeight = Math.max(20, Math.min(80, newHeight));
      setChartHeight(constrainedHeight);
    },
    [isResizing]
  );

  const handleMouseUp = useCallback(() => {
    setIsResizing(false);
  }, []);

  useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'row-resize';
      document.body.style.userSelect = 'none';

      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      };
    }
  }, [isResizing, handleMouseMove, handleMouseUp]);

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {/* Header */}
      <header className="flex-shrink-0 border-b border-zinc-800 bg-[#050507]">
        <div className="flex items-center justify-between px-6 py-3">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold">
              <span className="text-gold-400">OHLC</span>
              <span className="text-zinc-400"> Analyzer</span>
            </h1>
            <div className="h-4 w-px bg-zinc-800" />
            <span className="text-xs text-zinc-600">SMC Backtesting Platform</span>
          </div>
          
          <div className="flex items-center gap-4">
            {indicatorData.length > 0 && (
              <div className="text-xs text-zinc-500">
                {indicatorData.length.toLocaleString()} candles loaded
              </div>
            )}
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              <span className="text-xs text-zinc-500">Connected</span>
            </div>
          </div>
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div className="flex-shrink-0 bg-red-900/20 border-b border-red-900/50 px-6 py-3">
          <div className="flex items-center gap-2 text-red-400 text-sm">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            {error}
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar - Strategy Builder */}
        <aside className="w-80 flex-shrink-0 border-r border-zinc-800 bg-[#050507] overflow-hidden">
          <StrategyBuilder />
        </aside>

        {/* Main Area */}
        <main ref={mainRef} className="flex-1 flex flex-col overflow-hidden relative">
          {/* Chart */}
          <div 
            className="min-h-0 border-t border-zinc-800 bg-[#050507] overflow-hidden"
            style={{ height: `${chartHeight}%`, flexShrink: 0 }}
          >
            <TradingChart />
          </div>
          
          {/* Resizer Handle - More visible and functional */}
          <div
            onMouseDown={(e) => {
              e.preventDefault();
              e.stopPropagation();
              handleMouseDown();
            }}
            className={`bg-zinc-800 hover:bg-zinc-600 active:bg-gold-500/30 cursor-row-resize transition-colors relative select-none flex-shrink-0 ${
              isResizing ? 'bg-gold-500/50' : ''
            }`}
            style={{ 
              height: '16px', 
              minHeight: '16px',
              zIndex: 20,
              userSelect: 'none',
              WebkitUserSelect: 'none',
            }}
            title="Drag to resize chart height"
          >
            <div className="h-full w-full flex items-center justify-center">
              <div className="w-24 h-2 bg-zinc-500 hover:bg-zinc-400 rounded-full transition-colors" />
            </div>
            {/* Visual indicator dots for better visibility */}
            <div className="absolute inset-0 flex items-center justify-center gap-2 pointer-events-none">
              <div className="w-2 h-2 rounded-full bg-zinc-400" />
              <div className="w-2 h-2 rounded-full bg-zinc-400" />
              <div className="w-2 h-2 rounded-full bg-zinc-400" />
            </div>
          </div>
          
          {/* Results Panel */}
          <div 
            className="min-h-0 border-t border-zinc-800 bg-[#050507] overflow-auto flex-shrink-0"
            style={{ height: `${100 - chartHeight}%` }}
          >
            <ResultsTable />
          </div>
        </main>
      </div>

      {/* Footer */}
      <footer className="flex-shrink-0 border-t border-zinc-800 bg-[#050507] px-6 py-2">
        <div className="flex items-center justify-between text-xs text-zinc-600">
          <div className="flex items-center gap-4">
            <span>XAUUSD Gold Trading</span>
            <span>•</span>
            <span>23 Years Historical Data</span>
          </div>
          <div className="flex items-center gap-2">
            <span>Sessions:</span>
            <span className="flex items-center gap-1">
              <span className="session-dot asian" /> Asian
            </span>
            <span className="flex items-center gap-1">
              <span className="session-dot london" /> London
            </span>
            <span className="flex items-center gap-1">
              <span className="session-dot ny" /> NY
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}

