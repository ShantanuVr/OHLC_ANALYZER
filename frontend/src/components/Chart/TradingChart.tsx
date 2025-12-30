'use client';

import { useEffect, useRef, useCallback } from 'react';
import { 
  createChart, 
  IChartApi, 
  ISeriesApi, 
  CandlestickData,
  Time,
  ColorType,
} from 'lightweight-charts';
import { useBacktestStore } from '@/store/useBacktestStore';
import { IndicatorData } from '@/lib/api';

interface ChartMarker {
  time: Time;
  position: 'aboveBar' | 'belowBar';
  color: string;
  shape: 'circle' | 'square' | 'arrowUp' | 'arrowDown';
  text?: string;
}

export default function TradingChart() {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  
  const { indicatorData, showIndicators, params } = useBacktestStore();

  const formatDataForChart = useCallback((data: IndicatorData[]): CandlestickData<Time>[] => {
    return data.map(d => ({
      time: d.time as Time,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    }));
  }, []);

  const getMarkers = useCallback((data: IndicatorData[]): ChartMarker[] => {
    const markers: ChartMarker[] = [];
    
    data.forEach(d => {
      // Swing markers
      if (showIndicators.swings) {
        if (d.swing_high) {
          markers.push({
            time: d.time as Time,
            position: 'aboveBar',
            color: '#f59e0b',
            shape: 'circle',
            text: 'SH',
          });
        }
        if (d.swing_low) {
          markers.push({
            time: d.time as Time,
            position: 'belowBar',
            color: '#8b5cf6',
            shape: 'circle',
            text: 'SL',
          });
        }
      }
      
      // Sweep markers
      if (showIndicators.sweeps) {
        if (d.sweep_bullish) {
          markers.push({
            time: d.time as Time,
            position: 'belowBar',
            color: '#22c55e',
            shape: 'arrowUp',
            text: 'SSL',
          });
        }
        if (d.sweep_bearish) {
          markers.push({
            time: d.time as Time,
            position: 'aboveBar',
            color: '#ef4444',
            shape: 'arrowDown',
            text: 'BSL',
          });
        }
      }
    });
    
    return markers;
  }, [showIndicators]);

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#0a0a0f' },
        textColor: '#71717a',
        fontFamily: 'JetBrains Mono, monospace',
      },
      grid: {
        vertLines: { color: '#1a1a2e' },
        horzLines: { color: '#1a1a2e' },
      },
      crosshair: {
        mode: 1,
        vertLine: {
          color: '#fbbf24',
          width: 1,
          style: 2,
          labelBackgroundColor: '#fbbf24',
        },
        horzLine: {
          color: '#fbbf24',
          width: 1,
          style: 2,
          labelBackgroundColor: '#fbbf24',
        },
      },
      rightPriceScale: {
        borderColor: '#27272a',
        scaleMargins: {
          top: 0.1,
          bottom: 0.1,
        },
      },
      timeScale: {
        borderColor: '#27272a',
        timeVisible: true,
        secondsVisible: false,
      },
      handleScale: {
        axisPressedMouseMove: {
          time: true,
          price: true,
        },
      },
    });

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    });

    chartRef.current = chart;
    candlestickSeriesRef.current = candlestickSeries;

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ 
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight,
        });
      }
    };

    window.addEventListener('resize', handleResize);
    handleResize();

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  // Update data when indicatorData changes
  useEffect(() => {
    if (!candlestickSeriesRef.current || indicatorData.length === 0) return;

    const chartData = formatDataForChart(indicatorData);
    candlestickSeriesRef.current.setData(chartData);

    // Add markers
    const markers = getMarkers(indicatorData);
    candlestickSeriesRef.current.setMarkers(markers);

    // Fit content
    if (chartRef.current) {
      chartRef.current.timeScale().fitContent();
    }
  }, [indicatorData, formatDataForChart, getMarkers]);

  // Update markers when showIndicators changes
  useEffect(() => {
    if (!candlestickSeriesRef.current || indicatorData.length === 0) return;
    
    const markers = getMarkers(indicatorData);
    candlestickSeriesRef.current.setMarkers(markers);
  }, [showIndicators, indicatorData, getMarkers]);

  return (
    <div className="chart-container h-full w-full relative">
      {/* Chart header */}
      <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between px-4 py-2 bg-gradient-to-b from-[#0a0a0f] to-transparent">
        <div className="flex items-center gap-4">
          <span className="text-gold-400 font-semibold">{params.symbol}</span>
          <span className="text-zinc-500">{params.timeframe}</span>
        </div>
        <div className="flex items-center gap-2 text-xs">
          {showIndicators.swings && (
            <span className="px-2 py-1 rounded bg-zinc-800/50 text-zinc-400">Swings</span>
          )}
          {showIndicators.fvg && (
            <span className="px-2 py-1 rounded bg-green-900/30 text-green-400">FVG</span>
          )}
          {showIndicators.ob && (
            <span className="px-2 py-1 rounded bg-purple-900/30 text-purple-400">OB</span>
          )}
          {showIndicators.sweeps && (
            <span className="px-2 py-1 rounded bg-orange-900/30 text-orange-400">Sweeps</span>
          )}
        </div>
      </div>
      
      {/* Chart container */}
      <div ref={chartContainerRef} className="w-full h-full" />
      
      {/* No data message */}
      {indicatorData.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <div className="text-zinc-600 text-lg mb-2">No data loaded</div>
            <div className="text-zinc-700 text-sm">Load data to view the chart</div>
          </div>
        </div>
      )}
    </div>
  );
}

