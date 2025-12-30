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
import { IndicatorData, getIndicatorData } from '@/lib/api';

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
  const isLoadingMoreRef = useRef(false);
  
  const { 
    indicatorData, 
    showIndicators, 
    params,
    loadedDateRange,
    isLoadingMore,
    appendIndicatorData,
    setIsLoadingMore,
    setError,
  } = useBacktestStore();

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
    let pivotCount = 0;
    
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
      
      // Market Structure Pivots (HH, HL, LH, LL) - Always show when MSS or BoS is enabled
      if (showIndicators.mss || showIndicators.bos) {
        // Higher High (HH) - confirmed after bullish BoS
        // Handle various data types: boolean, number (0/1), string ('0'/'1')
        const hhValue = d.confirmed_hh;
        const isHH = hhValue === true || hhValue === 1 || hhValue === '1' || (typeof hhValue === 'number' && hhValue > 0);
        if (isHH) {
          pivotCount++;
          markers.push({
            time: d.time as Time,
            position: 'aboveBar',
            color: '#22c55e',
            shape: 'circle',
            text: 'HH',
          });
        }
        
        // Higher Low (HL) - confirmed after bullish BoS
        const hlValue = d.confirmed_hl_at_idx;
        const isHL = hlValue === true || hlValue === 1 || hlValue === '1' || (typeof hlValue === 'number' && hlValue > 0);
        if (isHL) {
          pivotCount++;
          markers.push({
            time: d.time as Time,
            position: 'belowBar',
            color: '#10b981',
            shape: 'circle',
            text: 'HL',
          });
        }
        
        // Lower High (LH) - confirmed after bearish BoS
        const lhValue = d.confirmed_lh_at_idx;
        const isLH = lhValue === true || lhValue === 1 || lhValue === '1' || (typeof lhValue === 'number' && lhValue > 0);
        if (isLH) {
          pivotCount++;
          markers.push({
            time: d.time as Time,
            position: 'aboveBar',
            color: '#f87171',
            shape: 'circle',
            text: 'LH',
          });
        }
        
        // Lower Low (LL) - confirmed after bearish BoS
        const llValue = d.confirmed_ll;
        const isLL = llValue === true || llValue === 1 || llValue === '1' || (typeof llValue === 'number' && llValue > 0);
        if (isLL) {
          pivotCount++;
          markers.push({
            time: d.time as Time,
            position: 'belowBar',
            color: '#ef4444',
            shape: 'circle',
            text: 'LL',
          });
        }
      }
      
      // MSS (Market Structure Shift) markers - only when they occur
      if (showIndicators.mss) {
        if (d.mss_bullish === true || d.mss_bullish === 1) {
          markers.push({
            time: d.time as Time,
            position: 'belowBar',
            color: '#3b82f6',
            shape: 'arrowUp',
            text: 'MSS↑',
          });
        }
        if (d.mss_bearish === true || d.mss_bearish === 1) {
          markers.push({
            time: d.time as Time,
            position: 'aboveBar',
            color: '#f97316',
            shape: 'arrowDown',
            text: 'MSS↓',
          });
        }
      }
      
      // BoS (Break of Structure) markers - only when they occur
      if (showIndicators.bos) {
        if (d.bos_bullish === true || d.bos_bullish === 1) {
          // High-Quality BoS gets a different marker
          const isHQ = d.bos_hq_bullish === true || d.bos_hq_bullish === 1;
          markers.push({
            time: d.time as Time,
            position: 'belowBar',
            color: isHQ ? '#10b981' : '#22c55e',
            shape: 'square',
            text: isHQ ? 'BoS+↑' : 'BoS↑',
          });
        }
        if (d.bos_bearish === true || d.bos_bearish === 1) {
          // High-Quality BoS gets a different marker
          const isHQ = d.bos_hq_bearish === true || d.bos_hq_bearish === 1;
          markers.push({
            time: d.time as Time,
            position: 'aboveBar',
            color: isHQ ? '#ef4444' : '#f87171',
            shape: 'square',
            text: isHQ ? 'BoS+↓' : 'BoS↓',
          });
        }
      }
    });
    
    // Debug: Log marker counts (remove in production)
    if (pivotCount > 0) {
      console.log(`[TradingChart] Created ${pivotCount} pivot markers, ${markers.length} total markers`);
    }
    
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

  // Lazy loading: Detect scroll to load more data
  useEffect(() => {
    if (!chartRef.current || indicatorData.length === 0) return;

    const chart = chartRef.current;
    const timeScale = chart.timeScale();

    const handleVisibleRangeChange = async () => {
      if (isLoadingMoreRef.current) return;

      const visibleRange = timeScale.getVisibleRange();
      if (!visibleRange || indicatorData.length === 0) return;

      // Get the time range of loaded data (Unix timestamps in seconds)
      const firstTime = indicatorData[0].time as number;
      const lastTime = indicatorData[indicatorData.length - 1].time as number;

      // Check if scrolled to start (need to load older data)
      const scrollThreshold = 0.15; // 15% from the start
      const rangeSize = visibleRange.to - visibleRange.from;
      const distanceFromStart = visibleRange.from - firstTime;

      if (distanceFromStart < rangeSize * scrollThreshold && distanceFromStart > -rangeSize) {
        // Load one year before the current start
        isLoadingMoreRef.current = true;
        setIsLoadingMore(true);
        
        try {
          const currentStart = new Date(firstTime * 1000);
          const newStart = new Date(currentStart);
          newStart.setFullYear(newStart.getFullYear() - 1);
          
          const formatDate = (date: Date) => {
            const year = date.getFullYear();
            const month = String(date.getMonth() + 1).padStart(2, '0');
            const day = String(date.getDate()).padStart(2, '0');
            return `${year}-${month}-${day}`;
          };

          const newData = await getIndicatorData(
            params.symbol,
            params.timeframe,
            formatDate(newStart),
            formatDate(currentStart)
          );

          if (newData.length > 0) {
            // Prepend older data - the store will update indicatorData
            appendIndicatorData(newData, true);
            // Chart will auto-update via the useEffect that watches indicatorData
          }
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Failed to load more data');
        } finally {
          isLoadingMoreRef.current = false;
          setIsLoadingMore(false);
        }
        return; // Prevent loading both directions at once
      }

      // Check if scrolled to end (need to load newer data)
      const distanceFromEnd = lastTime - visibleRange.to;
      if (distanceFromEnd < rangeSize * scrollThreshold && distanceFromEnd > -rangeSize) {
        // Load one year after the current end
        isLoadingMoreRef.current = true;
        setIsLoadingMore(true);
        
        try {
          const currentEnd = new Date(lastTime * 1000);
          const newEnd = new Date(currentEnd);
          newEnd.setFullYear(newEnd.getFullYear() + 1);
          
          const formatDate = (date: Date) => {
            const year = date.getFullYear();
            const month = String(date.getMonth() + 1).padStart(2, '0');
            const day = String(date.getDate()).padStart(2, '0');
            return `${year}-${month}-${day}`;
          };

          const newData = await getIndicatorData(
            params.symbol,
            params.timeframe,
            formatDate(currentEnd),
            formatDate(newEnd)
          );

          if (newData.length > 0) {
            // Append newer data - the store will update indicatorData
            appendIndicatorData(newData, false);
            // Chart will auto-update via the useEffect that watches indicatorData
          }
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Failed to load more data');
        } finally {
          isLoadingMoreRef.current = false;
          setIsLoadingMore(false);
        }
      }
    };

    // Use debounce to avoid too many calls
    let timeoutId: NodeJS.Timeout;
    const debouncedHandler = () => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(handleVisibleRangeChange, 300);
    };

    timeScale.subscribeVisibleTimeRangeChange(debouncedHandler);

    return () => {
      clearTimeout(timeoutId);
      timeScale.unsubscribeVisibleTimeRangeChange(debouncedHandler);
    };
  }, [indicatorData, params, appendIndicatorData, setIsLoadingMore, setError, formatDataForChart, getMarkers]);

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
          {showIndicators.mss && (
            <span className="px-2 py-1 rounded bg-blue-900/30 text-blue-400">MSS</span>
          )}
          {showIndicators.bos && (
            <span className="px-2 py-1 rounded bg-emerald-900/30 text-emerald-400">BoS</span>
          )}
        </div>
      </div>
      
      {/* Chart container */}
      <div ref={chartContainerRef} className="w-full h-full" />
      
      {/* Loading more data indicator */}
      {isLoadingMore && (
        <div className="absolute top-20 right-4 z-20 bg-zinc-800/90 border border-zinc-700 rounded-lg px-4 py-2 flex items-center gap-2">
          <div className="w-4 h-4 border-2 border-gold-400 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm text-zinc-300">Loading more data...</span>
        </div>
      )}
      
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

