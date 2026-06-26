/**
 * PriceChart.tsx — TradingView Lightweight Charts（ローソク足 + TP/entry/SL ライン）。
 *
 * 不変条件「テーマは単一の真実」(#3):
 *   チャートの色は **themes.ts の Theme オブジェクトからのみ** 取得する。色定数をここに持たない。
 *   DOM 側（var(--ax-*)）と同じ Theme を参照するので、切り替え時に色がドリフトしない。
 *   テーマ変更時は chart.applyOptions / series.applyOptions を呼び直し、価格線も貼り直す。
 *
 * 読み取り専用: 値（バリア水準・OHLC）は受け取って描画するだけ。UI で計算しない。
 */
import { useEffect, useRef } from "react";
import {
  ColorType,
  LineStyle,
  createChart,
  type IChartApi,
  type IPriceLine,
  type ISeriesApi,
  type UTCTimestamp,
} from "lightweight-charts";
import { useTheme } from "../theme/ThemeProvider";
import type { Theme } from "../theme/themes";
import type { Barriers, Candle } from "../data/types";

const CHART_HEIGHT = 360;

/** Theme -> チャート全体オプション（背景・文字・グリッド・枠線）。色は Theme のみ。 */
function chartOptions(theme: Theme) {
  return {
    layout: {
      background: { type: ColorType.Solid, color: theme.bg },
      textColor: theme.textSecondary,
      fontFamily: theme.fontBody,
    },
    grid: {
      vertLines: { color: theme.border },
      horzLines: { color: theme.border },
    },
    rightPriceScale: { borderColor: theme.border },
    timeScale: { borderColor: theme.border, timeVisible: true, secondsVisible: false },
  };
}

/** Theme -> ローソク色（up/down=candle*, wick=wick）。仕様のマッピングに一致。 */
function candleColors(theme: Theme) {
  return {
    upColor: theme.candleUp,
    downColor: theme.candleDown,
    borderUpColor: theme.candleUp,
    borderDownColor: theme.candleDown,
    wickUpColor: theme.wick,
    wickDownColor: theme.wick,
    borderVisible: true,
    wickVisible: true,
  };
}

type BarrierKind = "tp" | "entry" | "sl";

/** バリア種別 -> 色（TP=accent / entry=textSecondary / SL=negative）。色は Theme のみ。 */
function barrierColor(theme: Theme, kind: BarrierKind): string {
  switch (kind) {
    case "tp":
      return theme.accent;
    case "entry":
      return theme.textSecondary;
    case "sl":
      return theme.negative;
  }
}

export interface PriceChartProps {
  symbol: string;
  candles: Candle[];
  barriers: Barriers;
}

export default function PriceChart({ symbol, candles, barriers }: PriceChartProps) {
  const { theme } = useTheme();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const linesRef = useRef<IPriceLine[]>([]);

  // 生成時に最新テーマを使うための ref（生成 effect を再実行させないため）。
  const themeRef = useRef(theme);
  themeRef.current = theme;

  // --- チャート生成（マウント時 1 回） ---
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const chart = createChart(el, {
      autoSize: true, // 親サイズに追従（ResizeObserver）
      height: CHART_HEIGHT, // ResizeObserver 不可時のフォールバック
      ...chartOptions(themeRef.current),
    });
    const series = chart.addCandlestickSeries(candleColors(themeRef.current));
    chartRef.current = chart;
    seriesRef.current = series;
    return () => {
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      linesRef.current = [];
    };
  }, []);

  // --- データ更新 ---
  useEffect(() => {
    const chart = chartRef.current;
    const series = seriesRef.current;
    if (!chart || !series) return;
    series.setData(
      candles.map((c) => ({
        time: c.time as UTCTimestamp,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      })),
    );
    chart.timeScale().fitContent();
  }, [candles]);

  // --- テーマ反映（chart + series を貼り直し） ---
  useEffect(() => {
    const chart = chartRef.current;
    const series = seriesRef.current;
    if (!chart || !series) return;
    chart.applyOptions(chartOptions(theme));
    series.applyOptions(candleColors(theme));
  }, [theme]);

  // --- バリア線（テーマ/水準変更で貼り直し） ---
  useEffect(() => {
    const series = seriesRef.current;
    if (!series) return;
    for (const line of linesRef.current) series.removePriceLine(line);
    linesRef.current = [];
    const add = (price: number | null, kind: BarrierKind, title: string) => {
      if (price == null || !Number.isFinite(price)) return;
      linesRef.current.push(
        series.createPriceLine({
          price,
          color: barrierColor(theme, kind),
          lineWidth: 1,
          lineStyle: kind === "entry" ? LineStyle.Solid : LineStyle.Dashed,
          axisLabelVisible: true,
          title,
        }),
      );
    };
    add(barriers.takeProfit, "tp", "TP");
    add(barriers.entry, "entry", "entry");
    add(barriers.stopLoss, "sl", "SL");
  }, [barriers, theme]);

  return (
    <div className="ax-panel" style={{ padding: 12, display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
        <span className="ax-display" style={{ fontSize: 15, color: "var(--ax-text-primary)" }}>
          {symbol} · 5m
        </span>
        <span style={{ fontSize: 11, display: "inline-flex", gap: 12 }}>
          <Legend label="TP" color="var(--ax-accent)" />
          <Legend label="entry" color="var(--ax-text-secondary)" />
          <Legend label="SL" color="var(--ax-negative)" />
        </span>
      </div>
      <div ref={containerRef} style={{ width: "100%", height: CHART_HEIGHT }} />
    </div>
  );
}

/** 凡例の 1 項目（DOM 側は var(--ax-*) を参照）。 */
function Legend({ label, color }: { label: string; color: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 5, color: "var(--ax-text-tertiary)" }}>
      <span aria-hidden style={{ width: 14, height: 0, borderTop: `2px ${label === "entry" ? "solid" : "dashed"} ${color}` }} />
      {label}
    </span>
  );
}
