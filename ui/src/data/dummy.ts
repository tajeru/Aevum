/**
 * dummy.ts — 実装順 #3〜#6 用のダミー状態。
 *
 * これは「バックエンドが返す形」を模した固定データ。実装順 #7 で useLiveState の
 * WebSocket/REST 出力に差し替える。型は types.ts（=API 形状）に従うため差し替えは無痛。
 *
 * σ は per-bar → ×√48 → scaled をデータ側の値として持つ（UI は再計算しない）。
 * 仕様の例に合わせ perBar 0.0041 / horizon 48 / scaled 0.0284 を採用。
 */
import type { Candle, DashboardState } from "./types";

/**
 * 決定論的なダミーローソク足を生成（seed 固定の線形合同法）。
 * Math.random を使わないのは、再訪/HMR で表示が暴れないようにするため。
 * これは「バックエンドが返す ohlcv_bars」の代用で、UI 側の計算ロジックではない。
 */
function makeCandles(count: number, endEpochSec: number, stepSec: number, start: number): Candle[] {
  let seed = 0x9e3779b1; // 固定 seed
  const rnd = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 0xffffffff; // 0..1
  };
  const out: Candle[] = [];
  let close = start;
  const startTime = endEpochSec - (count - 1) * stepSec;
  for (let i = 0; i < count; i++) {
    const drift = Math.sin(i / 9) * 0.0008; // 緩い波
    const shock = (rnd() - 0.5) * 0.006; // ±0.3% 程度
    const open = close;
    close = open * (1 + drift + shock);
    const hi = Math.max(open, close) * (1 + rnd() * 0.0025);
    const lo = Math.min(open, close) * (1 - rnd() * 0.0025);
    out.push({
      time: startTime + i * stepSec,
      open: round1(open),
      high: round1(hi),
      low: round1(lo),
      close: round1(close),
    });
  }
  return out;
}

const round1 = (n: number) => Math.round(n * 10) / 10;

// 2026-06-26T09:40:00Z を末尾バーに固定（dummy なので決め打ち）。
const END_EPOCH = Math.floor(Date.parse("2026-06-26T09:40:00Z") / 1000);
const DUMMY_CANDLES = makeCandles(120, END_EPOCH, 300, 64000);

export const DUMMY_STATE: DashboardState = {
  connected: true,
  latencyMs: 12,
  updatedAt: "2026-06-26T09:41:07Z",
  metrics: {
    equity: 10342.18,
    openPnl: 127.44,
    winRate: 0.541,
    sigma: { perBar: 0.0041, horizon: 48, scaled: 0.0284 },
  },
  chart: {
    symbol: "BTC",
    candles: DUMMY_CANDLES,
    // entry は建玉の建値に一致。TP/SL は execution.py のブラケットを模した固定値。
    barriers: { entry: 64210.0, takeProfit: 65800.0, stopLoss: 63200.0 },
  },
  prediction: {
    symbol: "BTC",
    time: "2026-06-26T09:40:00Z",
    model_version: "v1",
    prob_down: 0.18,
    prob_flat: 0.29,
    prob_up: 0.53,
    signal: 1,
    sigma: 0.0041,
  },
  probThreshold: 0.5,
  // bar_features の代表的な列から少数を抜粋（カテゴリ横断）。
  features: [
    { name: "sigma_ewma", value: 0.0041 },
    { name: "obi_l5", value: 0.137 },
    { name: "rsi_14", value: 58.3 },
    { name: "macd_hist", value: 14.2 },
    { name: "spread_bps", value: 1.8 },
    { name: "funding_rate", value: 0.00012 },
  ],
  positions: [
    {
      symbol: "BTC",
      time: "2026-06-26T09:12:00Z",
      size: 0.15,
      entry_price: 64210.0,
      mark_price: 65060.0,
      unrealized_pnl: 127.5,
      realized_pnl: 0.0,
      leverage: 3.0,
      liquidation_px: 48230.0,
      margin_used: 3210.5,
    },
    {
      symbol: "ETH",
      time: "2026-06-26T08:55:00Z",
      size: -1.2,
      entry_price: 3420.0,
      mark_price: 3398.0,
      unrealized_pnl: 26.4,
      realized_pnl: 0.0,
      leverage: 2.0,
      liquidation_px: 4120.0,
      margin_used: 2038.8,
    },
  ],
};
