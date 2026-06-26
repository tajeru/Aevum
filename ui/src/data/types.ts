/**
 * types.ts — UI 表示用の型。**API（server.py）の配信形状に一致**させる。
 *
 * 目的: ダミー → 実データ差し替え（実装順 #7）を構造的に無痛にする。
 *   - Prediction = model_predictions 行（GET /predictions, WS snapshot）
 *   - Position   = positions 行（GET /positions, WS snapshot）
 *   スネークケースは DB/JSON 由来のためそのまま保持する。
 *
 * 不変条件: UI は読み取り専用・ロジックを持たない。σ は受け取った値を表示するだけで
 * 再計算しない（per-bar と ×√horizon 後の値はどちらもデータ側が供給する）。
 */

/** model_predictions 1 行（schema_v1.sql 準拠）。 */
export interface Prediction {
  symbol: string;
  time: string; // ISO8601
  model_version: string;
  prob_down: number; // P(label=-1)
  prob_flat: number; // P(label= 0)
  prob_up: number; // P(label=+1)
  signal: -1 | 0 | 1 | null;
  sigma: number; // per-bar σ（volatility.py 由来）
}

/** positions 1 行（schema_v1.sql 準拠）。size は符号付き（+ロング/-ショート）。 */
export interface Position {
  symbol: string;
  time: string; // ISO8601
  size: number;
  entry_price: number | null;
  mark_price: number | null;
  unrealized_pnl: number | null;
  realized_pnl: number | null;
  leverage: number | null;
  liquidation_px: number | null;
  margin_used: number | null;
}

/**
 * PositionsTable 表示用の行 = 生の Position + 表示専用の付帯情報。
 * 生の positions 行に無い列をデータ層（#7 では useLiveState）が結合して供給する:
 *   - slLevel: 損切バリア（execution.py のブラケット由来。positions には無い）
 *   - barsHeld / horizon: 縦バリアまでの保有経過（Triple-Barrier の time-stop 進捗）
 * UI では再計算しない（経過バーや SL を UI 内で導出しない）。
 */
export interface PositionView extends Position {
  slLevel: number | null;
  barsHeld: number; // エントリー以降の経過バー数
  horizon: number; // 縦バリア（=48）
}

/** ライブ特徴量の 1 項目（bar_features の列名と値）。SignalPanel 表示用。 */
export interface LiveFeature {
  name: string; // shared/feature_names.py の名称
  value: number;
}

/**
 * ローソク足 1 本。time は UTC エポック秒（lightweight-charts の UTCTimestamp 相当）。
 * バックエンドが ohlcv_bars 由来でこの形を返す想定。UI 側では値を加工しない。
 */
export interface Candle {
  time: number; // epoch seconds (UTC)
  open: number;
  high: number;
  low: number;
  close: number;
}

/**
 * チャートに重ねる Triple-Barrier の水準（建玉のブラケット）。
 * UI では再計算しない: entry/TP/SL は execution.py/labels.py 由来の値をそのまま受け取る。
 */
export interface Barriers {
  entry: number | null; // 建値
  takeProfit: number | null; // 上側バリア（指値・maker）
  stopLoss: number | null; // 下側バリア（成行トリガ）
}

/** PriceChart の表示ペイロード（主表示銘柄）。 */
export interface ChartData {
  symbol: string;
  candles: Candle[];
  barriers: Barriers;
}

/**
 * σ 表示用。per-bar と horizon スケール後をどちらもデータ側が供給する。
 * UI は perBar → ×√horizon → scaled を「表示」するだけ（Math.sqrt しない）。
 */
export interface SigmaView {
  perBar: number;
  horizon: number; // ラベル/モデルの horizon（=48）
  scaled: number; // shared/volatility.py の scale_to_horizon 後
}

/** ヘッダ/カードに出す集計メトリクス。 */
export interface Metrics {
  equity: number;
  openPnl: number;
  winRate: number; // 0..1
  sigma: SigmaView;
}

/** ダッシュボード全体の表示状態（WS スナップショット + REST をまとめた形）。 */
export interface DashboardState {
  connected: boolean;
  latencyMs: number | null;
  updatedAt: string; // ISO8601
  metrics: Metrics;
  chart: ChartData; // 主表示銘柄のローソク足 + バリア（PriceChart）
  prediction: Prediction; // 主表示銘柄の最新予測（SignalPanel）
  probThreshold: number; // execution.py ExecConfig.prob_threshold（表示用）
  features: LiveFeature[];
  positions: PositionView[];
}

/** signal(-1/0/1) -> 表示方向。 */
export type Direction = "LONG" | "FLAT" | "SHORT";

export function directionOf(signal: number | null): Direction {
  if (signal === 1) return "LONG";
  if (signal === -1) return "SHORT";
  return "FLAT";
}

/** 建玉サイズの符号 -> 売買方向（純粋な表示派生）。 */
export type Side = "LONG" | "SHORT" | "FLAT";

export function sideOf(size: number): Side {
  if (size > 0) return "LONG";
  if (size < 0) return "SHORT";
  return "FLAT";
}
