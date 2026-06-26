/**
 * themes.ts — 2テーマの【唯一の定義】。
 *
 * Aevum の不変条件「テーマは単一の真実」(#3): 色・タイポグラフィはここに1度だけ定義し、
 *   (a) CSS カスタムプロパティ (--ax-*) としてルート要素へ
 *   (b) TradingView チャートの applyOptions へ
 * の両方が同じ Theme オブジェクトを参照する。DOM側とチャート側で色を別々に書かない
 * （σ統一と同じ思想：切り替え時の色ドリフトを構造的に防ぐ）。
 */
export type ThemeName = "terminal" | "editorial";

export interface Theme {
  bg: string;
  panel: string;
  border: string;
  textPrimary: string;
  textSecondary: string;
  textTertiary: string;
  accent: string;
  positive: string;
  negative: string;
  candleUp: string;
  candleDown: string;
  wick: string;
  fontDisplay: string;
  fontBody: string;
}

// ラベル・数値フォントは両テーマ共通。
const FONT_BODY = "Arial, Helvetica, sans-serif";

export const themes: Record<ThemeName, Theme> = {
  // ダークターミナル（ティール差し色・サンセリフ）
  terminal: {
    bg: "#0a1117",
    panel: "#0d1820",
    border: "#1c2b36",
    textPrimary: "#e8eef0",
    textSecondary: "#6b8290",
    textTertiary: "#42606f",
    accent: "#5dcaa5",
    positive: "#5dcaa5",
    negative: "#e24b4a",
    candleUp: "#1d9e75",
    candleDown: "#d85a30",
    wick: "#33423a",
    fontDisplay: "'Inter', system-ui, sans-serif",
    fontBody: FONT_BODY,
  },
  // ダーク・エディトリアル（琥珀差し色・セリフ見出し・チャコール背景）
  editorial: {
    bg: "#211d18",
    panel: "#28231b",
    border: "#3a342b",
    textPrimary: "#f3efe6",
    textSecondary: "#9a8f7a",
    textTertiary: "#6e6555",
    accent: "#d4a23e",
    positive: "#d4a23e",
    negative: "#c87a52",
    candleUp: "#d4a23e",
    candleDown: "#7a7058",
    wick: "#4a4234",
    fontDisplay: "Georgia, 'Times New Roman', serif",
    fontBody: FONT_BODY,
  },
};

export const DEFAULT_THEME: ThemeName = "terminal";

// Theme のキー -> CSS 変数名。var 名を決めるのはここだけ（重複定義を防ぐ）。
export const CSS_VAR: Record<keyof Theme, string> = {
  bg: "--ax-bg",
  panel: "--ax-panel",
  border: "--ax-border",
  textPrimary: "--ax-text-primary",
  textSecondary: "--ax-text-secondary",
  textTertiary: "--ax-text-tertiary",
  accent: "--ax-accent",
  positive: "--ax-positive",
  negative: "--ax-negative",
  candleUp: "--ax-candle-up",
  candleDown: "--ax-candle-down",
  wick: "--ax-wick",
  fontDisplay: "--ax-font-display",
  fontBody: "--ax-font-body",
};

/** Theme を { "--ax-bg": "#..", ... } の CSS 変数マップへ変換。 */
export function themeToCssVars(theme: Theme): Record<string, string> {
  const out: Record<string, string> = {};
  (Object.keys(CSS_VAR) as (keyof Theme)[]).forEach((key) => {
    out[CSS_VAR[key]] = theme[key];
  });
  return out;
}
