/**
 * format.ts — 数値・時刻の表示整形（唯一の整形ヘルパ）。
 *
 * 仕様の不変条件 #4「数値は必ず丸めて表示」を一箇所に集約する。
 * コンポーネントは生の number/ISO 文字列をここに通してから描画する。
 * 値の計算（σの×√horizon など）はここでは行わない（UI側で再計算しない）。
 */

const USD = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

/** $12,345.67 */
export function usd(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return USD.format(n);
}

/** +$123.45 / -$123.45（符号付き）。PnL 表示用。 */
export function signedUsd(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  const s = USD.format(Math.abs(n));
  return n < 0 ? `-${s}` : `+${s}`;
}

/** 0.54 -> "54.0%" */
export function pct(n: number | null | undefined, digits = 1): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return `${(n * 100).toFixed(digits)}%`;
}

/** 任意桁の固定小数。 */
export function num(n: number | null | undefined, digits = 2): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

/** 価格（カンマ区切り・小数1桁）。 */
export function price(n: number | null | undefined, digits = 1): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return n.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

/** ISO 文字列 -> HH:MM:SS（ローカル時刻）。 */
export function clock(iso: string | null | undefined): string {
  if (!iso) return "—:—:—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—:—:—";
  return d.toLocaleTimeString("en-GB", { hour12: false });
}

/** Date -> HH:MM:SS（ヘッダのライブ時計用）。 */
export function clockFromDate(d: Date): string {
  return d.toLocaleTimeString("en-GB", { hour12: false });
}

/** 数値の符号で positive/negative の CSS 変数を返す（0 は secondary）。 */
export function signColor(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n) || n === 0) return "var(--ax-text-secondary)";
  return n > 0 ? "var(--ax-positive)" : "var(--ax-negative)";
}
