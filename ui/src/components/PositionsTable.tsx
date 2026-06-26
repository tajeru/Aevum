/**
 * PositionsTable.tsx — 建玉一覧: symbol / side / size / entry / SL / bars held(n/48) / PnL。
 *
 * 表示のみ。side は size 符号の純粋派生（sideOf）。SL・barsHeld・horizon は
 * データ層が供給した PositionView の値をそのまま表示し、UI では再計算しない。
 * 色は var(--ax-*) / Theme 由来のみ。
 */
import type { PositionView } from "../data/types";
import { sideOf } from "../data/types";
import { num, price, signColor, signedUsd } from "../format";

const SIDE_COLOR: Record<string, string> = {
  LONG: "var(--ax-positive)",
  SHORT: "var(--ax-negative)",
  FLAT: "var(--ax-text-secondary)",
};

const TH: React.CSSProperties = {
  textAlign: "right",
  padding: "8px 10px",
  fontSize: 10,
  letterSpacing: 1,
  textTransform: "uppercase",
  color: "var(--ax-text-secondary)",
  fontWeight: 400,
  borderBottom: "1px solid var(--ax-border)",
  whiteSpace: "nowrap",
};

const TD: React.CSSProperties = {
  textAlign: "right",
  padding: "9px 10px",
  fontSize: 13,
  color: "var(--ax-text-primary)",
  borderBottom: "1px solid var(--ax-border)",
  whiteSpace: "nowrap",
};

/** 縦バリアまでの保有経過バー（n/48）。 */
function HeldCell({ held, horizon }: { held: number; horizon: number }) {
  const frac = horizon > 0 ? Math.max(0, Math.min(1, held / horizon)) : 0;
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 8 }}>
      <span style={{ color: "var(--ax-text-secondary)", fontSize: 12 }}>
        {held}/{horizon}
      </span>
      <div style={{ width: 56, height: 6, background: "var(--ax-border)", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${frac * 100}%`, height: "100%", background: "var(--ax-accent)", borderRadius: 3 }} />
      </div>
    </div>
  );
}

export default function PositionsTable({ positions }: { positions: PositionView[] }) {
  return (
    <div className="ax-panel" style={{ padding: 0, marginTop: 14, overflowX: "auto" }}>
      <div style={{ padding: "12px 14px 0" }}>
        <span
          style={{
            fontSize: 10,
            letterSpacing: 1.2,
            textTransform: "uppercase",
            color: "var(--ax-text-secondary)",
          }}
        >
          positions
        </span>
      </div>
      <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 6 }}>
        <thead>
          <tr>
            <th style={{ ...TH, textAlign: "left" }}>symbol</th>
            <th style={{ ...TH, textAlign: "left" }}>side</th>
            <th style={TH}>size</th>
            <th style={TH}>entry</th>
            <th style={TH}>SL</th>
            <th style={TH}>bars held</th>
            <th style={TH}>PnL</th>
          </tr>
        </thead>
        <tbody>
          {positions.length === 0 && (
            <tr>
              <td
                colSpan={7}
                style={{ ...TD, textAlign: "center", color: "var(--ax-text-tertiary)", borderBottom: "none" }}
              >
                no open positions
              </td>
            </tr>
          )}
          {positions.map((p) => {
            const side = sideOf(p.size);
            return (
              <tr key={p.symbol}>
                <td style={{ ...TD, textAlign: "left", fontFamily: "var(--ax-font-display)" }}>{p.symbol}</td>
                <td style={{ ...TD, textAlign: "left" }}>
                  <span style={{ color: SIDE_COLOR[side], fontSize: 12, letterSpacing: 1 }}>{side}</span>
                </td>
                <td style={TD}>{num(Math.abs(p.size), 4)}</td>
                <td style={TD}>{price(p.entry_price)}</td>
                <td style={{ ...TD, color: "var(--ax-negative)" }}>{price(p.slLevel)}</td>
                <td style={TD}>
                  <HeldCell held={p.barsHeld} horizon={p.horizon} />
                </td>
                <td style={{ ...TD, color: signColor(p.unrealized_pnl) }}>{signedUsd(p.unrealized_pnl)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
