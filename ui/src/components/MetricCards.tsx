/**
 * MetricCards.tsx — equity / open PnL / win rate / σ。
 *
 * σ カードは設計判断 #3/#9 の可視化: per-bar → ×√horizon → scaled を「並べて表示」する。
 * σ の値（perBar・scaled）は shared/volatility.py 由来をそのまま受け取り、UI では再計算しない
 * （×√48 は表示ラベル、scaled はデータ供給値）。
 */
import type { Metrics } from "../data/types";
import { num, pct, signColor, signedUsd, usd } from "../format";

function Card({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="ax-panel" style={{ padding: "14px 16px", display: "flex", flexDirection: "column", gap: 8 }}>
      <span
        style={{
          fontSize: 10,
          letterSpacing: 1.2,
          textTransform: "uppercase",
          color: "var(--ax-text-secondary)",
        }}
      >
        {label}
      </span>
      {children}
    </div>
  );
}

const VALUE: React.CSSProperties = { fontSize: 24, color: "var(--ax-text-primary)", lineHeight: 1.1 };

export default function MetricCards({ metrics }: { metrics: Metrics }) {
  const { equity, openPnl, winRate, sigma } = metrics;
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
        gap: 14,
        marginBottom: 20,
      }}
    >
      <Card label="equity">
        <span style={VALUE}>{usd(equity)}</span>
      </Card>

      <Card label="open pnl">
        <span style={{ ...VALUE, color: signColor(openPnl) }}>{signedUsd(openPnl)}</span>
      </Card>

      <Card label="win rate">
        <span style={VALUE}>{pct(winRate, 1)}</span>
      </Card>

      {/* σ: per-bar → ×√horizon → scaled（UI は再計算しない） */}
      <Card label="σ (per-bar → horizon)">
        <span style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap" }}>
          <span style={{ ...VALUE, fontSize: 20 }}>{num(sigma.perBar, 4)}</span>
          <span style={{ fontSize: 12, color: "var(--ax-text-tertiary)" }}>
            ×√{sigma.horizon}
          </span>
          <span style={{ fontSize: 12, color: "var(--ax-text-tertiary)" }}>→</span>
          <span style={{ fontSize: 20, color: "var(--ax-accent)" }}>{num(sigma.scaled, 4)}</span>
        </span>
      </Card>
    </div>
  );
}
