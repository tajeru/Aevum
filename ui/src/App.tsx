/**
 * App.tsx — ダッシュボード合成（実データ・2テーマ切替対応）。
 *
 * 実装順 #3〜#7: Header(+ThemeToggle) / MetricCards / PriceChart / SignalPanel /
 * PositionsTable を useLiveState（WebSocket + REST GET）の DashboardState で描画する。
 * 接続前はダミーを offline プレースホルダとして表示（useLiveState 内）。
 *
 * 実装順 #14: IngestionMonitor タブを追加（収集監視ビュー）。
 */
import { useState } from "react";
import Header from "./components/Header";
import MetricCards from "./components/MetricCards";
import PriceChart from "./components/PriceChart";
import SignalPanel from "./components/SignalPanel";
import PositionsTable from "./components/PositionsTable";
import IngestionMonitor from "./components/IngestionMonitor";
import { useLiveState } from "./data/useLiveState";

type ViewName = "dashboard" | "ingestion";

function TabBar({ active, onChange }: { active: ViewName; onChange: (v: ViewName) => void }) {
  const tab = (name: ViewName, label: string) => (
    <button
      key={name}
      onClick={() => onChange(name)}
      style={{
        padding: "6px 18px",
        fontSize: 11,
        letterSpacing: 1,
        textTransform: "uppercase",
        cursor: "pointer",
        background: "none",
        border: "none",
        borderBottom: active === name ? "2px solid var(--ax-accent)" : "2px solid transparent",
        color: active === name ? "var(--ax-accent)" : "var(--ax-text-secondary)",
        fontFamily: "var(--ax-font-body)",
        outline: "none",
      }}
    >
      {label}
    </button>
  );
  return (
    <div style={{ display: "flex", gap: 4, borderBottom: "1px solid var(--ax-border)", marginBottom: 20 }}>
      {tab("dashboard", "Dashboard")}
      {tab("ingestion", "Ingestion")}
    </div>
  );
}

export default function App() {
  const state = useLiveState();
  const [view, setView] = useState<ViewName>("dashboard");

  return (
    <div style={{ minHeight: "100vh", maxWidth: 1180, margin: "0 auto", padding: "0 24px 32px" }}>
      <Header
        connected={state.connected}
        latencyMs={state.latencyMs}
        equity={state.metrics.equity}
        openPnl={state.metrics.openPnl}
      />
      <TabBar active={view} onChange={setView} />

      {view === "dashboard" && (
        <>
          <MetricCards metrics={state.metrics} />

          {/* チャート（広い）+ シグナル（横）。狭い画面では縦積み。 */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(0, 2fr) minmax(280px, 1fr)",
              gap: 14,
              alignItems: "start",
            }}
          >
            <PriceChart
              symbol={state.chart.symbol}
              candles={state.chart.candles}
              barriers={state.chart.barriers}
            />
            <SignalPanel
              prediction={state.prediction}
              probThreshold={state.probThreshold}
              features={state.features}
            />
          </div>

          <PositionsTable positions={state.positions} />
        </>
      )}

      {view === "ingestion" && <IngestionMonitor />}
    </div>
  );
}
