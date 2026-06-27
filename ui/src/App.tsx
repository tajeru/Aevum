/**
 * App.tsx — ダッシュボード合成（実データ・2テーマ切替対応）。
 *
 * 実装順 #3〜#7: Header(+ThemeToggle) / MetricCards / PriceChart / SignalPanel /
 * PositionsTable を useLiveState（WebSocket + REST GET）の DashboardState で描画する。
 * 接続前はダミーを offline プレースホルダとして表示（useLiveState 内）。
 */
import Header from "./components/Header";
import MetricCards from "./components/MetricCards";
import PriceChart from "./components/PriceChart";
import SignalPanel from "./components/SignalPanel";
import PositionsTable from "./components/PositionsTable";
import { useLiveState } from "./data/useLiveState";

export default function App() {
  const state = useLiveState();
  return (
    <div style={{ minHeight: "100vh", maxWidth: 1180, margin: "0 auto", padding: "0 24px 32px" }}>
      <Header
        connected={state.connected}
        latencyMs={state.latencyMs}
        equity={state.metrics.equity}
        openPnl={state.metrics.openPnl}
      />
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
    </div>
  );
}
