/**
 * App.tsx — ダッシュボード合成（terminal表示・ダミーデータ）。
 *
 * 実装順 #3: Header / MetricCards / SignalPanel をダミーで通す。
 * PriceChart(#4) / PositionsTable(#5) / ThemeToggle(#6) / useLiveState(#7) は後続。
 * 状態は今は DUMMY_STATE。#7 で WebSocket+REST 出力に差し替える（型は types.ts で同一）。
 */
import Header from "./components/Header";
import MetricCards from "./components/MetricCards";
import SignalPanel from "./components/SignalPanel";
import { DUMMY_STATE } from "./data/dummy";

export default function App() {
  const state = DUMMY_STATE;
  return (
    <div style={{ minHeight: "100vh", maxWidth: 1180, margin: "0 auto", padding: "0 24px 32px" }}>
      <Header
        connected={state.connected}
        latencyMs={state.latencyMs}
        equity={state.metrics.equity}
        openPnl={state.metrics.openPnl}
      />
      <MetricCards metrics={state.metrics} />
      <SignalPanel
        prediction={state.prediction}
        probThreshold={state.probThreshold}
        features={state.features}
      />
    </div>
  );
}
