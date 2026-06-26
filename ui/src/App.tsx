import { useTheme } from "./theme/ThemeProvider";

/**
 * 最小スキャフォールド（terminal テーマ）。
 * テーマ系（themes.ts + ThemeProvider）が CSS 変数を適用できることを示す土台。
 * Header / MetricCards / PriceChart / SignalPanel / PositionsTable は後続で追加する。
 */
export default function App() {
  const { name } = useTheme();
  return (
    <div style={{ minHeight: "100vh", padding: 24 }}>
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          borderBottom: "1px solid var(--ax-border)",
          paddingBottom: 12,
          marginBottom: 24,
        }}
      >
        <h1 className="ax-display" style={{ margin: 0, fontSize: 22, letterSpacing: 1 }}>
          Aevum
        </h1>
        <span style={{ color: "var(--ax-text-secondary)", fontSize: 13 }}>
          theme: <span style={{ color: "var(--ax-accent)" }}>{name}</span>
        </span>
      </header>

      <div className="ax-panel" style={{ padding: 20 }}>
        <p style={{ margin: 0, color: "var(--ax-text-secondary)" }}>
          monitoring dashboard scaffold — theme system ready (read-only).
        </p>
      </div>
    </div>
  );
}
