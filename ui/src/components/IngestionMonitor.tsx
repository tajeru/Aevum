/**
 * IngestionMonitor.tsx — ingestion health view (freshness / throughput / accumulation).
 *
 * 3 tables x 2 symbols. Polls GET /api/monitoring/ingestion every 5 s via useIngestionStatus.
 * Read-only. Colors from theme tokens only (no hardcoded values).
 */
import { useIngestionStatus, TableSymbolStats } from "../data/useIngestionStatus";

const TABLES = ["ohlcv_bars", "orderbook_snapshots", "funding_oi"] as const;
type TableName = (typeof TABLES)[number];

const TABLE_LABELS: Record<TableName, string> = {
  ohlcv_bars: "OHLCV (candle)",
  orderbook_snapshots: "Order Book",
  funding_oi: "Funding / OI",
};

const TABLE_CADENCE: Record<TableName, string> = {
  ohlcv_bars: "5 min bars",
  orderbook_snapshots: "real-time",
  funding_oi: "~1 min",
};

// Stale thresholds derived from write cadence (mirrors MONITORING_STALE_SECONDS in server.py).
const STALE_AFTER: Record<TableName, number> = {
  ohlcv_bars: 7 * 60,        // 7 min  (cadence: 5 min)
  orderbook_snapshots: 60,   // 60 s   (cadence: real-time)
  funding_oi: 3 * 60,        // 3 min  (cadence: ~1 min)
};

const SYMBOLS = ["BTC", "ETH"] as const;

// --------------------------------------------------------------------------- #
// Format helpers (local to this view)
// --------------------------------------------------------------------------- #

function formatAge(secs: number | null): string {
  if (secs == null) return "—";
  const s = Math.max(0, secs);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${(s / 3600).toFixed(1)}h ago`;
}

function formatSpan(secs: number | null): string {
  if (secs == null || secs < 0) return "—";
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m`;
  if (secs < 86400) return `${(secs / 3600).toFixed(1)}h`;
  return `${(secs / 86400).toFixed(1)}d`;
}

function fmtCount(n: number): string {
  return n.toLocaleString("en-US");
}

// --------------------------------------------------------------------------- #
// Symbol cell
// --------------------------------------------------------------------------- #

function SymbolCell({
  stats,
  staleAfter,
}: {
  stats: TableSymbolStats | null;
  staleAfter: number;
}) {
  const secs = stats?.seconds_since_last_write ?? null;
  const isStale = secs == null || secs > staleAfter;
  const dotColor = isStale ? "var(--ax-negative)" : "var(--ax-positive)";
  const label = isStale ? "STALE" : "FRESH";

  return (
    <div style={{ padding: "12px 16px" }}>
      {/* freshness row */}
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 10 }}>
        <span
          aria-hidden
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: dotColor,
            boxShadow: `0 0 5px ${dotColor}`,
            flexShrink: 0,
          }}
        />
        <span style={{ fontSize: 11, letterSpacing: 0.8, color: dotColor }}>{label}</span>
        <span style={{ fontSize: 11, color: "var(--ax-text-secondary)" }}>
          {formatAge(secs)}
        </span>
      </div>

      {/* 1h throughput */}
      <div style={{ fontSize: 12, color: "var(--ax-text-secondary)", marginBottom: 4 }}>
        <span style={{ color: "var(--ax-text-tertiary)", marginRight: 6, fontSize: 10 }}>1 H</span>
        {stats != null ? `${fmtCount(stats.rows_last_1h)} rows` : "—"}
      </div>

      {/* accumulation */}
      <div style={{ fontSize: 12, color: "var(--ax-text-secondary)" }}>
        <span style={{ color: "var(--ax-text-tertiary)", marginRight: 6, fontSize: 10 }}>TOTAL</span>
        {stats != null
          ? `${fmtCount(stats.rows_total)} / ${formatSpan(stats.span_seconds)}`
          : "—"}
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- #
// Per-table panel
// --------------------------------------------------------------------------- #

function TablePanel({
  tableName,
  symbols,
}: {
  tableName: TableName;
  symbols: Record<string, TableSymbolStats> | null;
}) {
  const staleAfter = STALE_AFTER[tableName];

  return (
    <div
      style={{
        background: "var(--ax-panel)",
        border: "1px solid var(--ax-border)",
        borderRadius: 6,
        overflow: "hidden",
      }}
    >
      {/* panel header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "8px 16px",
          borderBottom: "1px solid var(--ax-border)",
        }}
      >
        <span
          style={{
            fontSize: 12,
            letterSpacing: 1,
            textTransform: "uppercase",
            color: "var(--ax-text-primary)",
          }}
        >
          {TABLE_LABELS[tableName]}
        </span>
        <span style={{ fontSize: 10, color: "var(--ax-text-tertiary)" }}>
          {TABLE_CADENCE[tableName]}
          {" · stale > "}
          {formatSpan(staleAfter)}
        </span>
      </div>

      {/* BTC / ETH columns */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr" }}>
        {SYMBOLS.map((sym, i) => (
          <div
            key={sym}
            style={{ borderLeft: i > 0 ? "1px solid var(--ax-border)" : "none" }}
          >
            <div
              style={{
                padding: "5px 16px",
                fontSize: 10,
                letterSpacing: 1.2,
                textTransform: "uppercase",
                color: "var(--ax-text-tertiary)",
                borderBottom: "1px solid var(--ax-border)",
              }}
            >
              {sym}
            </div>
            <SymbolCell stats={symbols?.[sym] ?? null} staleAfter={staleAfter} />
          </div>
        ))}
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- #
// Root component
// --------------------------------------------------------------------------- #

export default function IngestionMonitor() {
  const status = useIngestionStatus();

  return (
    <div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 16,
        }}
      >
        <span style={{ fontSize: 13, color: "var(--ax-text-secondary)" }}>
          Ingestion health &mdash; 3 tables &times; 2 symbols &middot; poll every 5 s
        </span>
        {status == null && (
          <span style={{ fontSize: 11, color: "var(--ax-text-tertiary)" }}>loading&hellip;</span>
        )}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {TABLES.map((tbl) => (
          <TablePanel
            key={tbl}
            tableName={tbl}
            symbols={status?.tables[tbl] ?? null}
          />
        ))}
      </div>
    </div>
  );
}
