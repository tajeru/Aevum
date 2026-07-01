/**
 * useIngestionStatus.ts — GET /api/monitoring/ingestion を 5 秒ポーリング。
 *
 * 収集監視ビュー（IngestionMonitor）専用のデータフック。
 * 読み取り専用。DB 以外は触れない（server.py の制約と同じ精神）。
 */
import { useEffect, useState } from "react";

const INGESTION_STATUS_URL = "/api/monitoring/ingestion";
const POLL_INTERVAL_MS = 5000;

/** ohlcv_bars / orderbook_snapshots / funding_oi の 1 銘柄ぶん統計。 */
export interface TableSymbolStats {
  last_write_at: string | null;       // ISO8601 or null (no rows)
  seconds_since_last_write: number | null;
  rows_last_1h: number;
  rows_total: number;
  oldest_at: string | null;
  span_seconds: number | null;        // MAX(time) - MIN(time) in seconds
}

/** GET /api/monitoring/ingestion の応答形。 */
export interface IngestionStatus {
  tables: Record<string, Record<string, TableSymbolStats>>;
}

/**
 * useIngestionStatus — コンポーネントがマウントされている間だけポーリングする。
 * IngestionMonitor タブが非表示のときはアンマウントされポーリングも止まる。
 * エラー時は直前の値を維持する（チラつき防止）。
 */
export function useIngestionStatus(): IngestionStatus | null {
  const [status, setStatus] = useState<IngestionStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: number | null = null;

    const poll = () => {
      fetch(INGESTION_STATUS_URL)
        .then((r) => (r.ok ? (r.json() as Promise<IngestionStatus>) : Promise.reject(r.status)))
        .then((data) => {
          if (!cancelled) setStatus(data);
        })
        .catch(() => { /* keep stale data on fetch error */ })
        .finally(() => {
          if (!cancelled) timer = window.setTimeout(poll, POLL_INTERVAL_MS);
        });
    };

    poll();
    return () => {
      cancelled = true;
      if (timer != null) window.clearTimeout(timer);
    };
  }, []);

  return status;
}
