/**
 * useLiveState.ts — server.py（読み取り専用）への接続。
 *
 * 実装順 #7: ダミーを実データに置換する唯一の口。
 *   - 初回: REST GET /api/snapshot で即描画
 *   - 以降: WebSocket /ws の定期スナップショットで更新（自動再接続）
 * server は DashboardState から connected/latencyMs を除いた形を返す（build_snapshot）。
 * connected/latencyMs はクライアント側で付与する。
 *
 * 薄いクライアント: ここでは表示状態を受け取るだけ。発注/制御/σ再計算は一切しない。
 */
import { useEffect, useRef, useState } from "react";
import type { DashboardState } from "./types";
import { DUMMY_STATE } from "./dummy";

const SNAPSHOT_URL = "/api/snapshot";
const RECONNECT_MS = 2000;

/** server が返す形（DashboardState からクライアント専用フィールドを除いたもの）。 */
type ServerSnapshot = Omit<DashboardState, "connected" | "latencyMs">;

/** 接続前のプレースホルダ: レイアウトを即描画するためダミーを offline で使う。 */
const INITIAL: DashboardState = { ...DUMMY_STATE, connected: false, latencyMs: null };

function wsUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/ws`;
}

/** server スナップショット + クライアント由来メタ（接続状態・データ鮮度）。 */
function withClientMeta(payload: ServerSnapshot): DashboardState {
  const stamped = Date.parse(payload.updatedAt);
  // latency = データ鮮度（server stamp → 受信）。クロックずれは LAN/localhost では小さい。
  const latencyMs = Number.isFinite(stamped) ? Math.max(0, Date.now() - stamped) : null;
  return { ...payload, connected: true, latencyMs };
}

export function useLiveState(): DashboardState {
  const [state, setState] = useState<DashboardState>(INITIAL);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let cancelled = false;
    let retry: number | null = null;

    // 初回 REST 取得（高速初回描画。WS が来れば置き換わる）。
    fetch(SNAPSHOT_URL)
      .then((r) => (r.ok ? (r.json() as Promise<ServerSnapshot>) : Promise.reject(r.status)))
      .then((payload) => {
        if (!cancelled) setState(withClientMeta(payload));
      })
      .catch(() => {
        /* WS 経由で更新されるので握りつぶす */
      });

    const connect = () => {
      if (cancelled) return;
      const ws = new WebSocket(wsUrl());
      wsRef.current = ws;
      ws.onmessage = (ev) => {
        if (cancelled) return;
        try {
          setState(withClientMeta(JSON.parse(ev.data) as ServerSnapshot));
        } catch {
          /* 壊れたフレームは無視 */
        }
      };
      ws.onclose = () => {
        if (cancelled) return;
        setState((s) => ({ ...s, connected: false }));
        retry = window.setTimeout(connect, RECONNECT_MS); // 自動再接続
      };
      ws.onerror = () => ws.close(); // onclose 経由で再接続
    };
    connect();

    return () => {
      cancelled = true;
      if (retry) window.clearTimeout(retry);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, []);

  return state;
}
