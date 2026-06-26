/**
 * Header.tsx — ブランド・ライブ時刻・latency・equity・（後で）ThemeToggle。
 *
 * 表示のみ。色は var(--ax-*)、ブランドは見出しフォント（var(--ax-font-display)）。
 * ThemeToggle は実装順 #6 で右端スロットに差し込む（今はテーマ名チップを仮置き）。
 */
import { useEffect, useState } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { clockFromDate, signedUsd, usd } from "../format";

/** 1 秒ごとに更新するローカル時計。 */
function useNow(): Date {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(id);
  }, []);
  return now;
}

function Stat({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 2 }}>
      <span
        style={{
          fontSize: 10,
          letterSpacing: 1,
          textTransform: "uppercase",
          color: "var(--ax-text-tertiary)",
        }}
      >
        {label}
      </span>
      <span style={{ fontSize: 14, color: "var(--ax-text-primary)" }}>{children}</span>
    </div>
  );
}

export interface HeaderProps {
  connected: boolean;
  latencyMs: number | null;
  equity: number;
  openPnl: number;
}

export default function Header({ connected, latencyMs, equity, openPnl }: HeaderProps) {
  const { name } = useTheme();
  const now = useNow();
  const dotColor = connected ? "var(--ax-positive)" : "var(--ax-negative)";

  return (
    <header
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 24,
        padding: "14px 4px",
        borderBottom: "1px solid var(--ax-border)",
        marginBottom: 20,
      }}
    >
      {/* 左: ブランド */}
      <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
        <span
          className="ax-display"
          style={{ fontSize: 22, letterSpacing: 1.5, color: "var(--ax-text-primary)" }}
        >
          Aevum
        </span>
        <span style={{ fontSize: 11, letterSpacing: 1, color: "var(--ax-text-tertiary)" }}>
          monitoring · read-only
        </span>
      </div>

      {/* 右: 状態統計 + テーマ */}
      <div style={{ display: "flex", alignItems: "center", gap: 28 }}>
        <Stat label="time">{clockFromDate(now)}</Stat>

        <Stat label="latency">
          <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <span
              aria-hidden
              style={{
                width: 7,
                height: 7,
                borderRadius: "50%",
                background: dotColor,
                boxShadow: `0 0 6px ${dotColor}`,
              }}
            />
            {connected && latencyMs != null ? `${latencyMs} ms` : "offline"}
          </span>
        </Stat>

        <Stat label="equity">{usd(equity)}</Stat>

        <Stat label="open pnl">
          <span style={{ color: openPnl >= 0 ? "var(--ax-positive)" : "var(--ax-negative)" }}>
            {signedUsd(openPnl)}
          </span>
        </Stat>

        {/* 実装順 #6 で ThemeToggle に置換するスロット */}
        <span
          style={{
            fontSize: 11,
            letterSpacing: 1,
            textTransform: "uppercase",
            color: "var(--ax-accent)",
            border: "1px solid var(--ax-border)",
            borderRadius: 6,
            padding: "5px 10px",
          }}
        >
          {name}
        </span>
      </div>
    </header>
  );
}
