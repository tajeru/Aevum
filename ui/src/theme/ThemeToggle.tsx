/**
 * ThemeToggle.tsx — terminal / editorial 切り替え UI（ヘッダ右端）。
 *
 * 表示状態のみ: useTheme().setTheme を呼ぶだけ。永続化（localStorage）と CSS 変数/
 * チャート色の反映は ThemeProvider / 各コンポーネントが担う（ここはロジックを持たない）。
 * 色は var(--ax-*) のみ。アクティブは accent 背景＋bg 文字でテーマに追従。
 */
import { useTheme } from "./ThemeProvider";
import type { ThemeName } from "./themes";

const OPTIONS: { name: ThemeName; label: string }[] = [
  { name: "terminal", label: "terminal" },
  { name: "editorial", label: "editorial" },
];

export default function ThemeToggle() {
  const { name, setTheme } = useTheme();
  return (
    <div
      role="group"
      aria-label="theme"
      style={{
        display: "inline-flex",
        border: "1px solid var(--ax-border)",
        borderRadius: 6,
        overflow: "hidden",
      }}
    >
      {OPTIONS.map((opt) => {
        const active = opt.name === name;
        return (
          <button
            key={opt.name}
            type="button"
            onClick={() => setTheme(opt.name)}
            aria-pressed={active}
            style={{
              appearance: "none",
              border: "none",
              cursor: "pointer",
              padding: "5px 11px",
              fontSize: 11,
              letterSpacing: 1,
              textTransform: "uppercase",
              fontFamily: "var(--ax-font-body)",
              background: active ? "var(--ax-accent)" : "transparent",
              color: active ? "var(--ax-bg)" : "var(--ax-text-secondary)",
              transition: "background 120ms ease, color 120ms ease",
            }}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
