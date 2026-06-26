/**
 * ThemeProvider.tsx — テーマ状態の保持・CSS変数適用・localStorage 永続化。
 *
 * applyTheme は themes.ts の Theme を CSS 変数 (--ax-*) としてルート要素へ set する。
 * コンポーネントは var(--ax-*) を参照し、色をハードコードしない。
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  DEFAULT_THEME,
  themeToCssVars,
  themes,
  type Theme,
  type ThemeName,
} from "./themes";

const STORAGE_KEY = "aevum.theme";

interface ThemeContextValue {
  name: ThemeName;
  theme: Theme;
  setTheme: (name: ThemeName) => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  const vars = themeToCssVars(theme);
  for (const [name, value] of Object.entries(vars)) {
    root.style.setProperty(name, value);
  }
}

function loadInitialTheme(): ThemeName {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "terminal" || saved === "editorial") {
      return saved;
    }
  } catch {
    /* localStorage 不可環境は既定にフォールバック */
  }
  return DEFAULT_THEME;
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [name, setName] = useState<ThemeName>(loadInitialTheme);
  const theme = themes[name];

  useEffect(() => {
    applyTheme(theme);
    try {
      localStorage.setItem(STORAGE_KEY, name);
    } catch {
      /* 保存不可でも表示は継続 */
    }
  }, [name, theme]);

  const setTheme = useCallback((next: ThemeName) => setName(next), []);
  const value = useMemo<ThemeContextValue>(
    () => ({ name, theme, setTheme }),
    [name, theme, setTheme],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return ctx;
}
