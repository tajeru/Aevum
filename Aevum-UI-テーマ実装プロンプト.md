# Aevum UI — 監視ダッシュボード実装プロンプト（2テーマ切り替え）

> Claude Code に渡す指示。CLAUDE.md を読んだ上で `ui/` を実装する。

---

## 目的

Aevum の監視ダッシュボードを実装する。**2つのテーマを切り替え可能**にする。

- `terminal` — ダークターミナル（ティール差し色・サンセリフ）
- `editorial` — ダーク・エディトリアル（琥珀差し色・セリフ見出し・チャコール背景）

レイアウトは両テーマ共通。切り替わるのは**色とタイポグラフィ**のみ。
ユーザーがUIのトグルで切り替え、選択はlocalStorageに保存して再訪時に復元する。

---

## 技術スタック（固定）

- Vite + React + TypeScript
- TradingView Lightweight Charts（ローソク足・バリアライン）
- WebSocket（状態受信）+ REST（GET のみ）
- UIライブラリは原則追加しない（必要時のみ最小限）

---

## 設計原則（Aevum の不変条件に従う）

1. **薄いクライアント**：UIはロジックを持たない。テーマは純粋な表示状態で、トレーディングループ・server.py・risk.py には一切触れない。
2. **読み取り専用**：server.py は GET + WebSocket 配信のみ（#29）。UIに発注・kill-switch等の制御を実装しない。
3. **テーマは単一の真実**：両テーマを `src/theme/themes.ts` に1度だけ定義する。そこから
   (a) CSS カスタムプロパティをルート要素に適用し、
   (b) TradingView チャートの `applyOptions()` にも同じ値を渡す。
   **DOM側とチャート側で色を別々に書かない**（σ統一と同じ思想：切り替え時の色ドリフトを構造的に防ぐ）。
4. **数値は必ず丸めて表示**（toFixed / Intl.NumberFormat）。

---

## テーマ定義（`src/theme/themes.ts`）

両テーマを以下のトークンで定義する。キー名は共通、値だけが異なる。

| トークン | 用途 | terminal | editorial |
|---|---|---|---|
| `bg` | ページ背景 | `#0a1117` | `#211d18` |
| `panel` | パネル背景 | `#0d1820` | `#28231b` |
| `border` | 罫線 | `#1c2b36` | `#3a342b` |
| `textPrimary` | 見出し・主要数値 | `#e8eef0` | `#f3efe6` |
| `textSecondary` | ラベル・補助 | `#6b8290` | `#9a8f7a` |
| `textTertiary` | ヒント・フッタ | `#42606f` | `#6e6555` |
| `accent` | 差し色（シグナル等） | `#5dcaa5` | `#d4a23e` |
| `positive` | 利益・上昇 | `#5dcaa5` | `#d4a23e` |
| `negative` | 損失・SL | `#e24b4a` | `#c87a52` |
| `candleUp` | 陽線 | `#1d9e75` | `#d4a23e` |
| `candleDown` | 陰線 | `#d85a30` | `#7a7058` |
| `wick` | ヒゲ | `#33423a` | `#4a4234` |
| `fontDisplay` | 見出し・ブランド | `'Inter', system-ui, sans-serif` | `Georgia, 'Times New Roman', serif` |
| `fontBody` | ラベル・数値（共通） | `Arial, Helvetica, sans-serif` | 同左 |

TypeScript の型例：

```ts
export type ThemeName = 'terminal' | 'editorial';
export interface Theme {
  bg: string; panel: string; border: string;
  textPrimary: string; textSecondary: string; textTertiary: string;
  accent: string; positive: string; negative: string;
  candleUp: string; candleDown: string; wick: string;
  fontDisplay: string; fontBody: string;
}
export const themes: Record<ThemeName, Theme> = { terminal: {...}, editorial: {...} };
```

---

## 切り替え機構

- `ThemeProvider`（React context）で現在のテーマ名を保持。
- `applyTheme(theme)`：Theme の各値を `--ax-bg` 等の CSS 変数としてルート要素（`document.documentElement` か appルートdiv）に set する。
- コンポーネントの CSS は `var(--ax-bg)` 等を参照（ハードコード禁止）。
- 見出しは `font-family: var(--ax-font-display)`、ラベル・数値は `var(--ax-font-body)`。
- トグルUI：ヘッダーに小さな切り替えボタン or セレクト（"terminal / editorial"）。
- 選択を `localStorage`（例 `aevum.theme`）に保存し、初期化時に復元。なければ `terminal` 既定。

---

## TradingView チャートのテーマ連携

テーマ切り替え時、チャートにも同じ Theme オブジェクトを反映する。マッピング：

| チャート設定 | 使うトークン |
|---|---|
| `layout.background` (solid) | `bg` |
| `layout.textColor` | `textSecondary` |
| `grid.vertLines / horzLines.color` | `border` |
| candlestick `upColor` / `borderUpColor` / `wickUpColor` | `candleUp` / `wick` |
| candlestick `downColor` / `borderDownColor` / `wickDownColor` | `candleDown` / `wick` |
| priceLine: TP | `accent` |
| priceLine: entry | `textSecondary` |
| priceLine: SL | `negative` |

テーマ変更時は `chart.applyOptions()` と `series.applyOptions()` を呼び直す。
**チャート用の色定数を別ファイルに持たない**——必ず `themes.ts` の Theme から取る。

---

## コンポーネント構成（案）

```
ui/src/
├── theme/
│   ├── themes.ts          2テーマ定義（単一の真実）
│   ├── ThemeProvider.tsx  context + applyTheme + localStorage
│   └── ThemeToggle.tsx    切り替えUI
├── components/
│   ├── Header.tsx         ブランド・時刻・latency・equity・ThemeToggle
│   ├── MetricCards.tsx    equity / open PnL / win rate / σ(per-bar→×√48)
│   ├── PriceChart.tsx     TradingView LW Charts + TP/entry/SLライン
│   ├── SignalPanel.tsx    方向・確率・閾値バー・ライブ特徴量
│   └── PositionsTable.tsx symbol/side/size/entry/SL/bars held(n/48)/PnL
├── data/
│   └── useLiveState.ts    WebSocket購読 + REST GET（表示用のみ）
├── App.tsx
└── main.tsx
```

---

## σ表示の要件

メトリクスに per-bar σ を表示し、`per-bar 0.0041 → ×√48 → 0.0284` のように
**呼び出し側で√horizonスケールされる**ことが分かる形にする（設計判断 #3・#9 の可視化）。
表示するσは shared/volatility.py 由来の値をそのまま受け取り、UI側で再計算しない。

---

## Definition of Done

- [ ] `npm run dev` で起動し、ダッシュボードが表示される
- [ ] ヘッダーのトグルで terminal ⇄ editorial が即時に切り替わる（色・見出しフォント・チャート色すべて）
- [ ] リロード後も選択テーマが復元される（localStorage）
- [ ] チャートの色がDOMのテーマと一致する（切り替え時にズレない）
- [ ] 全数値が丸めて表示される
- [ ] 発注・制御UIが存在しない（読み取り専用）
- [ ] `npm run build` が型エラーなく通る（TypeScript strict）

---

## 制約

- 仕様にない機能・パネルを勝手に増やさない
- テーマ色をコンポーネントにハードコードしない（必ず CSS 変数 / themes.ts 経由）
- チャート色を themes.ts 以外で定義しない
- UIにトレーディングロジック・発注経路を持ち込まない
- まず `themes.ts` と `ThemeProvider` を作り、1テーマで全コンポーネントを通してから2テーマ目を足す

---

## 実装順序

1. Vite + React + TS プロジェクト作成、lightweight-charts 導入
2. `themes.ts`（まず terminal のみ値を入れる）+ `ThemeProvider` + CSS変数適用
3. Header / MetricCards / SignalPanel をダミーデータで実装（terminal表示）
4. PriceChart：TradingView LW Charts を themes.ts の色で初期化、TP/entry/SLライン
5. PositionsTable
6. `editorial` テーマの値を themes.ts に追加、ThemeToggle 実装、切り替え確認
7. useLiveState：WebSocket + REST GET を繋ぎ、ダミーを実データに置換
8. localStorage 永続化、build 確認
