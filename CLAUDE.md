# CLAUDE.md — Aevum

このファイルはClaude Codeが毎セッション自動で読み込む設計書。
実装前に必ずこのファイル全体を読むこと。

---

## プロジェクト概要

**Aevum** — Hyperliquid の無期限先物（BTC・ETH）を対象に、
Transformer モデルで売買シグナルを生成し自動発注するトレーディングシステム。

方法論の土台: López de Prado『Advances in Financial Machine Learning』
ラベル定義: Triple-Barrier Method

---

## ハードウェア構成

| マシン | 役割 |
|---|---|
| PC（RTX 2070 Super） | 特徴量計算・学習 |
| Raspberry Pi 5（NVMe） | 推論・発注・API・UI |

---

## 技術スタック

| 層 | 技術 |
|---|---|
| 言語 | Python 3.10 固定（hyperliquid-python-sdk の制約） |
| DB | TimescaleDB + asyncpg |
| 特徴量計算（PC） | Polars |
| 特徴量計算（Pi） | numpy / pandas（逐次） |
| 学習 | PyTorch |
| 推論 | ONNX Runtime（ARM最適化） |
| API | FastAPI + WebSocket |
| UI | React または Svelte + TradingView Lightweight Charts |
| 外部キュー | Redis または内部キュー |

---

## ディレクトリ構成

```
aevum/
├── CLAUDE.md               ← このファイル
├── data/
│   ├── ingestion.py        PC側: WebSocket受信 → TimescaleDB書き込み
│   ├── features.py         PC側: Polarsで特徴量バルク計算
│   └── labels.py           PC側: Triple-Barrierラベリング
├── model/
│   ├── dataset.py          PyTorch Dataset（正規化・分割）
│   ├── transformer.py      Transformerモデル定義
│   └── train.py            学習スクリプト + ONNXエクスポート
├── live/
│   ├── inference.py        Pi側: ONNX Runtime推論
│   ├── execution.py        Pi側: Hyperliquid発注
│   └── risk.py             リスク管理ゲート（全発注が通る）
├── api/
│   └── server.py           FastAPI + WebSocket配信
├── ui/                     監視ダッシュボード（React/Svelte）
├── shared/
│   └── volatility.py       ★ σ計算式の唯一の定義（全モジュールがここを呼ぶ）
├── schema/
│   └── schema_v1.sql       TimescaleDB DDL
└── tests/
```

---

## データの流れ

```
Hyperliquid WebSocket
        ↓
   ingestion.py  →  TimescaleDB（生データ3テーブル）
        ↓
   features.py   →  bar_features（正規化なし・生で保存）
        ↓
   labels.py     →  labels（σも一緒に保存）
        ↓
   dataset.py    →  PyTorch テンソル（ここで正規化）
        ↓
   transformer.py → 学習済みモデル → ONNX export
        ↓
   inference.py  →  model_predictions
        ↓
   execution.py  →  Hyperliquid 発注
        ↓
   server.py     →  監視UI へ配信
```

---

## DBスキーマ（TimescaleDB）

### 生データ（WebSocket購読に対応）
- `ohlcv_bars` ← candle チャンネル
- `orderbook_snapshots` ← l2Book チャンネル（板はFLOAT8配列。JSOBより高速）
- `funding_oi` ← activeAssetCtx チャンネル

### 計算・学習用
- `bar_features` — 特徴量（正規化なし）
- `labels` — Triple-Barrierラベル（使用したσも保存）

### 本番・執行用
- `model_predictions`
- `orders` / `positions`

---

## 特徴量（58個・10カテゴリ）

| カテゴリ | 数 |
|---|---|
| Price / Return | 11 |
| Volatility | 6 |
| Volume | 4 |
| Order Book Imbalance (OBI) | 10 |
| Spread | 3 |
| Technical Indicators | 8 |
| Microstructure | 5 |
| Cross-Asset | 3 |
| Temporal | 4 |
| Funding / OI | 4 |

**Cross-Assetは2パス計算**: 両銘柄の計算完了後に `cross_*` を埋める。

---

## 不変条件（絶対に守ること）

### ★ 最重要: σ計算式の統一

**`shared/volatility.py` に唯一の定義を置き、全モジュールから呼ぶ。**
以下の3箇所で式がズレると致命的な失敗モードになる:

1. `labels.py` — バリアサイズ決定
2. `execution.py` — ライブのバリア幅
3. 監視UI — σ表示

この3箇所で別々に計算を実装してはいけない。

### train/live 整合性

- 正規化は `dataset.py`（モデル入力時）のみ。DBには生で保存（リーク防止）
- `features.py` の計算式を train/live で同一に保つ
- Walk-forward検証のみ使用（**k-fold 禁止**）

### アーキテクチャ分離

- トレーディングループ（`execution.py`）はGUIから独立させる
- すべての注文は `risk.py`（リスク管理ゲート）を通す
- UI由来のコマンドもゲートを迂回できない
- `server.py` は状態配信のみ。ロジックを持たない

### 発注の非対称設計

- エントリー・利確: 指値（maker）
- 損切り: 成行（taker）で約定を保証

---

## 検証ルール

- **k-fold 禁止**。Walk-forward + Purging + Embargo のみ
- Purge長 = 最大ラベル保有期間
- Shadow-mode（実資金なし）を経てからlive

---

## 実装順序

依存関係の順に進める。前のモジュールが単体テストできる状態になってから次へ。

```
1. shared/volatility.py     ← σ計算式を最初に固定
2. schema/schema_v1.sql     ← DB定義
3. data/ingestion.py        ← WebSocket → DB
4. data/labels.py           ← Triple-Barrier
5. data/features.py         ← Polars特徴量計算
6. model/dataset.py         ← PyTorch Dataset
7. model/transformer.py     ← モデル定義
8. model/train.py           ← 学習 + ONNXエクスポート
9. live/inference.py        ← ONNX Runtime（Pi）
10. live/risk.py            ← リスク管理ゲート
11. live/execution.py       ← 発注
12. api/server.py           ← FastAPI
13. ui/                     ← 監視ダッシュボード
```

---

## Stop and Ask（実装を止めて質問する条件）

以下に該当する場合は実装せず、必ず確認を求める:

- σ計算式の定義が曖昧なとき
- 発注・執行ロジックの仕様が不明なとき
- DBスキーマへの破壊的変更が必要なとき
- `risk.py` を迂回する設計になりそうなとき
- train/live で異なる計算が必要に見えるとき

---

## 禁止事項

- 仕様にない機能を勝手に追加しない
- 無関係な整形やリファクタリングをしない
- k-fold を使わない
- σを複数箇所で別々に計算しない
- `risk.py` を通さない発注経路を作らない
- Python 3.10 以外を使わない

---

## セッション開始時のチェックリスト

1. `git status` で未コミット変更を確認
2. 今回実装するモジュールとその「完了定義」を確認
3. 依存するモジュールが完成しているか確認
4. baseline の検証結果（lint / typecheck / test）を記録
