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
| 特徴量計算（Pi） | Polars（`features.py` のロジックを再利用） |
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
│   ├── features.py         PC/Pi共通: Polars特徴量計算（バルク＝逐次で同一コード）
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
- **Pi側特徴量計算は `features.py` の Polars ロジックを再利用**（numpy/pandas で別実装しない）。
  train/live を「テスト担保」でなく「同一コード」で構造的に一致させる（Polars は ARM 対応）
- ライブはローリング窓で計算する。末尾行をバルクと一致させるため、最低
  `features.WARMUP_BARS + seq_len` バーの履歴を保持する（履歴依存: EWMA σ・Wilder TA・ret_240）
- 正規化（z-score + ±5σ クリップ）は ONNX グラフに同梱（`NormalizedModel`）。Pi は生特徴を渡すだけ
- Walk-forward検証のみ使用（**k-fold 禁止**）

### アーキテクチャ分離

- トレーディングループ（`execution.py`）はGUIから独立させる
- すべての注文は `risk.py`（リスク管理ゲート）を通す
- UI由来のコマンドもゲートを迂回できない
- `server.py` は状態配信のみ。ロジックを持たない

### 発注の非対称設計

- エントリー・利確: 指値（maker）
- 損切り: 成行（taker）で約定を保証

### ingestion の接続維持（死活監視 + 自動再接続）

- `ingestion.py` は WS の**死活監視**（受信スレッド生存）と**データ鮮度監視**（一定時間無受信で異常）を持ち、
  異常時に **Info を作り直して 3チャンネル（candle/l2Book/activeAssetCtx）を自動再購読**する。
  再接続時はログを残す（無言停止の再発を可視化）。リトライ上限・バックオフは設定可能。
- 理由: hyperliquid SDK は WS が `Expired` で切れると `run_forever()` が return して受信スレッドが
  無言終了する（**SDK は自動再接続しない**）。常時稼働（Pi）要件に内在する責務として補う。
- スコープは**接続維持のみ**。データ補完/バックフィルや書き込みロジック（`BatchWriter`）は持ち込まない。

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

## ローカル開発スタック（dev runbook）

ローカルで E2E を回す手順。Pi 本番ではなく開発機（Windows / .venv）想定。

- **DB**: `docker compose up -d`（TimescaleDB pg16）。DSN は環境変数
  `AEVUM_DB_DSN=postgresql://postgres:aevum@localhost:5432/aevum`。破棄は `docker compose down -v`。
- **Python**: `./.venv/Scripts/python.exe`（**3.10 固定**。既定の 3.12 を使わない）。
- **Windows コンソール**: cp932。スクリプト実行時は `PYTHONIOENCODING=utf-8`、print は ASCII 推奨（σ/µ/→ で落ちる）。
- スキーマ適用: `python scripts/apply_schema.py` ／ 静的 seed: `python scripts/seed_dev.py`
- candle 履歴 backfill（冪等・REST candleSnapshot。**穴埋めはこれ**）: `python scripts/backfill_candles.py --days N`
- ライブ収集（常駐・死活監視+自動再接続）: `python -m data.ingestion`
- 特徴量生成（features.py 無改修ランナー）: `python scripts/run_features.py`（`--no-store` で計算のみ）
- DB 確認: `python check_db.py`
- API: `uvicorn api.server:app`（read-only）／ UI: `cd ui && npm run dev`（vite proxy `/api`→8000, `/ws`→8000）
- テスト: `./.venv/Scripts/python.exe -m pytest -q`
- **注意**: セッションをまたぐと background プロセス（ingestion / uvicorn / vite）は全て停止する。再開時はこの runbook で起動し直す。

---

## 現状チェックポイント（最終更新 2026-06-27）

- **E2E 段階3まで到達**。`ingestion → ohlcv/book/funding` と `features → bar_features` を**実データで配線・検証済み**。
- features 核心検証は全 PASS: σ/TA の **train(bulk)↔live(rolling) 一致**、外部参照（pandas/polars）一致、Cross 2パス、Pi 計算時間に十分な余裕。
- ingestion に**接続維持スーパーバイザ**を追加し、実地で `Expired` の自動再接続を確認済み（不変条件「ingestion の接続維持」参照）。
- **未検証（残課題）**: features の **板/funding 特徴の rolling↔bulk 一致**（ライブ board/funding の蓄積待ち。ingestion 安定継続で解消可能）。
- equity / winRate は当面 null 表示（永続化テーブル無し・winRate 定義未確定）。
- **次の本流**: `labels`（実データ実行・現状 labels テーブルは空）→ `dataset` → `train` → `inference`(shadow) → `execution`。
- 詳細な経緯は memory（`features-stage3-verified` / `ingestion-reconnect-gap` / `ui-live-wiring-gaps`）を参照。

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
