# CLAUDE.md — Aevum

このファイルはClaude Codeが毎セッション自動で読み込む設計書。
実装前に必ずこのファイル全体を読むこと。

> **このドキュメントの読み方**
> 本書は3層構造になっている。
> 1. **判断哲学（第0章）** — なぜそう決めるのかの「軸」。実装判断はすべてここに照らす。
> 2. **不変条件・仕様** — 何を守り、何を作るか。
> 3. **手順・チェックリスト** — どう動かすか。
>
> 迷ったとき・「推奨」と異なる選択をしたくなったとき・新しい判断を求められたときは、
> **必ず第0章に立ち返る**こと。第0章と矛盾する実装・提案はしない。

---

## 第0章 — 判断哲学（最優先・全判断の土台）

この章は、Aevum におけるすべての技術判断の「軸」である。
個別ルールより上位にあり、ルールが想定していない状況での判断はこの軸で行う。

### 0.1 評価軸 — 「正答率」で測らない ★最重要の誤解防止★

**Aevum の成否は accuracy（正答率）では測らない。** これは飾りではなく、
誤った指標に最適化すると**システム全体が間違った方向に進む**ため、最上位に置く。

- ラベルは3クラス（+1 利確 / −1 損切り / 0 時間切れ）だが、**当たり外れの損益は非対称**。
  正答率が高くてもコスト後にマイナスになり得るし、正答率が低くてもプラスになり得る。
- 見るべき指標は次の3つ:
  1. **コスト後の期待値（EV）** — taker手数料・maker手数料・funding・スリッページを引いた後、
     1トレードあたりの期待値がプラスか。**これが最終判定。**
  2. **Sharpe比** — リターン ÷ ばらつき。安定して稼げるか。1.0で及第、2.0で優秀。
  3. **最大ドローダウン** — 最悪局面の下落幅が許容内か。
- accuracy を表示・参照すること自体は可。ただし**目的関数・成功基準にしない**。
  正答率60%超はむしろ過学習を疑うシグナルとして扱う。
- **失敗モードの自覚**: 最初のモデルはコスト後マイナスかトントンが普通。それが現実であり、
  「動いた=使える」ではない。**「shadow modeでコスト後EVがプラス」が確認できて初めて
  "使えるかもしれない"** と言える。

### 0.2 二大設計軸 — すべての判断はこの2つに帰着する

個別の設計判断（モデル規模・正規化・格納形式…）に迷ったら、
「この選択はどちらの軸を**守るか／脅かすか**」で評価する。

| 軸 | 意味 | 脅かすと起きること |
|---|---|---|
| **A. train/live 整合性** | 学習時と本番で計算がビット単位までズレない | バックテストが本番で再現せず、静かに資金を溶かす |
| **B. 過学習回避** | 少ないデータでモデルが暗記に走らない | 「エッジがある」と誤認し、out-of-sample で崩れる |

この2軸が、後述の個別ルール（σ統一・正規化リーク防止・small モデル等）すべての**理由**である。
ルールを丸暗記するのではなく、「なぜそのルールがあるか」をこの2軸で理解すること。

### 0.3 「推奨」を鵜呑みにしない — プロジェクト固有事情を優先

一般的な「推奨」「デフォルト」「ベストプラクティス」は出発点であって、結論ではない。
**Aevum 固有の事情（暗号資産のファットテール／データの少なさ／2マシン構成／コスト構造）に
照らして、推奨を覆すべき場合は覆す。** 過去に推奨を覆した実例（理由つき）:

- **bar_features 格納 → 名前付き58カラム**（推奨は FLOAT8 配列）。
  理由: 配列の位置インデックスは σ と同種のサイレントドリフト源。列圧縮とSQLデバッグ性を取る。
- **正規化 → z-score + ±5σ クリップ**（推奨は素の z-score）。
  理由: 暗号資産のファットテールで外れ値1個が学習を歪めるため、定義域に収める。
- **モデル規模 → small (d64/2層/4head)**（推奨は medium d128）。
  理由: データが少なくこれは baseline。過学習回避と切り分け可能性を優先。

これらは「好み」ではなく、**0.1/0.2 の軸からの能動的判断**。
新しい判断でも同じ作法: 推奨を確認 → 2軸に照らす → 固有事情があれば覆し、理由を記録。

### 0.4 「同一コードで構造的に一致」— テスト担保より構造担保

train/live 整合性は「テストで一致を確認する」より「**そもそも同じコードを使うから一致する**」
方を常に優先する。書かないコードはバグらない。

- σ → `shared/volatility.py` 唯一定義
- テクニカル指標 → `shared/` 共通定義（Wilder 初期値まで統一）
- 特徴量計算 → Pi 側も `features.py`(Polars) を再利用（numpy 別実装しない）
- 正規化 → ONNX グラフに同梱（Pi 側に正規化コードを置かない）

「一致するようテストする」設計を見たら、「一致するしかない（同一コード）」設計に寄せられないか問う。

### 0.5 一度に一つの未知 — 切り分け可能性を常に保つ

新しいものを導入するときは、**一度に一つだけ**。複数を同時に本番データへ晒さない。
詰まったとき「どこが原因か」を即座に切り分けられる状態を維持する。

- 例: ingestion → features → labels → dataset は1つずつ実データに通す。
- 合成データや縮小特徴での「空通し」は、本番構成と別物になり**誤った安心**を生むため避ける。
- 検証は「静的データで配線確認 → ライブデータ」の順に段階を割る。

### 0.6 基礎を固めてから上に積む / fail-closed

- 前段（土台）が単体で検証できてから次へ進む。土台に既知の欠陥を抱えたまま上を積まない。
- 安全方向に倒す（fail-closed）。サイズ超過は黙って調整せず**拒否**。kill-switch時も**決済は通す**。
- リスクゲート(`risk.py`)は「通すか止めるか」の番人。注文内容を黙って書き換えない。

### 0.7 仕様欠落の充足 vs 機能追加 — 区別する

「仕様にない機能を勝手に追加しない」は厳守する。ただし、
**要件に内在していたのに欠けていた責務を補う**のは「機能追加」ではなく「欠落の充足」であり、正当。

- 例: ingestion の自動再接続。「常時稼働」要件に内在する責務。SDKが再接続しないと判明 → 補った。
- 判断に迷えば Stop and Ask（後述）。充足なら理由を記録して進め、新機能なら止めて相談。

### 0.8 時間軸の現実 — データ蓄積は数日〜数ヶ月

板/funding は**過去取得できない**（取引所が提供しない）。リアルタイム蓄積が唯一の手段。

| 目的 | 必要な連続データ | 分かること |
|---|---|---|
| パイプライン確認 | 〜十数時間 | コードが端まで繋がるか |
| モデルが学習できるか | 2〜3週間 | （過学習の度合いはまだ不明） |
| **エッジがあるか判断** | **2〜3ヶ月** | walk-forward で初めて見える |
| 本番に出せる | 半年〜1年 | 複数レジームで検証 |

「dataset が通った」「train が走った」を「使えるAIができた」と誤認しない。
**連続が必要なのは1サンプル分（SEQ_LEN=128 ≈ 10.7時間）だけ**で、データ全体の連続は不要
（穴をまたぐ窓はスキップされ、連続区間からサンプルが作られる）。

---

## プロジェクト概要

**Aevum** — Hyperliquid の無期限先物（BTC・ETH）を対象に、
Transformer モデルで売買シグナルを生成し自動発注するトレーディングシステム。

方法論の土台: López de Prado『Advances in Financial Machine Learning』
ラベル定義: Triple-Barrier Method
**成功の定義: 正答率ではなく、shadow mode でコスト後 EV がプラス（→ 0.1）。**

---

## ハードウェア構成

| マシン | 役割 |
|---|---|
| PC（RTX 2070 Super） | 特徴量計算・学習（重い計算）。学習時は Pi の DB を LAN 経由で読む |
| Raspberry Pi 5（NVMe） | **ingestion + TimescaleDB + 推論・発注・API・UI（常時稼働）** |

学習済みモデルは **ONNX ファイル**として PC → Pi へ渡す（環境非依存フォーマット）。
PC↔Pi 通信: 同一Wi-Fi `192.168.0.7`（scp / DB接続）→ 必要になれば Tailscale。
**Pi が単独で板/funding を収集・保存。PC の常時稼働は不要になった（2026-07-01 移管完了）。**

---

## 技術スタック

| 層 | 技術 |
|---|---|
| 言語 | Python 3.10 固定（hyperliquid-python-sdk の制約） |
| DB | TimescaleDB + asyncpg |
| 特徴量計算（PC/Pi 共通） | Polars（同一コード再利用 → 0.4） |
| 学習 | PyTorch |
| 推論 | ONNX Runtime（ARM最適化） |
| API | FastAPI + WebSocket |
| UI | React + TradingView Lightweight Charts（terminal/editorial の2テーマ切替） |
| 外部キュー | Redis または内部キュー |

---

## ディレクトリ構成

```
aevum/
├── CLAUDE.md               ← このファイル
├── data/
│   ├── ingestion.py        PC側: WebSocket受信 → TimescaleDB書き込み（死活監視+自動再接続）
│   ├── features.py         PC/Pi共通: Polars特徴量計算（バルク＝逐次で同一コード）
│   └── labels.py           PC側: Triple-Barrierラベリング
├── model/
│   ├── dataset.py          PyTorch Dataset（正規化・walk-forward分割）
│   ├── transformer.py      Transformerモデル定義（small: d64/2層/4head）
│   └── train.py            学習スクリプト + ONNXエクスポート（正規化同梱）
├── live/
│   ├── inference.py        Pi側: ONNX Runtime推論
│   ├── execution.py        Pi側: Hyperliquid発注
│   └── risk.py             リスク管理ゲート（全発注が通る・番人）
├── api/
│   └── server.py           FastAPI + WebSocket配信（read-only）
├── ui/                     監視ダッシュボード（React・2テーマ）
├── shared/
│   ├── volatility.py       ★ σ計算式の唯一の定義（全モジュールがここを呼ぶ）
│   └── technical.py        テクニカル指標の共通定義（Wilder初期値まで統一）
├── schema/
│   └── schema_v1.sql       TimescaleDB DDL
└── tests/
```

---

## データの流れ

```
Hyperliquid WebSocket（candle / l2Book / activeAssetCtx）
        ↓
   ingestion.py  →  TimescaleDB（生データ3テーブル）
        ↓
   features.py   →  bar_features（正規化なし・生で保存）
        ↓
   labels.py     →  labels（σも一緒に保存）
        ↓
   dataset.py    →  PyTorch テンソル（ここで正規化）
        ↓
   transformer.py → 学習済みモデル → ONNX export（正規化同梱）
        ↓
   inference.py  →  model_predictions（Pi）
        ↓
   execution.py  →  Hyperliquid 発注（risk.py を通す）
        ↓
   server.py     →  監視UI へ配信（ロジックなし）
```

---

## DBスキーマ（TimescaleDB）

### 生データ（WebSocket購読に対応）
- `ohlcv_bars` ← candle（過去取得可・backfill可）
- `orderbook_snapshots` ← l2Book（板はFLOAT8配列。**過去取得不可**）
- `funding_oi` ← activeAssetCtx（funding rate + OI。**過去取得不可**）

### 計算・学習用
- `bar_features` — 特徴量（**名前付き58カラム・正規化なし** → 0.3）
- `labels` — Triple-Barrierラベル（使用したσも保存）

### 本番・執行用
- `model_predictions`
- `orders` / `positions`（SL水準は orders.price に記録）

---

## 特徴量（58個・10カテゴリ）

| カテゴリ | 数 | 過去取得 |
|---|---|---|
| Price / Return | 11 | 可（candle由来） |
| Volatility | 6 | 可 |
| Volume | 4 | 可 |
| Technical Indicators | 8 | 可 |
| Temporal | 4 | 可 |
| Order Book Imbalance (OBI) | 10 | **不可（板由来）** |
| Spread | 3 | **不可** |
| Microstructure | 5 | **不可** |
| Funding / OI | 4 | **不可** |
| Cross-Asset | 3 | 2パス計算 |

**Cross-Assetは2パス計算**: 両銘柄の計算完了後に `cross_*` を埋める。
**特徴量名の正準リスト（順序付き）を1箇所で定義**し、DDL・features.py・モデル入力が単一参照（→ 0.4）。

---

## 不変条件（絶対に守ること）

> これらはすべて第0章の軸（特に A: train/live整合性 / B: 過学習回避）の具体化である。
> ルールの背後の「なぜ」を見失ったら第0章へ。

### ★ 最重要: σ計算式の統一（軸A・0.4）

**`shared/volatility.py` に唯一の定義を置き、全モジュールから呼ぶ。**
per-bar σ を返し、√horizon スケールは呼び出し側が `scale_to_horizon` で行う（自前計算禁止）。
以下でズレると致命的な失敗モード:

1. `labels.py` — バリアサイズ決定（CUSUM閾値・バリア幅とも `scale_to_horizon` 経由）
2. `execution.py` — ライブのバリア幅
3. 監視UI — σ表示

### train/live 整合性（軸A・0.4）

- 正規化は `dataset.py`（モデル入力時）のみ。DBには生で保存（リーク防止）。
  正規化統計量は **train fold のみ**から算出し、保存して本番で再利用。
- Pi側特徴量計算は `features.py` の Polars ロジックを再利用（numpy別実装しない）。
- ライブはローリング窓で計算。末尾をバルクと一致させるため最低 `WARMUP_BARS + seq_len` バー保持。
- 正規化（z-score + ±5σ クリップ）は ONNX グラフに同梱（`NormalizedModel`）。Pi は生特徴を渡すだけ。
- テクニカル指標は `shared/technical.py` で手実装（Wilder初期値まで統一。ライブラリ依存禁止）。
- 検証は Walk-forward のみ（**k-fold 禁止**）。Purge長 = 最大ラベル保有期間（48バー）。

### 過学習回避（軸B・0.3）

- モデルは small（d64/2層/4head/FFN256）から。underfit の兆候が出たら段階的に拡大。
  `d_model`/`layers`/`heads`/`FFN` は設定で可変に。
- CUSUM フィルタで起点を間引き、相関サンプルを削減。
- 正規化は z-score + ±5σ クリップ（外れ値を定義域に収める）。
- accuracy を目的関数にしない（→ 0.1）。

### アーキテクチャ分離 / リスクゲート（0.6）

- トレーディングループ（`execution.py`）はGUIから独立。
- すべての注文は `risk.py`（番人）を通す。UI由来も迂回不可。
- `risk.py` は通すか止めるかのみ。**注文内容を黙って書き換えない**（サイズ超過=ハードリジェクト）。
- kill-switch 発動時も **take_profit/stop_loss（決済系）は常に許可**（建玉に閉じ込めない）。
- `server.py` は状態配信のみ（read-only）。ロジックを持たない。

### 発注の非対称設計

- エントリー・利確: 指値（maker）。損切り: 成行（taker）で約定を保証。
- サイジングはボラティリティターゲット（固定リスク%）。SL距離は labels と同一の `scale_to_horizon`。
- エグジットは TP指値 + SL成行 + 時間決済（horizon=48）。**labels と同式**（時間決済を外さない）。

### ingestion の接続維持（0.7 の実例: 仕様欠落の充足）

- WS の死活監視 + データ鮮度監視を持ち、異常時に Info を作り直し3チャンネルを自動再購読。
  再接続はログに残す。リトライ上限・バックオフは設定可能。
- 理由: hyperliquid SDK は `Expired` で無言終了し再接続しない。常時稼働要件に内在する責務。
- スコープは**接続維持のみ**。データ補完/バックフィルや書き込みロジックは持ち込まない。
- 実地で約3時間周期の `Expired` を10秒台で自動回復することを確認済み。

---

## Triple-Barrier / 学習の確定パラメータ

| 項目 | 値 | 軸 |
|---|---|---|
| バー間隔 | 5分足 | — |
| horizon（縦バリア） | 48バー（4時間）。purge と連動 | A |
| σ | EWMA・対数リターン・per-bar（√horizonは呼出側） | A |
| バリア倍率 | 対称 [1.0, 1.0]（baseline。pt/sl_mult可変） | B |
| イベント抽出 | CUSUM（閾値hはσ参照） | B |
| 系列長 | 128バー（≈10.7h） | — |
| 正規化 | z-score + ±5σクリップ（train foldのみでfit） | A,B |
| WF窓 | 拡張窓 + Purge48 + Embargo | A |
| 位置符号化 | 学習可能（固定長128） | — |
| 系列集約 | 最終トークン（ラベルは系列末に付く） | A |
| 損失 | sample_weight（一意性）× クラス重み | B |
| 本番モデル | WF評価後に全データ再学習 | — |

---

## 検証ルール

- **k-fold 禁止**。Walk-forward + Purging + Embargo のみ。Purge長 = 最大ラベル保有期間（48）。
- 開発順序: baseline → Transformer → **経済評価（コスト後EV・Sharpe → 0.1）** → Shadow-mode → live。
- Shadow-mode（実資金なし・長時間連続稼働）でコスト後EVプラスを確認してから live。

---

## 実装順序

依存関係の順に進める。前のモジュールが単体テストできてから次へ（→ 0.6）。

```
1. shared/volatility.py     ← σ計算式を最初に固定
2. shared/technical.py      ← テクニカル指標の共通定義
3. schema/schema_v1.sql     ← DB定義
4. data/ingestion.py        ← WebSocket → DB（死活監視+自動再接続）
5. data/labels.py           ← Triple-Barrier
6. data/features.py         ← Polars特徴量計算
7. model/dataset.py         ← PyTorch Dataset
8. model/transformer.py     ← モデル定義
9. model/train.py           ← 学習 + ONNXエクスポート（正規化同梱）
10. live/inference.py       ← ONNX Runtime（Pi）
11. live/risk.py            ← リスク管理ゲート
12. live/execution.py       ← 発注
13. api/server.py           ← FastAPI（read-only）
14. ui/                     ← 監視ダッシュボード（2テーマ）
```

---

## ローカル開発スタック（dev runbook）

開発機（Windows / .venv）想定。Pi 本番とは別。

### PC 側（学習・開発）

- **DB（学習時）**: Pi DB を LAN 経由で参照。`AEVUM_DB_DSN=postgresql://postgres:aevum@192.168.0.7:5432/aevum`
  PC ローカルの TimescaleDB コンテナは開発・テスト専用。`docker compose up -d` で起動し、
  `AEVUM_DB_DSN=postgresql://postgres:aevum@localhost:5432/aevum` で切り替えて使う。
- **Python**: `./.venv/Scripts/python.exe`（**3.10 固定**。既定の 3.12 を使わない）。
- **Windows コンソール**: cp932。`PYTHONIOENCODING=utf-8`、print は ASCII 推奨（σ/µ/→ で落ちる）。
  **生成ファイルに非ASCIIを混ぜない**（PowerShell が cp932 で読んで構文崩壊する）。
- スキーマ適用: `python scripts/apply_schema.py` ／ 静的 seed: `python scripts/seed_dev.py`
- candle 履歴 backfill（冪等・REST candleSnapshot。穴埋めはこれ）: `python scripts/backfill_candles.py --days N`
- 特徴量生成（features.py 無改修ランナー）: `python scripts/run_features.py`（`--no-store` で計算のみ）
- DB 確認: `python check_db.py`（Pi DB を見る場合は上記 DSN に切り替えて実行）
- API: `uvicorn api.server:app`（read-only）／ UI: `cd ui && npm run dev`
- テスト: `./.venv/Scripts/python.exe -m pytest -q`

### Pi 側（常時稼働）

- **SSH**: `ssh raspi5`（`~/.ssh/config` の `raspi5` エイリアス → `192.168.0.7`・鍵認証）
- **ingestion**: systemd `aevum-ingestion.service` で自動常駐（Pi 再起動後も自動復帰）。
  手動操作: `sudo systemctl start/stop/status aevum-ingestion`
  ログ: `journalctl -u aevum-ingestion -f`
- **DB**: `~/aevum/` の `docker compose`（`restart: unless-stopped`）で自動起動。
  手動: `cd ~/aevum && sudo docker compose up -d / ps / logs db`
- **Python 3.10**: `~/.pyenv/versions/3.10.17/bin/python3.10`（pyenv でビルド済み）
  venv: `~/aevum-env/bin/python`
- **容量監視**: `ssh raspi5 df -h` で NVMe 残量確認（235GB 中 7% 使用・2026-07-01 時点）
- **注意**: セッションをまたぐと background プロセス（ingestion は systemd なので停止しない）。

---

## 現状チェックポイント（最終更新 2026-07-01）

- **E2E 段階3+**。`ingestion → ohlcv/book/funding` と `features → bar_features → labels → dataset(配線)` を実データで検証済み。
- features 核心検証は全 PASS: σ/TA の **train(bulk)↔live(rolling) 一致**、外部参照一致、Cross 2パス。
- labels: σスケール二重実装を解消（`scale_to_horizon` に一本化）、一意性weight を `min(w,1.0)` クランプ。実データ投入済み。
- dataset: purge=48 ✓、正規化リーク防止の回帰テスト追加済み。配線健全。
- inference: book バッファを funding と同じフルWARMUP に統一（軸A）。回帰テスト追加済み（`59c78ce`）。
- ingestion: 接続維持スーパーバイザが**実地で約3時間周期の Expired を10秒台で自動回復**することを複数回確認。
- **ingestion + DB を Pi5 に完全移管済み（2026-07-01）**。PC の常時稼働は不要。
  Pi: systemd `aevum-ingestion.service` + Docker `restart: unless-stopped` で二重の自動復帰。
  PC からの学習時 DSN: `postgresql://postgres:aevum@192.168.0.7:5432/aevum`
- **律速はデータ量**。板/funding は warmup(60)・SEQ_LEN(128)窓をクリアする蓄積待ち（Pi で継続収集中）。
  ※ bar_features は features runner を走らせた時点までしか埋まらない。蓄積後に `run_features.py` 再実行が必要。
- equity / winRate は当面 null 表示（永続化テーブル無し・winRate 定義未確定 → 別途 0.1 に沿って設計）。
- **次の本流**: データ蓄積（あと2〜3週間） → `run_features.py` 再実行 → `labels`/`dataset` 再実行 → `train`(ONNX) → `inference`(shadow) → `execution`。
- 残課題: 板/funding 特徴の rolling↔bulk 一致（蓄積後に検証）／ 板/funding 欠損期間の学習からの扱い。

---

## Stop and Ask（実装を止めて質問する条件）

以下に該当したら実装せず確認を求める。**第0章と矛盾しそうなときも止める。**

- σ計算式の定義が曖昧なとき
- 発注・執行ロジックの仕様が不明なとき
- DBスキーマへの破壊的変更が必要なとき
- `risk.py` を迂回する設計になりそうなとき
- train/live で異なる計算が必要に見えるとき
- 「推奨」を覆すか／覆さないかの判断が固有事情に依存するとき（→ 0.3）
- 「機能追加」か「仕様欠落の充足」か判断がつかないとき（→ 0.7）
- accuracy を目的関数にする実装・評価になりそうなとき（→ 0.1）

---

## 禁止事項

- 仕様にない機能を勝手に追加しない（ただし「欠落の充足」は 0.7 で区別）
- 無関係な整形やリファクタリングをしない
- k-fold を使わない
- σを複数箇所で別々に計算しない（`scale_to_horizon` を必ず経由）
- `risk.py` を通さない発注経路を作らない
- Python 3.10 以外を使わない
- **accuracy を成功基準・目的関数にしない（→ 0.1）**
- **合成/縮小データの「空通し」で本番構成を検証したと見なさない（→ 0.5）**
- **生成ファイルに非ASCII文字を混ぜない（cp932 で構文崩壊）**

---

## セッション開始時のチェックリスト

1. `git status` で未コミット変更を確認
2. **第0章（判断哲学）を読み直す** — 今回の作業がどの軸に関わるか意識する
3. 今回実装するモジュールとその「完了定義」を確認
4. 依存するモジュールが完成しているか確認
5. baseline の検証結果（lint / typecheck / test）を記録
