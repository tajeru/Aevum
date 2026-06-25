-- =====================================================================
-- schema_v1.sql — Aevum TimescaleDB DDL (v1)
-- =====================================================================
-- 前提:
--   * TimescaleDB + asyncpg / Python 3.10
--   * 銘柄は BTC・ETH の無期限先物（perp）
--   * 時刻は TIMESTAMPTZ、各時系列テーブルは hypertable 化
--   * 単一バー間隔を仮定（interval 列は持たない。複数間隔が必要なら追加）
--
-- 不変条件（CLAUDE.md）:
--   * σ は shared/volatility.py の唯一定義。labels.sigma / model_predictions.sigma /
--     bar_features.sigma_ewma はすべて同じ σ を指す。
--   * bar_features の特徴量列順・名称は shared/feature_names.py の FEATURE_NAMES と一致
--     （tests/test_feature_names.py が DDL とリストの一致を強制）。
--   * 発注の非対称設計: entry / take_profit = 指値(limit/maker)、stop_loss = 成行(market/taker)
--     を orders の CHECK 制約で DB レベルでも保証。
--   * 全注文は risk.py を通る。orders.risk_passed にゲート通過を必ず記録。
--
-- 適用順:
--   psql -f schema/schema_v1.sql   （CREATE EXTENSION から実行）
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS timescaledb;


-- =====================================================================
-- 生データ（WebSocket 購読に対応）
-- =====================================================================

-- candle チャンネル
CREATE TABLE IF NOT EXISTS ohlcv_bars (
    symbol  TEXT        NOT NULL CHECK (symbol IN ('BTC', 'ETH')),
    time    TIMESTAMPTZ NOT NULL,
    open    FLOAT8      NOT NULL,
    high    FLOAT8      NOT NULL,
    low     FLOAT8      NOT NULL,
    close   FLOAT8      NOT NULL,
    volume  FLOAT8      NOT NULL,
    trades  INTEGER,                 -- candle 'n'（約定数）
    vwap    FLOAT8,                  -- 提供されれば
    PRIMARY KEY (symbol, time)
);
SELECT create_hypertable('ohlcv_bars', 'time',
    if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');

-- l2Book チャンネル（板スナップショット）。板は FLOAT8 配列で保存（JSON より高速）。
-- 配列は best→deep の順（index 0 が最良気配）。買い/売りで長さは可変。
CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    symbol  TEXT        NOT NULL CHECK (symbol IN ('BTC', 'ETH')),
    time    TIMESTAMPTZ NOT NULL,
    bid_px  FLOAT8[]    NOT NULL,
    bid_sz  FLOAT8[]    NOT NULL,
    ask_px  FLOAT8[]    NOT NULL,
    ask_sz  FLOAT8[]    NOT NULL,
    PRIMARY KEY (symbol, time)
);
-- 板は高頻度。chunk を小さめにして圧縮・クエリ効率を確保。
SELECT create_hypertable('orderbook_snapshots', 'time',
    if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');

-- activeAssetCtx チャンネル（funding / open interest など）
CREATE TABLE IF NOT EXISTS funding_oi (
    symbol        TEXT        NOT NULL CHECK (symbol IN ('BTC', 'ETH')),
    time          TIMESTAMPTZ NOT NULL,
    funding_rate  FLOAT8,
    open_interest FLOAT8,
    mark_price    FLOAT8,
    oracle_price  FLOAT8,
    premium       FLOAT8,
    PRIMARY KEY (symbol, time)
);
SELECT create_hypertable('funding_oi', 'time',
    if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');


-- =====================================================================
-- 計算・学習用
-- =====================================================================

-- 特徴量（正規化なし・生で保存）。
-- ★ 列順・名称は shared/feature_names.py の FEATURE_NAMES と一致させること。
--   特徴量はウォームアップ/2パス中に NULL になり得るため NOT NULL は付けない。
CREATE TABLE IF NOT EXISTS bar_features (
    symbol  TEXT        NOT NULL CHECK (symbol IN ('BTC', 'ETH')),
    time    TIMESTAMPTZ NOT NULL,
    -- Price / Return (11)
    ret_1 FLOAT8, ret_5 FLOAT8, ret_15 FLOAT8, ret_30 FLOAT8, ret_60 FLOAT8, ret_240 FLOAT8,
    ret_co FLOAT8, wick_up FLOAT8, wick_dn FLOAT8, price_pos_60 FLOAT8, ret_accel FLOAT8,
    -- Volatility (6)  ← sigma_ewma は shared/volatility.py の per-bar σ
    sigma_ewma FLOAT8, realized_vol_30 FLOAT8, parkinson_30 FLOAT8, garman_klass_30 FLOAT8,
    vol_of_vol_60 FLOAT8, downside_vol_30 FLOAT8,
    -- Volume (4)
    volume_log FLOAT8, volume_z_60 FLOAT8, volume_ratio_5_30 FLOAT8, trade_count_log FLOAT8,
    -- Order Book Imbalance (10)
    obi_l1 FLOAT8, obi_l5 FLOAT8, obi_l10 FLOAT8, bid_depth_5_log FLOAT8, ask_depth_5_log FLOAT8,
    depth_ratio_5 FLOAT8, obi_weighted FLOAT8, microprice_dev FLOAT8, bid_slope FLOAT8, ask_slope FLOAT8,
    -- Spread (3)
    spread_bps FLOAT8, spread_z_60 FLOAT8, spread_vol_30 FLOAT8,
    -- Technical Indicators (8)
    rsi_14 FLOAT8, macd_line FLOAT8, macd_signal FLOAT8, macd_hist FLOAT8, bb_pctb_20 FLOAT8,
    atr_14 FLOAT8, stoch_k_14 FLOAT8, adx_14 FLOAT8,
    -- Microstructure (5)
    kyle_lambda FLOAT8, amihud_illiq FLOAT8, roll_spread FLOAT8, ofi FLOAT8, vpin_50 FLOAT8,
    -- Cross-Asset (3)  ← 2パス: 両銘柄計算後に UPDATE で埋める
    cross_corr_60 FLOAT8, cross_beta_60 FLOAT8, cross_ret_spread FLOAT8,
    -- Temporal (4)
    hour_sin FLOAT8, hour_cos FLOAT8, dow_sin FLOAT8, dow_cos FLOAT8,
    -- Funding / OI (4)
    funding_rate FLOAT8, funding_z_60 FLOAT8, oi_log FLOAT8, oi_change FLOAT8,
    PRIMARY KEY (symbol, time)
);
SELECT create_hypertable('bar_features', 'time',
    if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');

-- Triple-Barrier ラベル。使用した σ も保存（shared/volatility.py 由来）。
CREATE TABLE IF NOT EXISTS labels (
    symbol        TEXT        NOT NULL CHECK (symbol IN ('BTC', 'ETH')),
    time          TIMESTAMPTZ NOT NULL,        -- t0: イベント発生バー
    label         SMALLINT    NOT NULL CHECK (label IN (-1, 0, 1)),
    ret           FLOAT8      NOT NULL,         -- バリア到達までの実現リターン(log)
    sigma         FLOAT8      NOT NULL,         -- 使用した per-bar σ（volatility.py）
    pt_level      FLOAT8,                       -- 上側バリア価格（利確）
    sl_level      FLOAT8,                       -- 下側バリア価格（損切）
    pt_mult       FLOAT8,                       -- 上側バリア倍率（× σ）
    sl_mult       FLOAT8,                       -- 下側バリア倍率（× σ）
    horizon_bars  INTEGER,                      -- 縦バリア（最大保有バー数）
    t1            TIMESTAMPTZ,                  -- 縦バリア時刻（t0 + horizon）
    touch_time    TIMESTAMPTZ,                  -- 実際にバリア到達した時刻
    touch_barrier TEXT CHECK (touch_barrier IN ('pt', 'sl', 'vertical')),
    sample_weight FLOAT8,                       -- AFML: 一意性/時間減衰重み
    PRIMARY KEY (symbol, time)
);
SELECT create_hypertable('labels', 'time',
    if_not_exists => TRUE, chunk_time_interval => INTERVAL '7 days');


-- =====================================================================
-- 本番・執行用
-- =====================================================================

CREATE TABLE IF NOT EXISTS model_predictions (
    symbol        TEXT        NOT NULL CHECK (symbol IN ('BTC', 'ETH')),
    time          TIMESTAMPTZ NOT NULL,
    model_version TEXT        NOT NULL,
    prob_down     FLOAT8,                       -- P(label = -1)
    prob_flat     FLOAT8,                       -- P(label =  0)
    prob_up       FLOAT8,                       -- P(label = +1)
    signal        SMALLINT CHECK (signal IN (-1, 0, 1)),  -- 最終シグナル
    sigma         FLOAT8,                       -- 推論時 σ（バリア幅算出用, volatility.py）
    PRIMARY KEY (symbol, time, model_version)
);
SELECT create_hypertable('model_predictions', 'time',
    if_not_exists => TRUE, chunk_time_interval => INTERVAL '7 days');

-- 注文。非対称設計を CHECK で保証: entry/take_profit=limit、stop_loss=market。
-- 全注文は risk.py を通る。risk_passed にゲート通過可否を必ず記録する。
CREATE TABLE IF NOT EXISTS orders (
    order_id       BIGINT      GENERATED ALWAYS AS IDENTITY,
    symbol         TEXT        NOT NULL CHECK (symbol IN ('BTC', 'ETH')),
    time           TIMESTAMPTZ NOT NULL,        -- 発注時刻
    side           TEXT        NOT NULL CHECK (side IN ('buy', 'sell')),
    intent         TEXT        NOT NULL CHECK (intent IN ('entry', 'take_profit', 'stop_loss')),
    order_type     TEXT        NOT NULL CHECK (order_type IN ('limit', 'market')),
    price          NUMERIC,                     -- 指値価格（成行は NULL）
    size           NUMERIC     NOT NULL,
    status         TEXT        NOT NULL
                   CHECK (status IN ('pending', 'open', 'filled', 'partial', 'cancelled', 'rejected')),
    filled_size    NUMERIC     DEFAULT 0,
    avg_fill_price NUMERIC,
    exchange_oid   TEXT,                        -- Hyperliquid 側 order id
    cloid          TEXT,                        -- client order id
    risk_passed    BOOLEAN     NOT NULL,        -- risk.py ゲート通過記録（必須）
    reason         TEXT,                        -- リジェクト理由 / メモ
    PRIMARY KEY (symbol, time, order_id),
    -- 発注の非対称設計（CLAUDE.md）を DB レベルでも強制
    CONSTRAINT orders_asymmetric_design CHECK (
        (intent IN ('entry', 'take_profit') AND order_type = 'limit')
        OR (intent = 'stop_loss' AND order_type = 'market')
    )
);
SELECT create_hypertable('orders', 'time',
    if_not_exists => TRUE, chunk_time_interval => INTERVAL '30 days');

-- 建玉スナップショット。金額/数量は NUMERIC（誤差回避）。
CREATE TABLE IF NOT EXISTS positions (
    symbol         TEXT        NOT NULL CHECK (symbol IN ('BTC', 'ETH')),
    time           TIMESTAMPTZ NOT NULL,
    size           NUMERIC     NOT NULL,        -- 符号付き（+ロング / -ショート）
    entry_price    NUMERIC,
    mark_price     NUMERIC,
    unrealized_pnl NUMERIC,
    realized_pnl   NUMERIC,
    leverage       FLOAT8,
    liquidation_px NUMERIC,
    margin_used    NUMERIC,
    PRIMARY KEY (symbol, time)
);
SELECT create_hypertable('positions', 'time',
    if_not_exists => TRUE, chunk_time_interval => INTERVAL '7 days');


-- =====================================================================
-- 圧縮ポリシー（HDD 保存のため早期有効化）
-- =====================================================================
-- 高頻度テーブルはストレージ削減のため圧縮（TimescaleDB Community）。
-- segmentby=symbol で銘柄ごとに、orderby=time DESC で時系列順に圧縮。
-- 圧縮対象は「閾値より古い」チャンクのみ。bar_features の cross_* 2パス UPDATE は
-- 投入直後（30日より新しい）に行うため、圧縮済みチャンクへの UPDATE は発生しない。

-- 板（最大容量）: 7日より古いチャンクを圧縮
ALTER TABLE orderbook_snapshots SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby   = 'time DESC');
SELECT add_compression_policy('orderbook_snapshots', INTERVAL '7 days');

-- 特徴量: 30日より古いチャンクを圧縮（2パス UPDATE 完了後）
ALTER TABLE bar_features SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby   = 'time DESC');
SELECT add_compression_policy('bar_features', INTERVAL '30 days');
