"""shared/feature_names.py — 特徴量名の【唯一の正準定義】(ordered).

CLAUDE.md の不変条件「train/live 整合性」を支える単一の真実。次の3つは必ず
このモジュールの ``FEATURE_NAMES`` を参照し、別々に列順を定義してはならない:

    1. schema/schema_v1.sql の bar_features 列（順序・名称が一致）
    2. data/features.py の出力列
    3. model/dataset.py → モデル入力テンソルの列順

整合は tests/test_feature_names.py が機械的に強制する（DDL とこのリストの一致を含む）。

σ について
---------
Volatility カテゴリの ``sigma_ewma`` が shared/volatility.py の per-bar σ
(``volatility.volatility``) に対応する唯一の特徴量。labels.sigma /
model_predictions.sigma も同じ σ 定義を指す。σ を別式で再計算してはならない。

注意
----
* 58個・10カテゴリ。カテゴリ別個数は CLAUDE.md と一致（``EXPECTED_CATEGORY_COUNTS``）。
* ``cross_*`` は2パス計算（両銘柄の非 cross 特徴量を計算後に UPDATE で埋める）。
* これは v1。features.py 実装で名称が変わる場合はスキーマのマイグレーションを伴う。
"""
from __future__ import annotations

__all__ = [
    "FEATURE_CATEGORIES",
    "EXPECTED_CATEGORY_COUNTS",
    "FEATURE_NAMES",
    "N_FEATURES",
    "CROSS_FEATURES",
    "SIGMA_FEATURE",
    "FEATURE_INDEX",
]

# カテゴリ -> 順序付き特徴量名（dict は挿入順を保持: Python 3.7+）。
# この順序が bar_features の列順、features.py の出力順、モデル入力順の基準。
FEATURE_CATEGORIES: dict[str, tuple[str, ...]] = {
    # Price / Return (11)
    "price_return": (
        "ret_1", "ret_5", "ret_15", "ret_30", "ret_60", "ret_240",
        "ret_co",        # ログ(終値/始値) イントラバー方向
        "wick_up",       # ログ(高値/max(始値,終値)) 上ヒゲ
        "wick_dn",       # ログ(min(始値,終値)/安値) 下ヒゲ
        "price_pos_60",  # 直近60本の高安レンジ内での終値位置(0..1)
        "ret_accel",     # ret_1 の階差（リターン加速度）
    ),
    # Volatility (6)
    "volatility": (
        "sigma_ewma",       # ★ shared/volatility.py の per-bar σ（唯一の定義）
        "realized_vol_30",  # 実現ボラ(二乗和の平方根, 30本)
        "parkinson_30",     # Parkinson 推定(高安)
        "garman_klass_30",  # Garman-Klass 推定(OHLC)
        "vol_of_vol_60",    # ボラのボラ
        "downside_vol_30",  # 下方半偏差
    ),
    # Volume (4)
    "volume": (
        "volume_log",         # log1p(出来高)
        "volume_z_60",        # 出来高 zscore(60)
        "volume_ratio_5_30",  # 出来高 MA5/MA30
        "trade_count_log",    # log1p(約定数 n)
    ),
    # Order Book Imbalance (10)
    "obi": (
        "obi_l1",          # L1 板不均衡
        "obi_l5",          # 上位5段
        "obi_l10",         # 上位10段
        "bid_depth_5_log", # log(買い深さ上位5)
        "ask_depth_5_log", # log(売り深さ上位5)
        "depth_ratio_5",   # 買い/売り 深さ比
        "obi_weighted",    # 距離加重 不均衡
        "microprice_dev",  # (microprice-mid)/mid
        "bid_slope",       # 買い側 デプススロープ
        "ask_slope",       # 売り側 デプススロープ
    ),
    # Spread (3)
    "spread": (
        "spread_bps",     # (ask1-bid1)/mid * 1e4
        "spread_z_60",    # スプレッド zscore
        "spread_vol_30",  # スプレッドの標準偏差
    ),
    # Technical Indicators (8)
    "technical": (
        "rsi_14",
        "macd_line",
        "macd_signal",
        "macd_hist",
        "bb_pctb_20",  # Bollinger %B
        "atr_14",      # ATR/close 正規化
        "stoch_k_14",
        "adx_14",
    ),
    # Microstructure (5)
    "microstructure": (
        "kyle_lambda",   # Kyle のラムダ(価格インパクト)
        "amihud_illiq",  # Amihud 非流動性 |ret|/dollar_vol
        "roll_spread",   # Roll の暗黙スプレッド
        "ofi",           # オーダーフロー不均衡
        "vpin_50",       # VPIN(50バケット)
    ),
    # Cross-Asset (3)  ← 2パス: 両銘柄計算後に UPDATE で埋める
    "cross_asset": (
        "cross_corr_60",     # BTC↔ETH リターン相関(60)
        "cross_beta_60",     # 相手銘柄に対するベータ
        "cross_ret_spread",  # ret_self - ret_other（リードラグ）
    ),
    # Temporal (4)
    "temporal": (
        "hour_sin", "hour_cos",  # 時刻の周期エンコード
        "dow_sin", "dow_cos",    # 曜日の周期エンコード
    ),
    # Funding / OI (4)
    "funding_oi": (
        "funding_rate",  # 現在のファンディングレート
        "funding_z_60",  # ファンディング zscore
        "oi_log",        # log(建玉)
        "oi_change",     # Δlog(建玉)
    ),
}

# CLAUDE.md 準拠のカテゴリ別個数（不変条件チェック用）。
EXPECTED_CATEGORY_COUNTS: dict[str, int] = {
    "price_return": 11,
    "volatility": 6,
    "volume": 4,
    "obi": 10,
    "spread": 3,
    "technical": 8,
    "microstructure": 5,
    "cross_asset": 3,
    "temporal": 4,
    "funding_oi": 4,
}

# 順序付きフラットリスト（単一の真実）。
FEATURE_NAMES: tuple[str, ...] = tuple(
    name for names in FEATURE_CATEGORIES.values() for name in names
)

# 特徴量総数（== 58）。
N_FEATURES: int = len(FEATURE_NAMES)

# 2パスで埋める cross-asset 特徴量。
CROSS_FEATURES: tuple[str, ...] = FEATURE_CATEGORIES["cross_asset"]

# shared/volatility.py の σ に対応する特徴量名。
SIGMA_FEATURE: str = "sigma_ewma"

# 名前 -> インデックス（テンソル列位置の参照用）。
FEATURE_INDEX: dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _validate() -> None:
    """import 時に不変条件を自己検証する（-O でも効くよう assert は使わない）。"""
    # カテゴリ別個数が CLAUDE.md と一致
    if set(FEATURE_CATEGORIES) != set(EXPECTED_CATEGORY_COUNTS):
        raise ValueError("FEATURE_CATEGORIES と EXPECTED_CATEGORY_COUNTS のキー不一致")
    for cat, expected in EXPECTED_CATEGORY_COUNTS.items():
        actual = len(FEATURE_CATEGORIES[cat])
        if actual != expected:
            raise ValueError(f"カテゴリ '{cat}' の個数が {actual}（期待 {expected}）")
    # 総数 58
    if N_FEATURES != 58:
        raise ValueError(f"特徴量総数が {N_FEATURES}（期待 58）")
    # 名称の一意性
    if len(set(FEATURE_NAMES)) != N_FEATURES:
        raise ValueError("FEATURE_NAMES に重複がある")
    # cross_* 命名規約
    for name in CROSS_FEATURES:
        if not name.startswith("cross_"):
            raise ValueError(f"cross-asset 特徴量 '{name}' は 'cross_' 始まりであること")
    # σ 特徴量の存在
    if SIGMA_FEATURE not in FEATURE_INDEX:
        raise ValueError(f"SIGMA_FEATURE '{SIGMA_FEATURE}' が FEATURE_NAMES に無い")


_validate()
