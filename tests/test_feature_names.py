"""shared/feature_names.py の単体テスト。

特徴量名の正準リスト（単一の真実）が
  1. CLAUDE.md のカテゴリ×個数に一致
  2. 一意・整合（index / cross / σ）
  3. schema_v1.sql の bar_features 列と完全一致（順序含む）
であることを強制する。3 が崩れると train/live で列順がズレる致命的な失敗モード。
"""
from __future__ import annotations

import re
from pathlib import Path

from shared.feature_names import (
    CROSS_FEATURES,
    EXPECTED_CATEGORY_COUNTS,
    FEATURE_CATEGORIES,
    FEATURE_INDEX,
    FEATURE_NAMES,
    N_FEATURES,
    SIGMA_FEATURE,
)

SCHEMA_SQL = Path(__file__).resolve().parents[1] / "schema" / "schema_v1.sql"


# --------------------------------------------------------------------------- #
# 個数・一意性・整合
# --------------------------------------------------------------------------- #
def test_total_is_58():
    assert N_FEATURES == 58
    assert len(FEATURE_NAMES) == 58


def test_category_counts_match_claude_md():
    assert set(FEATURE_CATEGORIES) == set(EXPECTED_CATEGORY_COUNTS)
    for cat, expected in EXPECTED_CATEGORY_COUNTS.items():
        assert len(FEATURE_CATEGORIES[cat]) == expected, cat
    assert sum(EXPECTED_CATEGORY_COUNTS.values()) == 58


def test_names_unique():
    assert len(set(FEATURE_NAMES)) == N_FEATURES


def test_flat_list_is_category_concatenation():
    flat = tuple(n for names in FEATURE_CATEGORIES.values() for n in names)
    assert FEATURE_NAMES == flat


def test_feature_index_consistent():
    assert len(FEATURE_INDEX) == N_FEATURES
    for i, name in enumerate(FEATURE_NAMES):
        assert FEATURE_INDEX[name] == i


def test_cross_features():
    assert len(CROSS_FEATURES) == 3
    assert all(n.startswith("cross_") for n in CROSS_FEATURES)
    assert all(n in FEATURE_INDEX for n in CROSS_FEATURES)


def test_sigma_feature_present():
    assert SIGMA_FEATURE in FEATURE_INDEX


def test_names_are_valid_sql_identifiers():
    # 小文字英数字とアンダースコアのみ・英字始まり（DDL 列名として安全）
    pat = re.compile(r"^[a-z][a-z0-9_]*$")
    for name in FEATURE_NAMES:
        assert pat.match(name), name


# --------------------------------------------------------------------------- #
# DDL ↔ FEATURE_NAMES の一致（単一の真実の強制）
# --------------------------------------------------------------------------- #
def _bar_features_columns_from_sql() -> list[str]:
    sql = SCHEMA_SQL.read_text(encoding="utf-8")
    marker = "CREATE TABLE IF NOT EXISTS bar_features"
    assert marker in sql, "bar_features の CREATE TABLE が見つからない"
    block = sql.split(marker, 1)[1].split(");", 1)[0]
    # bar_features で FLOAT8 型の列はすべて特徴量（symbol=TEXT, time=TIMESTAMPTZ）。
    # 配列(FLOAT8[])は bar_features に存在しないため \b で通常の FLOAT8 のみ拾う。
    return re.findall(r"(\w+)\s+FLOAT8\b", block)


def test_schema_file_exists():
    assert SCHEMA_SQL.is_file(), SCHEMA_SQL


def test_ddl_columns_match_feature_names_exactly():
    cols = _bar_features_columns_from_sql()
    assert cols == list(FEATURE_NAMES), (
        "schema_v1.sql の bar_features 列が FEATURE_NAMES と不一致（順序含む）。\n"
        f"DDL  : {cols}\n"
        f"NAMES: {list(FEATURE_NAMES)}"
    )
