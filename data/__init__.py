"""data — PC側のデータ取り込み・特徴量・ラベリング。

ingestion.py : Hyperliquid WebSocket → TimescaleDB（生データ3テーブル）
features.py   : Polars で特徴量バルク計算
labels.py     : Triple-Barrier ラベリング
"""
