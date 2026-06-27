"""data/ingestion.py — PC側: Hyperliquid WebSocket → TimescaleDB（生データ）.

3チャンネルを購読し、生データ3テーブルへ書き込む:
    candle         → ohlcv_bars            （5分足）
    l2Book         → orderbook_snapshots   （全更新を保存）
    activeAssetCtx → funding_oi            （funding / OI など）

設計
----
* SDK: hyperliquid-python-sdk の Info.subscribe を使用（CLAUDE.md の 3.10 制約由来）。
  コールバックは SDK のスレッドから同期的に呼ばれる。
* I/O から純粋な解析関数（parse_candle / parse_l2book / parse_active_ctx）を分離。
  これらは外部依存なしで単体テストできる。
* 書き込みは BatchWriter で非同期バッファリング → バッチ INSERT（HDD 追記効率）。
  冪等化のため ON CONFLICT を使用（candle/funding は UPDATE、board は DO NOTHING）。
* asyncpg / hyperliquid は遅延 import（解析関数のテストに両依存を要求しない）。

接続情報は環境変数（AEVUM_DB_DSN、無ければ標準 PG*）。
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

log = logging.getLogger("aevum.ingestion")

# --------------------------------------------------------------------------- #
# 定数
# --------------------------------------------------------------------------- #
SYMBOLS: tuple[str, ...] = ("BTC", "ETH")
CANDLE_INTERVAL: str = "5m"
TABLES: tuple[str, ...] = ("ohlcv_bars", "orderbook_snapshots", "funding_oi")

# 冪等な INSERT 文。プレースホルダ数は各 parse_* の返すタプル長と一致する
# （tests/test_ingestion.py が一致を強制）。
INSERT_SQL: dict[str, str] = {
    "ohlcv_bars": (
        "INSERT INTO ohlcv_bars "
        "(symbol, time, open, high, low, close, volume, trades, vwap) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) "
        "ON CONFLICT (symbol, time) DO UPDATE SET "
        "open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, "
        "close = EXCLUDED.close, volume = EXCLUDED.volume, "
        "trades = EXCLUDED.trades, vwap = EXCLUDED.vwap"
    ),
    "orderbook_snapshots": (
        "INSERT INTO orderbook_snapshots "
        "(symbol, time, bid_px, bid_sz, ask_px, ask_sz) "
        "VALUES ($1, $2, $3, $4, $5, $6) "
        "ON CONFLICT (symbol, time) DO NOTHING"
    ),
    "funding_oi": (
        "INSERT INTO funding_oi "
        "(symbol, time, funding_rate, open_interest, mark_price, oracle_price, premium) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7) "
        "ON CONFLICT (symbol, time) DO UPDATE SET "
        "funding_rate = EXCLUDED.funding_rate, open_interest = EXCLUDED.open_interest, "
        "mark_price = EXCLUDED.mark_price, oracle_price = EXCLUDED.oracle_price, "
        "premium = EXCLUDED.premium"
    ),
}


# --------------------------------------------------------------------------- #
# 解析（純粋関数・テスト可能）
# --------------------------------------------------------------------------- #
def _ms_to_dt(ms: int) -> datetime:
    """エポックミリ秒 → tz-aware(UTC) datetime。"""
    return datetime.fromtimestamp(int(ms) / 1000.0, tz=timezone.utc)


def _fnum(value: Any) -> Optional[float]:
    """文字列/数値/None を float|None に。Hyperliquid は数値を文字列で返す。"""
    if value is None:
        return None
    return float(value)


def parse_candle(msg: Any) -> Optional[tuple]:
    """candle メッセージ → ohlcv_bars 行。対象外/不正なら None。

    行: (symbol, time, open, high, low, close, volume, trades, vwap)
    time は candle の開始時刻 t（バーのキー）。vwap は未提供で None。
    """
    if not isinstance(msg, dict) or msg.get("channel") != "candle":
        return None
    d = msg.get("data")
    if not isinstance(d, dict):
        return None
    coin = d.get("s")
    if coin not in SYMBOLS or d.get("t") is None:
        return None
    try:
        return (
            coin,
            _ms_to_dt(d["t"]),
            _fnum(d["o"]),
            _fnum(d["h"]),
            _fnum(d["l"]),
            _fnum(d["c"]),
            _fnum(d["v"]),
            int(d["n"]) if d.get("n") is not None else None,
            None,  # vwap 未提供
        )
    except (KeyError, TypeError, ValueError):
        log.warning("parse_candle: 不正な candle ペイロード: %r", d)
        return None


def parse_l2book(msg: Any) -> Optional[tuple]:
    """l2Book メッセージ → orderbook_snapshots 行。対象外/不正なら None。

    行: (symbol, time, bid_px[], bid_sz[], ask_px[], ask_sz[])
    levels は [bids, asks]（各 index0 が最良気配）。
    """
    if not isinstance(msg, dict) or msg.get("channel") != "l2Book":
        return None
    d = msg.get("data")
    if not isinstance(d, dict):
        return None
    coin = d.get("coin")
    levels = d.get("levels")
    if coin not in SYMBOLS or d.get("time") is None:
        return None
    if not isinstance(levels, (list, tuple)) or len(levels) != 2:
        return None
    bids, asks = levels[0], levels[1]
    try:
        return (
            coin,
            _ms_to_dt(d["time"]),
            [_fnum(l["px"]) for l in bids],
            [_fnum(l["sz"]) for l in bids],
            [_fnum(l["px"]) for l in asks],
            [_fnum(l["sz"]) for l in asks],
        )
    except (KeyError, TypeError, ValueError):
        log.warning("parse_l2book: 不正な l2Book ペイロード: %r", d)
        return None


def parse_active_ctx(msg: Any, recv_time: datetime) -> Optional[tuple]:
    """activeAssetCtx メッセージ → funding_oi 行。対象外/不正なら None。

    ペイロードにタイムスタンプが無いため、受信時刻 recv_time(tz-aware) で打刻する。
    行: (symbol, time, funding_rate, open_interest, mark_price, oracle_price, premium)
    """
    if not isinstance(msg, dict) or msg.get("channel") != "activeAssetCtx":
        return None
    d = msg.get("data")
    if not isinstance(d, dict):
        return None
    coin = d.get("coin")
    ctx = d.get("ctx")
    if coin not in SYMBOLS or not isinstance(ctx, dict):
        return None
    try:
        return (
            coin,
            recv_time,
            _fnum(ctx.get("funding")),
            _fnum(ctx.get("openInterest")),
            _fnum(ctx.get("markPx")),
            _fnum(ctx.get("oraclePx")),
            _fnum(ctx.get("premium")),
        )
    except (TypeError, ValueError):
        log.warning("parse_active_ctx: 不正な ctx ペイロード: %r", d)
        return None


# --------------------------------------------------------------------------- #
# バッチ書き込み
# --------------------------------------------------------------------------- #
class BatchWriter:
    """テーブル別に行をバッファし、定期/しきい値でバッチ INSERT する。

    SDK コールバック（別スレッド）からは add() を呼ぶ。実際のバッファ操作は
    call_soon_threadsafe でイベントループ上に載せ替えるため、ロック不要で安全。
    """

    def __init__(
        self,
        pool: Any,
        *,
        flush_interval: float = 1.0,
        max_batch: int = 2000,
    ) -> None:
        self._pool = pool
        self._flush_interval = flush_interval
        self._max_batch = max_batch
        self._buffers: dict[str, list[tuple]] = {t: [] for t in TABLES}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._wake: Optional[asyncio.Event] = None
        self._stopping = False
        self.written = 0
        self.dropped = 0

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._wake = asyncio.Event()

    def add(self, table: str, row: tuple) -> None:
        """スレッドセーフな投入口（SDK コールバックから呼ぶ）。"""
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._append, table, row)

    def _append(self, table: str, row: tuple) -> None:
        buf = self._buffers[table]
        buf.append(row)
        if len(buf) >= self._max_batch and self._wake is not None:
            self._wake.set()

    async def flush_once(self) -> None:
        for table in TABLES:
            buf = self._buffers[table]
            if not buf:
                continue
            rows, self._buffers[table] = buf, []  # 先にスワップ（await 中の追記は新bufへ）
            try:
                await self._pool.executemany(INSERT_SQL[table], rows)
                self.written += len(rows)
            except Exception as exc:  # noqa: BLE001 - 1バッチの失敗で停止させない
                self.dropped += len(rows)
                log.error("flush 失敗 table=%s rows=%d: %s", table, len(rows), exc)

    async def run(self) -> None:
        """flush ループ。stop() まで定期/しきい値で flush する。"""
        assert self._wake is not None, "bind_loop() を先に呼ぶこと"
        while not self._stopping:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._flush_interval)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            await self.flush_once()
        await self.flush_once()  # 最終フラッシュ

    def stop(self) -> None:
        self._stopping = True
        if self._loop is not None and self._wake is not None:
            self._loop.call_soon_threadsafe(self._wake.set)


# --------------------------------------------------------------------------- #
# コールバック生成
# --------------------------------------------------------------------------- #
def make_callbacks(
    writer: BatchWriter,
    *,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Callable[[Any], None]]:
    """SDK 購読用の同期コールバック群を作る。now_fn はテストで差し替え可能。"""

    def on_candle(msg: Any) -> None:
        row = parse_candle(msg)
        if row is not None:
            writer.add("ohlcv_bars", row)

    def on_book(msg: Any) -> None:
        row = parse_l2book(msg)
        if row is not None:
            writer.add("orderbook_snapshots", row)

    def on_ctx(msg: Any) -> None:
        row = parse_active_ctx(msg, now_fn())
        if row is not None:
            writer.add("funding_oi", row)

    return {"candle": on_candle, "l2Book": on_book, "activeAssetCtx": on_ctx}


# --------------------------------------------------------------------------- #
# 設定
# --------------------------------------------------------------------------- #
def resolve_dsn(env: Optional[dict] = None) -> str:
    """接続 DSN を環境変数から解決。AEVUM_DB_DSN 優先、無ければ標準 PG*。"""
    env = os.environ if env is None else env
    dsn = env.get("AEVUM_DB_DSN")
    if dsn:
        return dsn
    host = env.get("PGHOST", "localhost")
    port = env.get("PGPORT", "5432")
    user = env.get("PGUSER", "postgres")
    pw = env.get("PGPASSWORD", "")
    db = env.get("PGDATABASE", "aevum")
    auth = f"{user}:{pw}@" if pw else f"{user}@"
    return f"postgresql://{auth}{host}:{port}/{db}"


# --------------------------------------------------------------------------- #
# 接続スーパーバイザ（責務 = 接続維持のみ）
# --------------------------------------------------------------------------- #
# 背景: hyperliquid SDK は WS が "Expired" で切れると run_forever() が return して
# 受信スレッドが無言で終了する（自動再接続しない）。本番(Pi)の常時稼働要件を満たすため、
# 死活/鮮度を監視して異常時に Info を作り直し、3チャンネルを張り直す。
# ここはあくまで「接続維持」のみ: データ補完/バックフィルや書き込みロジックは持たない
# （書き込みは BatchWriter が担う。本クラスは購読を張り直すだけ）。


@dataclass
class SupervisorConfig:
    """再接続スーパーバイザの設定（リトライ上限・バックオフは設定可能）。"""

    stale_after_seconds: float = 90.0       # 無受信がこの秒数を超えたら異常とみなす
    monitor_interval_seconds: float = 15.0  # 死活/鮮度のチェック間隔
    max_retries: int = 0                    # 連続再接続失敗の上限（0 = 無制限）
    backoff_initial_seconds: float = 1.0    # 再接続バックオフの初期遅延
    backoff_factor: float = 2.0             # 指数バックオフの倍率
    backoff_max_seconds: float = 60.0       # バックオフ遅延の上限


class ConnectionMonitor:
    """最後に WS メッセージを受信した時刻を追跡（データ鮮度監視）。

    bump() は SDK の受信スレッドから呼ばれる。float 代入は GIL 下で原子的なので
    ロック不要。idle_seconds() は監視ループ（イベントループ側）から読む。
    """

    def __init__(self, monotonic: Callable[[], float] = time.monotonic) -> None:
        self._monotonic = monotonic
        self._last_msg_at = monotonic()

    def bump(self) -> None:
        self._last_msg_at = self._monotonic()

    def idle_seconds(self) -> float:
        return self._monotonic() - self._last_msg_at


def should_reconnect(alive: bool, idle_seconds: float, stale_after: float) -> bool:
    """再接続すべきか。WS スレッド停止、または stale_after 超の無受信を異常とみなす。"""
    return (not alive) or (idle_seconds > stale_after)


def backoff_delay(attempt: int, initial: float, factor: float, max_delay: float) -> float:
    """指数バックオフ遅延（attempt は1始まり、max_delay で頭打ち）。"""
    if attempt < 1:
        attempt = 1
    return min(initial * (factor ** (attempt - 1)), max_delay)


def _freshness_wrap(cb: Callable[[Any], None], monitor: ConnectionMonitor) -> Callable[[Any], None]:
    """コールバックを包み、受信のたびに鮮度タイムスタンプを更新する（書き込みは不変）。"""

    def wrapped(msg: Any) -> None:
        monitor.bump()
        cb(msg)

    return wrapped


def _info_alive(info: Any) -> bool:
    """Info の WS 受信スレッドが生きているか（Expired 後は False）。"""
    mgr = getattr(info, "ws_manager", None)
    return bool(mgr is not None and mgr.is_alive())


def _default_make_info() -> Any:
    """本番用の Info ファクトリ（mainnet WS 接続）。テストでは差し替える。"""
    from hyperliquid.info import Info  # 遅延 import
    from hyperliquid.utils import constants

    return Info(constants.MAINNET_API_URL, skip_ws=False)


def _subscribe_all(info: Any, callbacks: dict[str, Callable[[Any], None]]) -> None:
    """candle / l2Book / activeAssetCtx を全銘柄ぶん購読する。"""
    for coin in SYMBOLS:
        info.subscribe({"type": "candle", "coin": coin, "interval": CANDLE_INTERVAL}, callbacks["candle"])
        info.subscribe({"type": "l2Book", "coin": coin}, callbacks["l2Book"])
        info.subscribe({"type": "activeAssetCtx", "coin": coin}, callbacks["activeAssetCtx"])


class WsSupervisor:
    """WS の死活/鮮度を監視し、異常時に Info を作り直して再購読する（接続維持のみ）。

    * 死活: info.ws_manager.is_alive()（Expired 後はスレッド終了 → False）
    * 鮮度: ConnectionMonitor.idle_seconds() が stale_after を超過
    異常検知で disconnect → 新 Info → 再購読。再接続はバックオフ付きで繰り返す。
    Info 構築は同期 HTTP を伴うため executor で実行し、書き込みループを止めない。
    """

    def __init__(
        self,
        callbacks: dict[str, Callable[[Any], None]],
        monitor: ConnectionMonitor,
        config: Optional[SupervisorConfig] = None,
        *,
        make_info: Callable[[], Any] = _default_make_info,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self._callbacks = callbacks
        self._monitor = monitor
        self._config = config or SupervisorConfig()
        self._make_info = make_info
        self._sleep = sleep
        self._info: Any = None
        self._stopping = False

    # -- 同期（executor 上で実行する）部分 -- #
    def _connect_blocking(self) -> Any:
        info = self._make_info()
        _subscribe_all(info, self._callbacks)
        return info

    @staticmethod
    def _disconnect_blocking(info: Any) -> None:
        if info is None:
            return
        try:
            info.disconnect_websocket()
        except Exception as exc:  # noqa: BLE001 - 切断失敗は無視（再接続を妨げない）
            log.debug("disconnect_websocket ignored error: %s", exc)

    def close(self) -> None:
        """現在の接続を閉じる（シャットダウン時）。"""
        self._disconnect_blocking(self._info)
        self._info = None

    def stop(self) -> None:
        self._stopping = True

    # -- 非同期ループ -- #
    async def _open(self) -> None:
        loop = asyncio.get_running_loop()
        self._info = await loop.run_in_executor(None, self._connect_blocking)
        self._monitor.bump()  # 接続直後は鮮度をリセット
        log.info("WS connected and subscribed: symbols=%s interval=%s", SYMBOLS, CANDLE_INTERVAL)

    async def _reconnect(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._disconnect_blocking, self._info)
        self._info = None
        attempt = 0
        while not self._stopping:
            attempt += 1
            try:
                self._info = await loop.run_in_executor(None, self._connect_blocking)
                self._monitor.bump()
                log.warning("WS reconnected and re-subscribed (attempt %d)", attempt)
                return
            except Exception as exc:  # noqa: BLE001 - 再接続失敗はバックオフして継続
                cfg = self._config
                if cfg.max_retries and attempt >= cfg.max_retries:
                    log.error("WS reconnect failed after %d attempts: %s", attempt, exc)
                    raise
                delay = backoff_delay(attempt, cfg.backoff_initial_seconds, cfg.backoff_factor, cfg.backoff_max_seconds)
                log.error("WS reconnect attempt %d failed: %s; retrying in %.1fs", attempt, exc, delay)
                await self._sleep(delay)

    async def run(self) -> None:
        """初回接続 → 死活/鮮度監視ループ（stop()/cancel まで常駐）。"""
        await self._open()
        while not self._stopping:
            await self._sleep(self._config.monitor_interval_seconds)
            if self._stopping:
                break
            alive = _info_alive(self._info)
            idle = self._monitor.idle_seconds()
            if should_reconnect(alive, idle, self._config.stale_after_seconds):
                log.warning(
                    "WS anomaly detected (alive=%s, idle=%.1fs, stale_after=%.1fs); reconnecting...",
                    alive, idle, self._config.stale_after_seconds,
                )
                await self._reconnect()


# --------------------------------------------------------------------------- #
# 実行
# --------------------------------------------------------------------------- #
async def run_ingestion(config: Optional[SupervisorConfig] = None) -> None:
    """購読を張り、書き込みループを回す（Ctrl-C / cancel まで継続）。

    WsSupervisor が WS の死活/鮮度を監視し、異常時に再接続・再購読する
    （SDK は自動再接続しないため。責務は接続維持のみ）。
    """
    import asyncpg  # 遅延 import

    dsn = resolve_dsn()
    pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
    writer = BatchWriter(pool)
    writer.bind_loop(asyncio.get_running_loop())
    writer_task = asyncio.create_task(writer.run())

    monitor = ConnectionMonitor()
    callbacks = {k: _freshness_wrap(cb, monitor) for k, cb in make_callbacks(writer).items()}
    supervisor = WsSupervisor(callbacks, monitor, config)

    try:
        await supervisor.run()  # 常駐: 死活/鮮度監視 + 自動再接続・再購読
    finally:
        supervisor.stop()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, supervisor.close)
        writer.stop()
        await writer_task
        await pool.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(run_ingestion())
    except KeyboardInterrupt:
        log.info("interrupted; shutting down")


if __name__ == "__main__":
    main()
