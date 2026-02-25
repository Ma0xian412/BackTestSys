"""Pickle格式的行情数据源实现。"""

import pickle
import math
import logging
from bisect import bisect_left, bisect_right
from typing import List, Any, Optional, Tuple
from ...core.port import IMarketDataFeed
from ...core.data_structure import NormalizedSnapshot, Level

# 设置模块级logger
logger = logging.getLogger(__name__)


class PickleMarketDataFeed_Impl(IMarketDataFeed):
    _UNSET = object()

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data: List[Any] = []
        self.idx = 0
        self._recv_ticks: List[int] = []
        self._parsed_cache: List[Any] = []
        self._query_hint = 0
        self._load_data()

    def _load_data(self):
        try:
            with open(self.file_path, "rb") as f:
                raw = pickle.load(f)
                self.data = raw.to_dict('records') if hasattr(raw, 'to_dict') else raw
            self.data.sort(key=self._read_recv_tick)
            self._recv_ticks = [self._read_recv_tick(row) for row in self.data]
            self._parsed_cache = [self._UNSET] * len(self.data)
            self._query_hint = 0
        except Exception as e:
            print(f"Error loading {self.file_path}: {e}")
            self.data = []
            self._recv_ticks = []
            self._parsed_cache = []
            self._query_hint = 0

    def next(self) -> Optional[NormalizedSnapshot]:
        if self.idx >= len(self.data):
            return None
        snap = self._get_snapshot_by_idx(self.idx)
        self.idx += 1
        return snap

    def reset(self) -> None:
        self.idx = 0
        self._query_hint = 0

    def query_data(self, t_start: int, t_end: int) -> List[NormalizedSnapshot]:
        t_start = int(t_start)
        t_end = int(t_end)
        if t_end < t_start or not self._recv_ticks:
            return []

        left = self._find_left_index(t_start)
        right = bisect_right(self._recv_ticks, t_end, lo=left)
        self._query_hint = left

        result: List[NormalizedSnapshot] = []
        for i in range(left, right):
            snap = self._get_snapshot_by_idx(i)
            if snap is None:
                continue
            ts = int(snap.ts_recv)
            if t_start <= ts <= t_end:
                result.append(snap)
        return result

    def __len__(self) -> int:
        """快照条数（用于进度条等场景）。"""
        return int(len(self.data))

    @staticmethod
    def _read_recv_tick(row: Any) -> int:
        val = row.get("RecvTick", None)
        if val is None:
            return 0
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return 0

    def _get_snapshot_by_idx(self, index: int) -> Optional[NormalizedSnapshot]:
        cached = self._parsed_cache[index]
        if cached is self._UNSET:
            parsed = self._parse_row(self.data[index])
            self._parsed_cache[index] = parsed
            return parsed
        return cached

    def _find_left_index(self, t_start: int) -> int:
        n = len(self._recv_ticks)
        if n == 0:
            return 0

        hint_candidates = [self.idx - 1, self._query_hint]
        for hint in hint_candidates:
            if hint < 0 or hint >= n:
                continue

            if self._recv_ticks[hint] <= t_start:
                i = hint
                steps = 0
                while i < n and self._recv_ticks[i] < t_start and steps < 64:
                    i += 1
                    steps += 1
                if i < n and (i == 0 or self._recv_ticks[i - 1] < t_start):
                    return i
            else:
                i = hint
                steps = 0
                while i > 0 and self._recv_ticks[i - 1] >= t_start and steps < 64:
                    i -= 1
                    steps += 1
                if i == 0 or self._recv_ticks[i - 1] < t_start:
                    return i

        return bisect_left(self._recv_ticks, t_start)

    def _parse_row(self, row) -> Optional[NormalizedSnapshot]:
        """把原始行解析成标准快照。

        兼容字段：
        - ExchTick / RecvTick
        - Bid, Bid2..Bid5 与 BidVol, BidVol2..BidVol5
        - Ask, Ask2..Ask5 与 AskVol, AskVol2..AskVol5
        - Last, Volume, Turnover, AveragePrice
        - LastVolSplit: [(price, qty), ...]
        """

        def _is_nan(x: Any) -> bool:
            try:
                return x is None or (isinstance(x, float) and math.isnan(x))
            except Exception:
                return False

        def _to_float(x: Any) -> Optional[float]:
            if _is_nan(x):
                return None
            try:
                return float(x)
            except Exception:
                return None

        def _to_int(x: Any) -> Optional[int]:
            if _is_nan(x):
                return None
            try:
                return int(x)
            except Exception:
                try:
                    return int(float(x))
                except Exception:
                    return None

        def _parse_levels(side: str) -> List[Level]:
            # side in {"Bid", "Ask"}
            out: List[Level] = []
            for i in range(1, 6):
                px_key = side if i == 1 else f"{side}{i}"
                vol_key = f"{side}Vol" if i == 1 else f"{side}Vol{i}"
                px = _to_float(row.get(px_key))
                vol = _to_int(row.get(vol_key))
                if px is None or vol is None:
                    continue
                out.append(Level(px, vol))
            # 保险：排序（Bid 降序，Ask 升序）
            out.sort(key=lambda x: x.price, reverse=(side == "Bid"))
            return out

        try:
            ts_exch = _to_int(row.get('ExchTick')) or 0
            ts_recv = _to_int(row.get('RecvTick'))
            
            # RecvTick is mandatory - error if missing or invalid
            if ts_recv is None:
                raise ValueError(
                    f"RecvTick field is required but missing or invalid. "
                    f"Expected integer timestamp in tick units (100ns per tick), got: {row.get('RecvTick')!r}"
                )
            
            bids = _parse_levels("Bid")
            asks = _parse_levels("Ask")

            last = _to_float(row.get('Last'))
            volume = _to_int(row.get('Volume'))
            turnover = _to_float(row.get('Turnover'))
            avg_px = _to_float(row.get('AveragePrice'))

            lvs: List[Tuple[float, int]] = []
            raw_split = row.get('LastVolSplit')
            if isinstance(raw_split, list):
                for it in raw_split:
                    try:
                        p, q = it
                        p2 = _to_float(p)
                        q2 = _to_int(q)
                        if p2 is not None and q2 is not None:
                            # Round price to 6 decimal places to fix floating-point precision issues
                            # e.g., 1050.199999999 -> 1050.2
                            p2 = round(p2, 6)
                            lvs.append((p2, q2))
                    except Exception:
                        continue

            return NormalizedSnapshot(
                ts_recv=ts_recv,  # 主时间线（必填）
                bids=bids,
                asks=asks,
                last_vol_split=lvs,
                ts_exch=ts_exch,  # 可选（仅记录）
                last=last,
                volume=volume,
                turnover=turnover,
                average_price=avg_px,
            )
        except Exception as e:
            # Re-raise ValueError for mandatory field errors
            if isinstance(e, ValueError):
                raise
            return None
