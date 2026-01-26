import pickle
import math
from typing import List, Any, Optional, Tuple
from .interfaces import IMarketDataFeed
from .types import NormalizedSnapshot, Level

class PickleMarketDataFeed(IMarketDataFeed):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data: List[Any] = []
        self._load_data()
        self.idx = 0

    def _load_data(self):
        try:
            with open(self.file_path, "rb") as f:
                raw = pickle.load(f)
                self.data = raw.to_dict('records') if hasattr(raw, 'to_dict') else raw
        except Exception as e:
            print(f"Error loading {self.file_path}: {e}")
            self.data = []

    def next(self) -> Optional[NormalizedSnapshot]:
        if self.idx >= len(self.data): return None
        row = self.data[self.idx]
        self.idx += 1
        return self._parse_row(row)

    def reset(self):
        self.idx = 0

    def __len__(self) -> int:
        """快照条数（用于进度条等场景）。"""
        return int(len(self.data))

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


class SnapshotDuplicatingFeed(IMarketDataFeed):
    """包装feed，实现快照复制逻辑。
    
    当两个快照之间的间隔超过500ms(SNAPSHOT_MIN_INTERVAL_TICK)时，
    将前一个快照向右复制以填充间隔，每个复制快照间隔500ms。
    复制的快照的last_vol_split为空（因为没有新成交）。
    
    例如：
    - 如果A(1000ms)和B(2000ms)间隔1000ms，会生成A(1000ms), A'(1500ms), B(2000ms)
    - 如果A(1000ms)和B(3000ms)间隔2000ms，会生成A(1000ms), A'(1500ms), A''(2000ms), A'''(2500ms), B(3000ms)
    
    时间单位：tick（每tick=100ns）。500ms = 5_000_000 ticks。
    """
    
    def __init__(self, inner_feed: IMarketDataFeed):
        """初始化包装feed。
        
        Args:
            inner_feed: 被包装的原始feed
        """
        from .types import SNAPSHOT_MIN_INTERVAL_TICK
        
        self.inner_feed = inner_feed
        self.min_interval = SNAPSHOT_MIN_INTERVAL_TICK
        
        # 内部状态
        self._buffer: List[NormalizedSnapshot] = []
        self._buffer_idx = 0
        self._prev_snapshot: Optional[NormalizedSnapshot] = None
        self._initialized = False
    
    def next(self) -> Optional[NormalizedSnapshot]:
        """获取下一个快照（可能是复制的）。"""
        # 如果buffer中有内容，直接返回
        if self._buffer_idx < len(self._buffer):
            snap = self._buffer[self._buffer_idx]
            self._buffer_idx += 1
            return snap
        
        # buffer为空，从内部feed获取下一个快照
        self._buffer.clear()
        self._buffer_idx = 0
        
        curr = self.inner_feed.next()
        if curr is None:
            return None
        
        # 第一个快照，直接返回
        if self._prev_snapshot is None:
            self._prev_snapshot = curr
            return curr
        
        # 计算间隔
        t_prev = self._prev_snapshot.ts_recv
        t_curr = curr.ts_recv
        gap = t_curr - t_prev
        
        # 如果间隔小于等于500ms，直接返回当前快照
        if gap <= self.min_interval:
            self._prev_snapshot = curr
            return curr
        
        # 间隔超过500ms，需要插入复制的快照
        # 计算需要插入的复制快照数量
        # 使用 (gap - 1) 而不是 gap 是为了确保边界正确处理：
        # - gap = 500ms: (500-1)//500 = 0 个复制
        # - gap = 501ms: (501-1)//500 = 1 个复制
        # - gap = 1000ms: (1000-1)//500 = 1 个复制
        # - gap = 1001ms: (1001-1)//500 = 2 个复制
        num_copies = int((gap - 1) // self.min_interval)
        
        # 生成复制的快照
        for i in range(num_copies):
            copy_time = t_prev + (i + 1) * self.min_interval
            if copy_time < t_curr:
                # 创建复制快照，last_vol_split为空
                copy_snap = NormalizedSnapshot(
                    ts_recv=copy_time,
                    bids=list(self._prev_snapshot.bids),  # 复制档位列表
                    asks=list(self._prev_snapshot.asks),
                    last_vol_split=[],  # 复制快照的last_vol_split为空
                    ts_exch=self._prev_snapshot.ts_exch,
                    last=self._prev_snapshot.last,
                    volume=self._prev_snapshot.volume,
                    turnover=self._prev_snapshot.turnover,
                    average_price=self._prev_snapshot.average_price,
                )
                self._buffer.append(copy_snap)
        
        # 最后添加当前快照
        self._buffer.append(curr)
        
        # 返回buffer中的第一个
        if self._buffer:
            snap = self._buffer[self._buffer_idx]
            self._buffer_idx += 1
            self._prev_snapshot = curr  # 更新prev为原始的curr
            return snap
        
        # 理论上不应该到这里
        self._prev_snapshot = curr
        return curr
    
    def reset(self):
        """重置feed。"""
        self.inner_feed.reset()
        self._buffer.clear()
        self._buffer_idx = 0
        self._prev_snapshot = None
        self._initialized = False