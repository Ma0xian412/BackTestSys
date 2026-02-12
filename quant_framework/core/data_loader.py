import pickle
import csv
import ast
import math
import logging
from typing import List, Any, Optional, Tuple
from .port import IMarketDataFeed
from .types import NormalizedSnapshot, Level
from ..utils.trading_hours import TradingHoursHelper

# 设置模块级logger
logger = logging.getLogger(__name__)


class CsvMarketDataFeed(IMarketDataFeed):
    """CSV格式的行情数据源。
    
    读取CSV文件，按RecvTick排序后提供快照数据。
    字段与PickleMarketDataFeed兼容：
    - ExchTick / RecvTick
    - Bid, Bid2..Bid5 与 BidVol, BidVol2..BidVol5
    - Ask, Ask2..Ask5 与 AskVol, AskVol2..AskVol5
    - Last, Volume, Turnover, AveragePrice
    - LastVolSplit: [(price, qty), ...]（CSV中存储为字符串形式）
    """
    
    def __init__(self, file_path: str):
        """初始化CSV数据源。
        
        Args:
            file_path: CSV文件路径
        """
        self.file_path = file_path
        self.data: List[Any] = []
        self._load_data()
        self.idx = 0
    
    def _load_data(self):
        """加载CSV数据并按RecvTick排序。"""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.data = list(reader)
            
            # 按RecvTick排序（处理可能的空值和类型转换）
            def get_recv_tick(row):
                val = row.get('RecvTick', '')
                if val == '' or val is None:
                    return 0
                try:
                    return int(float(val))
                except (ValueError, TypeError):
                    return 0
            
            self.data.sort(key=get_recv_tick)
        except Exception as e:
            print(f"Error loading {self.file_path}: {e}")
            self.data = []
    
    def next(self) -> Optional[NormalizedSnapshot]:
        """获取下一个快照。"""
        if self.idx >= len(self.data):
            return None
        row = self.data[self.idx]
        self.idx += 1
        return self._parse_row(row)
    
    def reset(self):
        """重置数据源。"""
        self.idx = 0
    
    def __len__(self) -> int:
        """快照条数（用于进度条等场景）。"""
        return int(len(self.data))
    
    def _parse_row(self, row) -> Optional[NormalizedSnapshot]:
        """把CSV行解析成标准快照。
        
        兼容字段：
        - ExchTick / RecvTick
        - Bid, Bid2..Bid5 与 BidVol, BidVol2..BidVol5
        - Ask, Ask2..Ask5 与 AskVol, AskVol2..AskVol5
        - Last, Volume, Turnover, AveragePrice
        - LastVolSplit: [(price, qty), ...]（CSV中存储为字符串形式）
        """
        
        def _is_nan(x: Any) -> bool:
            try:
                if x is None or x == '':
                    return True
                if isinstance(x, float) and math.isnan(x):
                    return True
                # CSV读取的值可能是字符串"nan"或"NaN"
                if isinstance(x, str) and x.lower() in ('nan', 'none', ''):
                    return True
                return False
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
        
        def _parse_last_vol_split(raw_split: Any) -> List[Tuple[float, int]]:
            """解析LastVolSplit字段。
            
            CSV中该字段可能以字符串形式存储，如 "[(100.0, 10), (101.0, 20)]"
            """
            lvs: List[Tuple[float, int]] = []
            
            if raw_split is None or raw_split == '':
                return lvs
            
            # 如果已经是列表，直接处理
            if isinstance(raw_split, list):
                data = raw_split
            elif isinstance(raw_split, str):
                # 尝试解析字符串表示的列表
                try:
                    data = ast.literal_eval(raw_split)
                    if not isinstance(data, list):
                        return lvs
                except (ValueError, SyntaxError):
                    return lvs
            else:
                return lvs
            
            for it in data:
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
            
            return lvs
        
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
            
            last_vol_split = _parse_last_vol_split(row.get('LastVolSplit'))
            
            return NormalizedSnapshot(
                ts_recv=ts_recv,  # 主时间线（必填）
                bids=bids,
                asks=asks,
                last_vol_split=last_vol_split,
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


class SnapshotDuplicatingFeed(IMarketDataFeed):
    """包装feed，实现快照复制逻辑。
    
    当两个快照之间的间隔超过500ms(SNAPSHOT_MIN_INTERVAL_TICK) + tolerance时，
    将前一个快照向右复制以填充间隔，每个复制快照间隔500ms。
    复制的快照的last_vol_split为空（因为没有新成交）。
    
    由于RecvTick可能存在误差，相邻快照间隔不一定刚好是500ms，
    因此支持tolerance参数来处理这种时间抖动。
    
    例如（假设tolerance=10ms）：
    - 间隔510ms: 在tolerance范围内，不复制
    - 间隔1000ms: 超过500ms+tolerance，生成1个复制快照
    - 间隔2000ms: 超过500ms+tolerance，生成3个复制快照
    
    交易时段支持：
    - 如果提供了交易时段配置（trading_hours），在两个交易时段之间不进行快照复制
    - 交易时段支持跨越午夜，例如晚上9点到凌晨2点半
    - 一天的交易可能包含多个不连续的时段
    
    时间单位：tick（每tick=100ns）。500ms = 5_000_000 ticks。
    """
    
    def __init__(self, inner_feed: IMarketDataFeed, tolerance_tick: int = None, 
                 trading_hours: List = None):
        """初始化包装feed。
        
        Args:
            inner_feed: 被包装的原始feed
            tolerance_tick: 时间容差（tick单位），默认为10ms。
                           如果间隔在 min_interval ± tolerance 范围内，
                           认为是正常的500ms间隔，不进行复制。
            trading_hours: 交易时段列表，每个元素是一个包含start_time和end_time的对象。
                          时间格式为 "HH:MM:SS"。如果为None，不进行交易时段检查。
        """
        from .types import SNAPSHOT_MIN_INTERVAL_TICK, DEFAULT_SNAPSHOT_TOLERANCE_TICK
        
        self.inner_feed = inner_feed
        self.min_interval = SNAPSHOT_MIN_INTERVAL_TICK
        self.tolerance = tolerance_tick if tolerance_tick is not None else DEFAULT_SNAPSHOT_TOLERANCE_TICK
        self.trading_hours = trading_hours or []
        
        # 使用TradingHoursHelper处理交易时段相关逻辑
        self._trading_hours_helper = TradingHoursHelper(self.trading_hours)
        
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
        
        # 如果间隔在 min_interval + tolerance 范围内，直接返回当前快照
        # 这处理了RecvTick的时间抖动
        threshold = self.min_interval + self.tolerance
        if gap <= threshold:
            self._prev_snapshot = curr
            return curr
        
        # 检查是否跨越交易时段间隔
        # 如果跨越了间隔，不进行快照复制
        if self._trading_hours_helper.spans_trading_session_gap(t_prev, t_curr):
            self._prev_snapshot = curr
            return curr
        
        # 间隔超过阈值，需要插入复制的快照
        # 计算需要插入的复制快照数量
        # 使用 (gap - tolerance - 1) 来考虑容差，确保边界正确处理：
        # 例如：min_interval=500ms, tolerance=10ms
        # - gap = 510ms: (510-10-1)//500 = 0 个复制（在容差范围内）
        # - gap = 520ms: (520-10-1)//500 = 1 个复制
        # - gap = 1000ms: (1000-10-1)//500 = 1 个复制
        # - gap = 1010ms: (1010-10-1)//500 = 1 个复制
        # - gap = 1020ms: (1020-10-1)//500 = 2 个复制
        num_copies = int((gap - self.tolerance - 1) // self.min_interval)
        
        # 确保至少复制0个（防止负数）
        num_copies = max(0, num_copies)
        
        # 生成复制的快照
        copies_count = 0
        for i in range(num_copies):
            copy_time = t_prev + (i + 1) * self.min_interval
            
            # 首先检查时间边界，确保复制快照不超过当前快照时间
            if copy_time >= t_curr:
                break  # 超出时间边界，停止生成复制快照
            
            # 检查复制快照的时间是否在交易时段内
            copy_seconds = self._trading_hours_helper.tick_to_day_seconds(copy_time)
            if not self._trading_hours_helper.is_in_any_trading_session(copy_seconds):
                continue  # 跳过非交易时间的复制快照
            
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
            copies_count += 1
        
        # 输出快照复制的debug信息
        if copies_count > 0:
            logger.debug(
                f"[SnapshotDuplicatingFeed] Snapshot copy: start={t_prev}, end={t_curr}, copies={copies_count}"
            )
        
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
    
    def __len__(self) -> int:
        """返回内部feed的长度（用于进度条初始化）。
        
        注意：由于快照复制逻辑，实际返回的快照数可能大于此值。
        进度条应使用动态更新来处理这种情况。
        """
        if hasattr(self.inner_feed, '__len__'):
            return len(self.inner_feed)
        raise TypeError(f"object of type '{type(self.inner_feed).__name__}' has no len()")