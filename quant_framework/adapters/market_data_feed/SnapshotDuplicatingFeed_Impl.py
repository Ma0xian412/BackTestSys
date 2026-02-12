"""快照复制数据源包装器实现。"""

import logging
from typing import List, Optional
from ...core.port import IMarketDataFeed
from ...core.data_structure import NormalizedSnapshot
from ...utils.trading_hours import TradingHoursHelper

# 设置模块级logger
logger = logging.getLogger(__name__)


class SnapshotDuplicatingFeed_Impl(IMarketDataFeed):
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
        from ...core.data_structure import SNAPSHOT_MIN_INTERVAL_TICK, DEFAULT_SNAPSHOT_TOLERANCE_TICK
        
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
                f"[SnapshotDuplicatingFeed_Impl] Snapshot copy: start={t_prev}, end={t_curr}, copies={copies_count}"
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
