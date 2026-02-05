"""事件循环模块 - 统一回测架构的核心运行器。

本模块实现主事件循环，协调以下组件：
- Tape构建
- 交易所仿真
- 策略回调
- 订单/回执路由（支持双时间线映射：exchtime <-> recvtime）

双时间线支持：
- exchtime: 交易所时间（事件实际发生的时间）
- recvtime: 接收时间（策略感知到事件的时间）
- 线性映射: recvtime = a * exchtime + b

延迟处理：
- delay_out: 策略 -> 交易所的延迟（订单发送延迟）
- delay_in: 交易所 -> 策略的延迟（回执接收延迟）
"""

import heapq
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Protocol, TYPE_CHECKING, Callable
from enum import Enum, auto

from ..core.interfaces import IMarketDataFeed, ITapeBuilder, IExchangeSimulator, IStrategy, IOrderManager
from ..core.types import NormalizedSnapshot, Order, OrderReceipt, TapeSegment, CancelRequest, AdvanceResult
from ..core.trading_hours import TradingHoursHelper

if TYPE_CHECKING:
    from ..trading.receipt_logger import ReceiptLogger

# 设置模块级logger
logger = logging.getLogger(__name__)


class IReceiptLogger(Protocol):
    """回执记录器接口协议。"""
    
    def register_order(self, order_id: str, qty: int) -> None:
        """注册订单。"""
        ...
    
    def log_receipt(self, receipt: OrderReceipt) -> None:
        """记录回执。"""
        ...


class EventType(Enum):
    """事件类型枚举。
    
    事件类型优先级（值越小优先级越高）：
    1. SEGMENT_END: 段结束（先完成内部撮合结果）
    2. ORDER_ARRIVAL: 订单到达（交易所先收到请求）
    3. CANCEL_ARRIVAL: 撤单到达（交易所先收到请求）
    4. RECEIPT_TO_STRATEGY: 回执到策略（策略后看到回执）
    5. INTERVAL_END: 区间结束（最后做边界对齐与快照回调）
    """
    SEGMENT_END = auto()          # 段结束
    ORDER_ARRIVAL = auto()        # 订单到达交易所
    CANCEL_ARRIVAL = auto()       # 撤单到达交易所
    RECEIPT_TO_STRATEGY = auto()  # 回执到达策略
    INTERVAL_END = auto()         # 区间结束


# 事件类型优先级映射（值越小优先级越高）
EVENT_TYPE_PRIORITY = {
    EventType.SEGMENT_END: 1,
    EventType.ORDER_ARRIVAL: 2,
    EventType.CANCEL_ARRIVAL: 3,
    EventType.RECEIPT_TO_STRATEGY: 4,
    EventType.INTERVAL_END: 5,
}

# 默认优先级：用于未知事件类型，设置为最低优先级确保已知类型总是优先处理
DEFAULT_EVENT_PRIORITY = 99

# 全局序列号计数器
# 注意：此模块设计为单线程使用（典型的回测场景）。
# 如果需要多线程支持，应使用threading.Lock或itertools.count()等线程安全替代方案。
_event_seq_counter = 0


def _get_next_seq() -> int:
    """获取下一个事件序列号。
    
    注意：此函数非线程安全，仅用于单线程回测场景。
    """
    global _event_seq_counter
    _event_seq_counter += 1
    return _event_seq_counter


def reset_event_seq_counter() -> None:
    """重置事件序列号计数器（用于测试）。
    
    注意：此函数非线程安全，仅用于单线程回测场景。
    """
    global _event_seq_counter
    _event_seq_counter = 0


@dataclass
class Event:
    """事件循环中的事件。

    Attributes:
        time: 事件时间（交易所事件用exchtime，策略事件用recvtime）
        event_type: 事件类型
        data: 事件数据
        priority: 同一时刻事件的优先级（值越小优先级越高）
        seq: 事件序列号（用于完全确定性排序）
        recv_time: 策略接收时间（用于RECEIPT_TO_STRATEGY事件，可选）
    """
    time: int
    event_type: EventType
    data: Any
    priority: int = None
    seq: int = None
    recv_time: Optional[int] = None

    def __post_init__(self):
        """初始化优先级和序列号。"""
        if self.priority is None:
            self.priority = EVENT_TYPE_PRIORITY.get(self.event_type, DEFAULT_EVENT_PRIORITY)
        if self.seq is None:
            self.seq = _get_next_seq()

    def __lt__(self, other):
        """按(time, priority, seq)比较事件（用于heapq）。
        
        确保同一时刻的事件有确定性的处理顺序：
        1. 首先按时间排序
        2. 时间相同时，按事件类型优先级排序
        3. 优先级也相同时，按序列号排序（先创建的先处理）
        """
        if self.time != other.time:
            return self.time < other.time
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.seq < other.seq


@dataclass
class TimelineConfig:
    """时间线配置（已弃用 - 保留用于向后兼容）。
    
    现在使用单一的recv timeline，此类仅为兼容性保留。
    所有时间都直接使用ts_recv，无需转换。
    
    Deprecated: This class is kept for backward compatibility.
    All time values now use ts_recv directly without conversion.
    """
    a: float = 1.0  # 保持为1，无转换
    b: float = 0.0  # 保持为0，无转换

    def exchtime_to_recvtime(self, exchtime: int) -> int:
        """恒等映射（已弃用）。"""
        return exchtime  # 直接返回，无转换

    def recvtime_to_exchtime(self, recvtime: int) -> int:
        """恒等映射（已弃用）。"""
        return recvtime  # 直接返回，无转换


# 进度回调类型定义
# 签名: (current: int, total: int) -> None
# 注意: total 可能为 0 如果 feed 不支持 __len__ 方法
ProgressCallback = Callable[[int, int], None]


@dataclass
class RunnerConfig:
    """事件循环运行器配置。
    
    所有时间使用统一的recv timeline (ts_recv)，单位为tick（每tick=100ns）。

    Attributes:
        delay_out: 策略 -> 交易所的延迟（tick单位）
        delay_in: 交易所 -> 策略的延迟（tick单位）
        timeline: 时间线配置（已弃用，保留用于兼容）
        show_progress: 是否显示进度条（需要tqdm库）
        progress_callback: 自定义进度回调函数，签名为 (current, total) -> None
                          注意: 如果feed不支持__len__方法，total将为0
                          回调函数抛出的异常将被记录但不会中断回测
    """
    delay_out: int = 0
    delay_in: int = 0
    timeline: TimelineConfig = None
    show_progress: bool = False
    progress_callback: Optional[ProgressCallback] = None

    def __post_init__(self):
        if self.timeline is None:
            self.timeline = TimelineConfig()


class EventLoopRunner:
    """事件循环运行器（单一recv时间线）。

    协调Tape构建、交易所仿真和策略执行的事件驱动循环。
    
    所有时间使用统一的recv timeline (ts_recv)，单位为tick（每tick=100ns）。

    主要功能：
    - 单一时间线：所有事件使用ts_recv
    - 延迟处理：delay_out（策略->交易所），delay_in（交易所->策略）
    - 交易时段支持：订单/撤单到达时间基于交易时段调整
    """

    def __init__(
        self,
        feed: IMarketDataFeed,
        tape_builder: ITapeBuilder,
        exchange: IExchangeSimulator,
        strategy: IStrategy,
        oms: IOrderManager,
        config: RunnerConfig = None,
        receipt_logger: Optional[IReceiptLogger] = None,
        trading_hours: List = None,
    ):
        """初始化运行器。

        Args:
            feed: 行情数据源
            tape_builder: Tape构建器
            exchange: 交易所模拟器
            strategy: 策略
            oms: 订单管理器
            config: 运行器配置
            receipt_logger: 回执记录器（可选）
            trading_hours: 交易时段列表，每个元素是一个TradingHour对象。
                          用于调整订单/撤单到达时间：
                          - 如果到达时间在交易时段之间，则调整到下一个交易时段开始
                          - 如果在最后一个交易时段结束之后，挂单直接失败，撤单也失败
        """
        self.feed = feed
        self.tape_builder = tape_builder
        self.exchange = exchange
        self.strategy = strategy
        self.oms = oms
        self.config = config or RunnerConfig()
        self.receipt_logger = receipt_logger
        self.trading_hours = trading_hours or []
        
        # 使用TradingHoursHelper处理交易时段相关逻辑
        self._trading_hours_helper = TradingHoursHelper(self.trading_hours)

        # 当前状态（统一使用recv time）
        self.current_time = 0  # 统一时间（ts_recv）
        self.current_snapshot: Optional[NormalizedSnapshot] = None
        self._pending_receipts: List[Tuple[int, OrderReceipt]] = []
        
        # 待处理的撤单请求列表（从策略获取）
        self._pending_cancels: List[Tuple[int, CancelRequest]] = []

        # 诊断信息
        self.diagnostics: Dict[str, Any] = {
            "intervals_processed": 0,
            "orders_submitted": 0,
            "orders_filled": 0,
            "receipts_generated": 0,
            "cancels_submitted": 0,
            "orders_rejected_after_trading": 0,
            "cancels_failed_after_trading": 0,
        }
    
    # ========== 交易时段相关方法 ==========
    
    def _adjust_arrival_time_for_trading_hours(
        self, 
        arrival_tick: int,
        is_order: bool = True
    ) -> Tuple[Optional[int], bool]:
        """根据交易时段调整到达时间。
        
        规则：
        1. 如果到达时间在交易时段内，保持不变
        2. 如果到达时间在两个交易时段之间，调整到下一个交易时段的开始
        3. 如果到达时间在最后一个交易时段结束之后：
           - 挂单：返回None表示订单被拒绝
           - 撤单：返回None表示撤单失败
        
        Args:
            arrival_tick: 原始到达时间（tick单位）
            is_order: True表示是挂单，False表示是撤单
            
        Returns:
            (adjusted_tick, success): 
            - adjusted_tick: 调整后的到达时间，如果应该被拒绝则为None
            - success: 是否成功（用于区分正常调整和被拒绝的情况）
        """
        if not self.trading_hours:
            return (arrival_tick, True)  # 未配置交易时段，不做调整
        
        helper = self._trading_hours_helper
        time_seconds = helper.tick_to_day_seconds(arrival_tick)
        
        # 检查是否在交易时段内
        if helper.is_in_any_trading_session(time_seconds):
            return (arrival_tick, True)  # 在交易时段内，不做调整
        
        # 检查是否在最后一个交易时段之后
        if helper.is_after_last_trading_session(time_seconds):
            return (None, False)  # 在最后时段之后，拒绝
        
        # 在两个交易时段之间，找下一个交易时段的开始
        next_start_seconds = helper.get_next_trading_session_start(time_seconds)
        
        if next_start_seconds is None:
            # 理论上不应该到这里（因为前面已经检查了是否在最后时段之后）
            logger.warning(f"Unexpected state: no next trading session found for time {time_seconds}s")
            return (None, False)
        
        # 计算调整后的到达时间
        # 需要保持同一天，只调整到下一个时段开始
        ticks_per_day = helper.SECONDS_PER_DAY * helper.TICKS_PER_SECOND
        current_day_start_tick = (arrival_tick // ticks_per_day) * ticks_per_day
        adjusted_tick = current_day_start_tick + helper.seconds_to_tick_offset(next_start_seconds)
        
        return (adjusted_tick, True)

    def run(self) -> Dict[str, Any]:
        """运行回测。

        Returns:
            包含回测结果的字典
        """
        self.feed.reset()
        
        # Full reset the exchange to ensure clean state for new backtest session
        # This clears both interval-specific state and the order registry
        if hasattr(self.exchange, 'full_reset'):
            self.exchange.full_reset()
        else:
            self.exchange.reset()
        
        prev = self.feed.next()

        if prev is None:
            return {"error": "No data"}

        self.current_snapshot = prev
        self.current_time = int(prev.ts_recv)  # 使用ts_recv

        # 初始快照回调
        orders = self.strategy.on_snapshot(prev, self.oms)
        for order in orders:
            # 如果订单已有create_time（预设发送时间），使用该时间
            # 否则使用当前时间
            submit_time = order.create_time if order.create_time > 0 else self.current_time
            self.oms.submit(order, submit_time)
            self.diagnostics["orders_submitted"] += 1
            # 注册到receipt_logger
            if self.receipt_logger:
                self.receipt_logger.register_order(order.order_id, order.qty)
        
        # 从策略获取待处理的撤单请求（如果策略支持）
        if hasattr(self.strategy, 'get_pending_cancels'):
            self._pending_cancels = self.strategy.get_pending_cancels()
            self.diagnostics["cancels_submitted"] = len(self._pending_cancels)

        interval_count = 0
        
        # 获取总数据量用于进度显示
        total_intervals = 0
        if hasattr(self.feed, '__len__'):
            total_intervals = len(self.feed) - 1  # -1 因为第一个快照已经处理
        
        # 设置进度条（如果启用）
        pbar = None
        if self.config.show_progress and total_intervals > 0:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=total_intervals, desc="Backtest Progress", unit="interval")
            except ImportError:
                logger.warning("tqdm not installed, progress bar disabled. Install with: pip install tqdm")

        while True:
            curr = self.feed.next()
            if curr is None:
                break

            # 运行一个区间
            self._run_interval(prev, curr)

            prev = curr
            self.current_snapshot = curr
            interval_count += 1
            
            # 更新进度
            if pbar:
                # 当实际处理数量超过初始估计时，动态扩展进度条总数
                # 这处理了 SnapshotDuplicatingFeed 等会产生额外快照的情况
                if pbar.total is not None and interval_count > pbar.total:
                    pbar.total = interval_count
                    pbar.refresh()
                pbar.update(1)
            if self.config.progress_callback:
                try:
                    self.config.progress_callback(interval_count, total_intervals)
                except Exception as e:
                    logger.warning(f"Progress callback raised exception: {e}")
        
        # 关闭进度条
        if pbar:
            pbar.close()

        self.diagnostics["intervals_processed"] = interval_count

        return {
            "intervals": interval_count,
            "final_time": self.current_time,  # 统一时间
            "diagnostics": self.diagnostics,
        }

    def _run_interval(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot) -> None:
        """运行一个区间[prev, curr]的事件循环。
        
        使用统一的recv timeline。
        通过_schedule_pending_receipts确保所有时间上早于t_a的事件在区间开始时被处理。

        Args:
            prev: 前一个快照（在T_A时刻）
            curr: 当前快照（在T_B时刻）
        """
        # 使用ts_recv作为统一时间线
        t_a = int(prev.ts_recv)
        t_b = int(curr.ts_recv)

        if t_b <= t_a:
            return

        if not self._has_order_events_between(t_a, t_b):
            needs_tape = self._process_receipts_without_tape(t_a, t_b)
            if not needs_tape:
                self._advance_without_tape(curr, t_b)
                return

        # 构建该区间的Tape
        tape = self.tape_builder.build(prev, curr)

        if not tape:
            # 无事件，直接推进到curr
            self._advance_without_tape(curr, t_b)
            return

        # 为新区间重置交易所并设置Tape
        self.exchange.reset()

        # 如果交易所支持set_tape方法则调用
        if hasattr(self.exchange, 'set_tape'):
            self.exchange.set_tape(tape, t_a, t_b)

        # 初始化事件队列（统一时间线）
        event_queue: List[Event] = []

        # 添加段结束事件（最后一个段的结束由INTERVAL_END代替，避免重复事件）
        for i, seg in enumerate(tape):
            is_last_segment = (i == len(tape) - 1)
            if is_last_segment and seg.t_end == t_b:
                # 最后一个段的结束时间等于区间结束时间，跳过SEGMENT_END事件
                # 因为INTERVAL_END已经代表了区间和最后一个段的结束
                continue
            heapq.heappush(event_queue, Event(
                time=seg.t_end,
                event_type=EventType.SEGMENT_END,
                data=seg,
            ))

        # 添加区间结束事件
        heapq.heappush(event_queue, Event(
            time=t_b,
            event_type=EventType.INTERVAL_END,
            data=curr,
        ))

        self._schedule_pending_receipts(event_queue, t_a, t_b)

        # 获取OMS中的待到达订单并调度到达事件
        # 优化：按create_time排序，支持早退出优化
        # 按create_time排序（等同于按arrival_time排序，因为delay_out是常量）
        pending_orders = sorted(self.oms.get_pending_orders(), key=lambda o: o.create_time)

        for order in pending_orders:
            # 计算订单到达交易所的时间（统一时间线）
            arrival_time = int(order.create_time) + int(self.config.delay_out)
            
            # 优化：如果arrival_time >= t_b，后续所有订单的arrival_time也都 >= t_b
            # 因为订单已按create_time排序，而arrival_time = create_time + delay_out（常量）
            # 这些订单将在后续区间处理
            if arrival_time >= t_b:
                break
            
            # 根据交易时段调整到达时间
            adjusted_arrival, success = self._adjust_arrival_time_for_trading_hours(
                arrival_time, is_order=True
            )
            
            if not success:
                # 到达时间在最后交易时段之后，订单被拒绝
                self.diagnostics["orders_rejected_after_trading"] += 1
                # 生成拒绝回执
                reject_receipt = OrderReceipt(
                    order_id=order.order_id,
                    receipt_type="REJECTED",
                    timestamp=arrival_time,
                    fill_qty=0,
                    fill_price=0.0,
                    remaining_qty=order.qty,
                )
                self._schedule_receipt(reject_receipt, event_queue, t_b)
                continue
            
            arrival_time = adjusted_arrival
            
            # 只处理到达时间在区间内的订单
            if arrival_time >= t_a and arrival_time < t_b:
                heapq.heappush(event_queue, Event(
                    time=arrival_time,
                    event_type=EventType.ORDER_ARRIVAL,
                    data=order,
                ))

        # 事件循环 - 使用"peek, advance, pop batch"范式
        # 避免"生成过去事件"的因果反转问题
        current_seg_idx = 0
        last_time = t_a
        
        # 调度区间内的撤单请求
        self._schedule_pending_cancels(event_queue, t_a, t_b)

        while event_queue:
            # Step 1: Peek 下一个事件的时间（不弹出）
            t_next = event_queue[0].time
            
            # Step 2: 将交易所推进到t_next，期间产生的回执会被调度
            # 使用advance_single实现单步推进，允许动态处理segment内部的订单到达
            if t_next > last_time and current_seg_idx < len(tape):
                # 处理到t_next为止的所有完整段
                while current_seg_idx < len(tape) and tape[current_seg_idx].t_end <= t_next:
                    seg = tape[current_seg_idx]
                    last_time = self._advance_segment_with_dynamic_orders(
                        last_time, seg.t_end, seg, event_queue, t_b
                    )
                    current_seg_idx += 1

                # 如果t_next落在当前段内部，需要先将交易所推进到t_next
                # 确保t_next时刻的事件发生之前的成交已被正确处理
                if current_seg_idx < len(tape):
                    seg = tape[current_seg_idx]
                    if seg.t_start <= t_next < seg.t_end and t_next > last_time:
                        # 推进到t_next（段内推进）
                        last_time = self._advance_segment_with_dynamic_orders(
                            last_time, t_next, seg, event_queue, t_b
                        )

            # Step 3: Pop并处理所有time==t_next的事件（批处理）
            # 这样新生成的回执（如果时间 <= t_next）会在heap中排在后面
            # 但因为我们是批处理同一时刻的事件，它们会被正确处理
            events_at_t_next = []
            while event_queue and event_queue[0].time == t_next:
                events_at_t_next.append(heapq.heappop(event_queue))
            
            # 处理同一时刻的所有事件（已按priority和seq排序）
            for event in events_at_t_next:
                self._process_single_event(event, event_queue, tape, t_b)

    def _advance_segment_with_dynamic_orders(
        self,
        t_from: int,
        t_to: int,
        segment: TapeSegment,
        event_queue: List[Event],
        t_b: int,
    ) -> int:
        """推进segment，支持动态订单到达。
        
        使用advance_single实现单步推进，在每次成交后：
        1. 调度回执到策略
        2. 检查并处理在[t_from, 成交时间]区间到达的订单
        3. 继续推进直到t_to
        
        这允许segment内部的动态交互：
        - A时刻开始 -> B时刻成交 -> 策略收到回执 -> 
        - C时刻新订单到达 -> 新订单参与[C, D]的撮合
        
        Args:
            t_from: 开始时间
            t_to: 结束时间  
            segment: Tape段
            event_queue: 事件队列
            t_b: 区间结束时间
            
        Returns:
            推进后的时间（应为t_to）
        """
        current_time = t_from
        
        # 检查交易所是否支持advance_single方法
        if not hasattr(self.exchange, 'advance_single'):
            # 回退到原有的batch模式
            receipts = self.exchange.advance(t_from, t_to, segment)
            for receipt in receipts:
                self._schedule_receipt(receipt, event_queue, t_b)
            return t_to
        
        while current_time < t_to:
            # Step 1: 检查是否有订单到达事件在当前时间点
            # 这些订单需要在继续advance之前先处理
            orders_to_process = []
            remaining_events = []
            
            for event in list(event_queue):
                if event.event_type == EventType.ORDER_ARRIVAL:
                    if event.time <= current_time:
                        orders_to_process.append(event)
                    else:
                        remaining_events.append(event)
                else:
                    remaining_events.append(event)
            
            # 处理当前时间点的订单到达
            for event in sorted(orders_to_process, key=lambda e: (e.time, e.priority, e.seq)):
                heapq.heappop(event_queue)  # Remove from queue
                self._process_single_event(event, event_queue, [], t_b)
            
            # Step 2: 使用advance_single推进到下一个成交或t_to
            result = self.exchange.advance_single(current_time, t_to, segment)
            
            if result.receipt:
                # 调度回执
                self._schedule_receipt(result.receipt, event_queue, t_b)
            
            current_time = result.stop_time
            
            # 如果没有更多成交，跳出循环
            if not result.has_more:
                break
        
        # 确保推进到t_to
        if current_time < t_to:
            # 可能还需要处理剩余时间的订单到达
            pass
        
        return max(current_time, t_to)

    def _process_single_event(
        self,
        event: Event,
        event_queue: List[Event],
        tape: List[TapeSegment],
        t_b: int,
    ) -> None:
        """处理单个事件。
        
        使用统一的recv时间线。
        
        Args:
            event: 要处理的事件
            event_queue: 事件队列（用于调度新事件）
            tape: Tape段列表
            t_b: 区间结束时间
        """
        # 更新当前时间（统一时间线）
        self.current_time = event.time
        
        if event.event_type == EventType.ORDER_ARRIVAL:
            # 订单到达
            order = event.data
            # 使用OMS的mark_order_arrived方法标记订单已到达
            self.oms.mark_order_arrived(order.order_id, event.time)

            # 从当前快照获取订单价位的市场队列深度
            market_qty = self._get_market_qty_from_snapshot(order, self.current_snapshot)
            receipt = self.exchange.on_order_arrival(order, event.time, market_qty)

            if receipt:
                # 所有回执都走统一RECEIPT_TO_STRATEGY调度
                # 在recv_time时刻再oms.on_receipt，确保延迟因果一致性
                self._schedule_receipt(receipt, event_queue, t_b)

        elif event.event_type == EventType.RECEIPT_TO_STRATEGY:
            receipt = event.data
            # 使用event.recv_time或receipt.recv_time作为权威时间
            if event.recv_time is not None:
                self.current_time = event.recv_time
            elif receipt.recv_time is not None:
                self.current_time = receipt.recv_time
            
            self.oms.on_receipt(receipt)
            
            # 记录回执到logger
            if self.receipt_logger:
                self.receipt_logger.log_receipt(receipt)

            if receipt.receipt_type in ["FILL", "PARTIAL"]:
                self.diagnostics["orders_filled"] += 1

            # 策略回调
            orders = self.strategy.on_receipt(receipt, self.current_snapshot, self.oms)
            for order in orders:
                self.oms.submit(order, self.current_time)
                self.diagnostics["orders_submitted"] += 1

                # 调度订单到达（统一时间线）
                arrival_time = self.current_time + self.config.delay_out

                if arrival_time < t_b:
                    heapq.heappush(event_queue, Event(
                        time=arrival_time,
                        event_type=EventType.ORDER_ARRIVAL,
                        data=order,
                    ))
        
        elif event.event_type == EventType.CANCEL_ARRIVAL:
            # 撤单到达
            cancel_request = event.data
            
            # 调用交易所处理撤单
            receipt = self.exchange.on_cancel_arrival(cancel_request.order_id, event.time)
            
            if receipt:
                # 调度撤单回执到策略
                self._schedule_receipt(receipt, event_queue, t_b)

        elif event.event_type == EventType.INTERVAL_END:
            snapshot = event.data

            # 在边界对齐交易所状态
            self.exchange.align_at_boundary(snapshot)

            # 策略回调
            self.current_snapshot = snapshot

            orders = self.strategy.on_snapshot(snapshot, self.oms)
            for order in orders:
                self.oms.submit(order, self.current_time)
                self.diagnostics["orders_submitted"] += 1

    def _advance_without_tape(self, snapshot: NormalizedSnapshot, time: int) -> None:
        """无订单区间快速推进并触发快照回调。"""
        self.current_time = time
        self.current_snapshot = snapshot
        orders = self.strategy.on_snapshot(snapshot, self.oms)
        for order in orders:
            self.oms.submit(order, time)
            self.diagnostics["orders_submitted"] += 1
    
    def _has_order_events_between(self, t_a: int, t_b: int) -> bool:
        """判断区间内是否存在需要处理的订单或撤单事件。
        
        使用统一时间线。
        
        优化策略：
        1. 直接使用OMS维护的已到达和待到达订单集合，避免每次调用都重新分类
        2. 对于待到达的订单，按计算出的arrival_time排序后检查
           - 如果arrival_time >= t_b，由于已排序，后续订单都会 >= t_b，直接break
        """
        # 检查已到达的订单（直接从OMS获取）
        for order in self.oms.get_arrived_orders():
            if order.arrival_time < t_b:
                return True
        
        # 获取待到达的订单并按arrival_time排序
        pending_orders = self.oms.get_pending_orders()
        pending_with_arrival = [
            (int(order.create_time) + int(self.config.delay_out), order)
            for order in pending_orders
        ]
        pending_with_arrival.sort(key=lambda x: x[0])
        
        # 检查待到达的订单（已按arrival_time升序排列）
        for arrival_time, order in pending_with_arrival:
            # 优化：如果arrival_time >= t_b，后续所有订单也都 >= t_b，直接退出
            if arrival_time >= t_b:
                break
            
            # 只处理arrival_time在区间[t_a, t_b)内的订单
            if arrival_time >= t_a:
                return True
        
        # 检查待处理的撤单
        for cancel_sent_time, cancel_request in self._pending_cancels:
            arrival_time = cancel_sent_time + self.config.delay_out
            if arrival_time >= t_a and arrival_time < t_b:
                return True

        return False

    def _schedule_pending_receipts(self, event_queue: List[Event], t_a: int, t_b: int) -> None:
        """将到达时间落在区间内的回执加入事件队列，并处理所有早于t_a的历史回执。
        
        使用统一时间线。确保所有时间上早于当前区间的回执都被正确处理，维护因果一致性。
        """
        remaining: List[Tuple[int, OrderReceipt]] = []
        historical: List[Tuple[int, OrderReceipt]] = []
        
        for recv_time, receipt in self._pending_receipts:
            if recv_time < t_a:
                # 历史回执：早于当前区间，需要立即处理
                historical.append((recv_time, receipt))
            elif t_a <= recv_time <= t_b:
                # 当前区间内的回执：加入事件队列
                heapq.heappush(event_queue, Event(
                    time=recv_time,
                    event_type=EventType.RECEIPT_TO_STRATEGY,
                    data=receipt,
                    recv_time=recv_time,  # 携带接收时间
                ))
            else:
                # 未来回执：保留到后续区间处理
                remaining.append((recv_time, receipt))
        
        self._pending_receipts = remaining
        
        # 使用通用处理方法处理历史回执，确保因果一致性
        self._process_receipt_batch(historical)

    def _queue_receipt_for_future(self, recv_time: int, receipt: OrderReceipt) -> None:
        """缓存将在后续区间到达的回执。"""
        receipt.recv_time = recv_time
        self._pending_receipts.append((recv_time, receipt))

    def _schedule_pending_cancels(self, event_queue: List[Event], t_a: int, t_b: int) -> None:
        """调度区间内的撤单请求到事件队列。
        
        使用统一时间线。将撤单到达时间落在区间[t_a, t_b)内的撤单请求加入事件队列。
        
        根据交易时段调整到达时间：
        - 如果到达时间在交易时段之间，调整到下一个交易时段开始
        - 如果在最后一个交易时段结束之后，撤单失败
        
        Args:
            event_queue: 事件队列
            t_a: 区间开始时间
            t_b: 区间结束时间
        """
        remaining: List[Tuple[int, CancelRequest]] = []
        
        for cancel_sent_time, cancel_request in self._pending_cancels:
            # 计算撤单到达时间（统一时间线）
            arrival_time = cancel_sent_time + self.config.delay_out
            
            # 根据交易时段调整到达时间
            adjusted_arrival, success = self._adjust_arrival_time_for_trading_hours(
                arrival_time, is_order=False
            )
            
            if not success:
                # 到达时间在最后交易时段之后，撤单失败
                self.diagnostics["cancels_failed_after_trading"] += 1
                # 生成撤单失败回执
                reject_receipt = OrderReceipt(
                    order_id=cancel_request.order_id,
                    receipt_type="REJECTED",
                    timestamp=arrival_time,
                    fill_qty=0,
                    fill_price=0.0,
                    remaining_qty=0,
                )
                self._schedule_receipt(reject_receipt, event_queue, t_b)
                continue
            
            arrival_time = adjusted_arrival
            
            if arrival_time >= t_a and arrival_time < t_b:
                # 撤单到达时间在区间内，加入事件队列
                heapq.heappush(event_queue, Event(
                    time=arrival_time,
                    event_type=EventType.CANCEL_ARRIVAL,
                    data=cancel_request,
                ))
            elif arrival_time >= t_b:
                # 撤单到达时间在未来区间，保留
                remaining.append((cancel_sent_time, cancel_request))
            # else: 撤单到达时间早于当前区间，已过期，丢弃
        
        self._pending_cancels = remaining

    def _schedule_receipt(self, receipt: OrderReceipt, event_queue: List[Event], t_b: int) -> None:
        """统一调度回执到策略的事件。
        
        使用统一时间线。所有回执都通过此方法调度，确保delay_in延迟因果一致性。
        回执会在recv_time时刻通过RECEIPT_TO_STRATEGY事件传递给策略。
        
        因果一致性保证：
        - 如果由于int截断导致计算出的事件时间早于当前时间，
          会自动将事件时间调整为当前时间，避免因果反转。
        
        Args:
            receipt: 订单回执
            event_queue: 事件队列
            t_b: 当前区间的时间上界
        """
        # 统一时间线：直接使用timestamp加延迟
        recv_time = receipt.timestamp + self.config.delay_in
        
        # 设置回执的接收时间
        receipt.recv_time = recv_time
        
        if recv_time <= t_b:
            # 因果一致性保证：确保新事件时间 >= 当前时间
            if recv_time < self.current_time:
                recv_time = self.current_time
                receipt.recv_time = recv_time
            
            heapq.heappush(event_queue, Event(
                time=recv_time,
                event_type=EventType.RECEIPT_TO_STRATEGY,
                data=receipt,
                recv_time=recv_time,
            ))
        else:
            self._queue_receipt_for_future(recv_time, receipt)
        
        self.diagnostics["receipts_generated"] += 1

    def _process_receipt_batch(self, receipts: List[Tuple[int, OrderReceipt]], t_b: int = None) -> bool:
        """处理一批回执，按时间顺序处理并调用策略回调。
        
        使用统一时间线。
        
        Args:
            receipts: (recv_time, receipt) 元组列表
            t_b: 区间结束时间，用于判断是否需要构建tape（可选）
            
        Returns:
            如果提供了t_b，返回是否需要构建tape；否则返回False
        """
        if not receipts:
            return False
        
        receipts.sort(key=lambda item: item[0])
        needs_tape = False
        
        for recv_time, receipt in receipts:
            self.current_time = recv_time
            self.oms.on_receipt(receipt)
            
            if receipt.receipt_type in ["FILL", "PARTIAL"]:
                self.diagnostics["orders_filled"] += 1
            
            # 策略回调
            # 注意：使用current_snapshot，这是当前最新的市场状态
            orders = self.strategy.on_receipt(receipt, self.current_snapshot, self.oms)
            for order in orders:
                self.oms.submit(order, recv_time)
                self.diagnostics["orders_submitted"] += 1
                
                if t_b is not None:
                    arrival_time = recv_time + self.config.delay_out
                    if arrival_time < t_b:
                        needs_tape = True
        
        return needs_tape

    def _process_receipts_without_tape(self, t_a: int, t_b: int) -> bool:
        """仅处理区间内回执，判断是否需要继续构建tape。
        
        使用统一时间线。
        """
        if not self._pending_receipts:
            return False

        in_window: List[Tuple[int, OrderReceipt]] = []
        remaining: List[Tuple[int, OrderReceipt]] = []
        for recv_time, receipt in self._pending_receipts:
            if t_a <= recv_time <= t_b:
                in_window.append((recv_time, receipt))
            else:
                remaining.append((recv_time, receipt))

        self._pending_receipts = remaining
        if not in_window:
            return False

        return self._process_receipt_batch(in_window, t_b)

    def _get_market_qty_from_snapshot(self, order: Order, snapshot: NormalizedSnapshot) -> int:
        """从快照获取订单价位的市场队列深度。

        Args:
            order: 订单
            snapshot: 当前市场快照

        Returns:
            订单价位的市场队列深度
        """
        if snapshot is None:
            return 0

        levels = snapshot.bids if order.side.value == "BUY" else snapshot.asks

        for level in levels:
            if abs(float(level.price) - float(order.price)) < 1e-12:
                return int(level.qty)

        return 0
