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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum, auto

from ..core.interfaces import IMarketDataFeed, ITapeBuilder, IExchangeSimulator, IStrategy, IOrderManager
from ..core.types import NormalizedSnapshot, Order, OrderReceipt, TapeSegment


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

# 全局序列号计数器
_event_seq_counter = 0


def _get_next_seq() -> int:
    """获取下一个事件序列号。"""
    global _event_seq_counter
    _event_seq_counter += 1
    return _event_seq_counter


def reset_event_seq_counter() -> None:
    """重置事件序列号计数器（用于测试）。"""
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
            self.priority = EVENT_TYPE_PRIORITY.get(self.event_type, 99)
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
    """双时间线配置。

    映射公式:
    - recvtime = a * exchtime + b
    - exchtime = (recvtime - b) / a

    要求: a > 0（严格单调）

    Attributes:
        a: 缩放因子（默认1.0）
        b: 偏移量（默认0.0）
    """
    a: float = 1.0
    b: float = 0.0

    def exchtime_to_recvtime(self, exchtime: int) -> int:
        """将交易所时间转换为接收时间。"""
        return int(self.a * exchtime + self.b)

    def recvtime_to_exchtime(self, recvtime: int) -> int:
        """将接收时间转换为交易所时间。"""
        if self.a <= 0:
            raise ValueError("时间线缩放因子'a'必须为正数")
        return int((recvtime - self.b) / self.a)


@dataclass
class RunnerConfig:
    """事件循环运行器配置。

    Attributes:
        delay_out: 策略 -> 交易所的延迟（recvtime单位）
        delay_in: 交易所 -> 策略的延迟（recvtime单位）
        timeline: 双时间线映射配置
    """
    delay_out: int = 0
    delay_in: int = 0
    timeline: TimelineConfig = None

    def __post_init__(self):
        if self.timeline is None:
            self.timeline = TimelineConfig()


class EventLoopRunner:
    """事件循环运行器（支持双时间线）。

    协调Tape构建、交易所仿真和策略执行的事件驱动循环。

    主要功能：
    - 双时间线支持：exchtime（交易所）和recvtime（策略）
    - 延迟处理：delay_out（策略->交易所），delay_in（交易所->策略）
    - 线性时间映射：recvtime = a * exchtime + b
    """

    def __init__(
        self,
        feed: IMarketDataFeed,
        tape_builder: ITapeBuilder,
        exchange: IExchangeSimulator,
        strategy: IStrategy,
        oms: IOrderManager,
        config: RunnerConfig = None,
    ):
        """初始化运行器。

        Args:
            feed: 行情数据源
            tape_builder: Tape构建器
            exchange: 交易所模拟器
            strategy: 策略
            oms: 订单管理器
            config: 运行器配置
        """
        self.feed = feed
        self.tape_builder = tape_builder
        self.exchange = exchange
        self.strategy = strategy
        self.oms = oms
        self.config = config or RunnerConfig()

        # 验证时间线配置
        if self.config.timeline.a <= 0:
            raise ValueError("时间线缩放因子'a'必须为正数")

        # 当前状态
        self.current_exchtime = 0
        self.current_recvtime = 0
        self.current_snapshot: Optional[NormalizedSnapshot] = None
        self._pending_receipts: List[Tuple[int, OrderReceipt]] = []

        # 诊断信息
        self.diagnostics: Dict[str, Any] = {
            "intervals_processed": 0,
            "orders_submitted": 0,
            "orders_filled": 0,
            "receipts_generated": 0,
        }

    def run(self) -> Dict[str, Any]:
        """运行回测。

        Returns:
            包含回测结果的字典
        """
        self.feed.reset()
        prev = self.feed.next()

        if prev is None:
            return {"error": "No data"}

        self.current_snapshot = prev
        self.current_exchtime = int(prev.ts_exch)
        self.current_recvtime = self.config.timeline.exchtime_to_recvtime(self.current_exchtime)

        # 初始快照回调
        orders = self.strategy.on_snapshot(prev, self.oms)
        for order in orders:
            self.oms.submit(order, self.current_recvtime)
            self.diagnostics["orders_submitted"] += 1

        interval_count = 0

        while True:
            curr = self.feed.next()
            if curr is None:
                break

            # 运行一个区间
            self._run_interval(prev, curr)

            prev = curr
            self.current_snapshot = curr
            interval_count += 1

        self.diagnostics["intervals_processed"] = interval_count

        return {
            "intervals": interval_count,
            "final_exchtime": self.current_exchtime,
            "final_recvtime": self.current_recvtime,
            "diagnostics": self.diagnostics,
        }

    def _run_interval(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot) -> None:
        """运行一个区间[prev, curr]的事件循环。
        
        通过_schedule_pending_receipts确保所有时间上早于t_a的事件在区间开始时被处理。

        Args:
            prev: 前一个快照（在T_A时刻）
            curr: 当前快照（在T_B时刻）
        """
        t_a = int(prev.ts_exch)
        t_b = int(curr.ts_exch)

        if t_b <= t_a:
            return

        # 转换为接收时间
        recv_a = self.config.timeline.exchtime_to_recvtime(t_a)
        recv_b = self.config.timeline.exchtime_to_recvtime(t_b)

        if not self._has_order_events_between(recv_a, recv_b, t_a, t_b):
            needs_tape = self._process_receipts_without_tape(recv_a, recv_b, t_b)
            if not needs_tape:
                self._advance_without_tape(curr, t_b, recv_b)
                return

        # 构建该区间的Tape
        tape = self.tape_builder.build(prev, curr)

        if not tape:
            # 无事件，直接推进到curr
            self._advance_without_tape(curr, t_b, recv_b)
            return

        # 为新区间重置交易所并设置Tape
        self.exchange.reset()

        # 如果交易所支持set_tape方法则调用
        if hasattr(self.exchange, 'set_tape'):
            self.exchange.set_tape(tape, t_a, t_b)

        # 初始化事件队列（使用exchtime）
        event_queue: List[Event] = []

        # 添加段结束事件
        for seg in tape:
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

        self._schedule_pending_receipts(event_queue, recv_a, recv_b)

        # 获取OMS中的待处理订单并调度到达事件
        # 确保所有时间上早于t_a的订单都已被处理
        pending_orders = self.oms.get_active_orders()

        for order in pending_orders:
            # 计算订单到达交易所的时间
            # 注意：时间戳假设为整数（毫秒或微秒精度的时间戳）
            # Note: timestamps are assumed to be integers (ms or μs precision)
            recv_arr = int(order.create_time) + int(self.config.delay_out)
            exchtime_arr = self.config.timeline.recvtime_to_exchtime(recv_arr)
            
            # 跳过已经有arrival_time且已处理过的订单
            if order.arrival_time is not None:
                # 订单已经到达过交易所，不需要重复调度
                continue
            
            # 只处理到达时间在区间内的订单
            if exchtime_arr >= t_a and exchtime_arr < t_b:
                heapq.heappush(event_queue, Event(
                    time=exchtime_arr,
                    event_type=EventType.ORDER_ARRIVAL,
                    data=order,
                ))

        # 事件循环 - 使用"peek, advance, pop batch"范式
        # 避免"生成过去事件"的因果反转问题
        current_seg_idx = 0
        last_exchtime = t_a

        while event_queue:
            # Step 1: Peek 下一个事件的时间（不弹出）
            t_next = event_queue[0].time
            
            # Step 2: 将交易所推进到t_next，期间产生的回执会被调度
            # 如果回执时间 <= t_next，会在同一轮被处理
            if t_next > last_exchtime and current_seg_idx < len(tape):
                # 处理到t_next为止的所有完整段
                while current_seg_idx < len(tape) and tape[current_seg_idx].t_end <= t_next:
                    seg = tape[current_seg_idx]
                    receipts = self.exchange.advance(last_exchtime, seg.t_end, seg)

                    # 调度回执发送到策略
                    for receipt in receipts:
                        self._schedule_receipt(receipt, event_queue, recv_b)

                    last_exchtime = seg.t_end
                    current_seg_idx += 1

                # 如果t_next落在当前段内部，需要先将交易所推进到t_next
                # 确保t_next时刻的事件发生之前的成交已被正确处理
                if current_seg_idx < len(tape):
                    seg = tape[current_seg_idx]
                    if seg.t_start <= t_next < seg.t_end and t_next > last_exchtime:
                        # 推进到t_next（段内推进）
                        receipts = self.exchange.advance(last_exchtime, t_next, seg)
                        for receipt in receipts:
                            self._schedule_receipt(receipt, event_queue, recv_b)
                        last_exchtime = t_next

            # Step 3: Pop并处理所有time==t_next的事件（批处理）
            # 这样新生成的回执（如果时间 <= t_next）会在heap中排在后面
            # 但因为我们是批处理同一时刻的事件，它们会被正确处理
            events_at_t_next = []
            while event_queue and event_queue[0].time == t_next:
                events_at_t_next.append(heapq.heappop(event_queue))
            
            # 处理同一时刻的所有事件（已按priority和seq排序）
            for event in events_at_t_next:
                self._process_single_event(event, event_queue, tape, t_b, recv_b)

    def _process_single_event(
        self,
        event: Event,
        event_queue: List[Event],
        tape: List[TapeSegment],
        t_b: int,
        recv_b: int,
    ) -> None:
        """处理单个事件。
        
        Args:
            event: 要处理的事件
            event_queue: 事件队列（用于调度新事件）
            tape: Tape段列表
            t_b: 区间结束时间
            recv_b: 区间结束的接收时间
        """
        # 更新当前时间
        self.current_exchtime = event.time
        
        if event.event_type == EventType.ORDER_ARRIVAL:
            # 订单到达：使用exchtime计算recvtime
            self.current_recvtime = self.config.timeline.exchtime_to_recvtime(event.time)
            order = event.data
            order.arrival_time = event.time

            # 从当前快照获取订单价位的市场队列深度
            market_qty = self._get_market_qty_from_snapshot(order, self.current_snapshot)
            receipt = self.exchange.on_order_arrival(order, event.time, market_qty)

            if receipt:
                # 所有回执都走统一RECEIPT_TO_STRATEGY调度
                # 在recv_time时刻再oms.on_receipt，确保延迟因果一致性
                self._schedule_receipt(receipt, event_queue, recv_b)

        elif event.event_type == EventType.RECEIPT_TO_STRATEGY:
            receipt = event.data
            # Issue 3 修复：使用event.recv_time或receipt.recv_time作为权威的策略侧时间
            # 而不是从event.time推导
            if event.recv_time is not None:
                self.current_recvtime = event.recv_time
            elif receipt.recv_time is not None:
                self.current_recvtime = receipt.recv_time
            else:
                # 回退：从exchtime计算（理论上不应该发生）
                self.current_recvtime = self.config.timeline.exchtime_to_recvtime(event.time)
            
            self.oms.on_receipt(receipt)

            if receipt.receipt_type in ["FILL", "PARTIAL"]:
                self.diagnostics["orders_filled"] += 1

            # 策略回调
            orders = self.strategy.on_receipt(receipt, self.current_snapshot, self.oms)
            for order in orders:
                self.oms.submit(order, self.current_recvtime)
                self.diagnostics["orders_submitted"] += 1

                # 调度订单到达
                recv_arr = self.current_recvtime + self.config.delay_out
                exchtime_arr = self.config.timeline.recvtime_to_exchtime(recv_arr)

                if exchtime_arr < t_b:
                    heapq.heappush(event_queue, Event(
                        time=exchtime_arr,
                        event_type=EventType.ORDER_ARRIVAL,
                        data=order,
                    ))

        elif event.event_type == EventType.INTERVAL_END:
            snapshot = event.data

            # 在边界对齐交易所状态
            self.exchange.align_at_boundary(snapshot)

            # 策略回调
            self.current_snapshot = snapshot
            self.current_recvtime = self.config.timeline.exchtime_to_recvtime(event.time)

            orders = self.strategy.on_snapshot(snapshot, self.oms)
            for order in orders:
                self.oms.submit(order, self.current_recvtime)
                self.diagnostics["orders_submitted"] += 1

    def _advance_without_tape(self, snapshot: NormalizedSnapshot, exchtime: int, recvtime: int) -> None:
        """无订单区间快速推进并触发快照回调。"""
        self.current_exchtime = exchtime
        self.current_recvtime = recvtime
        self.current_snapshot = snapshot
        orders = self.strategy.on_snapshot(snapshot, self.oms)
        for order in orders:
            self.oms.submit(order, recvtime)
            self.diagnostics["orders_submitted"] += 1
    
    def _has_order_events_between(self, recv_a: int, recv_b: int, t_a: int, t_b: int) -> bool:
        """判断区间内是否存在需要处理的订单事件。"""
        active_orders = self.oms.get_active_orders()
        if not active_orders:
            return False

        for order in active_orders:
            if order.arrival_time is not None:
                if order.arrival_time < t_b:
                    return True
                continue

            recv_arr = int(order.create_time) + int(self.config.delay_out)
            if recv_arr < recv_a:
                continue

            exchtime_arr = self.config.timeline.recvtime_to_exchtime(recv_arr)
            if exchtime_arr < t_b:
                return True

        return False

    def _schedule_pending_receipts(self, event_queue: List[Event], recv_a: int, recv_b: int) -> None:
        """将到达时间落在区间内的回执加入事件队列，并处理所有早于recv_a的历史回执。
        
        确保所有时间上早于当前区间的回执都被正确处理，维护因果一致性。
        """
        remaining: List[Tuple[int, OrderReceipt]] = []
        historical: List[Tuple[int, OrderReceipt]] = []
        
        for recv_time, receipt in self._pending_receipts:
            if recv_time < recv_a:
                # 历史回执：早于当前区间，需要立即处理
                historical.append((recv_time, receipt))
            elif recv_a <= recv_time <= recv_b:
                # 当前区间内的回执：加入事件队列
                exchtime_recv = self.config.timeline.recvtime_to_exchtime(recv_time)
                heapq.heappush(event_queue, Event(
                    time=exchtime_recv,
                    event_type=EventType.RECEIPT_TO_STRATEGY,
                    data=receipt,
                    recv_time=recv_time,  # 携带策略侧接收时间
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

    def _schedule_receipt(self, receipt: OrderReceipt, event_queue: List[Event], recv_b: int) -> None:
        """统一调度回执到策略的事件。
        
        所有回执都通过此方法调度，确保delay_in延迟因果一致性。
        回执会在recv_time时刻通过RECEIPT_TO_STRATEGY事件传递给策略。
        
        Args:
            receipt: 订单回执
            event_queue: 事件队列
            recv_b: 当前区间的接收时间上界
        """
        recv_fill = self.config.timeline.exchtime_to_recvtime(receipt.timestamp)
        recv_recv = recv_fill + self.config.delay_in
        
        # 设置回执的接收时间
        receipt.recv_time = recv_recv
        
        if recv_recv <= recv_b:
            # 转换回exchtime用于事件队列
            exchtime_recv = self.config.timeline.recvtime_to_exchtime(recv_recv)
            heapq.heappush(event_queue, Event(
                time=exchtime_recv,
                event_type=EventType.RECEIPT_TO_STRATEGY,
                data=receipt,
                recv_time=recv_recv,  # 携带策略侧接收时间
            ))
        else:
            self._queue_receipt_for_future(recv_recv, receipt)
        
        self.diagnostics["receipts_generated"] += 1

    def _process_receipt_batch(self, receipts: List[Tuple[int, OrderReceipt]], t_b: int = None) -> bool:
        """处理一批回执，按时间顺序处理并调用策略回调。
        
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
            self.current_recvtime = recv_time
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
                    recv_arr = recv_time + self.config.delay_out
                    exchtime_arr = self.config.timeline.recvtime_to_exchtime(recv_arr)
                    if exchtime_arr < t_b:
                        needs_tape = True
        
        return needs_tape

    def _process_receipts_without_tape(self, recv_a: int, recv_b: int, t_b: int) -> bool:
        """仅处理区间内回执，判断是否需要继续构建tape。"""
        if not self._pending_receipts:
            return False

        in_window: List[Tuple[int, OrderReceipt]] = []
        remaining: List[Tuple[int, OrderReceipt]] = []
        for recv_time, receipt in self._pending_receipts:
            if recv_a <= recv_time <= recv_b:
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
