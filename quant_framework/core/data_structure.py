"""核心类型定义模块。"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum

# 基本类型别名
Price = float      # 价格类型
Qty = int          # 数量类型
OrderId = str      # 订单ID类型
Timestamp = int    # 时间戳类型（单位：tick，每tick=100ns，从0000-00-00开始计数）

# 快照推送最小间隔（tick单位，500ms = 5_000_000 ticks）
# 1ms = 10_000 ticks (100ns per tick)
TICK_PER_MS = 10_000
SNAPSHOT_MIN_INTERVAL_TICK = 500 * TICK_PER_MS  # 500ms in ticks

# 快照时间容差（tick单位）
# 由于RecvTick可能存在误差，相邻快照间隔不一定刚好是500ms
# 默认容差为10ms = 100_000 ticks
DEFAULT_SNAPSHOT_TOLERANCE_TICK = 10 * TICK_PER_MS  # 10ms in ticks


class Side(Enum):
    """买卖方向枚举。"""
    BUY = "BUY"    # 买入
    SELL = "SELL"  # 卖出


class OrderStatus(Enum):
    """订单状态枚举。"""
    NEW = "NEW"                        # 新订单
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # 部分成交
    FILLED = "FILLED"                  # 完全成交
    CANCELED = "CANCELED"              # 已撤销
    REJECTED = "REJECTED"              # 已拒绝
    LIVE = "LIVE"                      # 在交易所队列中活跃


class TimeInForce(Enum):
    """订单有效期类型枚举。"""
    GTC = "GTC"  # 撤销前有效（默认）
    IOC = "IOC"  # 立即成交否则撤销


class RequestType(Enum):
    """策略请求类型枚举。
    
    策略可以发出两类请求：
    - ORDER: 挂单请求（新订单）
    - CANCEL: 撤单请求
    """
    ORDER = "ORDER"    # 挂单请求
    CANCEL = "CANCEL"  # 撤单请求


class ReceiptType(Enum):
    """订单回执类型枚举。

    用于EventLoop架构中的回执消息。
    
    回执分为两大类：
    1. 成交类回执：FILL, PARTIAL
    2. 撤单类回执：CANCELED, REJECTED
    
    撤单回执的判断方法：
    - fill_qty > 0: 撤单成功，且在撤单前有部分成交
    - fill_qty == 0: 撤单成功，撤单前无成交
    - receipt_type == REJECTED: 撤单失败（订单不存在或已完成）
    """
    FILL = "FILL"          # 完全成交
    PARTIAL = "PARTIAL"    # 部分成交
    CANCELED = "CANCELED"  # 已撤销（撤单成功）
    REJECTED = "REJECTED"  # 已拒绝（撤单失败或订单被拒）

@dataclass(frozen=True)
class Level:
    """价格档位。

    Attributes:
        price: 价格
        qty: 数量
    """
    price: Price
    qty: Qty


@dataclass(frozen=True)
class NormalizedSnapshot:
    """标准化快照数据。
    
    所有时间戳使用统一的recv timeline（ts_recv），单位为tick（每tick=100ns）。
    ts_exch保留用于记录交易所时间，但所有事件调度使用ts_recv。

    Attributes:
        ts_recv: 接收时间戳（主时间线，tick单位，必填）
        bids: 买盘档位列表
        asks: 卖盘档位列表
        last_vol_split: 最近成交量在各价位的分布
        ts_exch: 交易所时间戳（可选，仅用于记录）
        last: 最新价（可选）
        volume: 成交量（可选）
        turnover: 成交额（可选）
        average_price: 均价（可选）
    """
    ts_recv: Timestamp  # 主时间线（必填）
    bids: Tuple[Level, ...]
    asks: Tuple[Level, ...]
    last_vol_split: Tuple[Tuple[Price, Qty], ...] = field(default_factory=tuple)

    # 可选字段
    ts_exch: Optional[Timestamp] = None  # 交易所时间戳（仅记录）
    last: Optional[Price] = None
    volume: Optional[int] = None
    turnover: Optional[float] = None
    average_price: Optional[float] = None

    def __post_init__(self) -> None:
        # 允许调用方传 list，内部统一冻结为 tuple。
        object.__setattr__(self, "bids", tuple(self.bids))
        object.__setattr__(self, "asks", tuple(self.asks))
        object.__setattr__(self, "last_vol_split", tuple(self.last_vol_split))

    @property
    def best_bid(self) -> Optional[Price]:
        if not self.bids:
            return None
        return max(level.price for level in self.bids)

    @property
    def best_ask(self) -> Optional[Price]:
        if not self.asks:
            return None
        return min(level.price for level in self.asks)

    @property
    def mid_price(self) -> Optional[Price]:
        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    @property
    def spread(self) -> Optional[Price]:
        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None:
            return None
        return ask - bid


@dataclass
class Order:
    """订单数据。

    Attributes:
        order_id: 订单ID
        side: 买卖方向
        price: 价格
        qty: 数量
        type: 订单类型（默认LIMIT）
        tif: 有效期类型
        filled_qty: 已成交数量
        status: 订单状态
        create_time: 创建时间
        arrival_time: 到达交易所时间（可选）
    """
    order_id: OrderId
    side: Side
    price: Price
    qty: Qty
    type: str = "LIMIT"
    tif: TimeInForce = TimeInForce.GTC
    filled_qty: Qty = 0
    status: OrderStatus = OrderStatus.NEW
    create_time: Timestamp = 0
    arrival_time: Optional[Timestamp] = None

    @property
    def remaining_qty(self) -> int:
        """获取剩余数量。"""
        return max(0, self.qty - self.filled_qty)

    @property
    def is_active(self) -> bool:
        """判断订单是否活跃。"""
        return self.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.LIVE]


@dataclass
class CancelRequest:
    """撤单请求。
    
    策略发出的撤单请求，包含要撤销的订单ID。
    
    Attributes:
        order_id: 要撤销的订单ID
        create_time: 撤单请求创建时间
    """
    order_id: OrderId
    create_time: Timestamp = 0


@dataclass
class Fill:
    """成交记录。

    Attributes:
        fill_id: 成交ID
        order_id: 订单ID
        side: 买卖方向
        price: 成交价
        qty: 成交数量
        ts: 成交时间戳
        liquidity: 流动性类型（maker/taker）
    """
    fill_id: str
    order_id: OrderId
    side: Side
    price: Price
    qty: Qty
    ts: Timestamp
    liquidity: str


@dataclass
class TapeSegment:
    """统一Tape模型的时间段。

    Attributes:
        index: 段索引（从1开始）
        t_start: 开始时间（交易所时间）
        t_end: 结束时间（交易所时间）
        bid_price: 该段内的最优买价
        ask_price: 该段内的最优卖价
        trades: 各(方向, 价格)的成交量 -> M_{s,i}(p)
        cancels: 各(方向, 价格)的撤单量 -> C_{s,i}(p)
        net_flow: 各(方向, 价格)的净流入量（挂单-撤单）-> N_{s,i}(p)
        activation_bid: 激活的买价集合（从最优价起的前5档）
        activation_ask: 激活的卖价集合（从最优价起的前5档）
    """
    index: int
    t_start: int
    t_end: int
    bid_price: Price
    ask_price: Price
    trades: Dict[Tuple[Side, Price], Qty] = field(default_factory=dict)
    cancels: Dict[Tuple[Side, Price], Qty] = field(default_factory=dict)
    net_flow: Dict[Tuple[Side, Price], Qty] = field(default_factory=dict)
    activation_bid: Set[Price] = field(default_factory=set)
    activation_ask: Set[Price] = field(default_factory=set)


@dataclass
class OrderReceipt:
    """订单回执（用于EventLoop架构）。
    
    回执分为两大类：
    
    1. 成交类回执（对应挂单请求）:
       - FILL: 完全成交，fill_qty = 订单数量
       - PARTIAL: 部分成交，fill_qty = 已成交数量，remaining_qty = 剩余数量
    
    2. 撤单类回执（对应撤单请求）:
       - CANCELED: 撤单成功
         - fill_qty > 0: 撤单前有部分成交
         - fill_qty == 0: 撤单前无成交
       - REJECTED: 撤单失败（订单不存在、已完成或已撤销）

    Attributes:
        order_id: 订单ID
        receipt_type: 回执类型（FILL/PARTIAL/CANCELED/REJECTED）
        timestamp: 事件发生的交易所时间
        fill_qty: 成交数量（默认0）
        fill_price: 成交价格（默认0.0）
        remaining_qty: 剩余数量（默认0）
        recv_time: 策略接收到该回执的时间（可选）
    """
    order_id: str
    receipt_type: str
    timestamp: int
    fill_qty: Qty = 0
    fill_price: Price = 0.0
    remaining_qty: Qty = 0
    recv_time: Optional[int] = None


@dataclass
class FillDetail:
    """成交详情（用于诊断）。

    Attributes:
        fill_qty: 成交数量
        fill_price: 成交价格
        exchtime_fill: 成交的交易所时间
        recvtime_recv: 收到回执的接收时间
    """
    fill_qty: Qty
    fill_price: Price
    exchtime_fill: int
    recvtime_recv: int


@dataclass
class OrderDiagnostics:
    """订单诊断信息。

    Attributes:
        order_id: 订单ID
        side: 买卖方向
        price: 价格
        qty: 数量
        exchtime_arr: 到达交易所的时间
        recvtime_send: 发送订单的接收时间
        status: 订单状态
        filled_qty: 已成交数量
        fills: 成交详情列表
        activation_time_ratio: 在激活窗口内的时间比例
        x_threshold: pos + qty（完全成交的阈值）
        x_final: 区间结束时的X(px, T_B)
        q_truncation_count: Q < 0 截断的次数
    """
    order_id: str
    side: Side
    price: Price
    qty: Qty
    exchtime_arr: int
    recvtime_send: int
    status: OrderStatus
    filled_qty: Qty
    fills: List[FillDetail] = field(default_factory=list)
    activation_time_ratio: float = 0.0
    x_threshold: float = 0.0
    x_final: float = 0.0
    q_truncation_count: int = 0


@dataclass(frozen=True)
class OrderSnapshot:
    """订单只读快照。"""

    order_id: str
    side: Side
    price: Price
    qty: Qty
    type: str
    tif: TimeInForce
    filled_qty: Qty
    status: OrderStatus
    create_time: Timestamp
    arrival_time: Optional[Timestamp] = None

    @property
    def remaining_qty(self) -> int:
        return max(0, self.qty - self.filled_qty)

    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.LIVE]


@dataclass(frozen=True)
class PortfolioSnapshot:
    """投资组合只读快照。"""

    cash: float
    position: int
    realized_pnl: float


class ReadOnlyOMSView:
    """OMS 只读访问视图。"""

    def __init__(self, oms: object):
        self._oms = oms

    def get_active_orders(self) -> List[OrderSnapshot]:
        orders = self._oms.get_active_orders()
        return [self._to_order_snapshot(o) for o in orders]

    def get_order(self, order_id: str) -> Optional[OrderSnapshot]:
        order = self._oms.get_order(order_id)
        if order is None:
            return None
        return self._to_order_snapshot(order)

    def get_portfolio(self) -> PortfolioSnapshot:
        portfolio = self._oms.portfolio
        return PortfolioSnapshot(
            cash=portfolio.cash,
            position=portfolio.position,
            realized_pnl=portfolio.realized_pnl,
        )

    @staticmethod
    def _to_order_snapshot(order) -> OrderSnapshot:
        return OrderSnapshot(
            order_id=order.order_id,
            side=order.side,
            price=order.price,
            qty=order.qty,
            type=order.type,
            tif=order.tif,
            filled_qty=order.filled_qty,
            status=order.status,
            create_time=order.create_time,
            arrival_time=order.arrival_time,
        )


EVENT_KIND_SNAPSHOT_ARRIVAL = "SnapshotArrival"
EVENT_KIND_ACTION_ARRIVAL = "ActionArrival"
EVENT_KIND_RECEIPT_DELIVERY = "ReceiptDelivery"


class ActionType(Enum):
    """策略动作类型。"""

    PLACE_ORDER = "PLACE_ORDER"
    CANCEL_ORDER = "CANCEL_ORDER"


@dataclass
class Action:
    """策略动作纯数据结构。"""

    action_type: ActionType
    create_time: int = 0
    payload: Any = None

    def get_type(self) -> ActionType:
        return self.action_type

    def set_type(self, action_type: ActionType) -> None:
        self.action_type = action_type

    def get_create_time(self) -> int:
        return int(self.create_time)

    def set_create_time(self, create_time: int) -> None:
        self.create_time = int(create_time)

    def get_payload(self) -> Any:
        return self.payload

    def set_payload(self, payload: Any) -> None:
        self.payload = payload


_event_seq_counter = 0


def _next_seq() -> int:
    global _event_seq_counter
    _event_seq_counter += 1
    return _event_seq_counter


def reset_event_seq() -> None:
    global _event_seq_counter
    _event_seq_counter = 0


@dataclass
class Event:
    """调度器中的统一事件。"""

    time: int
    kind: str
    payload: object
    priority: int = 0
    seq: int = field(default_factory=_next_seq)

    def __lt__(self, other: "Event") -> bool:
        if self.time != other.time:
            return self.time < other.time
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.seq < other.seq


@dataclass(frozen=True)
class StrategyContext:
    """策略回调上下文（只读）。"""

    t: int
    snapshot: Optional[NormalizedSnapshot]
    omsView: ReadOnlyOMSView


@dataclass
class EventSpecRegistry:
    """事件规范与优先级注册中心。"""

    _priorities: Dict[str, int]
    _validators: Dict[str, Callable[[object], bool]]

    @classmethod
    def default(cls) -> "EventSpecRegistry":
        return cls(
            _priorities={
                EVENT_KIND_SNAPSHOT_ARRIVAL: 10,
                EVENT_KIND_ACTION_ARRIVAL: 20,
                EVENT_KIND_RECEIPT_DELIVERY: 30,
            },
            _validators={
                EVENT_KIND_SNAPSHOT_ARRIVAL: lambda payload: isinstance(payload, NormalizedSnapshot),
                EVENT_KIND_ACTION_ARRIVAL: lambda payload: isinstance(payload, Action),
                EVENT_KIND_RECEIPT_DELIVERY: lambda payload: isinstance(payload, OrderReceipt),
            },
        )

    def priorityOf(self, kind: str) -> int:
        return self._priorities.get(kind, 99)

    def validate(self, kind: str, payload: object) -> bool:
        validator = self._validators.get(kind)
        if validator is None:
            return True
        return bool(validator(payload))

    def register(self, kind: str, priority: int, validator: Optional[Callable[[object], bool]] = None) -> None:
        self._priorities[kind] = priority
        if validator is not None:
            self._validators[kind] = validator


@dataclass(frozen=True)
class StepOutcome:
    """执行场所 step 结果。"""

    next_time: int
    receipts_generated: List[OrderReceipt]


@dataclass
class RuntimeContext:
    """运行上下文（由 CompositionRoot 组装）。"""

    feed: Any
    venue: Any
    strategy: Any
    oms: Any
    timeModel: Any
    obs: Any
    dispatcher: Any
    eventSpec: EventSpecRegistry
    last_snapshot: Optional[NormalizedSnapshot] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
