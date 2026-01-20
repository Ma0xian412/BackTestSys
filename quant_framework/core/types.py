"""核心类型定义模块。

本模块定义回测系统中使用的基础类型：
- 基本类型别名：Price, Qty, OrderId, Timestamp
- 枚举类型：Side, OrderStatus, TimeInForce, ReceiptType
- 数据类：Level, NormalizedSnapshot, Order, Fill, TapeSegment, OrderReceipt等
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum

# 基本类型别名
Price = float      # 价格类型
Qty = int          # 数量类型
OrderId = str      # 订单ID类型
Timestamp = int    # 时间戳类型


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


class ReceiptType(Enum):
    """订单回执类型枚举。

    用于EventLoop架构中的回执消息。
    与OrderStatus分开定义，用于区分：
    - OrderStatus: 内部订单状态跟踪
    - ReceiptType: 交易所到策略的消息类型

    虽然部分值重叠，但在系统中服务于不同目的。
    """
    FILL = "FILL"          # 完全成交
    PARTIAL = "PARTIAL"    # 部分成交
    CANCELED = "CANCELED"  # 已撤销
    REJECTED = "REJECTED"  # 已拒绝

@dataclass
class Level:
    """价格档位。

    Attributes:
        price: 价格
        qty: 数量
    """
    price: Price
    qty: Qty


@dataclass
class NormalizedSnapshot:
    """标准化快照数据。

    Attributes:
        ts_exch: 交易所时间戳
        bids: 买盘档位列表
        asks: 卖盘档位列表
        last_vol_split: 最近成交量在各价位的分布
        ts_recv: 接收时间戳（可选）
        last: 最新价（可选）
        volume: 成交量（可选）
        turnover: 成交额（可选）
        average_price: 均价（可选）
    """
    ts_exch: Timestamp
    bids: List[Level]
    asks: List[Level]
    last_vol_split: List[Tuple[Price, Qty]] = field(default_factory=list)

    # 可选字段
    ts_recv: Optional[Timestamp] = None
    last: Optional[Price] = None
    volume: Optional[int] = None
    turnover: Optional[float] = None
    average_price: Optional[float] = None


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
