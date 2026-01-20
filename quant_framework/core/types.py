from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum

Price = float
Qty = int
OrderId = str
Timestamp = int

class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"

class ReceiptType(Enum):
    """Order receipt types for new architecture"""
    FILL = "FILL"
    PARTIAL = "PARTIAL"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"

@dataclass
class Level:
    price: Price
    qty: Qty

@dataclass
class NormalizedSnapshot:
    ts_exch: Timestamp
    bids: List[Level]
    asks: List[Level]
    last_vol_split: List[Tuple[Price, Qty]] = field(default_factory=list)

    # 可选字段：仅在数据源提供时填充，用于更强的桥约束与统计
    ts_recv: Optional[Timestamp] = None
    last: Optional[Price] = None
    volume: Optional[int] = None
    turnover: Optional[float] = None
    average_price: Optional[float] = None

@dataclass
class Order:
    order_id: OrderId
    side: Side
    price: Price
    qty: Qty
    type: str = "LIMIT"
    filled_qty: Qty = 0
    status: OrderStatus = OrderStatus.NEW
    # create_time: 策略“提交/生成”该订单的时间（本地时间轴，单位与 ts_exch 一致）
    create_time: Timestamp = 0
    # arrival_time: 订单“到达交易所/进入撮合”的时间（受 order_latency 影响）
    # 仅在订单实际进入撮合器时由 runner 填充。
    arrival_time: Optional[Timestamp] = None

    @property
    def remaining_qty(self) -> int:
        return max(0, self.qty - self.filled_qty)
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]

@dataclass
class Fill:
    fill_id: str
    order_id: OrderId
    side: Side
    price: Price
    qty: Qty
    ts: Timestamp
    liquidity: str

@dataclass(frozen=True)
class TapeSegment:
    """A time segment of market state for the unified tape model"""
    index: int
    t_start: int
    t_end: int
    bid_price: Price
    ask_price: Price
    trades: Dict[Tuple[Side, Price], Qty] = field(default_factory=dict)    # M_{s,i}(p)
    cancels: Dict[Tuple[Side, Price], Qty] = field(default_factory=dict)   # C_{s,i}(p)

@dataclass
class OrderReceipt:
    """Order receipt for the new EventLoop architecture"""
    order_id: str
    receipt_type: str   # 'FILL' | 'PARTIAL' | 'CANCELED' | 'REJECTED'
    timestamp: int
    fill_qty: Qty = 0
    fill_price: Price = 0.0
    remaining_qty: Qty = 0