from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
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
    LIVE = "LIVE"  # Active in exchange queue

class TimeInForce(Enum):
    """Time-in-force for orders."""
    GTC = "GTC"  # Good-til-canceled (default)
    IOC = "IOC"  # Immediate-or-cancel

class ReceiptType(Enum):
    """Order receipt types for new architecture.
    
    Note: This enum is used for receipt messages in the EventLoop architecture.
    It is separate from OrderStatus to distinguish between:
    - OrderStatus: Internal order state tracking
    - ReceiptType: Message types from exchange to strategy
    
    While some values overlap, they serve different purposes in the system.
    """
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

    # Optional fields
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
    tif: TimeInForce = TimeInForce.GTC  # Time-in-force
    filled_qty: Qty = 0
    status: OrderStatus = OrderStatus.NEW
    create_time: Timestamp = 0
    arrival_time: Optional[Timestamp] = None

    @property
    def remaining_qty(self) -> int:
        return max(0, self.qty - self.filled_qty)

    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.LIVE]

@dataclass
class Fill:
    fill_id: str
    order_id: OrderId
    side: Side
    price: Price
    qty: Qty
    ts: Timestamp
    liquidity: str

@dataclass
class TapeSegment:
    """A time segment of market state for the unified tape model.
    
    Attributes:
        index: Segment index (1-based)
        t_start: Start time (exchtime)
        t_end: End time (exchtime)
        bid_price: Best bid price during this segment
        ask_price: Best ask price during this segment
        trades: Trade volume consumed at each (side, price) -> M_{s,i}(p)
        cancels: Cancel volume at each (side, price) -> C_{s,i}(p)
        net_flow: Net flow (adds - cancels) at each (side, price) -> N_{s,i}(p)
        activation_bid: Set of activated bid prices (top5 from best)
        activation_ask: Set of activated ask prices (top5 from best)
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
    """Order receipt for the new EventLoop architecture"""
    order_id: str
    receipt_type: str   # 'FILL' | 'PARTIAL' | 'CANCELED' | 'REJECTED'
    timestamp: int      # exchtime of the event
    fill_qty: Qty = 0
    fill_price: Price = 0.0
    remaining_qty: Qty = 0
    recv_time: Optional[int] = None  # recvtime when strategy receives this

@dataclass
class FillDetail:
    """Detailed fill information for diagnostics."""
    fill_qty: Qty
    fill_price: Price
    exchtime_fill: int
    recvtime_recv: int

@dataclass
class OrderDiagnostics:
    """Diagnostic information for an order."""
    order_id: str
    side: Side
    price: Price
    qty: Qty
    exchtime_arr: int
    recvtime_send: int
    status: OrderStatus
    filled_qty: Qty
    fills: List[FillDetail] = field(default_factory=list)
    activation_time_ratio: float = 0.0  # Time ratio in activation window
    x_threshold: float = 0.0  # pos + qty (threshold for complete fill)
    x_final: float = 0.0  # X(px, T_B) at interval end
    q_truncation_count: int = 0  # Number of Q < 0 truncations
