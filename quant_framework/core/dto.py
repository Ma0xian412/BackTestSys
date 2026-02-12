"""数据传输对象(DTO)和只读视图模块。

本模块实现策略与其他组件之间的解耦：
- SnapshotDTO: 快照数据传输对象，提供不可变的市场数据
- OrderInfoDTO: 订单信息传输对象，提供只读的订单状态
- PortfolioDTO: 投资组合传输对象，提供只读的持仓信息
- ReadOnlyOMSView: OMS的只读视图，策略只能查询不能修改订单状态

所有时间使用统一的recv timeline (ts_recv)，单位为tick（每tick=100ns）。
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Sequence
from .types import Price, Qty, Timestamp, Side, OrderStatus, TimeInForce


@dataclass(frozen=True)
class LevelDTO:
    """价格档位数据传输对象（不可变）。

    Attributes:
        price: 价格
        qty: 数量
    """
    price: Price
    qty: Qty


@dataclass(frozen=True)
class SnapshotDTO:
    """快照数据传输对象（不可变）。

    为策略提供只读的市场快照数据，确保策略无法修改原始数据。
    使用统一的recv timeline (ts_recv)作为主时间线。

    Attributes:
        ts_recv: 接收时间戳（主时间线，tick单位）
        bids: 买盘档位列表（只读）
        asks: 卖盘档位列表（只读）
        last_vol_split: 最近成交量分布（只读）
        ts_exch: 交易所时间戳（可选，仅记录）
        last: 最新价（可选）
        volume: 成交量（可选）
        turnover: 成交额（可选）
        average_price: 均价（可选）
    """
    ts_recv: Timestamp  # 主时间线
    bids: Tuple[LevelDTO, ...]
    asks: Tuple[LevelDTO, ...]
    last_vol_split: Tuple[Tuple[Price, Qty], ...] = field(default_factory=tuple)
    ts_exch: Optional[Timestamp] = None  # 可选
    last: Optional[Price] = None
    volume: Optional[int] = None
    turnover: Optional[float] = None
    average_price: Optional[float] = None

    @property
    def best_bid(self) -> Optional[Price]:
        """获取最优买价。"""
        if not self.bids:
            return None
        return max(level.price for level in self.bids)

    @property
    def best_ask(self) -> Optional[Price]:
        """获取最优卖价。"""
        if not self.asks:
            return None
        return min(level.price for level in self.asks)

    @property
    def mid_price(self) -> Optional[Price]:
        """获取中间价。"""
        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    @property
    def spread(self) -> Optional[Price]:
        """获取买卖价差。"""
        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None:
            return None
        return ask - bid


@dataclass(frozen=True)
class OrderInfoDTO:
    """订单信息数据传输对象（不可变）。

    为策略提供只读的订单信息，确保策略无法直接修改订单状态。

    Attributes:
        order_id: 订单ID
        side: 买卖方向
        price: 价格
        qty: 数量
        type: 订单类型
        tif: 有效期类型
        filled_qty: 已成交数量
        status: 订单状态
        create_time: 创建时间
        arrival_time: 到达交易所时间（可选）
    """
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
        """获取剩余数量。"""
        return max(0, self.qty - self.filled_qty)

    @property
    def is_active(self) -> bool:
        """判断订单是否活跃。"""
        return self.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.LIVE]


@dataclass(frozen=True)
class PortfolioDTO:
    """投资组合数据传输对象（不可变）。

    为策略提供只读的持仓信息。

    Attributes:
        cash: 现金余额
        position: 持仓数量
        realized_pnl: 已实现盈亏
    """
    cash: float
    position: int
    realized_pnl: float


class ReadOnlyOMSView:
    """OMS只读视图。

    为策略提供只读的订单管理系统访问接口，
    策略只能查询订单和持仓状态，不能直接修改。

    这种设计确保了策略与OMS之间的解耦：
    - 策略通过返回Order对象来提交新订单
    - 策略不能直接操作OMS内部状态
    """

    def __init__(self, oms: object):
        """初始化只读视图。

        Args:
            oms: 支持查询接口的 OMS 实例
        """
        self._oms = oms

    def get_active_orders(self) -> List[OrderInfoDTO]:
        """获取所有活跃订单的只读信息。

        Returns:
            活跃订单信息列表（不可变）
        """
        orders = self._oms.get_active_orders()
        return [self._to_order_dto(o) for o in orders]

    def get_order(self, order_id: str) -> Optional[OrderInfoDTO]:
        """根据订单ID获取订单信息。

        Args:
            order_id: 订单ID

        Returns:
            订单信息（不可变），如果不存在则返回None
        """
        order = self._oms.get_order(order_id)
        if order is None:
            return None
        return self._to_order_dto(order)

    def get_portfolio(self) -> PortfolioDTO:
        """获取投资组合信息。

        Returns:
            投资组合信息（不可变）
        """
        portfolio = self._oms.portfolio
        return PortfolioDTO(
            cash=portfolio.cash,
            position=portfolio.position,
            realized_pnl=portfolio.realized_pnl,
        )

    @staticmethod
    def _to_order_dto(order) -> OrderInfoDTO:
        """将Order对象转换为OrderInfoDTO。

        Args:
            order: 原始Order对象

        Returns:
            不可变的OrderInfoDTO
        """
        return OrderInfoDTO(
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


def to_snapshot_dto(snapshot) -> SnapshotDTO:
    """将NormalizedSnapshot转换为SnapshotDTO。

    Args:
        snapshot: 原始NormalizedSnapshot对象

    Returns:
        不可变的SnapshotDTO
    """
    bids = tuple(LevelDTO(price=l.price, qty=l.qty) for l in snapshot.bids)
    asks = tuple(LevelDTO(price=l.price, qty=l.qty) for l in snapshot.asks)
    last_vol_split = tuple(snapshot.last_vol_split) if snapshot.last_vol_split else ()

    return SnapshotDTO(
        ts_recv=snapshot.ts_recv,  # 主时间线
        bids=bids,
        asks=asks,
        last_vol_split=last_vol_split,
        ts_exch=snapshot.ts_exch,  # 可选
        last=snapshot.last,
        volume=snapshot.volume,
        turnover=snapshot.turnover,
        average_price=snapshot.average_price,
    )
