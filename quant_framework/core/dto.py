"""只读视图与 OMS 相关 DTO。"""

from dataclasses import dataclass
from typing import List, Optional

from .types import OrderStatus, Price, Qty, Side, TimeInForce, Timestamp


@dataclass(frozen=True)
class OrderInfoDTO:
    """订单信息数据传输对象（不可变）。"""

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
class PortfolioDTO:
    """投资组合数据传输对象（不可变）。"""

    cash: float
    position: int
    realized_pnl: float


class ReadOnlyOMSView:
    """OMS 只读视图。"""

    def __init__(self, oms: object):
        self._oms = oms

    def get_active_orders(self) -> List[OrderInfoDTO]:
        orders = self._oms.get_active_orders()
        return [self._to_order_dto(o) for o in orders]

    def get_order(self, order_id: str) -> Optional[OrderInfoDTO]:
        order = self._oms.get_order(order_id)
        if order is None:
            return None
        return self._to_order_dto(order)

    def get_portfolio(self) -> PortfolioDTO:
        portfolio = self._oms.portfolio
        return PortfolioDTO(
            cash=portfolio.cash,
            position=portfolio.position,
            realized_pnl=portfolio.realized_pnl,
        )

    @staticmethod
    def _to_order_dto(order) -> OrderInfoDTO:
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
