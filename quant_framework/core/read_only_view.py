"""OMS 只读视图与只读数据结构。"""

from dataclasses import dataclass
from typing import List, Optional

from .types import OrderStatus, Price, Qty, Side, TimeInForce, Timestamp


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
