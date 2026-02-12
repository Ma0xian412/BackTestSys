"""OMS 端口适配器。"""

from __future__ import annotations

from typing import Any

from ..core.dto import ReadOnlyOMSView
from ..core.interfaces import IOMS, IOrderManager
from ..core.types import Order, OrderReceipt


class OrderStateMachineOMS(IOMS):
    """将现有 OrderManager 适配到 IOMS 端口。"""

    def __init__(self, oms: IOrderManager):
        self._oms = oms

    @property
    def raw(self) -> IOrderManager:
        return self._oms

    def submit_action(self, action: Any, send_time: int) -> None:
        if isinstance(action, Order):
            self._oms.submit(action, int(send_time))

    def apply_receipt(self, receipt: OrderReceipt) -> None:
        self._oms.on_receipt(receipt)

    def view(self) -> ReadOnlyOMSView:
        return ReadOnlyOMSView(self._oms)
