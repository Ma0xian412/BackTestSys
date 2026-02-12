"""时间模型适配器。"""

from __future__ import annotations

from typing import Any

from ..core.interfaces import ITimeModel
from ..core.types import OrderReceipt


class DelayTimeModel(ITimeModel):
    """固定出入向时延模型。"""

    def __init__(self, delay_out: int = 0, delay_in: int = 0) -> None:
        self._delay_out = int(delay_out)
        self._delay_in = int(delay_in)

    def action_arrival_time(self, send_time: int, action: Any) -> int:
        action_create_time = int(getattr(action, "create_time", 0) or 0)
        base_send = action_create_time if action_create_time > 0 else int(send_time)
        return base_send + self._delay_out

    def receipt_delivery_time(self, receipt: OrderReceipt) -> int:
        return int(receipt.timestamp) + self._delay_in
