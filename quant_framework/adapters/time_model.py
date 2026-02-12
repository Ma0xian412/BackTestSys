"""时间模型适配器。"""

from __future__ import annotations

from ..core.interfaces import ITimeModel


class TimeModelImpl(ITimeModel):
    """固定出入向时延模型。"""

    def __init__(self, delay_out: int = 0, delay_in: int = 0) -> None:
        self._delay_out = int(delay_out)
        self._delay_in = int(delay_in)

    def delayout(self, local_time: int) -> int:
        return int(local_time) + self._delay_out

    def delayin(self, exchange_time: int) -> int:
        return int(exchange_time) + self._delay_in
