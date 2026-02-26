"""执行场所端口适配器（纯框架转发层）。"""

from __future__ import annotations

from typing import List

from ...core.data_structure import Action, OrderReceipt
from ...core.port import IExecutionVenue, IMarketDataQuery, ISimulator


class ExecutionVenue_Impl(IExecutionVenue):
    """IExecutionVenue 适配器：仅转发到 ISimulator。"""

    def __init__(self, simulator: ISimulator) -> None:
        self._simulator = simulator

    def set_market_data_query(self, market_data_query: IMarketDataQuery) -> None:
        self._simulator.set_market_data_query(market_data_query)

    def start_run(self) -> None:
        self._simulator.start_run()

    def start_session(self) -> None:
        self._simulator.start_session()

    def on_action(self, action: Action) -> List[OrderReceipt]:
        return self._simulator.on_action(action)

    def step(self, until_time: int) -> List[OrderReceipt]:
        return self._simulator.step(until_time)

    def flush_window(self) -> object:
        return self._simulator.flush_window()
