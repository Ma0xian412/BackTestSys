"""执行场所端口适配器（纯框架转发层）。"""

from __future__ import annotations

from ...core.data_structure import Action, Result
from ...core.port import IExecutionVenue, IMarketDataFeed, ISimulator


class ExecutionVenue_Impl(IExecutionVenue):
    """IExecutionVenue 适配器：仅转发到 ISimulator。"""

    def __init__(self, simulator: ISimulator) -> None:
        self._simulator = simulator

    def set_market_data_feed(self, market_data_feed: IMarketDataFeed) -> None:
        self._simulator.set_market_data_feed(market_data_feed)

    def start_session(self) -> None:
        self._simulator.start_session()

    def set_time_window(self, t_start: int, t_end: int) -> None:
        self._simulator.set_time_window(t_start, t_end)

    def on_action(self, action: Action) -> Result:
        return self._simulator.on_action(action)

    def step(self, until_time: int) -> Result:
        return self._simulator.step(until_time)

    def flush_window(self) -> object:
        return self._simulator.flush_window()
