"""SnapMatch 适配公共类型与桥接器。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ...core.port import IMarketDataQuery

MarketData = Dict[str, Any]
EngineFactory = Callable[..., Any]


class OrderTriggerType(Enum):
    """引擎触发类型。"""

    MarketData = "MarketData"


class MatchEngineTradeCallback(ABC):
    """成交/撤单回调基类。"""

    @abstractmethod
    def on_driver_order_traded(
        self,
        driver_order_id: int,
        event_id: int,
        traded_price: float,
        net_traded_vol: int,
        recv_tick: int,
        exch_tick: int,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_driver_order_canceled(
        self,
        driver_order_id: int,
        event_id: int,
        recv_tick: int,
        exch_tick: int,
    ) -> None:
        raise NotImplementedError


class MatchEngineFeedCallback(ABC):
    """行情查询回调基类。"""

    @abstractmethod
    def get_next_market_data(self, contract_id: int, tick_time: int) -> Optional[Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class MatchEngineEvent:
    """引擎回调事件（先缓存，再由 handler 转回执）。"""

    event_type: str
    driver_order_id: int
    event_id: int
    recv_tick: int
    exch_tick: int
    traded_price: float = 0.0
    net_traded_vol: int = 0

    def sort_key(self) -> tuple[int, int, int]:
        return int(self.recv_tick), int(self.exch_tick), int(self.event_id)


class MatchEngineTradeCallbackBridge(MatchEngineTradeCallback):
    """将引擎回调缓存为事件列表。"""

    def __init__(self, event_sink: List[MatchEngineEvent]) -> None:
        self._event_sink = event_sink

    def on_driver_order_traded(
        self,
        driver_order_id: int,
        event_id: int,
        traded_price: float,
        net_traded_vol: int,
        recv_tick: int,
        exch_tick: int,
    ) -> None:
        self._event_sink.append(
            MatchEngineEvent(
                event_type="TRADE",
                driver_order_id=int(driver_order_id),
                event_id=int(event_id),
                recv_tick=int(recv_tick),
                exch_tick=int(exch_tick),
                traded_price=float(traded_price),
                net_traded_vol=max(0, int(net_traded_vol)),
            )
        )

    def on_driver_order_canceled(
        self,
        driver_order_id: int,
        event_id: int,
        recv_tick: int,
        exch_tick: int,
    ) -> None:
        self._event_sink.append(
            MatchEngineEvent(
                event_type="CANCEL",
                driver_order_id=int(driver_order_id),
                event_id=int(event_id),
                recv_tick=int(recv_tick),
                exch_tick=int(exch_tick),
            )
        )


class MatchEngineFeedCallbackBridge(MatchEngineFeedCallback):
    """行情回调桥：取 recv_time > tick_time 且按 recv/exch 排序。"""

    def __init__(
        self,
        query_provider: Callable[[], Optional[IMarketDataQuery]],
        snapshot_mapper: Callable[[object, int], MarketData],
        to_match_market_data: Optional[Callable[[MarketData], Any]] = None,
        query_window: int = 128,
    ) -> None:
        self._query_provider = query_provider
        self._snapshot_mapper = snapshot_mapper
        self._to_match_market_data = to_match_market_data or (lambda x: x)
        self._query_window = max(1, int(query_window))

    def get_next_market_data(self, contract_id: int, tick_time: int) -> Optional[Any]:
        query = self._query_provider()
        if query is None:
            return None
        snapshots = list(query.query_data(self._query_window) or [])
        if not snapshots:
            return None
        candidates: List[MarketData] = []
        for snapshot in snapshots:
            market_data = self._snapshot_mapper(snapshot, int(contract_id))
            if int(market_data["recv_time"]) > int(tick_time):
                candidates.append(market_data)
        if not candidates:
            return None
        candidates.sort(key=lambda item: (int(item["recv_time"]), int(item["exch_time"])))
        return self._to_match_market_data(candidates[0])


def default_engine_factory(**kwargs: Any) -> Any:
    """默认引擎工厂：按候选模块导入 SnapMatchEngineWrapper。"""
    candidate_modules = ("snap_match_engine", "snap_match", "cpp")
    for module_name in candidate_modules:
        try:
            module = __import__(module_name, fromlist=["SnapMatchEngineWrapper"])
        except ImportError:
            continue
        engine_cls = getattr(module, "SnapMatchEngineWrapper", None)
        if engine_cls is not None:
            return engine_cls(**kwargs)
    raise ImportError(
        "Cannot import SnapMatchEngineWrapper. "
        "Please install snap engine bindings or pass engine/engine_factory explicitly."
    )
