"""核心端口定义。"""

from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Optional, TYPE_CHECKING

from .data_structure import (
    Action,
    CancelRequest,
    NormalizedSnapshot,
    Order,
    OrderReceipt,
    ShadowOrder,
    TapeSegment,
)

if TYPE_CHECKING:
    from .data_structure import ReadOnlyOMSView


class IMarketDataStream(ABC):
    """行情流端口：负责 next/reset。"""

    @abstractmethod
    def next(self) -> Optional[Any]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class IMarketDataQuery(ABC):
    """行情查询端口：负责窗口查询。"""

    @abstractmethod
    def query_data(self, n: int) -> List[Any]:
        raise NotImplementedError


class IIntervalModel(ABC):
    """区间模型端口。"""

    @abstractmethod
    def build(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot) -> List[TapeSegment]:
        raise NotImplementedError


class IExecutionVenue(ABC):
    """执行场所端口。"""

    @abstractmethod
    def set_market_data_query(self, market_data_query: IMarketDataQuery) -> None:
        raise NotImplementedError

    @abstractmethod
    def start_run(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def start_session(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_action(self, action: Action) -> List[OrderReceipt]:
        raise NotImplementedError

    @abstractmethod
    def step(self, until_time: int) -> List[OrderReceipt]:
        raise NotImplementedError

    @abstractmethod
    def flush_window(self) -> object:
        raise NotImplementedError


class ISimulator(ABC):
    """模拟交易所框架端口。"""

    @abstractmethod
    def set_market_data_query(self, market_data_query: IMarketDataQuery) -> None:
        raise NotImplementedError

    @abstractmethod
    def start_run(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def start_session(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_action(self, action: Action) -> List[OrderReceipt]:
        raise NotImplementedError

    @abstractmethod
    def step(self, until_time: int) -> List[OrderReceipt]:
        raise NotImplementedError

    @abstractmethod
    def flush_window(self) -> object:
        raise NotImplementedError


class IMatchAlgorithm(ABC):
    """撮合算法端口。"""

    @abstractmethod
    def set_market_data_query(self, market_data_query: IMarketDataQuery) -> None:
        raise NotImplementedError

    @abstractmethod
    def start_run(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def start_session(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_order_action_impl(
        self,
        order: ShadowOrder,
        active_orders: Mapping[str, ShadowOrder],
    ) -> List[OrderReceipt]:
        raise NotImplementedError

    @abstractmethod
    def on_step(
        self,
        active_orders: Mapping[str, ShadowOrder],
        start_time: int,
        until_time: int,
    ) -> List[OrderReceipt]:
        raise NotImplementedError

    @abstractmethod
    def flush_window(self) -> object:
        raise NotImplementedError


class IOMS(ABC):
    """OMS 端口。"""

    @abstractmethod
    def submit_order(self, order: Order, send_time: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def submit_cancel(self, request: CancelRequest, send_time: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_receipt(self, receipt: OrderReceipt) -> None:
        raise NotImplementedError

    @abstractmethod
    def view(self) -> "ReadOnlyOMSView":
        raise NotImplementedError


class ITimeModel(ABC):
    """时延模型端口。"""

    @abstractmethod
    def delayout(self, local_time: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def delayin(self, exchange_time: int) -> int:
        raise NotImplementedError


class IObservabilitySinks(ABC):
    """可观测性端口。

    纯观察者角色：记录事件 + 生成报表。
    不维护订单状态，统计数据通过 set_oms 注入的 OMS 引用查询。
    """

    @abstractmethod
    def set_oms(self, oms: IOMS) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_order_submitted(self, order: Order) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_cancel_submitted(self, request: CancelRequest) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_receipt_generated(self, receipt: OrderReceipt) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_receipt_delivered(self, receipt: OrderReceipt) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_interval_end(self, stats: object) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_run_end(self, context: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_diagnostics(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_run_result(self) -> dict:
        raise NotImplementedError


class IStrategy(ABC):
    """策略端口。"""

    @abstractmethod
    def on_event(self, e: Any, ctx: Any) -> List[Action]:
        raise NotImplementedError
