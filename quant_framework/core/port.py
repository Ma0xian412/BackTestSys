"""核心端口定义。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, TYPE_CHECKING

from .model import Action
from .types import CancelRequest, NormalizedSnapshot, Order, OrderReceipt, TapeSegment

if TYPE_CHECKING:
    from .read_only_view import ReadOnlyOMSView


class IMarketDataFeed(ABC):
    """行情数据源端口。"""

    @abstractmethod
    def next(self) -> Optional[NormalizedSnapshot]:
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class ITapeBuilder(ABC):
    """Tape 构建器端口。"""

    @abstractmethod
    def build(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot) -> List[TapeSegment]:
        raise NotImplementedError


@dataclass(frozen=True)
class StepOutcome:
    """执行场所 step 结果。"""

    next_time: int
    receipts_generated: List[OrderReceipt]


class IExecutionVenue(ABC):
    """执行场所端口。"""

    @abstractmethod
    def startSession(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def beginInterval(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot) -> None:
        raise NotImplementedError

    @abstractmethod
    def onActionArrival(self, action: Action, t_arrive: int) -> List[OrderReceipt]:
        raise NotImplementedError

    @abstractmethod
    def step(self, t_cur: int, t_limit: int) -> StepOutcome:
        raise NotImplementedError

    @abstractmethod
    def endInterval(self, snapshot_end: NormalizedSnapshot) -> object:
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
    """可观测性端口。"""

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
    def on_run_end(self, final_time: int, error: str | None = None) -> None:
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
