"""核心端口定义。"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, TYPE_CHECKING

from .data_structure import Action, CancelRequest, NormalizedSnapshot, Order, OrderReceipt, StepOutcome, TapeSegment

if TYPE_CHECKING:
    from .data_structure import ReadOnlyOMSView


class IMarketDataFeed(ABC):
    """行情数据源端口。"""

    @abstractmethod
    def next(self) -> Optional[Any]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def query_data(self, t_start: int, t_end: int) -> List[Any]:
        raise NotImplementedError

    # 兼容旧命名（与需求文档中的 Query_Data 对齐）
    def Query_Data(self, T_Start: int, T_End: int) -> List[Any]:
        return self.query_data(int(T_Start), int(T_End))


class IIntervalModel(ABC):
    """区间模型端口。"""

    @abstractmethod
    def build(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot) -> List[TapeSegment]:
        raise NotImplementedError


class IExecutionVenue(ABC):
    """执行场所端口。"""

    @abstractmethod
    def start_session(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_time_window(self, t_start: int, t_end: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_action(self, action: Action) -> List[OrderReceipt]:
        raise NotImplementedError

    @abstractmethod
    def step(self, until_time: int) -> StepOutcome:
        raise NotImplementedError

    @abstractmethod
    def flush_window(self) -> object:
        raise NotImplementedError

    # --- backward compatibility helpers ---
    def startSession(self) -> None:
        self.start_session()

    def beginInterval(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot) -> None:
        self.set_time_window(int(prev.ts_recv), int(curr.ts_recv))

    def onActionArrival(self, action: Action, t_arrive: int) -> List[OrderReceipt]:
        action.create_time = int(t_arrive)
        return self.on_action(action)

    def endInterval(self, snapshot_end: NormalizedSnapshot) -> object:
        _ = snapshot_end
        return self.flush_window()


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
