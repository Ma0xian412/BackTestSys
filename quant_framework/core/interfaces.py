"""核心端口接口定义（面向新架构）。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, TYPE_CHECKING

from .actions import Action
from .types import CancelRequest, NormalizedSnapshot, Order, OrderReceipt, TapeSegment

if TYPE_CHECKING:
    from .dto import ReadOnlyOMSView


class IMarketDataFeed(ABC):
    """行情数据源接口。"""

    @abstractmethod
    def next(self) -> Optional[NormalizedSnapshot]:
        """获取下一个快照。"""
        pass

    @abstractmethod
    def reset(self):
        """重置数据源（支持多次回测）。"""
        raise NotImplementedError


class ITapeBuilder(ABC):
    """Tape构建器接口（纯函数，无状态）。"""

    @abstractmethod
    def build(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot) -> List[TapeSegment]:
        """从A/B快照构建Tape段。

        Args:
            prev: 前一个快照（A）
            curr: 当前快照（B）

        Returns:
            按时间排序的TapeSegment列表
        """
        raise NotImplementedError


@dataclass(frozen=True)
class StepOutcome:
    """执行场所 step 结果。"""

    next_time: int
    receipts_generated: List[OrderReceipt]


class IExecutionVenue(ABC):
    """执行场所端口（由核心事件循环调用）。"""

    @abstractmethod
    def startSession(self) -> None:
        """开启回测会话。"""
        raise NotImplementedError

    @abstractmethod
    def beginInterval(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot) -> None:
        """开始处理区间 [prev.ts_recv, curr.ts_recv]。"""
        raise NotImplementedError

    @abstractmethod
    def onActionArrival(self, action: Action, t_arrive: int) -> List[OrderReceipt]:
        """动作到达执行场所，返回即时生成的回执。"""
        raise NotImplementedError

    @abstractmethod
    def step(self, t_cur: int, t_limit: int) -> StepOutcome:
        """推进执行场所内部时钟到 t_limit（或更早停止点）。"""
        raise NotImplementedError

    @abstractmethod
    def endInterval(self, snapshot_end: NormalizedSnapshot) -> object:
        """结束当前区间，返回统计对象。"""
        raise NotImplementedError


class IOMS(ABC):
    """OMS 端口（核心只依赖该抽象）。"""

    @abstractmethod
    def submit_order(self, order: Order, send_time: int) -> None:
        """登记订单动作。"""
        raise NotImplementedError

    @abstractmethod
    def submit_cancel(self, request: CancelRequest, send_time: int) -> None:
        """登记撤单动作。"""
        raise NotImplementedError

    @abstractmethod
    def apply_receipt(self, receipt: OrderReceipt) -> None:
        """应用回执，推进订单状态机。"""
        raise NotImplementedError

    @abstractmethod
    def view(self) -> 'ReadOnlyOMSView':
        """返回只读 OMS 视图供策略查询。"""
        raise NotImplementedError


class ITimeModel(ABC):
    """时延模型端口。"""

    @abstractmethod
    def action_arrival_time(self, send_time: int, action: Action) -> int:
        """策略动作到达执行场所的时间。"""
        raise NotImplementedError

    @abstractmethod
    def receipt_delivery_time(self, receipt: OrderReceipt) -> int:
        """回执送达策略的时间。"""
        raise NotImplementedError


class IObservabilitySinks(ABC):
    """可观测性汇聚端口。"""

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
    def get_diagnostics(self) -> dict:
        raise NotImplementedError


class IStrategy(ABC):
    """策略接口：单入口 on_event。"""

    @abstractmethod
    def on_event(self, e: Any, ctx: Any) -> List[Action]:
        """统一事件入口，返回动作列表。"""
        raise NotImplementedError
