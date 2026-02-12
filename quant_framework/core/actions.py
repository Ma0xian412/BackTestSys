"""动作模型：策略输出的可扩展 Action 抽象。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Protocol

from .types import CancelRequest, Order, OrderReceipt


class ActionVenueBridge(Protocol):
    """Action 与 Venue 的桥接协议（避免类型探测）。"""

    def execute_place_order(self, order: Order, t_arrive: int) -> List[OrderReceipt]:
        ...

    def execute_cancel_order(self, request: CancelRequest, t_arrive: int) -> List[OrderReceipt]:
        ...


class ActionOMSBridge(Protocol):
    """Action 与 OMS 的桥接协议（避免类型探测）。"""

    def submit_order(self, order: Order, send_time: int) -> None:
        ...

    def submit_cancel(self, request: CancelRequest, send_time: int) -> None:
        ...


class ActionObservabilityBridge(Protocol):
    """Action 与 Observability 的桥接协议（避免类型探测）。"""

    def on_order_submitted(self, order: Order) -> None:
        ...

    def on_cancel_submitted(self, request: CancelRequest) -> None:
        ...


class Action(ABC):
    """策略动作抽象。"""

    @property
    @abstractmethod
    def kind(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def resolve_send_time(self, fallback_send_time: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def submit_to_oms(self, oms: ActionOMSBridge, send_time: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def execute_at_venue(self, venue: ActionVenueBridge, t_arrive: int) -> List[OrderReceipt]:
        raise NotImplementedError

    @abstractmethod
    def record_submission(self, obs: ActionObservabilityBridge) -> None:
        raise NotImplementedError


@dataclass
class PlaceOrderAction(Action):
    """下单动作。"""

    order: Order

    @property
    def kind(self) -> str:
        return "PlaceOrder"

    def resolve_send_time(self, fallback_send_time: int) -> int:
        if int(self.order.create_time) > 0:
            return int(self.order.create_time)
        return int(fallback_send_time)

    def submit_to_oms(self, oms: ActionOMSBridge, send_time: int) -> None:
        oms.submit_order(self.order, int(send_time))

    def execute_at_venue(self, venue: ActionVenueBridge, t_arrive: int) -> List[OrderReceipt]:
        return venue.execute_place_order(self.order, int(t_arrive))

    def record_submission(self, obs: ActionObservabilityBridge) -> None:
        obs.on_order_submitted(self.order)


@dataclass
class CancelOrderAction(Action):
    """撤单动作。"""

    request: CancelRequest

    @property
    def kind(self) -> str:
        return "CancelOrder"

    def resolve_send_time(self, fallback_send_time: int) -> int:
        if int(self.request.create_time) > 0:
            return int(self.request.create_time)
        return int(fallback_send_time)

    def submit_to_oms(self, oms: ActionOMSBridge, send_time: int) -> None:
        oms.submit_cancel(self.request, int(send_time))

    def execute_at_venue(self, venue: ActionVenueBridge, t_arrive: int) -> List[OrderReceipt]:
        return venue.execute_cancel_order(self.request, int(t_arrive))

    def record_submission(self, obs: ActionObservabilityBridge) -> None:
        obs.on_cancel_submitted(self.request)
