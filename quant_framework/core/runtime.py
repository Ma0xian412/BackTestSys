"""核心运行时模型：事件、上下文、注册中心。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .actions import Action
from .dto import ReadOnlyOMSView, SnapshotDTO
from .types import OrderReceipt


EVENT_KIND_SNAPSHOT_ARRIVAL = "SnapshotArrival"
EVENT_KIND_ACTION_ARRIVAL = "ActionArrival"
EVENT_KIND_RECEIPT_DELIVERY = "ReceiptDelivery"


_event_seq_counter = 0


def _next_seq() -> int:
    """获取单线程事件序号（用于稳定排序）。"""
    global _event_seq_counter
    _event_seq_counter += 1
    return _event_seq_counter


def reset_event_seq() -> None:
    """重置事件序号（主要用于测试）。"""
    global _event_seq_counter
    _event_seq_counter = 0


@dataclass
class Event:
    """调度器中的统一事件。"""

    time: int
    kind: str
    payload: object
    priority: int = 0
    seq: int = field(default_factory=_next_seq)

    def __lt__(self, other: "Event") -> bool:
        if self.time != other.time:
            return self.time < other.time
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.seq < other.seq


@dataclass(frozen=True)
class StrategyContext:
    """策略回调上下文（只读）。"""

    t: int
    snapshot: Optional[SnapshotDTO]
    omsView: ReadOnlyOMSView


@dataclass
class EventSpecRegistry:
    """事件规范与优先级注册中心。"""

    _priorities: Dict[str, int]
    _validators: Dict[str, Callable[[object], bool]]

    @classmethod
    def default(cls) -> "EventSpecRegistry":
        return cls(
            _priorities={
                EVENT_KIND_SNAPSHOT_ARRIVAL: 10,
                EVENT_KIND_ACTION_ARRIVAL: 20,
                EVENT_KIND_RECEIPT_DELIVERY: 30,
            },
            _validators={
                EVENT_KIND_SNAPSHOT_ARRIVAL: lambda payload: isinstance(payload, SnapshotDTO),
                EVENT_KIND_ACTION_ARRIVAL: lambda payload: isinstance(payload, Action),
                EVENT_KIND_RECEIPT_DELIVERY: lambda payload: isinstance(payload, OrderReceipt),
            },
        )

    def priorityOf(self, kind: str) -> int:
        return self._priorities.get(kind, 99)

    def validate(self, kind: str, payload: object) -> bool:
        validator = self._validators.get(kind)
        if validator is None:
            return True
        return bool(validator(payload))

    def register(self, kind: str, priority: int, validator: Optional[Callable[[object], bool]] = None) -> None:
        self._priorities[kind] = priority
        if validator is not None:
            self._validators[kind] = validator


@dataclass
class RuntimeContext:
    """运行上下文（由 CompositionRoot 组装）。"""

    feed: Any
    venue: Any
    strategy: Any
    oms: Any
    timeModel: Any
    obs: Any
    dispatcher: Any
    eventSpec: EventSpecRegistry
    last_snapshot_dto: Optional[SnapshotDTO] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
