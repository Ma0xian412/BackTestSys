"""核心数据模型：Action/Event/Context。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

from .read_only_view import ReadOnlyOMSView
from .types import NormalizedSnapshot, OrderReceipt


EVENT_KIND_SNAPSHOT_ARRIVAL = "SnapshotArrival"
EVENT_KIND_ACTION_ARRIVAL = "ActionArrival"
EVENT_KIND_RECEIPT_DELIVERY = "ReceiptDelivery"


class ActionType(Enum):
    """策略动作类型。"""

    PLACE_ORDER = "PLACE_ORDER"
    CANCEL_ORDER = "CANCEL_ORDER"


@dataclass
class Action:
    """策略动作纯数据结构。"""

    action_type: ActionType
    create_time: int = 0
    payload: Any = None

    def get_type(self) -> ActionType:
        return self.action_type

    def set_type(self, action_type: ActionType) -> None:
        self.action_type = action_type

    def get_create_time(self) -> int:
        return int(self.create_time)

    def set_create_time(self, create_time: int) -> None:
        self.create_time = int(create_time)

    def get_payload(self) -> Any:
        return self.payload

    def set_payload(self, payload: Any) -> None:
        self.payload = payload


_event_seq_counter = 0


def _next_seq() -> int:
    global _event_seq_counter
    _event_seq_counter += 1
    return _event_seq_counter


def reset_event_seq() -> None:
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
    snapshot: Optional[NormalizedSnapshot]
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
                EVENT_KIND_SNAPSHOT_ARRIVAL: lambda payload: isinstance(payload, NormalizedSnapshot),
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
    last_snapshot: Optional[NormalizedSnapshot] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
