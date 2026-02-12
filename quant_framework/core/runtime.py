"""核心运行时模型：事件封装、上下文与策略事件。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .dto import ReadOnlyOMSView, SnapshotDTO
from .types import NormalizedSnapshot, OrderReceipt


EVENT_KIND_SNAPSHOT_ARRIVAL = "SnapshotArrival"
EVENT_KIND_ACTION_ARRIVAL = "ActionArrival"
EVENT_KIND_RECEIPT_DELIVERY = "ReceiptDelivery"


_event_seq_counter = 0


def _next_seq() -> int:
    """获取单线程事件序号（用于稳定排序）。"""
    global _event_seq_counter
    _event_seq_counter += 1
    return _event_seq_counter


def reset_event_envelope_seq() -> None:
    """重置事件序号（主要用于测试）。"""
    global _event_seq_counter
    _event_seq_counter = 0


@dataclass
class EventEnvelope:
    """调度器中的统一事件封装。"""

    time: int
    kind: str
    payload: object
    priority: int = 0
    seq: int = field(default_factory=_next_seq)

    def __lt__(self, other: "EventEnvelope") -> bool:
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


@dataclass(frozen=True)
class StrategyEvent:
    """策略事件基类。"""

    time: int
    kind: str


@dataclass(frozen=True)
class SnapshotStrategyEvent(StrategyEvent):
    """快照驱动策略事件。"""

    snapshot: SnapshotDTO

    def __init__(self, time: int, snapshot: SnapshotDTO):
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "kind", EVENT_KIND_SNAPSHOT_ARRIVAL)
        object.__setattr__(self, "snapshot", snapshot)


@dataclass(frozen=True)
class ReceiptStrategyEvent(StrategyEvent):
    """回执驱动策略事件。"""

    receipt: OrderReceipt

    def __init__(self, time: int, receipt: OrderReceipt):
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "kind", EVENT_KIND_RECEIPT_DELIVERY)
        object.__setattr__(self, "receipt", receipt)


@dataclass
class EngineState:
    """运行时共享状态。"""

    lastSnapshotDTO: Optional[SnapshotDTO] = None

    def update_snapshot(self, snapshotDTO: SnapshotDTO) -> None:
        self.lastSnapshotDTO = snapshotDTO


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
                EVENT_KIND_ACTION_ARRIVAL: lambda payload: payload is not None,
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
    state: EngineState
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def ensure_default_diagnostics(self) -> None:
        self.diagnostics.setdefault("intervals_processed", 0)
        self.diagnostics.setdefault("orders_submitted", 0)
        self.diagnostics.setdefault("orders_filled", 0)
        self.diagnostics.setdefault("receipts_generated", 0)
        self.diagnostics.setdefault("cancels_submitted", 0)
