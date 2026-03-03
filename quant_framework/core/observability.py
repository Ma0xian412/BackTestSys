"""可观测性事件与订阅数据结构定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Tuple


EVENT_TYPE_RUN_STARTED = "run.started"
EVENT_TYPE_RUN_ENDED = "run.ended"
EVENT_TYPE_ORDER_SUBMITTED = "order.submitted"
EVENT_TYPE_CANCEL_SUBMITTED = "cancel.submitted"
EVENT_TYPE_RECEIPT_GENERATED = "receipt.generated"
EVENT_TYPE_RECEIPT_DELIVERED = "receipt.delivered"
EVENT_TYPE_INTERVAL_ENDED = "interval.ended"
EVENT_TYPE_OMS_ORDER_CHANGED = "oms.order.changed"
EVENT_TYPE_SUBSCRIBER_ERRORED = "subscriber.errored"


class ObsStartPosition(str, Enum):
    """订阅起始位置。"""

    BEGINNING = "beginning"
    LATEST = "latest"


class ObsSubscriptionState(str, Enum):
    """订阅状态。"""

    ACTIVE = "active"
    ERRORED = "errored"
    CLOSED = "closed"


@dataclass(frozen=True)
class ObsEventEnvelope:
    """对外发布的可观测事件包。"""

    run_id: str
    seq: int
    event_type: str
    sim_time: int
    wall_time: float
    payload: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1


@dataclass(frozen=True)
class ObsSubscriptionOptions:
    """订阅参数。"""

    topics: Tuple[str, ...] = field(default_factory=tuple)
    start_position: ObsStartPosition = ObsStartPosition.BEGINNING
    max_memory_bytes: int = 8 * 1024 * 1024


@dataclass(frozen=True)
class ObsSubscriptionStatus:
    """订阅状态快照。"""

    subscription_id: str
    state: ObsSubscriptionState
    error: Optional[str] = None
    buffered_events: int = 0
    buffered_bytes: int = 0


@dataclass(frozen=True)
class OMSOrderChange:
    """OMS 订单变化事件。"""

    order_id: str
    prev_status: str
    new_status: str
    prev_filled_qty: int
    new_filled_qty: int
    prev_remaining_qty: int
    new_remaining_qty: int
    timestamp: int
