"""订阅者状态与内存计量。"""

from __future__ import annotations

import pickle
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Set, Tuple

from ...core.observability import ObsEventEnvelope, ObsSubscriptionState

_DEFAULT_EVENT_BYTES = 512


def estimate_event_size_bytes(event: ObsEventEnvelope) -> int:
    try:
        return len(pickle.dumps(event, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return max(_DEFAULT_EVENT_BYTES, len(repr(event)))


@dataclass
class SubscriberState:
    subscription_id: str
    topics: Optional[Set[str]]
    max_memory_bytes: int
    live_min_seq: int
    replay_seq: int
    state: ObsSubscriptionState = ObsSubscriptionState.ACTIVE
    error: Optional[str] = None
    buffer: Deque[Tuple[ObsEventEnvelope, int]] = field(default_factory=deque)
    buffer_bytes: int = 0
    condition: threading.Condition = field(default_factory=threading.Condition)

    def matches(self, event_type: str) -> bool:
        if not self.topics:
            return True
        return event_type in self.topics

    def buffer_len(self) -> int:
        return len(self.buffer)
