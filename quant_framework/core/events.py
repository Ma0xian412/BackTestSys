from enum import Enum, auto
from dataclasses import dataclass
from typing import Any
from .types import Timestamp

class EventType(Enum):
    SNAPSHOT_ARRIVAL = auto()
    TRADE_TICK = auto()
    QUOTE_UPDATE = auto()
    TIME_ADVANCE = auto()

@dataclass
class SimulationEvent:
    ts: Timestamp
    type: EventType
    data: Any = None