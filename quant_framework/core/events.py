"""仿真事件模块。

定义仿真过程中使用的事件类型和事件数据结构。
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Any
from .types import Timestamp


class EventType(Enum):
    """仿真事件类型枚举。"""
    SNAPSHOT_ARRIVAL = auto()  # 快照到达
    TRADE_TICK = auto()        # 成交tick
    QUOTE_UPDATE = auto()      # 报价更新
    TIME_ADVANCE = auto()      # 时间推进


@dataclass
class SimulationEvent:
    """仿真事件。

    Attributes:
        ts: 事件时间戳
        type: 事件类型
        data: 事件数据（可选）
    """
    ts: Timestamp
    type: EventType
    data: Any = None