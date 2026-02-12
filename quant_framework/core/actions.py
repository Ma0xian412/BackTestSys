"""动作模型：策略输出的纯数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ActionType(Enum):
    """动作类型。"""

    PLACE_ORDER = "PLACE_ORDER"
    CANCEL_ORDER = "CANCEL_ORDER"


@dataclass
class Action:
    """策略动作：type/time/payload 纯数据载体。"""

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
