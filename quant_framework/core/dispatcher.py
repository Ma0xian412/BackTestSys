"""事件分发器与处理器接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from .runtime import Event, EventSpecRegistry, RuntimeContext


class IEventHandler(ABC):
    """事件处理器接口。"""

    @abstractmethod
    def handle(self, e: Event, ctx: RuntimeContext) -> List[Event]:
        raise NotImplementedError


class Dispatcher:
    """基于 kind 的事件分发。"""

    def __init__(self, event_spec: EventSpecRegistry) -> None:
        self._event_spec = event_spec
        self._handlers: Dict[str, IEventHandler] = {}

    def register(self, kind: str, handler: IEventHandler) -> None:
        self._handlers[kind] = handler

    def dispatch(self, e: Event, ctx: RuntimeContext) -> List[Event]:
        if not self._event_spec.validate(e.kind, e.payload):
            raise ValueError(f"Invalid payload for event kind={e.kind!r}")

        handler = self._handlers.get(e.kind)
        if handler is None:
            return []
        return handler.handle(e, ctx)
