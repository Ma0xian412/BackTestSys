"""事件分发器与处理器接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from .data_structure import Event, EventSpecRegistry, RuntimeContext


class IEventHandler(ABC):
    """事件处理器接口。"""

    @abstractmethod
    def handle(self, e: Event, ctx: RuntimeContext) -> List[Event]:
        raise NotImplementedError


@dataclass(frozen=True)
class _HandlerBinding:
    """事件处理器绑定信息。"""

    handler: IEventHandler
    order: Optional[int]
    register_seq: int


class Dispatcher:
    """基于 type 的事件分发。"""

    def __init__(self, event_spec: EventSpecRegistry) -> None:
        self._event_spec = event_spec
        self._handlers: Dict[str, List[_HandlerBinding]] = {}
        self._register_seq = 0

    def register(self, event_type: str, handler: IEventHandler, order: Optional[int] = None) -> None:
        """覆盖注册事件处理器。"""
        self._handlers[event_type] = [self._create_binding(handler=handler, order=order)]

    def subscribe(self, event_type: str, handler: IEventHandler, order: Optional[int] = None) -> None:
        """追加注册事件处理器。"""
        bucket = self._handlers.setdefault(event_type, [])
        bucket.append(self._create_binding(handler=handler, order=order))

    def dispatch(self, e: Event, ctx: RuntimeContext) -> List[Event]:
        if not self._event_spec.validate(e.type, e.payload):
            raise ValueError(f"Invalid payload for event type={e.type!r}")

        bindings = self._handlers.get(e.type)
        if not bindings:
            return []
        emitted: List[Event] = []
        for binding in self._sorted_bindings(bindings):
            emitted.extend(binding.handler.handle(e, ctx) or [])
        return emitted

    def _create_binding(self, handler: IEventHandler, order: Optional[int]) -> _HandlerBinding:
        self._register_seq += 1
        return _HandlerBinding(handler=handler, order=order, register_seq=self._register_seq)

    @staticmethod
    def _sorted_bindings(bindings: List[_HandlerBinding]) -> List[_HandlerBinding]:
        max_order = max((b.order for b in bindings if b.order is not None), default=0)

        def key_fn(binding: _HandlerBinding) -> tuple[int, int]:
            order_value = binding.order if binding.order is not None else max_order + 1
            return order_value, binding.register_seq

        return sorted(bindings, key=key_fn)
