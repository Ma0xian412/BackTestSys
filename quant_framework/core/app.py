"""应用装配层：CompositionRoot 与 BacktestApp。

CompositionRoot 只负责通用 wiring（连接接口间的 callback、注册 handler）。
具体 adapter 实例由外部创建并通过 RuntimeBuildConfig 传入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .dispatcher import Dispatcher
from .handlers import ActionArrivalHandler, MDArriveHandler, ReceiptDeliveryHandler
from .kernel import EventLoopKernel
from .data_structure import (
    EVENT_KIND_ACTION_ARRIVAL,
    EVENT_KIND_RECEIPT_DELIVERY,
    EVENT_KIND_MDARRIVE,
    EventSpecRegistry,
    RuntimeContext,
)


@dataclass
class RuntimeBuildConfig:
    """RuntimeContext 直接组装参数。"""

    feed: Any
    venue: Any
    strategy: Any
    oms: Any
    timeModel: Any
    obs: Any
    eventSpec: Optional[EventSpecRegistry] = None
    dispatcher: Optional[Dispatcher] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CompositionRoot:
    """通用框架 wiring：从已创建的组件构建 RuntimeContext。"""

    def build(self, config: RuntimeBuildConfig) -> RuntimeContext:
        config.venue.set_market_data_query(config.feed)
        config.oms.subscribe_new(config.obs.on_order_submitted)
        config.oms.subscribe_receipt(config.obs.on_receipt_delivered)

        event_spec = config.eventSpec or EventSpecRegistry.default()
        dispatcher = config.dispatcher or Dispatcher(event_spec)

        dispatcher.register(EVENT_KIND_MDARRIVE, MDArriveHandler())
        dispatcher.register(EVENT_KIND_ACTION_ARRIVAL, ActionArrivalHandler())
        dispatcher.register(EVENT_KIND_RECEIPT_DELIVERY, ReceiptDeliveryHandler())

        return RuntimeContext(
            feed=config.feed,
            venue=config.venue,
            strategy=config.strategy,
            oms=config.oms,
            timeModel=config.timeModel,
            obs=config.obs,
            dispatcher=dispatcher,
            eventSpec=event_spec,
            metadata=dict(config.metadata or {}),
        )


class BacktestApp:
    """应用入口。"""

    def __init__(
        self,
        config: RuntimeBuildConfig,
        composition_root: Optional[CompositionRoot] = None,
        kernel: Optional[EventLoopKernel] = None,
    ) -> None:
        self._config = config
        self._composition_root = composition_root or CompositionRoot()
        self._kernel = kernel or EventLoopKernel()
        self._last_context: Optional[RuntimeContext] = None

    def run(self) -> Dict[str, Any]:
        ctx = self._composition_root.build(self._config)
        self._last_context = ctx
        return self._kernel.run(ctx)

    @property
    def last_context(self) -> Optional[RuntimeContext]:
        return self._last_context
