"""应用装配层：CompositionRoot 与 BacktestApp。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..adapters.strategy_adapter import LegacyStrategyAdapter
from .dispatcher import Dispatcher
from .handlers import ActionArrivalHandler, ReceiptDeliveryHandler, SnapshotArrivalHandler
from .kernel import EventLoopKernel
from .runtime import (
    EVENT_KIND_ACTION_ARRIVAL,
    EVENT_KIND_RECEIPT_DELIVERY,
    EVENT_KIND_SNAPSHOT_ARRIVAL,
    EngineState,
    EventSpecRegistry,
    RuntimeContext,
)


@dataclass
class RuntimeBuildConfig:
    """RuntimeContext 组装参数。"""

    feed: Any
    venue: Any
    strategy: Any
    oms: Any
    timeModel: Any
    obs: Any
    eventSpec: Optional[EventSpecRegistry] = None
    dispatcher: Optional[Dispatcher] = None
    state: Optional[EngineState] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    wrap_legacy_strategy: bool = True


class CompositionRoot:
    """统一组装入口。"""

    def build(self, config: RuntimeBuildConfig) -> RuntimeContext:
        event_spec = config.eventSpec or EventSpecRegistry.default()
        dispatcher = config.dispatcher or Dispatcher(event_spec)
        state = config.state or EngineState()

        dispatcher.register(EVENT_KIND_SNAPSHOT_ARRIVAL, SnapshotArrivalHandler())
        dispatcher.register(EVENT_KIND_ACTION_ARRIVAL, ActionArrivalHandler())
        dispatcher.register(EVENT_KIND_RECEIPT_DELIVERY, ReceiptDeliveryHandler())

        strategy = config.strategy
        if config.wrap_legacy_strategy:
            strategy = LegacyStrategyAdapter(strategy)

        return RuntimeContext(
            feed=config.feed,
            venue=config.venue,
            strategy=strategy,
            oms=config.oms,
            timeModel=config.timeModel,
            obs=config.obs,
            dispatcher=dispatcher,
            eventSpec=event_spec,
            state=state,
            diagnostics=dict(config.diagnostics or {}),
        )


class BacktestApp:
    """应用入口（按配置构建 context 并运行 kernel）。"""

    def __init__(
        self,
        composition_root: Optional[CompositionRoot] = None,
        kernel: Optional[EventLoopKernel] = None,
    ) -> None:
        self._composition_root = composition_root or CompositionRoot()
        self._kernel = kernel or EventLoopKernel()

    def run(self, config: RuntimeBuildConfig) -> Dict[str, Any]:
        ctx = self._composition_root.build(config)
        return self._kernel.run(ctx)
