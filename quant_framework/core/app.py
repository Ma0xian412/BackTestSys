"""应用装配层：CompositionRoot 与 BacktestApp。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

from ..adapters import ExecutionVenueImpl, ObservabilityImpl, TimeModelImpl
from ..config import BacktestConfig
from .data_loader import CsvMarketDataFeed, PickleMarketDataFeed, SnapshotDuplicatingFeed
from ..exchange.simulator import FIFOExchangeSimulator
from ..tape.builder import TapeConfig as BuilderTapeConfig, UnifiedTapeBuilder
from ..trading.oms import OMSImpl, Portfolio
from ..trading.receipt_logger import ReceiptLogger
from ..trading.strategy import SimpleStrategyImpl
from .dispatcher import Dispatcher
from .handlers import ActionArrivalHandler, ReceiptDeliveryHandler, SnapshotArrivalHandler
from .kernel import EventLoopKernel
from .runtime import (
    EVENT_KIND_ACTION_ARRIVAL,
    EVENT_KIND_RECEIPT_DELIVERY,
    EVENT_KIND_SNAPSHOT_ARRIVAL,
    EventSpecRegistry,
    RuntimeContext,
)


@dataclass
class RuntimeBuildConfig:
    """RuntimeContext 直接组装参数（测试/高级用法）。"""

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
    """统一组装入口。"""

    def build(self, config: Union[RuntimeBuildConfig, BacktestConfig]) -> RuntimeContext:
        """构建 RuntimeContext。

        - BacktestConfig: 标准应用入口（生产路径）
        - RuntimeBuildConfig: 直接注入组件（测试/高级路径）
        """
        runtime_cfg = self._to_runtime_build_config(config)

        event_spec = runtime_cfg.eventSpec or EventSpecRegistry.default()
        dispatcher = runtime_cfg.dispatcher or Dispatcher(event_spec)

        dispatcher.register(EVENT_KIND_SNAPSHOT_ARRIVAL, SnapshotArrivalHandler())
        dispatcher.register(EVENT_KIND_ACTION_ARRIVAL, ActionArrivalHandler())
        dispatcher.register(EVENT_KIND_RECEIPT_DELIVERY, ReceiptDeliveryHandler())

        return RuntimeContext(
            feed=runtime_cfg.feed,
            venue=runtime_cfg.venue,
            strategy=runtime_cfg.strategy,
            oms=runtime_cfg.oms,
            timeModel=runtime_cfg.timeModel,
            obs=runtime_cfg.obs,
            dispatcher=dispatcher,
            eventSpec=event_spec,
            metadata=dict(runtime_cfg.metadata or {}),
        )

    def _to_runtime_build_config(self, config: Union[RuntimeBuildConfig, BacktestConfig]) -> RuntimeBuildConfig:
        if isinstance(config, RuntimeBuildConfig):
            return config
        if isinstance(config, BacktestConfig):
            return self._build_from_backtest_config(config)
        raise TypeError(f"Unsupported config type for CompositionRoot.build: {type(config)!r}")

    def _build_from_backtest_config(self, config: BacktestConfig) -> RuntimeBuildConfig:
        feed = self._create_feed(config)
        tape_builder = self._create_tape_builder(config)
        venue = self._create_venue(config, tape_builder)
        strategy = self._create_strategy(config)
        oms = self._create_oms(config)
        time_model = self._create_time_model(config)
        obs = self._create_observability(config)
        return RuntimeBuildConfig(
            feed=feed,
            venue=venue,
            strategy=strategy,
            oms=oms,
            timeModel=time_model,
            obs=obs,
        )

    @staticmethod
    def _create_feed(config: BacktestConfig):
        if config.data.format == "csv":
            inner_feed = CsvMarketDataFeed(config.data.path)
        else:
            inner_feed = PickleMarketDataFeed(config.data.path)

        trading_hours = None
        if config.contract.contract_info and config.contract.contract_info.trading_hours:
            trading_hours = config.contract.contract_info.trading_hours

        return SnapshotDuplicatingFeed(
            inner_feed=inner_feed,
            tolerance_tick=config.snapshot.tolerance_tick,
            trading_hours=trading_hours,
        )

    @staticmethod
    def _create_tape_builder(config: BacktestConfig) -> UnifiedTapeBuilder:
        tape_cfg = BuilderTapeConfig(
            ghost_rule=config.tape.ghost_rule,
            ghost_alpha=config.tape.ghost_alpha,
            epsilon=config.tape.epsilon,
            segment_iterations=config.tape.segment_iterations,
            time_scale_lambda=config.tape.time_scale_lambda,
            cancel_front_ratio=config.tape.cancel_front_ratio,
            crossing_order_policy=config.tape.crossing_order_policy,
            top_k=config.tape.top_k,
        )
        return UnifiedTapeBuilder(config=tape_cfg, tick_size=config.tape.tick_size)

    @staticmethod
    def _create_venue(config: BacktestConfig, tape_builder: UnifiedTapeBuilder) -> ExecutionVenueImpl:
        simulator = FIFOExchangeSimulator(cancel_bias_k=config.exchange.cancel_front_ratio)
        return ExecutionVenueImpl(simulator=simulator, tape_builder=tape_builder)

    @staticmethod
    def _create_strategy(config: BacktestConfig):
        # 当前默认策略实现；后续可扩展 registry/factory
        return SimpleStrategyImpl(name=config.strategy.name)

    @staticmethod
    def _create_oms(config: BacktestConfig) -> OMSImpl:
        portfolio = Portfolio(cash=config.portfolio.initial_cash)
        return OMSImpl(portfolio=portfolio)

    @staticmethod
    def _create_time_model(config: BacktestConfig) -> TimeModelImpl:
        return TimeModelImpl(
            delay_out=config.runner.delay_out,
            delay_in=config.runner.delay_in,
        )

    @staticmethod
    def _create_observability(config: BacktestConfig) -> ObservabilityImpl:
        receipt_logger = ReceiptLogger(
            output_file=config.receipt_logger.output_file or None,
            verbose=config.receipt_logger.verbose,
        )
        return ObservabilityImpl(receipt_logger)


class BacktestApp:
    """应用入口（按配置构建 context 并运行 kernel）。"""

    def __init__(
        self,
        config: Union[RuntimeBuildConfig, BacktestConfig],
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
