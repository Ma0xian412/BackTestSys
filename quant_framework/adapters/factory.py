"""BacktestConfig → RuntimeBuildConfig 工厂。

从 BacktestConfig 创建所有具体 adapter 实例并组装为 RuntimeBuildConfig。
所有 adapter 的具体 import 集中在此文件。
"""

from __future__ import annotations

from typing import Any

from ..config import BacktestConfig
from ..core.app import RuntimeBuildConfig
from .execution_venue import ExecutionVenue_Impl, SegmentBaseAlgorithm, Simulator_Impl
from .interval_model import TapeConfig as BuilderTapeConfig, UnifiedIntervalModel_impl
from .market_data_feed import CsvMarketDataFeed_Impl, PickleMarketDataFeed_Impl, SnapshotDuplicatingFeed_Impl
from .IOMS.oms import OMS_Impl, Portfolio
from .IStrategy import SimpleStrategy_Impl
from .observability.Observability_Impl import Observability_Impl
from .time_model import TimeModel_Impl


class BacktestConfigFactory:
    """从 BacktestConfig 创建 RuntimeBuildConfig。"""

    def create(self, config: BacktestConfig) -> RuntimeBuildConfig:
        feed = self._create_feed(config)
        tape_builder = self._create_tape_builder(config)
        venue = self._create_venue(config, tape_builder, feed)
        strategy = self._create_strategy(config)
        oms = self._create_oms(config)
        time_model = self._create_time_model(config)
        obs = self._create_observability(config, oms)
        return RuntimeBuildConfig(
            feed=feed,
            venue=venue,
            strategy=strategy,
            oms=oms,
            timeModel=time_model,
            obs=obs,
        )

    @staticmethod
    def _create_feed(config: BacktestConfig) -> Any:
        if config.data.format == "csv":
            inner_feed = CsvMarketDataFeed_Impl(config.data.path)
        else:
            inner_feed = PickleMarketDataFeed_Impl(config.data.path)

        trading_hours = None
        if config.contract.contract_info and config.contract.contract_info.trading_hours:
            trading_hours = config.contract.contract_info.trading_hours

        return SnapshotDuplicatingFeed_Impl(
            inner_feed=inner_feed,
            tolerance_tick=config.snapshot.tolerance_tick,
            trading_hours=trading_hours,
        )

    @staticmethod
    def _create_tape_builder(config: BacktestConfig) -> UnifiedIntervalModel_impl:
        tape_cfg = BuilderTapeConfig(
            epsilon=config.tape.epsilon,
            time_scale_lambda=config.tape.time_scale_lambda,
            top_k=config.tape.top_k,
        )
        return UnifiedIntervalModel_impl(config=tape_cfg, tick_size=config.tape.tick_size)

    @staticmethod
    def _create_venue(
        config: BacktestConfig,
        tape_builder: UnifiedIntervalModel_impl,
        feed: Any,
    ) -> ExecutionVenue_Impl:
        match_algo = SegmentBaseAlgorithm(
            cancel_bias_k=config.exchange.cancel_bias_k,
            tape_builder=tape_builder,
            market_data_query=feed,
        )
        simulator = Simulator_Impl(match_algo=match_algo)
        return ExecutionVenue_Impl(simulator=simulator)

    @staticmethod
    def _create_strategy(config: BacktestConfig) -> Any:
        return SimpleStrategy_Impl(name=config.strategy.name)

    @staticmethod
    def _create_oms(config: BacktestConfig) -> OMS_Impl:
        portfolio = Portfolio(cash=config.portfolio.initial_cash)
        return OMS_Impl(portfolio=portfolio)

    @staticmethod
    def _create_time_model(config: BacktestConfig) -> TimeModel_Impl:
        return TimeModel_Impl(
            delay_out=config.runner.delay_out,
            delay_in=config.runner.delay_in,
        )

    @staticmethod
    def _create_observability(config: BacktestConfig, oms: OMS_Impl) -> Observability_Impl:
        obs = Observability_Impl(
            output_file=config.receipt_logger.output_file or None,
            verbose=config.receipt_logger.verbose,
            history_dir=config.observability_stream.history_dir,
            keep_history_files=bool(config.logging.debug),
            contract_info=config.contract.contract_info,
            default_subscriber_memory_bytes=int(
                max(1, config.observability_stream.subscriber_max_memory_mb) * 1024 * 1024
            ),
        )
        obs.set_oms(oms)
        return obs
