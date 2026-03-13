"""BacktestConfig → RuntimeBuildConfig 工厂。

从 BacktestConfig 创建所有具体 adapter 实例并组装为 RuntimeBuildConfig。
所有 adapter 的具体 import 集中在此文件。
"""

from __future__ import annotations

from typing import Any

from ..config import BacktestConfig, ContractInfo
from ..core.app import RuntimeBuildConfig
from .execution_venue import ExecutionVenue_Impl, SegmentBaseAlgorithm, Simulator_Impl
from .interval_model import TapeConfig as BuilderTapeConfig, UnifiedIntervalModel_impl
from .market_data_feed import CsvMarketDataFeed_Impl, PickleMarketDataFeed_Impl, SnapshotDuplicatingFeed_Impl
from .IOMS.oms import OMS_Impl, Portfolio
from .IStrategy import SimpleStrategy_Impl, ReplayStrategy_Impl
from .observability.Observability_Impl import Observability_Impl
from .time_model import TimeModel_Impl


class BacktestConfigFactory:
    """从 BacktestConfig 创建 RuntimeBuildConfig。"""

    def create(self, config: BacktestConfig) -> RuntimeBuildConfig:
        feed = self._create_feed(config)
        tape_builder = self._create_tape_builder(config)
        venue = self._create_venue(config, tape_builder, feed)
        strategy = self._create_strategy(config)
        inferred_metadata = self._collect_inferred_result_metadata(feed, strategy)
        oms = self._create_oms(config)
        time_model = self._create_time_model(config)
        obs = self._create_observability(config, oms, inferred_metadata)
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
        name = config.strategy.name
        params = config.strategy.params
        if name in ("ReplayStrategy", "ReplayStrategy_Impl"):
            order_file = params.order_file or None
            cancel_file = params.cancel_file or None
            return ReplayStrategy_Impl(
                name=name,
                order_file=order_file,
                cancel_file=cancel_file,
            )
        return SimpleStrategy_Impl(name=name)

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
    def _create_observability(
        config: BacktestConfig,
        oms: OMS_Impl,
        inferred_metadata: dict[str, object],
    ) -> Observability_Impl:
        resolved_result_contract_id = BacktestConfigFactory._resolve_result_contract_id(
            config.contract.contract_info,
            inferred_metadata,
        )
        resolved_contract_info = BacktestConfigFactory._resolve_contract_info(
            config.contract.contract_info,
            inferred_metadata,
        )
        resolved_machine_name = str(config.contract.machine_name or "").strip()
        if not resolved_machine_name:
            resolved_machine_name = str(inferred_metadata.get("machine_name", "")).strip()
        obs = Observability_Impl(
            output_file=config.receipt_logger.output_file or None,
            verbose=config.receipt_logger.verbose,
            history_dir=config.observability_stream.history_dir,
            keep_history_files=bool(config.logging.debug),
            contract_info=resolved_contract_info,
            machine_name=resolved_machine_name,
            result_contract_id=resolved_result_contract_id,
            default_subscriber_memory_bytes=int(
                max(1, config.observability_stream.subscriber_max_memory_mb) * 1024 * 1024
            ),
        )
        obs.set_oms(oms)
        return obs

    @staticmethod
    def _collect_inferred_result_metadata(feed: Any, strategy: Any) -> dict[str, object]:
        merged: dict[str, object] = {}
        for source in (strategy, getattr(feed, "inner_feed", None), feed):
            inferred = BacktestConfigFactory._read_inferred_result_metadata(source)
            for key, value in inferred.items():
                merged.setdefault(key, value)
        return merged

    @staticmethod
    def _read_inferred_result_metadata(source: Any) -> dict[str, object]:
        getter = getattr(source, "get_inferred_result_metadata", None)
        if getter is None or not callable(getter):
            return {}
        raw = getter()
        if not isinstance(raw, dict):
            return {}
        contract_id = BacktestConfigFactory._non_zero_text(raw.get("contract_id"))
        partition_day = BacktestConfigFactory._positive_int(raw.get("partition_day"))
        machine_name = str(raw.get("machine_name", "")).strip()
        out: dict[str, object] = {}
        if contract_id:
            out["contract_id"] = contract_id
        if partition_day > 0:
            out["partition_day"] = partition_day
        if machine_name:
            out["machine_name"] = machine_name
        return out

    @staticmethod
    def _resolve_result_contract_id(
        base_info: ContractInfo | None,
        inferred_metadata: dict[str, object],
    ) -> str:
        if base_info is not None:
            base_contract_id = BacktestConfigFactory._non_zero_text(base_info.contract_id)
            if base_contract_id:
                return base_contract_id
        return BacktestConfigFactory._non_zero_text(inferred_metadata.get("contract_id"))

    @staticmethod
    def _resolve_contract_info(
        base_info: ContractInfo | None,
        inferred_metadata: dict[str, object],
    ) -> ContractInfo | None:
        inferred_partition_day = BacktestConfigFactory._positive_int(inferred_metadata.get("partition_day"))
        inferred_machine_name = str(inferred_metadata.get("machine_name", "")).strip()
        if base_info is None:
            if inferred_partition_day <= 0 and not inferred_machine_name:
                return None
            return ContractInfo(
                partition_day=inferred_partition_day,
                machine_name=inferred_machine_name,
            )
        resolved_partition_day = int(base_info.partition_day)
        if resolved_partition_day <= 0 and inferred_partition_day > 0:
            resolved_partition_day = inferred_partition_day
        resolved_machine_name = str(base_info.machine_name).strip() or inferred_machine_name
        return ContractInfo(
            contract_id=int(base_info.contract_id),
            partition_day=resolved_partition_day,
            tick_size=float(base_info.tick_size),
            exchange_code=base_info.exchange_code,
            machine_name=resolved_machine_name,
            trading_hours=list(base_info.trading_hours),
        )

    @staticmethod
    def _positive_int(value: object) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 0
        return parsed if parsed > 0 else 0

    @staticmethod
    def _non_zero_text(value: object) -> str:
        text = str(value).strip()
        if not text or text == "0":
            return ""
        return text
