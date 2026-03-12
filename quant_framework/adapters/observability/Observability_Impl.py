"""回执记录器 + 流式可观测实现（ingest 单入口）。"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Mapping, Optional

from ...config import ContractInfo
from ...core.data_structure import Event, OrderReceipt
from ...core.observability import (
    EVENT_TYPE_CANCEL_SUBMITTED,
    EVENT_TYPE_INTERVAL_ENDED,
    EVENT_TYPE_OBS_EVENT_INVALID,
    EVENT_TYPE_OMS_ORDER_CHANGED,
    EVENT_TYPE_ORDER_SUBMITTED,
    EVENT_TYPE_RECEIPT_DELIVERED,
    EVENT_TYPE_RECEIPT_GENERATED,
    EVENT_TYPE_RUN_ENDED,
    EVENT_TYPE_RUN_STARTED,
    ObsEventEnvelope,
    ObsSubscriptionOptions,
    ObsSubscriptionStatus,
)
from ...core.port import IObservability, IOMS
from ...core.run_result import BacktestRunResult, RunResultMetadata
from .receipt_logger_utils import (
    ReceiptRecord,
    build_statistics,
    calculate_fill_rate,
    calculate_fill_rate_by_count,
    print_receipt_line,
    print_summary as print_receipt_summary,
    records_as_dicts,
    save_receipts_csv,
)
from .receipt_logger_ingest_utils import (
    payload_for_publish,
    payload_mapping,
    receipt_from_payload,
    validate_cancel_payload,
    validate_oms_change_payload,
    validate_order_payload,
)
from .run_result_builder import RunResultBuilder
from .stream_runtime import ObsStreamRuntime

logger = logging.getLogger(__name__)
_DEFAULT_HISTORY_DIR = ".obs_history"
_DEFAULT_SUBSCRIBER_MEMORY = 8 * 1024 * 1024
ReceiptCallback = Callable[[OrderReceipt], None]


class Observability_Impl(IObservability):
    """可观测实现：接收框架事件流并发布订阅流。"""

    def __init__(
        self,
        output_file: Optional[str] = None,
        verbose: bool = False,
        callback: Optional[ReceiptCallback] = None,
        contract_info: Optional[ContractInfo] = None,
        machine_name: str = "",
        history_dir: str = _DEFAULT_HISTORY_DIR,
        keep_history_files: bool = False,
        default_subscriber_memory_bytes: int = _DEFAULT_SUBSCRIBER_MEMORY,
    ) -> None:
        self.output_file = output_file
        self.verbose = verbose
        self.callback = callback
        self.records: List[ReceiptRecord] = []
        self._oms: Optional[IOMS] = None
        self._runtime = ObsStreamRuntime(
            history_dir=history_dir,
            keep_history_files=keep_history_files,
            default_max_memory_bytes=default_subscriber_memory_bytes,
        )
        self._result_builder = RunResultBuilder(
            self._build_result_metadata(contract_info, machine_name)
        )
        self._run_open = False
        self._run_id: Optional[str] = None
        self._final_time = 0
        self._error: Optional[str] = None
        self._status = "completed"
        self._interrupted = False
        self._interrupt_reason: Optional[str] = None
        self._diagnostics = {
            "intervals_processed": 0,
            "orders_submitted": 0,
            "orders_filled": 0,
            "receipts_generated": 0,
            "cancels_submitted": 0,
        }
        self._event_handlers: Dict[str, Callable[[Event], bool]] = {}
        self._register_builtin_handlers()

    def set_oms(self, oms: IOMS) -> None:
        self._oms = oms

    @property
    def receipt_logger(self) -> "Observability_Impl":
        return self

    def ingest(self, event: Event) -> None:
        if event.type == EVENT_TYPE_RUN_STARTED:
            self._ingest_run_started(event)
            return
        self._ensure_run_started()
        try:
            should_publish = self._dispatch_event(event)
        except ValueError as exc:
            self._publish_invalid_event(event, str(exc))
            return
        if should_publish:
            self._publish_event(event)

    def register_event_handler(self, event_type: str, handler: Callable[[Event], bool]) -> None:
        if not event_type:
            raise ValueError("event_type must not be empty")
        self._event_handlers[str(event_type)] = handler

    def _register_builtin_handlers(self) -> None:
        self.register_event_handler(EVENT_TYPE_ORDER_SUBMITTED, self._handle_order_submitted)
        self.register_event_handler(EVENT_TYPE_CANCEL_SUBMITTED, self._handle_cancel_submitted)
        self.register_event_handler(EVENT_TYPE_RECEIPT_GENERATED, self._handle_receipt_generated)
        self.register_event_handler(EVENT_TYPE_RECEIPT_DELIVERED, self._handle_receipt_delivered)
        self.register_event_handler(EVENT_TYPE_INTERVAL_ENDED, self._handle_interval_ended)
        self.register_event_handler(EVENT_TYPE_OMS_ORDER_CHANGED, self._handle_oms_order_changed)
        self.register_event_handler(EVENT_TYPE_RUN_ENDED, self._handle_run_ended)

    def _dispatch_event(self, event: Event) -> bool:
        handler = self._event_handlers.get(event.type)
        if handler is None:
            logger.warning("Unknown observability event type: %s", event.type)
            return True
        return bool(handler(event))

    def _ingest_run_started(self, event: Event) -> None:
        payload = payload_mapping(event.payload)
        if self._run_open:
            self._runtime.finish_run()
        self._result_builder.reset()
        run_id = payload.get("run_id")
        seed_id = run_id if isinstance(run_id, str) and run_id else (event.run_id or None)
        assigned = self._runtime.start_run(seed_id)
        self._run_open = True
        self._run_id = assigned
        start_payload = dict(payload)
        start_payload["run_id"] = assigned
        self._runtime.publish(EVENT_TYPE_RUN_STARTED, sim_time=int(event.time), payload=start_payload)

    def _handle_order_submitted(self, event: Event) -> bool:
        payload = payload_mapping(event.payload)
        validate_order_payload(payload)
        self._diagnostics["orders_submitted"] += 1
        self._result_builder.record_order_submitted(event.time, payload)
        return True

    def _handle_cancel_submitted(self, event: Event) -> bool:
        payload = payload_mapping(event.payload)
        validate_cancel_payload(payload)
        self._diagnostics["cancels_submitted"] += 1
        self._result_builder.record_cancel_submitted(event.time, payload)
        return True

    def _handle_receipt_generated(self, event: Event) -> bool:
        _ = receipt_from_payload(payload_mapping(event.payload))
        self._diagnostics["receipts_generated"] += 1
        return True

    def _handle_receipt_delivered(self, event: Event) -> bool:
        payload = payload_mapping(event.payload)
        receipt = receipt_from_payload(payload)
        self._log_receipt(receipt)
        order_lookup = getattr(self._oms, "orders", {}) if self._oms is not None else {}
        self._result_builder.record_receipt_delivered(payload, order_lookup)
        if receipt.receipt_type in {"FILL", "PARTIAL"}:
            self._diagnostics["orders_filled"] += 1
        return True

    def _handle_interval_ended(self, event: Event) -> bool:
        _ = payload_mapping(event.payload)
        self._diagnostics["intervals_processed"] += 1
        return True

    def _handle_oms_order_changed(self, event: Event) -> bool:
        validate_oms_change_payload(payload_mapping(event.payload))
        return True

    def _handle_run_ended(self, event: Event) -> bool:
        self._ingest_run_ended(event)
        return False

    def _ingest_run_ended(self, event: Event) -> None:
        payload = payload_mapping(event.payload)
        self._final_time = int(payload.get("final_time", event.time))
        self._error = payload.get("error")
        self._status = str(payload.get("status", "completed"))
        self._interrupted = bool(payload.get("interrupted", self._status == "interrupted"))
        self._interrupt_reason = payload.get("interrupt_reason")
        end_payload = dict(payload)
        if self._run_id:
            end_payload["run_id"] = self._run_id
        self._runtime.publish(EVENT_TYPE_RUN_ENDED, sim_time=int(event.time), payload=end_payload)
        self._runtime.finish_run()
        self._run_open = False

    def _publish_event(self, event: Event) -> None:
        payload = payload_for_publish(event.payload)
        self._runtime.publish(event.type, sim_time=int(event.time), payload=payload)

    def _publish_invalid_event(self, source_event: Event, reason: str) -> None:
        logger.warning("Invalid event payload for type=%s: %s", source_event.type, reason)
        payload = {
            "reason": reason,
            "source_type": source_event.type,
            "source_time": int(source_event.time),
            "source_payload": repr(source_event.payload),
        }
        self._runtime.publish(EVENT_TYPE_OBS_EVENT_INVALID, sim_time=int(source_event.time), payload=payload)

    def _ensure_run_started(self) -> None:
        if self._run_id is not None:
            return
        seed = Event(type=EVENT_TYPE_RUN_STARTED, time=0, payload={"auto_started": True})
        self._ingest_run_started(seed)

    def __del__(self) -> None:
        if not self._run_open:
            return
        try:
            self._runtime.finish_run()
        except Exception:
            return

    def subscribe(self, options: Optional[ObsSubscriptionOptions] = None) -> str:
        return self._runtime.subscribe(options)

    def poll(self, subscription_id: str, max_items: int = 1, timeout_ms: int = 0) -> List[ObsEventEnvelope]:
        try:
            return self._runtime.poll(subscription_id, max_items=max_items, timeout_ms=timeout_ms)
        except RuntimeError as exc:
            if "closed" in str(exc):
                return []
            raise

    def unsubscribe(self, subscription_id: str) -> None:
        self._runtime.unsubscribe(subscription_id)

    def get_subscription_status(self, subscription_id: str) -> ObsSubscriptionStatus:
        return self._runtime.get_subscription_status(subscription_id)

    def get_diagnostics(self) -> dict:
        return dict(self._diagnostics)

    def get_run_result(self) -> BacktestRunResult:
        return self._result_builder.build(self._oms, self._final_time)

    def _log_receipt(self, receipt: OrderReceipt) -> None:
        self.records.append(
            ReceiptRecord(
                order_id=receipt.order_id,
                exch_time=receipt.timestamp,
                recv_time=receipt.recv_time if receipt.recv_time else 0,
                receipt_type=receipt.receipt_type,
                fill_qty=receipt.fill_qty,
                fill_price=receipt.fill_price,
                remaining_qty=receipt.remaining_qty,
            )
        )
        if self.verbose:
            print_receipt_line(receipt)
        if self.callback:
            try:
                self.callback(receipt)
            except Exception as exc:
                logger.warning("Receipt callback raised exception: %s", exc)

    def get_statistics(self) -> dict:
        return build_statistics(self.records, self._oms)

    def calculate_fill_rate(self) -> float:
        return calculate_fill_rate(self._oms)

    def calculate_fill_rate_by_count(self) -> float:
        return calculate_fill_rate_by_count(self._oms)

    def save_to_file(self, filepath: Optional[str] = None) -> None:
        output_path = filepath or self.output_file
        if not output_path:
            raise ValueError("No output file specified")
        save_receipts_csv(self.records, output_path)

    def print_summary(self) -> None:
        print_receipt_summary(self.get_statistics())

    def get_records_as_dicts(self) -> List[Dict]:
        return records_as_dicts(self.records)

    def clear(self) -> None:
        self.records.clear()
        self._result_builder.reset()

    @staticmethod
    def _build_result_metadata(
        contract_info: Optional[ContractInfo],
        machine_name: str,
    ) -> RunResultMetadata:
        resolved_machine_name = str(machine_name).strip()
        if not resolved_machine_name and contract_info is not None:
            resolved_machine_name = str(contract_info.machine_name).strip()
        if contract_info is None:
            return RunResultMetadata(machine_name=resolved_machine_name)
        return RunResultMetadata(
            partition_day=int(contract_info.partition_day),
            contract_id=contract_info.contract_id,
            machine_name=resolved_machine_name,
        )
