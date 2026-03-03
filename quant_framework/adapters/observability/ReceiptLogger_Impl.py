"""回执记录器 + 流式可观测对外接口实现。"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Mapping, Optional

from ...core.data_structure import CancelRequest, Order, OrderReceipt
from ...core.observability import (
    EVENT_TYPE_CANCEL_SUBMITTED,
    EVENT_TYPE_INTERVAL_ENDED,
    EVENT_TYPE_OMS_ORDER_CHANGED,
    EVENT_TYPE_ORDER_SUBMITTED,
    EVENT_TYPE_RECEIPT_DELIVERED,
    EVENT_TYPE_RECEIPT_GENERATED,
    EVENT_TYPE_RUN_ENDED,
    EVENT_TYPE_RUN_STARTED,
    OMSOrderChange,
    ObsEventEnvelope,
    ObsSubscriptionOptions,
    ObsSubscriptionStatus,
)
from ...core.port import IObservability, IOMS
from .receipt_logger_utils import (
    build_statistics,
    calculate_fill_rate,
    calculate_fill_rate_by_count,
    print_receipt_line,
    print_summary as print_receipt_summary,
    ReceiptRecord,
    records_as_dicts,
    save_receipts_csv,
)
from .stream_runtime import ObsStreamRuntime
logger = logging.getLogger(__name__)
_DEFAULT_HISTORY_DIR = ".obs_history"
_DEFAULT_SUBSCRIBER_MEMORY = 8 * 1024 * 1024
ReceiptCallback = Callable[[OrderReceipt], None]

class ReceiptLogger_Impl(IObservability):
    """可观测性实现：统计 + 流式事件分发。"""

    def __init__(
        self,
        output_file: Optional[str] = None,
        verbose: bool = False,
        callback: Optional[ReceiptCallback] = None,
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

    def set_oms(self, oms: IOMS) -> None:
        self._oms = oms

    @property
    def receipt_logger(self) -> "ReceiptLogger_Impl":
        return self

    def on_run_started(self, context: dict) -> None:
        if self._run_open:
            self._runtime.finish_run()
        run_id = context.get("run_id")
        assigned = self._runtime.start_run(run_id if isinstance(run_id, str) and run_id else None)
        self._run_open = True
        self._run_id = assigned
        payload = dict(context)
        payload["run_id"] = assigned
        self._runtime.publish(EVENT_TYPE_RUN_STARTED, sim_time=int(context.get("sim_time", 0)), payload=payload)

    def on_order_submitted(self, order: Order) -> None:
        self._ensure_run_started()
        self._diagnostics["orders_submitted"] += 1
        self._runtime.publish(
            EVENT_TYPE_ORDER_SUBMITTED,
            sim_time=int(order.create_time),
            payload={
                "order_id": order.order_id,
                "side": str(order.side.value),
                "price": float(order.price),
                "qty": int(order.qty),
                "status": str(order.status.value),
            },
        )

    def on_cancel_submitted(self, request: CancelRequest) -> None:
        self._ensure_run_started()
        self._diagnostics["cancels_submitted"] += 1
        self._runtime.publish(
            EVENT_TYPE_CANCEL_SUBMITTED,
            sim_time=int(request.create_time),
            payload={"order_id": request.order_id, "create_time": int(request.create_time)},
        )

    def on_receipt_generated(self, receipt: OrderReceipt) -> None:
        self._ensure_run_started()
        self._diagnostics["receipts_generated"] += 1
        self._runtime.publish(
            EVENT_TYPE_RECEIPT_GENERATED,
            sim_time=int(receipt.timestamp),
            payload=self._receipt_payload(receipt),
        )

    def on_receipt_delivered(self, receipt: OrderReceipt) -> None:
        self._ensure_run_started()
        self._log_receipt(receipt)
        if receipt.receipt_type in {"FILL", "PARTIAL"}:
            self._diagnostics["orders_filled"] += 1
        self._runtime.publish(
            EVENT_TYPE_RECEIPT_DELIVERED,
            sim_time=int(receipt.recv_time or receipt.timestamp),
            payload=self._receipt_payload(receipt),
        )

    def on_interval_end(self, stats: object) -> None:
        self._ensure_run_started()
        self._diagnostics["intervals_processed"] += 1
        payload = stats if isinstance(stats, Mapping) else {"stats_repr": repr(stats)}
        sim_time = 0
        if isinstance(stats, Mapping):
            sim_time = int(stats.get("interval_end", 0) or 0)
        self._runtime.publish(EVENT_TYPE_INTERVAL_ENDED, sim_time=sim_time, payload=dict(payload))

    def on_oms_order_changed(self, change: OMSOrderChange) -> None:
        self._ensure_run_started()
        self._runtime.publish(
            EVENT_TYPE_OMS_ORDER_CHANGED,
            sim_time=int(change.timestamp),
            payload={
                "order_id": change.order_id,
                "prev_status": change.prev_status,
                "new_status": change.new_status,
                "prev_filled_qty": change.prev_filled_qty,
                "new_filled_qty": change.new_filled_qty,
                "prev_remaining_qty": change.prev_remaining_qty,
                "new_remaining_qty": change.new_remaining_qty,
                "timestamp": change.timestamp,
            },
        )

    def on_run_end(self, context: dict) -> None:
        self._ensure_run_started()
        self._final_time = int(context.get("final_time", 0))
        self._error = context.get("error")
        self._status = str(context.get("status", "completed"))
        self._interrupted = bool(context.get("interrupted", self._status == "interrupted"))
        self._interrupt_reason = context.get("interrupt_reason")
        payload = dict(context)
        if self._run_id:
            payload["run_id"] = self._run_id
        self._runtime.publish(
            EVENT_TYPE_RUN_ENDED,
            sim_time=int(context.get("final_time", 0)),
            payload=payload,
        )
        self._runtime.finish_run()
        self._run_open = False

    def emit_custom(self, event_type: str, sim_time: int, payload: Mapping[str, object]) -> None:
        self._ensure_run_started()
        if not event_type:
            raise ValueError("event_type must not be empty")
        self._runtime.publish(event_type=event_type, sim_time=sim_time, payload=payload)

    def get_diagnostics(self) -> dict:
        return dict(self._diagnostics)

    def get_run_result(self) -> dict:
        result: Dict[str, object] = {
            "intervals": self._diagnostics["intervals_processed"],
            "final_time": self._final_time,
            "status": self._status,
            "interrupted": self._interrupted,
            "interrupt_reason": self._interrupt_reason,
            "diagnostics": self.get_diagnostics(),
        }
        if self._error is not None:
            result["error"] = self._error
        if self._run_id is not None:
            result["run_id"] = self._run_id
        return result

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

    @staticmethod
    def _receipt_payload(receipt: OrderReceipt) -> Dict[str, object]:
        return {
            "order_id": receipt.order_id,
            "receipt_type": receipt.receipt_type,
            "timestamp": int(receipt.timestamp),
            "fill_qty": int(receipt.fill_qty),
            "fill_price": float(receipt.fill_price),
            "remaining_qty": int(receipt.remaining_qty),
            "pos": int(receipt.pos),
            "recv_time": int(receipt.recv_time or 0),
        }

    def _ensure_run_started(self) -> None:
        if self._run_id is not None:
            return
        assigned = self._runtime.start_run(None)
        self._run_open = True
        self._run_id = assigned
        self._runtime.publish(
            EVENT_TYPE_RUN_STARTED,
            sim_time=0,
            payload={"run_id": assigned, "auto_started": True},
        )

    def __del__(self) -> None:
        if not self._run_open:
            return
        try:
            self._runtime.finish_run()
        except Exception:
            return

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

