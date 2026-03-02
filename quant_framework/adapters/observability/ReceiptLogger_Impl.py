"""回执记录器 — IObservabilitySinks 实现。

纯观察者：记录回执、维护轻量计数、生成报表。
不维护订单状态，统计数据通过 OMS 引用查询。
"""

from __future__ import annotations

import csv
import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable, TYPE_CHECKING

from ...core.port import IObservabilitySinks, IOMS
from ...core.data_structure import CancelRequest, Order, OrderReceipt, OrderStatus

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ReceiptRecord:
    """回执记录。"""
    order_id: str
    exch_time: int
    recv_time: int
    receipt_type: str
    fill_qty: int
    fill_price: float
    remaining_qty: int


@dataclass(frozen=True)
class ObservabilityUpdate:
    event_type: str
    payload: dict


ReceiptCallback = Callable[[OrderReceipt], None]


class ReceiptLogger_Impl(IObservabilitySinks):
    """回执记录器。

    职责：
    1. 记录所有回执到内存 / CSV
    2. 维护轻量诊断计数
    3. 从 OMS 查询统计数据生成报表
    """

    def __init__(
        self,
        output_file: Optional[str] = None,
        verbose: bool = False,
        callback: Optional[ReceiptCallback] = None,
    ):
        self.output_file = output_file
        self.verbose = verbose
        self.callback = callback
        self.records: List[ReceiptRecord] = []

        self._oms: Optional[IOMS] = None
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
        self._next_subscriber_id = 1
        self._subscriber_queues: Dict[int, deque[ObservabilityUpdate]] = {}

    # ── IObservabilitySinks 接口 ────────────────────────────────

    def set_oms(self, oms: IOMS) -> None:
        self._oms = oms

    def on_order_submitted(self, order: Order) -> None:
        self._diagnostics["orders_submitted"] += 1
        self._push_update(
            "order_submitted",
            {"order_id": order.order_id, "qty": order.qty, "price": order.price, "side": order.side.value},
        )

    def on_cancel_submitted(self, request: CancelRequest) -> None:
        self._diagnostics["cancels_submitted"] += 1
        self._push_update("cancel_submitted", {"order_id": request.order_id, "create_time": request.create_time})

    def on_receipt_generated(self, receipt: OrderReceipt) -> None:
        self._diagnostics["receipts_generated"] += 1
        self._push_update("receipt_generated", self._receipt_payload(receipt))

    def on_receipt_delivered(self, receipt: OrderReceipt) -> None:
        self._log_receipt(receipt)
        if receipt.receipt_type in {"FILL", "PARTIAL"}:
            self._diagnostics["orders_filled"] += 1
        self._push_update("receipt_delivered", self._receipt_payload(receipt))

    def on_interval_end(self, stats: object) -> None:
        self._diagnostics["intervals_processed"] += 1
        self._push_update("interval_end", {"stats": stats})

    def on_run_end(self, context: dict) -> None:
        self._final_time = int(context.get("final_time", 0))
        self._error = context.get("error")
        self._status = str(context.get("status", "completed"))
        self._interrupted = bool(context.get("interrupted", self._status == "interrupted"))
        self._interrupt_reason = context.get("interrupt_reason")
        self._push_update("run_end", {"context": dict(context)})

    def get_diagnostics(self) -> dict:
        return dict(self._diagnostics)

    def get_run_result(self) -> dict:
        result: dict = {
            "intervals": self._diagnostics["intervals_processed"],
            "final_time": self._final_time,
            "status": self._status,
            "interrupted": self._interrupted,
            "interrupt_reason": self._interrupt_reason,
            "diagnostics": self.get_diagnostics(),
        }
        if self._error is not None:
            result["error"] = self._error
        return result

    def subscribe_updates(self) -> int:
        subscriber_id = self._next_subscriber_id
        self._next_subscriber_id += 1
        self._subscriber_queues[subscriber_id] = deque()
        return subscriber_id

    def unsubscribe_updates(self, subscriber_id: int) -> None:
        self._subscriber_queues.pop(subscriber_id, None)

    def pull_updates(self, subscriber_id: int) -> List[dict]:
        queue = self._subscriber_queues.get(subscriber_id)
        if queue is None:
            return []
        updates = list(queue)
        queue.clear()
        return [{"event_type": update.event_type, "payload": update.payload} for update in updates]

    # ── 兼容属性 ────────────────────────────────────────────────

    @property
    def receipt_logger(self) -> ReceiptLogger_Impl:
        return self

    # ── 回执记录 ────────────────────────────────────────────────

    def _log_receipt(self, receipt: OrderReceipt) -> None:
        record = ReceiptRecord(
            order_id=receipt.order_id,
            exch_time=receipt.timestamp,
            recv_time=receipt.recv_time if receipt.recv_time else 0,
            receipt_type=receipt.receipt_type,
            fill_qty=receipt.fill_qty,
            fill_price=receipt.fill_price,
            remaining_qty=receipt.remaining_qty,
        )
        self.records.append(record)

        if self.verbose:
            self._print_receipt(receipt)
        if self.callback:
            try:
                self.callback(receipt)
            except Exception as e:
                logger.warning(f"Receipt callback raised exception: {e}")

    def _push_update(self, event_type: str, payload: dict) -> None:
        update = ObservabilityUpdate(event_type=event_type, payload=payload)
        for subscriber_queue in self._subscriber_queues.values():
            subscriber_queue.append(update)

    @staticmethod
    def _receipt_payload(receipt: OrderReceipt) -> dict:
        return {
            "order_id": receipt.order_id,
            "receipt_type": receipt.receipt_type,
            "timestamp": receipt.timestamp,
            "recv_time": receipt.recv_time,
            "fill_qty": receipt.fill_qty,
            "fill_price": receipt.fill_price,
            "remaining_qty": receipt.remaining_qty,
        }

    @staticmethod
    def _print_receipt(receipt: OrderReceipt) -> None:
        print(
            f"[Receipt] {receipt.receipt_type:8s} | "
            f"order_id={receipt.order_id} | "
            f"fill_qty={receipt.fill_qty} | "
            f"fill_price={receipt.fill_price:.2f} | "
            f"remaining={receipt.remaining_qty} | "
            f"timestamp={receipt.timestamp}"
        )

    # ── 统计（从 OMS 查询）────────────────────────────────────

    def get_statistics(self) -> dict:
        receipt_counts = self._count_receipts()

        if self._oms is None:
            return {
                "total_receipts": len(self.records),
                **receipt_counts,
            }

        orders = dict(self._oms.orders)
        total_orders = len(orders)
        total_qty = sum(o.qty for o in orders.values())
        filled_qty = sum(o.filled_qty for o in orders.values())

        fully_filled = 0
        partially_filled = 0
        unfilled = 0
        for o in orders.values():
            if o.qty <= 0:
                continue
            if o.status == OrderStatus.FILLED:
                fully_filled += 1
            elif o.filled_qty > 0:
                partially_filled += 1
            else:
                unfilled += 1

        countable = fully_filled + partially_filled + unfilled
        full_fill_rate = fully_filled / countable if countable else 0.0
        partial_fill_rate = partially_filled / countable if countable else 0.0
        fill_rate_by_qty = filled_qty / total_qty if total_qty else 0.0
        fill_rate_by_count = fully_filled / total_orders if total_orders else 0.0

        return {
            "total_receipts": len(self.records),
            "total_orders": total_orders,
            "total_order_qty": total_qty,
            "total_filled_qty": filled_qty,
            "fill_rate_by_qty": fill_rate_by_qty,
            "fill_rate_by_count": fill_rate_by_count,
            "full_fill_rate": full_fill_rate,
            "partial_fill_rate": partial_fill_rate,
            "fully_filled_orders": fully_filled,
            "partially_filled_orders": partially_filled,
            "unfilled_orders": unfilled,
            **receipt_counts,
        }

    def _count_receipts(self) -> dict:
        partial = full = cancel = reject = 0
        for r in self.records:
            if r.receipt_type == "PARTIAL":
                partial += 1
            elif r.receipt_type == "FILL":
                full += 1
            elif r.receipt_type == "CANCELED":
                cancel += 1
            elif r.receipt_type == "REJECTED":
                reject += 1
        return {
            "partial_fill_count": partial,
            "full_fill_count": full,
            "cancel_count": cancel,
            "reject_count": reject,
        }

    def calculate_fill_rate(self) -> float:
        if self._oms is None:
            return 0.0
        orders = self._oms.orders
        total_qty = sum(o.qty for o in orders.values())
        if total_qty == 0:
            return 0.0
        filled_qty = sum(o.filled_qty for o in orders.values())
        return filled_qty / total_qty

    def calculate_fill_rate_by_count(self) -> float:
        if self._oms is None:
            return 0.0
        orders = self._oms.orders
        total = len(orders)
        if total == 0:
            return 0.0
        fully_filled = sum(1 for o in orders.values() if o.status == OrderStatus.FILLED)
        return fully_filled / total

    # ── CSV / 打印 ─────────────────────────────────────────────

    def save_to_file(self, filepath: Optional[str] = None) -> None:
        output_path = filepath or self.output_file
        if not output_path:
            raise ValueError("No output file specified")
        dir_path = os.path.dirname(output_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "order_id", "exch_time", "recv_time", "receipt_type",
                "fill_qty", "fill_price", "remaining_qty",
            ])
            writer.writeheader()
            for r in self.records:
                writer.writerow({
                    "order_id": r.order_id,
                    "exch_time": r.exch_time,
                    "recv_time": r.recv_time,
                    "receipt_type": r.receipt_type,
                    "fill_qty": r.fill_qty,
                    "fill_price": r.fill_price,
                    "remaining_qty": r.remaining_qty,
                })

    def print_summary(self) -> None:
        stats = self.get_statistics()
        print("\n" + "=" * 60)
        print("Receipt Logger Summary")
        print("=" * 60)
        print(f"Total Receipts: {stats['total_receipts']}")
        if "total_orders" in stats:
            print(f"Total Orders: {stats['total_orders']}")
            print(f"Total Filled Qty: {stats['total_filled_qty']}")
            print(f"Full Fill Rate: {stats.get('full_fill_rate', 0):.2%}")
            print(f"Fill Rate (by qty): {stats.get('fill_rate_by_qty', 0):.2%}")
        print("=" * 60)

    def get_records_as_dicts(self) -> List[Dict]:
        return [
            {
                "order_id": r.order_id,
                "exch_time": r.exch_time,
                "recv_time": r.recv_time,
                "receipt_type": r.receipt_type,
                "fill_qty": r.fill_qty,
                "fill_price": r.fill_price,
                "remaining_qty": r.remaining_qty,
            }
            for r in self.records
        ]

    def clear(self) -> None:
        self.records.clear()


class NullObservability_Impl(ReceiptLogger_Impl):
    """无输出观测实现（保留诊断计数）。"""

    def __init__(self) -> None:
        super().__init__(output_file=None, verbose=False, callback=None)
