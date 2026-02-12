"""可观测性端口适配器实现。"""

from __future__ import annotations

from ...core.port import IObservabilitySinks
from ...core.data_structure import CancelRequest, Order, OrderReceipt
from .receipt_logger import ReceiptLogger


class ObservabilityImpl(IObservabilitySinks):
    """默认可观测性实现：日志 + 诊断计数。"""

    def __init__(self, receipt_logger: ReceiptLogger | None = None):
        self._logger = receipt_logger
        self._final_time = 0
        self._error: str | None = None
        self._diagnostics = {
            "intervals_processed": 0,
            "orders_submitted": 0,
            "orders_filled": 0,
            "receipts_generated": 0,
            "cancels_submitted": 0,
        }

    @property
    def receipt_logger(self) -> ReceiptLogger | None:
        return self._logger

    def on_order_submitted(self, order: Order) -> None:
        self._diagnostics["orders_submitted"] += 1
        if self._logger is not None:
            self._logger.register_order(order.order_id, order.qty)

    def on_cancel_submitted(self, request: CancelRequest) -> None:
        self._diagnostics["cancels_submitted"] += 1

    def on_receipt_generated(self, receipt: OrderReceipt) -> None:
        self._diagnostics["receipts_generated"] += 1

    def on_receipt_delivered(self, receipt: OrderReceipt) -> None:
        if self._logger is not None:
            self._logger.log_receipt(receipt)
        if receipt.receipt_type in {"FILL", "PARTIAL"}:
            self._diagnostics["orders_filled"] += 1

    def on_interval_end(self, stats: object) -> None:
        self._diagnostics["intervals_processed"] += 1

    def on_run_end(self, final_time: int, error: str | None = None) -> None:
        self._final_time = int(final_time)
        self._error = error

    def get_diagnostics(self) -> dict:
        return dict(self._diagnostics)

    def get_run_result(self) -> dict:
        result = {
            "intervals": self._diagnostics["intervals_processed"],
            "final_time": self._final_time,
            "diagnostics": self.get_diagnostics(),
        }
        if self._error is not None:
            result["error"] = self._error
        return result


class NullObservabilityImpl(ObservabilityImpl):
    """无输出观测实现（但保留诊断计数）。"""

    def __init__(self) -> None:
        super().__init__(receipt_logger=None)
