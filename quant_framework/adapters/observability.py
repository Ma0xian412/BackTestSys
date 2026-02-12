"""可观测性端口适配器。"""

from __future__ import annotations

from ..core.interfaces import IObservabilitySinks
from ..core.types import OrderReceipt
from ..trading.receipt_logger import ReceiptLogger


class NullObservabilitySinks(IObservabilitySinks):
    """空实现，用于测试或禁用观测。"""

    def on_receipt_generated(self, receipt: OrderReceipt) -> None:
        return

    def on_receipt_delivered(self, receipt: OrderReceipt) -> None:
        return

    def on_interval_end(self, stats: object) -> None:
        return


class ReceiptLoggerSink(IObservabilitySinks):
    """将 IObservabilitySinks 适配到现有 ReceiptLogger。"""

    def __init__(self, receipt_logger: ReceiptLogger):
        self._logger = receipt_logger

    @property
    def receipt_logger(self) -> ReceiptLogger:
        return self._logger

    def register_order(self, order_id: str, qty: int) -> None:
        self._logger.register_order(order_id, qty)

    def on_receipt_generated(self, receipt: OrderReceipt) -> None:
        return

    def on_receipt_delivered(self, receipt: OrderReceipt) -> None:
        self._logger.log_receipt(receipt)

    def on_interval_end(self, stats: object) -> None:
        return
