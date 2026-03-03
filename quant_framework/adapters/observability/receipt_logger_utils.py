"""ReceiptLogger 统计与输出工具。"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ...core.data_structure import OrderStatus


@dataclass
class ReceiptRecord:
    order_id: str
    exch_time: int
    recv_time: int
    receipt_type: str
    fill_qty: int
    fill_price: float
    remaining_qty: int


def count_receipts(records: Iterable[Any]) -> dict:
    counts = {"PARTIAL": 0, "FILL": 0, "CANCELED": 0, "REJECTED": 0}
    for record in records:
        receipt_type = getattr(record, "receipt_type", "")
        if receipt_type in counts:
            counts[receipt_type] += 1
    return {
        "partial_fill_count": counts["PARTIAL"],
        "full_fill_count": counts["FILL"],
        "cancel_count": counts["CANCELED"],
        "reject_count": counts["REJECTED"],
    }


def split_order_states(orders: Mapping[str, Any]) -> tuple[int, int, int]:
    full = partial = unfilled = 0
    for order in orders.values():
        if order.qty <= 0:
            continue
        if order.status == OrderStatus.FILLED:
            full += 1
        elif order.filled_qty > 0:
            partial += 1
        else:
            unfilled += 1
    return full, partial, unfilled


def build_statistics(records: List[Any], oms: Optional[Any]) -> dict:
    receipt_counts = count_receipts(records)
    if oms is None:
        return {"total_receipts": len(records), **receipt_counts}

    orders = dict(oms.orders)
    total_orders = len(orders)
    total_qty = sum(o.qty for o in orders.values())
    filled_qty = sum(o.filled_qty for o in orders.values())
    fully_filled, partially_filled, unfilled = split_order_states(orders)
    countable = fully_filled + partially_filled + unfilled
    return {
        "total_receipts": len(records),
        "total_orders": total_orders,
        "total_order_qty": total_qty,
        "total_filled_qty": filled_qty,
        "fill_rate_by_qty": filled_qty / total_qty if total_qty else 0.0,
        "fill_rate_by_count": fully_filled / total_orders if total_orders else 0.0,
        "full_fill_rate": fully_filled / countable if countable else 0.0,
        "partial_fill_rate": partially_filled / countable if countable else 0.0,
        "fully_filled_orders": fully_filled,
        "partially_filled_orders": partially_filled,
        "unfilled_orders": unfilled,
        **receipt_counts,
    }


def calculate_fill_rate(oms: Optional[Any]) -> float:
    if oms is None:
        return 0.0
    total_qty = sum(o.qty for o in oms.orders.values())
    if total_qty == 0:
        return 0.0
    return sum(o.filled_qty for o in oms.orders.values()) / total_qty


def calculate_fill_rate_by_count(oms: Optional[Any]) -> float:
    if oms is None:
        return 0.0
    total = len(oms.orders)
    if total == 0:
        return 0.0
    full = sum(1 for o in oms.orders.values() if o.status == OrderStatus.FILLED)
    return full / total


def save_receipts_csv(records: List[Any], output_path: str) -> None:
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["order_id", "exch_time", "recv_time", "receipt_type", "fill_qty", "fill_price", "remaining_qty"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def print_summary(stats: Mapping[str, Any]) -> None:
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


def records_as_dicts(records: List[Any]) -> List[Dict]:
    return [record.__dict__.copy() for record in records]


def print_receipt_line(receipt: Any) -> None:
    print(
        f"[Receipt] {receipt.receipt_type:8s} | order_id={receipt.order_id} | "
        f"fill_qty={receipt.fill_qty} | fill_price={receipt.fill_price:.2f} | "
        f"remaining={receipt.remaining_qty} | timestamp={receipt.timestamp}"
    )
