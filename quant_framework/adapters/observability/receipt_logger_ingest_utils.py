"""ReceiptLogger ingest 解析与校验工具。"""

from __future__ import annotations

from typing import Dict, Mapping

from ...core.data_structure import OrderReceipt


def payload_mapping(payload: object) -> Mapping[str, object]:
    if isinstance(payload, Mapping):
        return payload
    raise ValueError("payload must be a mapping")


def payload_for_publish(payload: object) -> Dict[str, object]:
    if isinstance(payload, Mapping):
        return dict(payload)
    return {"value_repr": repr(payload)}


def validate_order_payload(payload: Mapping[str, object]) -> None:
    required = ("order_id", "side", "price", "qty", "status")
    if any(key not in payload for key in required):
        raise ValueError(f"missing keys: {required}")


def validate_cancel_payload(payload: Mapping[str, object]) -> None:
    required = ("order_id", "create_time")
    if any(key not in payload for key in required):
        raise ValueError(f"missing keys: {required}")


def validate_oms_change_payload(payload: Mapping[str, object]) -> None:
    required = (
        "order_id",
        "prev_status",
        "new_status",
        "prev_filled_qty",
        "new_filled_qty",
        "prev_remaining_qty",
        "new_remaining_qty",
        "timestamp",
    )
    if any(key not in payload for key in required):
        raise ValueError(f"missing keys: {required}")


def receipt_from_payload(payload: Mapping[str, object]) -> OrderReceipt:
    required = ("order_id", "receipt_type", "timestamp", "fill_qty", "fill_price", "remaining_qty", "pos")
    if any(key not in payload for key in required):
        raise ValueError(f"missing keys: {required}")
    return OrderReceipt(
        order_id=str(payload["order_id"]),
        receipt_type=str(payload["receipt_type"]),
        timestamp=int(payload["timestamp"]),
        fill_qty=int(payload["fill_qty"]),
        fill_price=float(payload["fill_price"]),
        remaining_qty=int(payload["remaining_qty"]),
        pos=int(payload["pos"]),
        recv_time=int(payload.get("recv_time") or 0),
    )
