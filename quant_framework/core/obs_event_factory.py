"""框架观测事件构造器。"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .data_structure import CancelRequest, Event, Order, OrderReceipt
from .observability import (
    EVENT_TYPE_CANCEL_SUBMITTED,
    EVENT_TYPE_INTERVAL_ENDED,
    EVENT_TYPE_OMS_ORDER_CHANGED,
    EVENT_TYPE_ORDER_SUBMITTED,
    EVENT_TYPE_RECEIPT_DELIVERED,
    EVENT_TYPE_RECEIPT_GENERATED,
    EVENT_TYPE_RUN_ENDED,
    EVENT_TYPE_RUN_STARTED,
    OMSOrderChange,
)


def make_obs_event(event_type: str, sim_time: int, payload: Mapping[str, Any]) -> Event:
    return Event(type=event_type, time=int(sim_time), payload=dict(payload), priority=0)


def make_run_started_event(sim_time: int, context: Mapping[str, Any]) -> Event:
    return make_obs_event(EVENT_TYPE_RUN_STARTED, sim_time, context)


def make_run_ended_event(sim_time: int, context: Mapping[str, Any]) -> Event:
    return make_obs_event(EVENT_TYPE_RUN_ENDED, sim_time, context)


def make_order_submitted_event(order: Order) -> Event:
    payload: Dict[str, Any] = {
        "order_id": order.order_id,
        "side": str(order.side.value),
        "price": float(order.price),
        "qty": int(order.qty),
        "status": str(order.status.value),
    }
    return make_obs_event(EVENT_TYPE_ORDER_SUBMITTED, int(order.create_time), payload)


def make_cancel_submitted_event(request: CancelRequest) -> Event:
    payload: Dict[str, Any] = {
        "order_id": request.order_id,
        "create_time": int(request.create_time),
    }
    return make_obs_event(EVENT_TYPE_CANCEL_SUBMITTED, int(request.create_time), payload)


def make_receipt_generated_event(receipt: OrderReceipt) -> Event:
    return make_obs_event(EVENT_TYPE_RECEIPT_GENERATED, int(receipt.timestamp), _receipt_payload(receipt))


def make_receipt_delivered_event(receipt: OrderReceipt) -> Event:
    recv_time = int(receipt.recv_time or receipt.timestamp)
    return make_obs_event(EVENT_TYPE_RECEIPT_DELIVERED, recv_time, _receipt_payload(receipt))


def make_interval_ended_event(stats: object) -> Event:
    if isinstance(stats, Mapping):
        payload = dict(stats)
        sim_time = int(payload.get("interval_end", 0) or 0)
    else:
        payload = {"stats_repr": repr(stats)}
        sim_time = 0
    return make_obs_event(EVENT_TYPE_INTERVAL_ENDED, sim_time, payload)


def make_oms_order_changed_event(change: OMSOrderChange) -> Event:
    payload: Dict[str, Any] = {
        "order_id": change.order_id,
        "prev_status": change.prev_status,
        "new_status": change.new_status,
        "prev_filled_qty": int(change.prev_filled_qty),
        "new_filled_qty": int(change.new_filled_qty),
        "prev_remaining_qty": int(change.prev_remaining_qty),
        "new_remaining_qty": int(change.new_remaining_qty),
        "timestamp": int(change.timestamp),
    }
    return make_obs_event(EVENT_TYPE_OMS_ORDER_CHANGED, int(change.timestamp), payload)


def _receipt_payload(receipt: OrderReceipt) -> Dict[str, Any]:
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
