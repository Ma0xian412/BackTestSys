"""核心事件处理器：快照、动作、回执。"""

from __future__ import annotations

from typing import List

from .dispatcher import IEventHandler
from .runtime import (
    EVENT_KIND_ACTION_ARRIVAL,
    EVENT_KIND_RECEIPT_DELIVERY,
    EventEnvelope,
    ReceiptStrategyEvent,
    RuntimeContext,
    SnapshotStrategyEvent,
    StrategyContext,
)
from .types import Order, OrderReceipt
from .types import CancelRequest
from .dto import to_snapshot_dto


def _clamp_time(t: int, floor: int) -> int:
    return t if t >= floor else floor


class SnapshotArrivalHandler(IEventHandler):
    """处理 SnapshotArrival 事件。"""

    def handle(self, e: EventEnvelope, ctx: RuntimeContext) -> List[EventEnvelope]:
        snapshot = e.payload
        snapshot_dto = to_snapshot_dto(snapshot)
        ctx.state.update_snapshot(snapshot_dto)

        oms_view = ctx.oms.view()
        sctx = StrategyContext(
            t=e.time,
            snapshot=ctx.state.lastSnapshotDTO,
            omsView=oms_view,
        )
        sev = SnapshotStrategyEvent(time=e.time, snapshot=snapshot_dto)

        actions = ctx.strategy.on_event(sev, sctx) or []
        emitted: List[EventEnvelope] = []
        for action in actions:
            ctx.oms.submit_action(action, e.time)
            if isinstance(action, Order):
                if hasattr(ctx.obs, "register_order"):
                    ctx.obs.register_order(action.order_id, action.qty)
                ctx.diagnostics["orders_submitted"] = ctx.diagnostics.get("orders_submitted", 0) + 1
            if isinstance(action, CancelRequest):
                ctx.diagnostics["cancels_submitted"] = ctx.diagnostics.get("cancels_submitted", 0) + 1
            t_arrive = ctx.timeModel.action_arrival_time(send_time=e.time, action=action)
            emitted.append(
                EventEnvelope(
                    time=_clamp_time(int(t_arrive), e.time),
                    kind=EVENT_KIND_ACTION_ARRIVAL,
                    priority=ctx.eventSpec.priorityOf(EVENT_KIND_ACTION_ARRIVAL),
                    payload=action,
                )
            )
        return emitted


class ActionArrivalHandler(IEventHandler):
    """处理 ActionArrival 事件。"""

    def handle(self, e: EventEnvelope, ctx: RuntimeContext) -> List[EventEnvelope]:
        action = e.payload
        receipts = ctx.venue.onActionArrival(action, t_arrive=e.time) or []

        emitted: List[EventEnvelope] = []
        for receipt in receipts:
            ctx.obs.on_receipt_generated(receipt)
            ctx.diagnostics["receipts_generated"] = ctx.diagnostics.get("receipts_generated", 0) + 1
            t_deliver = ctx.timeModel.receipt_delivery_time(receipt)
            emitted.append(
                EventEnvelope(
                    time=_clamp_time(int(t_deliver), e.time),
                    kind=EVENT_KIND_RECEIPT_DELIVERY,
                    priority=ctx.eventSpec.priorityOf(EVENT_KIND_RECEIPT_DELIVERY),
                    payload=receipt,
                )
            )
        return emitted


class ReceiptDeliveryHandler(IEventHandler):
    """处理 ReceiptDelivery 事件。"""

    def handle(self, e: EventEnvelope, ctx: RuntimeContext) -> List[EventEnvelope]:
        receipt: OrderReceipt = e.payload
        receipt.recv_time = e.time

        ctx.obs.on_receipt_delivered(receipt)
        ctx.oms.apply_receipt(receipt)

        if receipt.receipt_type in {"FILL", "PARTIAL"}:
            ctx.diagnostics["orders_filled"] = ctx.diagnostics.get("orders_filled", 0) + 1

        oms_view = ctx.oms.view()
        sctx = StrategyContext(
            t=e.time,
            snapshot=ctx.state.lastSnapshotDTO,
            omsView=oms_view,
        )
        sev = ReceiptStrategyEvent(time=e.time, receipt=receipt)
        actions = ctx.strategy.on_event(sev, sctx) or []

        emitted: List[EventEnvelope] = []
        for action in actions:
            ctx.oms.submit_action(action, e.time)
            if isinstance(action, Order):
                if hasattr(ctx.obs, "register_order"):
                    ctx.obs.register_order(action.order_id, action.qty)
                ctx.diagnostics["orders_submitted"] = ctx.diagnostics.get("orders_submitted", 0) + 1
            if isinstance(action, CancelRequest):
                ctx.diagnostics["cancels_submitted"] = ctx.diagnostics.get("cancels_submitted", 0) + 1

            t_arrive = ctx.timeModel.action_arrival_time(send_time=e.time, action=action)
            emitted.append(
                EventEnvelope(
                    time=_clamp_time(int(t_arrive), e.time),
                    kind=EVENT_KIND_ACTION_ARRIVAL,
                    priority=ctx.eventSpec.priorityOf(EVENT_KIND_ACTION_ARRIVAL),
                    payload=action,
                )
            )
        return emitted
