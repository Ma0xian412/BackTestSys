"""核心事件处理器：快照、动作、回执。"""

from __future__ import annotations

from typing import List

from .data_structure import (
    CancelRequest,
    EVENT_KIND_ACTION_ARRIVAL,
    EVENT_KIND_RECEIPT_DELIVERY,
    Action,
    ActionType,
    Event,
    Order,
    OrderReceipt,
    RuntimeContext,
    StrategyContext,
)
from .dispatcher import IEventHandler


def _clamp_time(t: int, floor: int) -> int:
    return t if t >= floor else floor


class SnapshotArrivalHandler(IEventHandler):
    """处理 SnapshotArrival 事件。"""

    def handle(self, e: Event, ctx: RuntimeContext) -> List[Event]:
        snapshot = e.payload
        ctx.last_snapshot = snapshot
        oms_view = ctx.oms.view()
        sctx = StrategyContext(
            t=e.time,
            snapshot=ctx.last_snapshot,
            omsView=oms_view,
        )
        actions: List[Action] = ctx.strategy.on_event(e, sctx) or []
        emitted: List[Event] = []
        for action in actions:
            send_time = int(action.create_time) if int(action.create_time) > 0 else int(e.time)
            action.create_time = send_time
            if action.action_type == ActionType.PLACE_ORDER:
                order: Order = action.payload
                ctx.oms.submit_order(order, send_time)
                ctx.obs.on_order_submitted(order)
            elif action.action_type == ActionType.CANCEL_ORDER:
                request: CancelRequest = action.payload
                ctx.oms.submit_cancel(request, send_time)
                ctx.obs.on_cancel_submitted(request)
            else:
                raise ValueError(f"Unsupported action type: {action.action_type!r}")
            t_arrive = ctx.timeModel.delayout(send_time)
            emitted.append(
                Event(
                    time=_clamp_time(int(t_arrive), e.time),
                    kind=EVENT_KIND_ACTION_ARRIVAL,
                    priority=ctx.eventSpec.priorityOf(EVENT_KIND_ACTION_ARRIVAL),
                    payload=action,
                )
            )
        return emitted


class ActionArrivalHandler(IEventHandler):
    """处理 ActionArrival 事件。"""

    def handle(self, e: Event, ctx: RuntimeContext) -> List[Event]:
        action: Action = e.payload
        receipts = ctx.venue.onActionArrival(action, t_arrive=e.time) or []

        emitted: List[Event] = []
        for receipt in receipts:
            ctx.obs.on_receipt_generated(receipt)
            t_deliver = ctx.timeModel.delayin(int(receipt.timestamp))
            emitted.append(
                Event(
                    time=_clamp_time(int(t_deliver), e.time),
                    kind=EVENT_KIND_RECEIPT_DELIVERY,
                    priority=ctx.eventSpec.priorityOf(EVENT_KIND_RECEIPT_DELIVERY),
                    payload=receipt,
                )
            )
        return emitted


class ReceiptDeliveryHandler(IEventHandler):
    """处理 ReceiptDelivery 事件。"""

    def handle(self, e: Event, ctx: RuntimeContext) -> List[Event]:
        receipt: OrderReceipt = e.payload
        receipt.recv_time = e.time

        ctx.obs.on_receipt_delivered(receipt)
        ctx.oms.apply_receipt(receipt)

        oms_view = ctx.oms.view()
        sctx = StrategyContext(
            t=e.time,
            snapshot=ctx.last_snapshot,
            omsView=oms_view,
        )
        actions: List[Action] = ctx.strategy.on_event(e, sctx) or []

        emitted: List[Event] = []
        for action in actions:
            send_time = int(action.create_time) if int(action.create_time) > 0 else int(e.time)
            action.create_time = send_time
            if action.action_type == ActionType.PLACE_ORDER:
                order: Order = action.payload
                ctx.oms.submit_order(order, send_time)
                ctx.obs.on_order_submitted(order)
            elif action.action_type == ActionType.CANCEL_ORDER:
                request: CancelRequest = action.payload
                ctx.oms.submit_cancel(request, send_time)
                ctx.obs.on_cancel_submitted(request)
            else:
                raise ValueError(f"Unsupported action type: {action.action_type!r}")
            t_arrive = ctx.timeModel.delayout(send_time)
            emitted.append(
                Event(
                    time=_clamp_time(int(t_arrive), e.time),
                    kind=EVENT_KIND_ACTION_ARRIVAL,
                    priority=ctx.eventSpec.priorityOf(EVENT_KIND_ACTION_ARRIVAL),
                    payload=action,
                )
            )
        return emitted
