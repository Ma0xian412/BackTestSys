"""新架构事件调度单元测试。"""

import heapq

from quant_framework.core.types import (
    RequestType, ReceiptType, CancelRequest, OrderReceipt, TICK_PER_MS,
)
from quant_framework.core.runtime import (
    EVENT_KIND_ACTION_ARRIVAL,
    EVENT_KIND_RECEIPT_DELIVERY,
    EVENT_KIND_SNAPSHOT_ARRIVAL,
    Event,
    EventSpecRegistry,
    reset_event_seq,
)
from quant_framework.core.scheduler import HeapScheduler


def test_event_priority_ordering():
    """同一时刻按 (priority, seq) 进行确定性排序。"""
    reset_event_seq()
    event_spec = EventSpecRegistry.default()

    t = 1000 * TICK_PER_MS
    events = [
        Event(time=t, kind=EVENT_KIND_RECEIPT_DELIVERY, priority=event_spec.priorityOf(EVENT_KIND_RECEIPT_DELIVERY), payload="receipt1"),
        Event(time=t, kind=EVENT_KIND_SNAPSHOT_ARRIVAL, priority=event_spec.priorityOf(EVENT_KIND_SNAPSHOT_ARRIVAL), payload="snapshot"),
        Event(time=t, kind=EVENT_KIND_ACTION_ARRIVAL, priority=event_spec.priorityOf(EVENT_KIND_ACTION_ARRIVAL), payload="order"),
        Event(time=t, kind=EVENT_KIND_RECEIPT_DELIVERY, priority=event_spec.priorityOf(EVENT_KIND_RECEIPT_DELIVERY), payload="receipt2"),
    ]

    heap = []
    for e in events:
        heapq.heappush(heap, e)

    popped = []
    while heap:
        popped.append(heapq.heappop(heap))

    expected = [
        EVENT_KIND_SNAPSHOT_ARRIVAL,
        EVENT_KIND_ACTION_ARRIVAL,
        EVENT_KIND_RECEIPT_DELIVERY,
        EVENT_KIND_RECEIPT_DELIVERY,
    ]
    actual = [e.kind for e in popped]
    assert actual == expected, f"事件顺序不正确\n期望: {expected}\n实际: {actual}"

    # 同类型按 seq 排序
    receipts = [e for e in popped if e.kind == EVENT_KIND_RECEIPT_DELIVERY]
    assert receipts[0].payload == "receipt1" and receipts[1].payload == "receipt2"

    # 多次运行确定性一致
    for _ in range(3):
        reset_event_seq()
        h = []
        for e in events:
            heapq.heappush(
                h,
                Event(time=e.time, kind=e.kind, priority=e.priority, payload=e.payload),
            )
        result = []
        while h:
            result.append(heapq.heappop(h).kind)
        assert result == expected


def test_scheduler_pop_all_at_time():
    """HeapScheduler 可批量弹出同一时间事件。"""
    reset_event_seq()
    event_spec = EventSpecRegistry.default()
    scheduler = HeapScheduler()

    scheduler.push(Event(time=100, kind=EVENT_KIND_SNAPSHOT_ARRIVAL, priority=event_spec.priorityOf(EVENT_KIND_SNAPSHOT_ARRIVAL), payload=1))
    scheduler.push(Event(time=100, kind=EVENT_KIND_ACTION_ARRIVAL, priority=event_spec.priorityOf(EVENT_KIND_ACTION_ARRIVAL), payload=2))
    scheduler.push(Event(time=200, kind=EVENT_KIND_RECEIPT_DELIVERY, priority=event_spec.priorityOf(EVENT_KIND_RECEIPT_DELIVERY), payload=3))

    at_100 = scheduler.popAllAtTime(100)
    assert len(at_100) == 2
    assert all(e.time == 100 for e in at_100)
    assert scheduler.nextDueTimeOrDefault(999) == 200


def test_request_receipt_types():
    """RequestType / ReceiptType 枚举和 CancelRequest 数据类。"""
    assert RequestType.ORDER.value == "ORDER"
    assert RequestType.CANCEL.value == "CANCEL"

    cancel = CancelRequest(order_id="o1", create_time=1000)
    assert cancel.order_id == "o1"
    assert cancel.create_time == 1000

    # 各种回执类型
    r1 = OrderReceipt(order_id="o1", receipt_type="CANCELED", timestamp=1100,
                      fill_qty=5, remaining_qty=0)
    assert r1.receipt_type == "CANCELED" and r1.fill_qty > 0

    r2 = OrderReceipt(order_id="o2", receipt_type="CANCELED", timestamp=1200,
                      fill_qty=0, remaining_qty=0)
    assert r2.receipt_type == "CANCELED" and r2.fill_qty == 0

    r3 = OrderReceipt(order_id="o3", receipt_type="REJECTED", timestamp=1300)
    assert r3.receipt_type == "REJECTED"
