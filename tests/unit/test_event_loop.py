"""事件循环单元测试。

验证内容：
- 事件优先级排序（确定性）
- SEGMENT_END / INTERVAL_END 不重复
- 请求类型与回执类型枚举
"""

import heapq

from quant_framework.core.types import (
    RequestType, ReceiptType, CancelRequest, OrderReceipt, TapeSegment,
    Side, TICK_PER_MS,
)
from quant_framework.runner.event_loop import (
    Event, EventType, EVENT_TYPE_PRIORITY, reset_event_seq_counter,
)


def test_event_priority_ordering():
    """事件优先级：同一时刻的事件按 (priority, seq) 确定性排序。

    优先级：SEGMENT_END > ORDER_ARRIVAL > CANCEL_ARRIVAL > RECEIPT_TO_STRATEGY > INTERVAL_END
    """
    reset_event_seq_counter()

    t = 1000 * TICK_PER_MS
    events = [
        Event(time=t, event_type=EventType.INTERVAL_END, data="interval"),
        Event(time=t, event_type=EventType.RECEIPT_TO_STRATEGY, data="receipt1"),
        Event(time=t, event_type=EventType.SEGMENT_END, data="segment"),
        Event(time=t, event_type=EventType.ORDER_ARRIVAL, data="order"),
        Event(time=t, event_type=EventType.RECEIPT_TO_STRATEGY, data="receipt2"),
        Event(time=t, event_type=EventType.CANCEL_ARRIVAL, data="cancel"),
    ]

    heap = []
    for e in events:
        heapq.heappush(heap, e)

    popped = []
    while heap:
        popped.append(heapq.heappop(heap))

    expected = [
        EventType.SEGMENT_END,
        EventType.ORDER_ARRIVAL,
        EventType.CANCEL_ARRIVAL,
        EventType.RECEIPT_TO_STRATEGY,
        EventType.RECEIPT_TO_STRATEGY,
        EventType.INTERVAL_END,
    ]
    actual = [e.event_type for e in popped]
    assert actual == expected, f"事件顺序不正确\n期望: {[e.name for e in expected]}\n实际: {[e.name for e in actual]}"

    # 同类型按 seq 排序
    receipts = [e for e in popped if e.event_type == EventType.RECEIPT_TO_STRATEGY]
    assert receipts[0].data == "receipt1" and receipts[1].data == "receipt2"

    # 多次运行确定性一致
    for _ in range(3):
        reset_event_seq_counter()
        h = []
        for e in events:
            heapq.heappush(h, Event(time=e.time, event_type=e.event_type, data=e.data))
        result = []
        while h:
            result.append(heapq.heappop(h).event_type)
        assert result == expected


def test_no_duplicate_segment_and_interval_end():
    """SEGMENT_END / INTERVAL_END：最后一段的结束由 INTERVAL_END 代表，不重复。"""
    reset_event_seq_counter()

    tape = [
        TapeSegment(index=1, t_start=0, t_end=100, bid_price=100.0, ask_price=101.0,
                    trades={}, cancels={}, net_flow={},
                    activation_bid={100.0}, activation_ask={101.0}),
        TapeSegment(index=2, t_start=100, t_end=200, bid_price=100.0, ask_price=101.0,
                    trades={}, cancels={}, net_flow={},
                    activation_bid={100.0}, activation_ask={101.0}),
        TapeSegment(index=3, t_start=200, t_end=300, bid_price=100.0, ask_price=101.0,
                    trades={}, cancels={}, net_flow={},
                    activation_bid={100.0}, activation_ask={101.0}),
    ]
    t_b = 300

    queue = []
    for seg in tape:
        if seg.t_end == t_b:
            continue
        heapq.heappush(queue, Event(time=seg.t_end, event_type=EventType.SEGMENT_END, data=seg))
    heapq.heappush(queue, Event(time=t_b, event_type=EventType.INTERVAL_END, data=None))

    seg_end_count = sum(1 for e in queue if e.event_type == EventType.SEGMENT_END)
    assert seg_end_count == len(tape) - 1, f"SEGMENT_END 应为 {len(tape) - 1}，实际 {seg_end_count}"
    assert not any(e.time == t_b and e.event_type == EventType.SEGMENT_END for e in queue), (
        "t_b 时刻不应有 SEGMENT_END"
    )

    # 单段场景
    reset_event_seq_counter()
    single_tape = [tape[0]]
    q2 = []
    for seg in single_tape:
        if seg.t_end == single_tape[-1].t_end:
            continue
        heapq.heappush(q2, Event(time=seg.t_end, event_type=EventType.SEGMENT_END, data=seg))
    heapq.heappush(q2, Event(time=single_tape[-1].t_end, event_type=EventType.INTERVAL_END, data=None))

    assert sum(1 for e in q2 if e.event_type == EventType.SEGMENT_END) == 0
    assert sum(1 for e in q2 if e.event_type == EventType.INTERVAL_END) == 1


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
