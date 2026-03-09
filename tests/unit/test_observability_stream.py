"""可观测流式订阅单元测试。"""

from __future__ import annotations

import os
import time

from quant_framework.adapters.observability import Observability_Impl
from quant_framework.core.data_structure import Event, Order, Side
from quant_framework.core.obs_event_factory import (
    make_order_submitted_event,
    make_run_ended_event,
    make_run_started_event,
)
from quant_framework.core.observability import (
    EVENT_TYPE_ORDER_SUBMITTED,
    ObsStartPosition,
    ObsSubscriptionOptions,
    ObsSubscriptionState,
)


def _mk_order(order_id: str, t: int) -> Order:
    return Order(order_id=order_id, side=Side.BUY, price=100.0, qty=1, create_time=t)


def _list_history_files(history_dir: str) -> list[str]:
    if not os.path.exists(history_dir):
        return []
    return sorted(
        os.path.join(history_dir, name)
        for name in os.listdir(history_dir)
        if name.startswith("run_") and ".sqlite" in name
    )


def test_stream_beginning_replay_after_run_end(tmp_path):
    obs = Observability_Impl(history_dir=str(tmp_path), keep_history_files=True)
    obs.ingest(make_run_started_event(sim_time=1, context={"sim_time": 1}))
    obs.ingest(make_order_submitted_event(_mk_order("1", 10)))
    obs.ingest(make_run_ended_event(sim_time=20, context={"status": "completed", "final_time": 20}))

    sub = obs.subscribe(ObsSubscriptionOptions(start_position=ObsStartPosition.BEGINNING))
    events = obs.poll(sub, max_items=10, timeout_ms=100)
    assert [e.type for e in events] == ["run.started", "order.submitted", "run.ended"]
    assert [e.seq for e in events] == [1, 2, 3]


def test_topic_exact_match(tmp_path):
    obs = Observability_Impl(history_dir=str(tmp_path), keep_history_files=True)
    obs.ingest(make_run_started_event(sim_time=1, context={"sim_time": 1}))
    sub = obs.subscribe(
        ObsSubscriptionOptions(
            topics=(EVENT_TYPE_ORDER_SUBMITTED,),
            start_position=ObsStartPosition.BEGINNING,
        )
    )
    obs.ingest(make_order_submitted_event(_mk_order("2", 11)))
    obs.ingest(make_run_ended_event(sim_time=21, context={"status": "completed", "final_time": 21}))

    events = obs.poll(sub, max_items=10, timeout_ms=200)
    assert len(events) == 1
    assert events[0].type == EVENT_TYPE_ORDER_SUBMITTED


def test_subscriber_memory_limit_isolated(tmp_path):
    obs = Observability_Impl(
        history_dir=str(tmp_path),
        keep_history_files=True,
        default_subscriber_memory_bytes=1024 * 1024,
    )
    obs.ingest(make_run_started_event(sim_time=1, context={"sim_time": 1}))
    good_sub = obs.subscribe()
    tiny_sub = obs.subscribe(ObsSubscriptionOptions(max_memory_bytes=1))

    obs.ingest(make_order_submitted_event(_mk_order("3", 12)))

    deadline = time.time() + 1.0
    tiny_status = obs.get_subscription_status(tiny_sub)
    while tiny_status.state == ObsSubscriptionState.ACTIVE and time.time() < deadline:
        time.sleep(0.01)
        tiny_status = obs.get_subscription_status(tiny_sub)
    assert tiny_status.state == ObsSubscriptionState.ERRORED

    good_events = obs.poll(good_sub, max_items=10, timeout_ms=300)
    assert any(e.type == EVENT_TYPE_ORDER_SUBMITTED for e in good_events)
    obs.ingest(make_run_ended_event(sim_time=22, context={"status": "completed", "final_time": 22}))


def test_history_cleanup_normal_mode_after_unsubscribe(tmp_path):
    history_dir = str(tmp_path / "normal")
    obs = Observability_Impl(history_dir=history_dir, keep_history_files=False)
    obs.ingest(make_run_started_event(sim_time=1, context={"sim_time": 1}))
    obs.ingest(make_order_submitted_event(_mk_order("4", 10)))
    obs.ingest(make_run_ended_event(sim_time=30, context={"status": "completed", "final_time": 30}))

    sub = obs.subscribe()
    replay_events = obs.poll(sub, max_items=10, timeout_ms=200)
    assert replay_events, "run 结束后应可继续读取历史"
    assert _list_history_files(history_dir), "读取前应存在历史文件"
    obs.unsubscribe(sub)
    assert not _list_history_files(history_dir), "正常模式下订阅者关闭后应自动清理历史文件"


def test_history_keep_in_debug_mode(tmp_path):
    history_dir = str(tmp_path / "debug")
    obs = Observability_Impl(history_dir=history_dir, keep_history_files=True)
    obs.ingest(make_run_started_event(sim_time=1, context={"sim_time": 1}))
    obs.ingest(make_order_submitted_event(_mk_order("5", 10)))
    obs.ingest(make_run_ended_event(sim_time=40, context={"status": "completed", "final_time": 40}))
    sub = obs.subscribe()
    _ = obs.poll(sub, max_items=10, timeout_ms=200)
    obs.unsubscribe(sub)
    assert _list_history_files(history_dir), "debug 模式应保留历史文件"


def test_unknown_event_type_is_forwarded_with_warning(tmp_path):
    obs = Observability_Impl(history_dir=str(tmp_path), keep_history_files=True)
    obs.ingest(make_run_started_event(sim_time=1, context={"sim_time": 1}))
    obs.ingest(Event(type="custom.unknown", time=2, payload={"k": "v"}))
    sub = obs.subscribe(ObsSubscriptionOptions(start_position=ObsStartPosition.BEGINNING))
    events = obs.poll(sub, max_items=10, timeout_ms=200)
    assert any(e.type == "custom.unknown" for e in events)


def test_invalid_payload_becomes_obs_invalid_event(tmp_path):
    obs = Observability_Impl(history_dir=str(tmp_path), keep_history_files=True)
    obs.ingest(make_run_started_event(sim_time=1, context={"sim_time": 1}))
    obs.ingest(Event(type=EVENT_TYPE_ORDER_SUBMITTED, time=2, payload={"bad": "payload"}))
    sub = obs.subscribe(ObsSubscriptionOptions(start_position=ObsStartPosition.BEGINNING))
    events = obs.poll(sub, max_items=10, timeout_ms=200)
    assert any(e.type == "obs.event.invalid" for e in events)


def test_dynamic_event_handler_registration(tmp_path):
    obs = Observability_Impl(history_dir=str(tmp_path), keep_history_files=True)
    handled_values: list[str] = []

    def _handle_custom(event: Event) -> bool:
        value = str(event.payload["k"])
        handled_values.append(value)
        return True

    obs.register_event_handler("custom.dynamic", _handle_custom)
    obs.ingest(make_run_started_event(sim_time=1, context={"sim_time": 1}))
    obs.ingest(Event(type="custom.dynamic", time=2, payload={"k": "v"}))
    sub = obs.subscribe(ObsSubscriptionOptions(start_position=ObsStartPosition.BEGINNING))
    events = obs.poll(sub, max_items=10, timeout_ms=200)

    assert handled_values == ["v"]
    assert any(e.type == "custom.dynamic" and e.payload.get("k") == "v" for e in events)
