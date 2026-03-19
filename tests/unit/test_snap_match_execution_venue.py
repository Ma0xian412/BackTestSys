"""SnapMatchExecutionVenueAdapter 单元测试。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from quant_framework.adapters.IOMS.oms import OMS_Impl
from quant_framework.adapters.execution_venue import (
    OrderTriggerType,
    SnapMatchExecutionVenueAdapter,
)
from quant_framework.core.data_structure import (
    Action,
    ActionType,
    CancelRequest,
    Event,
    EventSpecRegistry,
    Level,
    NormalizedSnapshot,
    Order,
    OrderStatus,
    RuntimeContext,
    Side,
    TimeInForce,
)
from quant_framework.core.handlers import SnapMDArriveHandler
from quant_framework.core.obs_event_factory import make_receipt_delivered_event
from quant_framework.core.observability import (
    EVENT_TYPE_RECEIPT_DELIVERED,
    EVENT_TYPE_RECEIPT_GENERATED,
)


@dataclass
class _ObsCollector:
    events: List[Event]

    def ingest(self, event: Event) -> None:
        self.events.append(event)


class _NoopTimeModel:
    def delayout(self, local_time: int) -> int:
        return int(local_time)

    def delayin(self, exchange_time: int) -> int:
        return int(exchange_time)


class _NoopStrategy:
    def on_event(self, e: Event, ctx: Any):
        return []


class _NoopFeed:
    def __init__(self, snapshots: List[NormalizedSnapshot]):
        self._snapshots = snapshots

    def next(self):
        return None

    def reset(self):
        return None

    def query_data(self, n: int):
        return list(self._snapshots[: max(0, int(n))])


class _FakeSnapEngine:
    def __init__(self) -> None:
        self.trade_callback = None
        self.feed_callback = None
        self.initialized: tuple[int, float] | None = None
        self.submit_calls: List[dict[str, Any]] = []
        self.cancel_calls: List[dict[str, Any]] = []
        self.before_calls: List[dict[str, Any]] = []
        self.after_calls: List[dict[str, Any]] = []

    def initialize(self, contract_id: int, tick_size: float) -> None:
        self.initialized = (int(contract_id), float(tick_size))

    def set_trade_callback(self, callback: Any) -> None:
        self.trade_callback = callback

    def set_feed_callback(self, callback: Any) -> None:
        self.feed_callback = callback

    def submit_order(self, **kwargs: Any) -> None:
        self.submit_calls.append(dict(kwargs))

    def cancel_order(self, **kwargs: Any) -> None:
        self.cancel_calls.append(dict(kwargs))

    def on_market_data_before_trade(self, market_data: dict[str, Any]) -> None:
        self.before_calls.append(dict(market_data))
        if self.trade_callback is None:
            return
        self.trade_callback.on_driver_order_traded(
            driver_order_id=11,
            event_id=2,
            traded_price=100.2,
            net_traded_vol=5,
            recv_tick=130,
            exch_tick=121,
        )
        self.trade_callback.on_driver_order_traded(
            driver_order_id=11,
            event_id=1,
            traded_price=100.1,
            net_traded_vol=10,
            recv_tick=125,
            exch_tick=119,
        )

    def on_market_data_after_trade(self, market_data: dict[str, Any]) -> None:
        self.after_calls.append(dict(market_data))


def _build_context(venue: SnapMatchExecutionVenueAdapter, oms: OMS_Impl, obs: _ObsCollector) -> RuntimeContext:
    return RuntimeContext(
        feed=_NoopFeed([]),
        venue=venue,
        strategy=_NoopStrategy(),
        oms=oms,
        timeModel=_NoopTimeModel(),
        obs=obs,
        dispatcher=None,
        eventSpec=EventSpecRegistry.default(),
    )


def test_snap_match_venue_maps_order_ioc_cancel_and_step() -> None:
    engine = _FakeSnapEngine()
    venue = SnapMatchExecutionVenueAdapter(contract_id=99, tick_size=0.2, engine=engine)
    venue.start_run()
    assert engine.initialized == (99, 0.2)

    order = Order(order_id=7, side=Side.SELL, price=101.5, qty=3, tif=TimeInForce.IOC)
    action = Action(action_type=ActionType.PLACE_ORDER, create_time=12345, payload=order)
    order_receipts = venue.on_action(action)
    assert order_receipts[0].receipt_type == "NONE"
    assert order_receipts[0].pos == -1
    assert len(engine.submit_calls) == 1
    submit_call = engine.submit_calls[0]
    assert submit_call["driver_order_id"] == 7
    assert submit_call["direction"] == "Sell"
    assert submit_call["is_ioc"] is True
    assert submit_call["trigger_type"] == OrderTriggerType.MarketData

    cancel = CancelRequest(order_id=7, create_time=12360)
    cancel_action = Action(action_type=ActionType.CANCEL_ORDER, create_time=12360, payload=cancel)
    cancel_receipts = venue.on_action(cancel_action)
    assert cancel_receipts[0].receipt_type == "NONE"
    assert cancel_receipts[0].order_id == 7
    assert engine.cancel_calls == [{"driver_order_id": 7, "cancel_time": 12360}]

    step_receipts = venue.step(20000)
    assert len(step_receipts) == 1
    assert step_receipts[0].receipt_type == "NONE"
    assert step_receipts[0].timestamp == 20000
    assert step_receipts[0].pos == -1


def test_snap_md_arrive_handler_applies_past_receipt_directly_to_oms() -> None:
    engine = _FakeSnapEngine()
    venue = SnapMatchExecutionVenueAdapter(contract_id=501, tick_size=0.5, engine=engine)
    venue.start_run()
    oms = OMS_Impl()
    obs = _ObsCollector(events=[])
    oms.subscribe_receipt(lambda receipt: obs.ingest(make_receipt_delivered_event(receipt)))
    ctx = _build_context(venue, oms, obs)

    order = Order(order_id=11, side=Side.BUY, price=100.0, qty=10)
    oms.submit_order(order, send_time=100)
    venue.on_action(Action(action_type=ActionType.PLACE_ORDER, create_time=100, payload=order))

    snapshot = NormalizedSnapshot(
        ts_recv=200,
        ts_exch=190,
        bids=[Level(100.0, 20)],
        asks=[Level(101.0, 30)],
    )
    md_event = Event(type="md.arrive", time=200, priority=10, payload=snapshot)
    emitted = SnapMDArriveHandler().handle(md_event, ctx)
    assert emitted == []

    applied_order = oms.get_order(11)
    assert applied_order is not None
    assert applied_order.filled_qty == 10
    assert applied_order.status == OrderStatus.FILLED

    event_types = [event.type for event in obs.events]
    assert event_types.count(EVENT_TYPE_RECEIPT_GENERATED) == 1
    assert event_types.count(EVENT_TYPE_RECEIPT_DELIVERED) == 1
    delivered_event = next(event for event in obs.events if event.type == EVENT_TYPE_RECEIPT_DELIVERED)
    assert delivered_event.time == 125
    assert delivered_event.payload["timestamp"] == 119

    flush_stats = venue.flush_window()
    assert flush_stats["interval_end"] == 200
    assert len(engine.after_calls) == 1


def test_feed_callback_returns_recv_gt_tick_and_exchtime_sorted() -> None:
    engine = _FakeSnapEngine()
    venue = SnapMatchExecutionVenueAdapter(contract_id=9, tick_size=0.1, engine=engine)
    snapshots = [
        NormalizedSnapshot(ts_recv=100, ts_exch=100, bids=[Level(10.0, 1)], asks=[Level(10.2, 1)]),
        NormalizedSnapshot(ts_recv=120, ts_exch=130, bids=[Level(10.1, 2)], asks=[Level(10.3, 2)]),
        NormalizedSnapshot(ts_recv=120, ts_exch=125, bids=[Level(10.1, 3)], asks=[Level(10.3, 3)]),
        NormalizedSnapshot(ts_recv=140, ts_exch=140, bids=[Level(10.2, 4)], asks=[Level(10.4, 4)]),
    ]
    venue.set_market_data_query(_NoopFeed(snapshots))

    callback = engine.feed_callback
    assert callback is not None
    first = callback.get_next_market_data(contract_id=9, tick_time=100)
    assert first is not None
    assert first["contract_id"] == 9
    assert first["recv_time"] == 120
    assert first["exch_time"] == 125
    assert first["tick_size"] == 0.1

    second = callback.get_next_market_data(contract_id=9, tick_time=120)
    assert second is not None
    assert second["recv_time"] == 140

