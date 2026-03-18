"""Dispatcher 行为单元测试。"""

import pytest

from quant_framework.core.data_structure import Event, EventSpecRegistry, RuntimeContext
from quant_framework.core.dispatcher import Dispatcher, IEventHandler


class _Dummy:
    def __getattr__(self, _: str):
        return self

    def __call__(self, *args, **kwargs):
        return None


class _RecordingHandler(IEventHandler):
    def __init__(self, name: str, trace: list[str], emitted: list[Event] | None = None) -> None:
        self._name = name
        self._trace = trace
        self._emitted = emitted or []

    def handle(self, e: Event, ctx: RuntimeContext) -> list[Event]:
        self._trace.append(self._name)
        return list(self._emitted)


class _RaisingHandler(IEventHandler):
    def __init__(self, name: str, trace: list[str]) -> None:
        self._name = name
        self._trace = trace

    def handle(self, e: Event, ctx: RuntimeContext) -> list[Event]:
        self._trace.append(self._name)
        raise RuntimeError("boom")


def _build_context(dispatcher: Dispatcher, event_spec: EventSpecRegistry) -> RuntimeContext:
    return RuntimeContext(
        feed=_Dummy(),
        venue=_Dummy(),
        strategy=_Dummy(),
        oms=_Dummy(),
        timeModel=_Dummy(),
        obs=_Dummy(),
        dispatcher=dispatcher,
        eventSpec=event_spec,
    )


def _md_event() -> Event:
    return Event(type="md.arrive", time=1, priority=0, payload={"x": 1})


def test_register_keeps_override_semantics() -> None:
    trace: list[str] = []
    event_spec = EventSpecRegistry.default()
    dispatcher = Dispatcher(event_spec)
    dispatcher.register("md.arrive", _RecordingHandler("h1", trace))
    dispatcher.register("md.arrive", _RecordingHandler("h2", trace))

    emitted = dispatcher.dispatch(_md_event(), _build_context(dispatcher, event_spec))

    assert emitted == []
    assert trace == ["h2"]


def test_subscribe_supports_one_to_many_with_registration_order() -> None:
    trace: list[str] = []
    event_spec = EventSpecRegistry.default()
    dispatcher = Dispatcher(event_spec)
    dispatcher.subscribe("md.arrive", _RecordingHandler("h1", trace))
    dispatcher.subscribe("md.arrive", _RecordingHandler("h2", trace))

    dispatcher.dispatch(_md_event(), _build_context(dispatcher, event_spec))

    assert trace == ["h1", "h2"]


def test_handlers_with_explicit_order_run_before_unordered() -> None:
    trace: list[str] = []
    event_spec = EventSpecRegistry.default()
    dispatcher = Dispatcher(event_spec)
    dispatcher.subscribe("md.arrive", _RecordingHandler("late-1", trace))
    dispatcher.subscribe("md.arrive", _RecordingHandler("late-2", trace))
    dispatcher.subscribe("md.arrive", _RecordingHandler("early", trace), order=1)

    dispatcher.dispatch(_md_event(), _build_context(dispatcher, event_spec))

    assert trace == ["early", "late-1", "late-2"]


def test_same_order_keeps_registration_order() -> None:
    trace: list[str] = []
    event_spec = EventSpecRegistry.default()
    dispatcher = Dispatcher(event_spec)
    dispatcher.subscribe("md.arrive", _RecordingHandler("a", trace), order=10)
    dispatcher.subscribe("md.arrive", _RecordingHandler("b", trace), order=10)
    dispatcher.subscribe("md.arrive", _RecordingHandler("c", trace))

    dispatcher.dispatch(_md_event(), _build_context(dispatcher, event_spec))

    assert trace == ["a", "b", "c"]


def test_dispatch_merges_emitted_events_in_execution_order() -> None:
    trace: list[str] = []
    event_spec = EventSpecRegistry.default()
    dispatcher = Dispatcher(event_spec)

    out1 = Event(type="x", time=2, priority=1, payload=1)
    out2 = Event(type="y", time=3, priority=1, payload=2)
    dispatcher.subscribe("md.arrive", _RecordingHandler("first", trace, [out1]), order=2)
    dispatcher.subscribe("md.arrive", _RecordingHandler("second", trace, [out2]), order=3)

    emitted = dispatcher.dispatch(_md_event(), _build_context(dispatcher, event_spec))

    assert trace == ["first", "second"]
    assert emitted == [out1, out2]


def test_duplicate_handler_registration_is_allowed() -> None:
    trace: list[str] = []
    event_spec = EventSpecRegistry.default()
    dispatcher = Dispatcher(event_spec)
    handler = _RecordingHandler("dup", trace)
    dispatcher.subscribe("md.arrive", handler)
    dispatcher.subscribe("md.arrive", handler)

    dispatcher.dispatch(_md_event(), _build_context(dispatcher, event_spec))

    assert trace == ["dup", "dup"]


def test_dispatch_is_fail_fast_when_handler_raises() -> None:
    trace: list[str] = []
    event_spec = EventSpecRegistry.default()
    dispatcher = Dispatcher(event_spec)
    dispatcher.subscribe("md.arrive", _RecordingHandler("ok", trace), order=1)
    dispatcher.subscribe("md.arrive", _RaisingHandler("boom", trace), order=2)
    dispatcher.subscribe("md.arrive", _RecordingHandler("never", trace), order=3)

    with pytest.raises(RuntimeError, match="boom"):
        dispatcher.dispatch(_md_event(), _build_context(dispatcher, event_spec))

    assert trace == ["ok", "boom"]


def test_dispatch_returns_empty_when_no_handler_registered() -> None:
    event_spec = EventSpecRegistry.default()
    dispatcher = Dispatcher(event_spec)

    emitted = dispatcher.dispatch(_md_event(), _build_context(dispatcher, event_spec))

    assert emitted == []


def test_dispatch_validates_payload() -> None:
    event_spec = EventSpecRegistry.default()
    dispatcher = Dispatcher(event_spec)
    dispatcher.subscribe("md.arrive", _RecordingHandler("h", []))
    event = Event(type="md.arrive", time=1, priority=0, payload=None)

    with pytest.raises(ValueError, match="Invalid payload"):
        dispatcher.dispatch(event, _build_context(dispatcher, event_spec))
