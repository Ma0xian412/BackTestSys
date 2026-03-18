"""SnapMatch 引擎的 IExecutionVenue 适配器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...core.data_structure import (
    Action,
    ActionType,
    CancelRequest,
    Level,
    NormalizedSnapshot,
    Order,
    OrderReceipt,
    Side,
    TimeInForce,
)
from ...core.port import IExecutionVenue, IMarketDataQuery
from .snap_match_types import (
    EngineFactory,
    MarketData,
    MatchEngineEvent,
    MatchEngineFeedCallbackBridge,
    MatchEngineTradeCallbackBridge,
    OrderTriggerType,
    default_engine_factory,
)


@dataclass
class _OrderState:
    orig_qty: int
    cum_filled: int = 0
    is_terminal: bool = False
    last_fill_price: float = 0.0


class SnapMatchExecutionVenueAdapter(IExecutionVenue):
    """SnapMatchEngineWrapper 适配为 IExecutionVenue。"""

    def __init__(
        self,
        queue_ratio: float = 0.5,
        match_delay: int = 0,
        match_queue_delay: int = 0,
        response_delay: int = 0,
        cancel_ratio: float = 0.5,
        mask_decay_halflife: int = 1000,
        *,
        contract_id: int = 0,
        tick_size: float = 1.0,
        engine: Optional[Any] = None,
        engine_factory: Optional[EngineFactory] = None,
        to_match_market_data: Optional[Any] = None,
    ) -> None:
        self._market_data_query: Optional[IMarketDataQuery] = None
        self._contract_id = int(contract_id)
        self._tick_size = float(tick_size)
        self._pending_events: List[MatchEngineEvent] = []
        self._order_states: Dict[int, _OrderState] = {}
        self._last_market_data: Optional[MarketData] = None
        self._engine = self._create_engine(
            engine=engine,
            engine_factory=engine_factory,
            queue_ratio=queue_ratio,
            match_delay=match_delay,
            match_queue_delay=match_queue_delay,
            response_delay=response_delay,
            cancel_ratio=cancel_ratio,
            mask_decay_halflife=mask_decay_halflife,
        )
        self._engine.set_trade_callback(MatchEngineTradeCallbackBridge(self._pending_events))
        self._engine.set_feed_callback(
            MatchEngineFeedCallbackBridge(
                query_provider=lambda: self._market_data_query,
                snapshot_mapper=self._snapshot_to_market_data_with_contract,
                to_match_market_data=to_match_market_data,
            )
        )

    def set_market_data_query(self, market_data_query: IMarketDataQuery) -> None:
        self._market_data_query = market_data_query

    def start_run(self) -> None:
        self._pending_events.clear()
        self._order_states.clear()
        self._last_market_data = None
        self._engine.initialize(contract_id=int(self._contract_id), tick_size=float(self._tick_size))

    def start_session(self) -> None:
        return None

    def on_action(self, action: Action) -> List[OrderReceipt]:
        trigger_time = int(action.create_time)
        if action.action_type in (ActionType.ORDER_NEW, ActionType.PLACE_ORDER):
            return [self._handle_order_action(action, trigger_time)]
        if action.action_type in (ActionType.ORDER_CANCEL, ActionType.CANCEL_ORDER):
            return [self._handle_cancel_action(action, trigger_time)]
        raise ValueError(f"Unsupported action type: {action.action_type!r}")

    def step(self, until_time: int) -> List[OrderReceipt]:
        return [_none_receipt(timestamp=int(until_time))]

    def flush_window(self) -> object:
        if self._last_market_data is not None:
            self._engine.on_market_data_after_trade(self._last_market_data)
            return {"interval_end": int(self._last_market_data["recv_time"])}
        return {"interval_end": 0}

    def on_market_data_before_trade(self, market_data: MarketData) -> None:
        self._last_market_data = dict(market_data)
        self._engine.on_market_data_before_trade(self._last_market_data)

    def on_market_data_after_trade(self, market_data: MarketData) -> None:
        self._last_market_data = dict(market_data)
        self._engine.on_market_data_after_trade(self._last_market_data)

    def drain_trade_events(self) -> List[MatchEngineEvent]:
        events = list(self._pending_events)
        self._pending_events.clear()
        return events

    def build_receipt_from_event(self, event: MatchEngineEvent) -> Optional[OrderReceipt]:
        if event.event_type == "TRADE":
            return self._build_trade_receipt(event)
        if event.event_type == "CANCEL":
            return self._build_cancel_receipt(event)
        raise ValueError(f"Unsupported match engine event type: {event.event_type!r}")

    def snapshot_to_market_data(self, snapshot: object) -> MarketData:
        return self._snapshot_to_market_data_with_contract(snapshot, self._contract_id)

    def _handle_order_action(self, action: Action, trigger_time: int) -> OrderReceipt:
        order = _extract_order(action)
        self._register_order(order)
        self._engine.submit_order(
            driver_order_id=int(order.order_id),
            price=float(order.price),
            volume=int(order.qty),
            direction=_to_engine_direction(order.side),
            trigger_time=trigger_time,
            is_ioc=bool(order.tif == TimeInForce.IOC),
            trigger_type=OrderTriggerType.MarketData,
        )
        return _none_receipt(timestamp=trigger_time, order_id=int(order.order_id))

    def _handle_cancel_action(self, action: Action, trigger_time: int) -> OrderReceipt:
        request = _extract_cancel(action)
        self._engine.cancel_order(driver_order_id=int(request.order_id), cancel_time=trigger_time)
        return _none_receipt(timestamp=trigger_time, order_id=int(request.order_id))

    def _snapshot_to_market_data_with_contract(self, snapshot: object, contract_id: int) -> MarketData:
        if not isinstance(snapshot, NormalizedSnapshot):
            raise TypeError(f"Expected NormalizedSnapshot, got {type(snapshot)!r}")
        bid_levels = sorted(list(snapshot.bids), key=lambda item: float(item.price), reverse=True)
        ask_levels = sorted(list(snapshot.asks), key=lambda item: float(item.price))
        bid_price, bid_volume = _level_head(bid_levels)
        ask_price, ask_volume = _level_head(ask_levels)
        recv_time = int(snapshot.ts_recv)
        exch_time = int(snapshot.ts_exch) if snapshot.ts_exch is not None else recv_time
        last_price = float(snapshot.last) if snapshot.last is not None else (bid_price + ask_price) / 2.0
        return {
            "contract_id": int(contract_id),
            "recv_time": recv_time,
            "exch_time": exch_time,
            "bid_price": bid_price,
            "ask_price": ask_price,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "bid_prices": [float(item.price) for item in bid_levels],
            "ask_prices": [float(item.price) for item in ask_levels],
            "last_price": float(last_price),
            "last_volume": int(snapshot.volume or 0),
            "turnover": int(snapshot.turnover) if snapshot.turnover is not None else 0,
            "total_volume": int(snapshot.volume) if snapshot.volume is not None else 0,
            "depth": max(len(bid_levels), len(ask_levels)),
            "tick_size": float(self._tick_size),
        }

    def _register_order(self, order: Order) -> None:
        self._order_states[int(order.order_id)] = _OrderState(orig_qty=max(0, int(order.qty)))

    def _build_trade_receipt(self, event: MatchEngineEvent) -> Optional[OrderReceipt]:
        state = self._order_states.setdefault(int(event.driver_order_id), _OrderState(orig_qty=0))
        if state.is_terminal:
            return None
        next_cum = max(state.cum_filled, int(event.net_traded_vol))
        fill_delta = next_cum - int(state.cum_filled)
        if fill_delta <= 0:
            return None
        state.cum_filled = int(next_cum)
        state.last_fill_price = float(event.traded_price)
        remaining_qty = max(0, int(state.orig_qty) - int(state.cum_filled))
        receipt_type = "FILL" if int(state.orig_qty) > 0 and remaining_qty == 0 else "PARTIAL"
        state.is_terminal = receipt_type == "FILL"
        return OrderReceipt(
            order_id=int(event.driver_order_id),
            receipt_type=receipt_type,
            timestamp=int(event.exch_tick),
            fill_qty=int(fill_delta),
            fill_price=float(event.traded_price),
            remaining_qty=int(remaining_qty),
            pos=-1,
            recv_time=int(event.recv_tick),
        )

    def _build_cancel_receipt(self, event: MatchEngineEvent) -> Optional[OrderReceipt]:
        state = self._order_states.get(int(event.driver_order_id))
        if state is not None:
            state.is_terminal = True
        fill_price = float(state.last_fill_price) if state is not None else 0.0
        return OrderReceipt(
            order_id=int(event.driver_order_id),
            receipt_type="CANCELED",
            timestamp=int(event.exch_tick),
            fill_qty=0,
            fill_price=fill_price,
            remaining_qty=0,
            pos=-1,
            recv_time=int(event.recv_tick),
        )

    @staticmethod
    def _create_engine(
        *,
        engine: Optional[Any],
        engine_factory: Optional[EngineFactory],
        queue_ratio: float,
        match_delay: int,
        match_queue_delay: int,
        response_delay: int,
        cancel_ratio: float,
        mask_decay_halflife: int,
    ) -> Any:
        if engine is not None:
            return engine
        factory = engine_factory or default_engine_factory
        return factory(
            queue_ratio=queue_ratio,
            match_delay=match_delay,
            match_queue_delay=match_queue_delay,
            response_delay=response_delay,
            cancel_ratio=cancel_ratio,
            mask_decay_halflife=mask_decay_halflife,
        )


def _extract_order(action: Action) -> Order:
    if isinstance(action.payload, Order):
        return action.payload
    raise TypeError(f"ORDER_NEW payload must be Order, got {type(action.payload)!r}")


def _extract_cancel(action: Action) -> CancelRequest:
    if isinstance(action.payload, CancelRequest):
        return action.payload
    if isinstance(action.payload, int):
        return CancelRequest(order_id=action.payload, create_time=int(action.create_time))
    raise TypeError(f"ORDER_CANCEL payload must be CancelRequest or int, got {type(action.payload)!r}")


def _to_engine_direction(side: Side) -> str:
    raw = side.value if isinstance(side, Side) else str(side)
    normalized = raw.strip().upper()
    if normalized == "BUY":
        return "Buy"
    if normalized == "SELL":
        return "Sell"
    raise ValueError(f"Unsupported order side: {side!r}")


def _level_head(levels: List[Level]) -> tuple[float, int]:
    if not levels:
        return 0.0, 0
    return float(levels[0].price), int(levels[0].qty)


def _none_receipt(timestamp: int, order_id: Optional[int] = None) -> OrderReceipt:
    return OrderReceipt(
        order_id=order_id,
        receipt_type="NONE",
        timestamp=int(timestamp),
        fill_qty=0,
        fill_price=0.0,
        remaining_qty=0,
        pos=-1,
    )
