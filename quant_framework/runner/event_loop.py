"""EventLoop-based runner for unified backtest architecture.

This module implements the main event loop that coordinates:
- Tape building
- Exchange simulation
- Strategy callbacks
- Order/receipt routing
"""

import heapq
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum, auto

from ..core.interfaces import IMarketDataFeed, ITapeBuilder, IExchangeSimulator, IStrategyNew, IOrderManager
from ..core.types import NormalizedSnapshot, Order, OrderReceipt, TapeSegment


class EventType(Enum):
    """Event types for the event loop."""
    SEGMENT_END = auto()
    ORDER_ARRIVAL = auto()
    CANCEL_ARRIVAL = auto()
    RECEIPT_TO_STRATEGY = auto()
    INTERVAL_END = auto()


@dataclass
class Event:
    """An event in the event loop."""
    time: int
    event_type: EventType
    data: Any
    
    def __lt__(self, other):
        """Compare events by time for heapq."""
        return self.time < other.time


@dataclass
class RunnerConfig:
    """Configuration for the event loop runner."""
    delay_out: int = 0  # Strategy -> Exchange delay
    delay_in: int = 0   # Exchange -> Strategy delay


class EventLoopRunner:
    """Event loop runner for unified backtest.
    
    Coordinates tape building, exchange simulation, and strategy execution
    in a single event-driven loop.
    """
    
    def __init__(
        self,
        feed: IMarketDataFeed,
        tape_builder: ITapeBuilder,
        exchange: IExchangeSimulator,
        strategy: IStrategyNew,
        oms: IOrderManager,
        config: RunnerConfig = None,
    ):
        """Initialize the runner.
        
        Args:
            feed: Market data feed
            tape_builder: Tape builder
            exchange: Exchange simulator
            strategy: Strategy
            oms: Order manager
            config: Runner configuration
        """
        self.feed = feed
        self.tape_builder = tape_builder
        self.exchange = exchange
        self.strategy = strategy
        self.oms = oms
        self.config = config or RunnerConfig()
        
        # Current state
        self.current_time = 0
        self.current_snapshot: Optional[NormalizedSnapshot] = None
    
    def run(self) -> Dict[str, Any]:
        """Run the backtest.
        
        Returns:
            Dictionary with backtest results
        """
        self.feed.reset()
        prev = self.feed.next()
        
        if prev is None:
            return {"error": "No data"}
        
        self.current_snapshot = prev
        self.current_time = int(prev.ts_exch)
        
        # Initial snapshot callback
        orders = self.strategy.on_snapshot(prev, self.oms)
        for order in orders:
            self.oms.submit(order, self.current_time)
        
        interval_count = 0
        
        while True:
            curr = self.feed.next()
            if curr is None:
                break
            
            # Run one interval
            self._run_interval(prev, curr)
            
            prev = curr
            self.current_snapshot = curr
            interval_count += 1
        
        return {
            "intervals": interval_count,
            "final_time": self.current_time,
        }
    
    def _run_interval(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot) -> None:
        """Run event loop for one interval [prev, curr].
        
        Args:
            prev: Previous snapshot
            curr: Current snapshot
        """
        t_a = int(prev.ts_exch)
        t_b = int(curr.ts_exch)
        
        if t_b <= t_a:
            return
        
        # Build tape for this interval
        tape = self.tape_builder.build(prev, curr)
        
        if not tape:
            # No events, just advance to curr
            self.current_time = t_b
            self.current_snapshot = curr
            orders = self.strategy.on_snapshot(curr, self.oms)
            for order in orders:
                self.oms.submit(order, t_b)
            return
        
        # Reset exchange for new interval
        self.exchange.reset()
        
        # Initialize event queue
        event_queue: List[Event] = []
        
        # Add segment end events
        for seg in tape:
            heapq.heappush(event_queue, Event(
                time=seg.t_end,
                event_type=EventType.SEGMENT_END,
                data=seg,
            ))
        
        # Add interval end event
        heapq.heappush(event_queue, Event(
            time=t_b,
            event_type=EventType.INTERVAL_END,
            data=curr,
        ))
        
        # Get pending orders from OMS
        pending_orders = self.oms.get_active_orders()
        
        # Schedule order arrivals
        for order in pending_orders:
            if order.create_time >= t_a:
                arrival_time = order.create_time + self.config.delay_out
                if arrival_time < t_b:
                    heapq.heappush(event_queue, Event(
                        time=arrival_time,
                        event_type=EventType.ORDER_ARRIVAL,
                        data=order,
                    ))
        
        # Event loop
        current_seg_idx = 0
        last_time = t_a
        
        while event_queue:
            event = heapq.heappop(event_queue)
            
            # Advance exchange to event time
            if event.time > last_time and current_seg_idx < len(tape):
                # Process segments up to event time
                while current_seg_idx < len(tape) and tape[current_seg_idx].t_end <= event.time:
                    seg = tape[current_seg_idx]
                    receipts = self.exchange.advance(last_time, seg.t_end, seg)
                    
                    # Schedule receipt delivery to strategy
                    for receipt in receipts:
                        receipt_time = receipt.timestamp + self.config.delay_in
                        if receipt_time <= t_b:
                            heapq.heappush(event_queue, Event(
                                time=receipt_time,
                                event_type=EventType.RECEIPT_TO_STRATEGY,
                                data=receipt,
                            ))
                    
                    last_time = seg.t_end
                    current_seg_idx += 1
            
            # Process event
            self.current_time = event.time
            
            if event.event_type == EventType.ORDER_ARRIVAL:
                order = event.data
                # Get market qty at order price
                market_qty = self._get_market_qty(order, tape, current_seg_idx)
                receipt = self.exchange.on_order_arrival(order, event.time, market_qty)
                
                if receipt:
                    # Immediate rejection
                    self.oms.on_receipt(receipt)
                    
            elif event.event_type == EventType.RECEIPT_TO_STRATEGY:
                receipt = event.data
                self.oms.on_receipt(receipt)
                
                # Strategy callback
                orders = self.strategy.on_receipt(receipt, self.current_snapshot, self.oms)
                for order in orders:
                    self.oms.submit(order, event.time)
                    # Schedule arrival
                    arrival_time = event.time + self.config.delay_out
                    if arrival_time < t_b:
                        heapq.heappush(event_queue, Event(
                            time=arrival_time,
                            event_type=EventType.ORDER_ARRIVAL,
                            data=order,
                        ))
                        
            elif event.event_type == EventType.INTERVAL_END:
                snapshot = event.data
                
                # Align exchange at boundary
                self.exchange.align_at_boundary(snapshot)
                
                # Strategy callback
                self.current_snapshot = snapshot
                orders = self.strategy.on_snapshot(snapshot, self.oms)
                for order in orders:
                    self.oms.submit(order, event.time)
    
    def _get_market_qty(self, order: Order, tape: List[TapeSegment], current_seg_idx: int) -> int:
        """Get market queue depth at order price.
        
        Args:
            order: The order
            tape: Tape segments
            current_seg_idx: Current segment index
            
        Returns:
            Market queue depth
        """
        if current_seg_idx >= len(tape):
            return 0
        
        # Use exchange's queue depth
        return self.exchange.get_queue_depth(order.side, order.price)
