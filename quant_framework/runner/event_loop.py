"""EventLoop-based runner for unified backtest architecture with two-timeline support.

This module implements the main event loop that coordinates:
- Tape building
- Exchange simulation
- Strategy callbacks
- Order/receipt routing with two-timeline mapping (exchtime <-> recvtime)
"""

import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
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
    time: int  # Time in the relevant timeline (exchtime for exchange, recvtime for strategy)
    event_type: EventType
    data: Any
    
    def __lt__(self, other):
        """Compare events by time for heapq."""
        return self.time < other.time


@dataclass
class TimelineConfig:
    """Configuration for two-timeline mapping.
    
    recvtime = a * exchtime + b
    exchtime = (recvtime - b) / a
    
    Requires: a > 0 (strictly monotonic)
    """
    a: float = 1.0  # Scale factor
    b: float = 0.0  # Offset
    
    def exchtime_to_recvtime(self, exchtime: int) -> int:
        """Convert exchange time to strategy receive time."""
        return int(self.a * exchtime + self.b)
    
    def recvtime_to_exchtime(self, recvtime: int) -> int:
        """Convert strategy receive time to exchange time."""
        if self.a <= 0:
            raise ValueError("Timeline scale factor 'a' must be positive")
        return int((recvtime - self.b) / self.a)


@dataclass
class RunnerConfig:
    """Configuration for the event loop runner."""
    delay_out: int = 0  # Strategy -> Exchange delay (in recvtime units)
    delay_in: int = 0   # Exchange -> Strategy delay (in recvtime units)
    timeline: TimelineConfig = None  # Two-timeline mapping
    
    def __post_init__(self):
        if self.timeline is None:
            self.timeline = TimelineConfig()


class EventLoopRunner:
    """Event loop runner for unified backtest with two-timeline support.
    
    Coordinates tape building, exchange simulation, and strategy execution
    in a single event-driven loop.
    
    Key features:
    - Two-timeline support: exchtime (exchange) and recvtime (strategy)
    - Delay handling: delay_out (strategy->exchange), delay_in (exchange->strategy)
    - Linear time mapping: recvtime = a * exchtime + b
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
        
        # Validate timeline config
        if self.config.timeline.a <= 0:
            raise ValueError("Timeline scale factor 'a' must be positive")
        
        # Current state
        self.current_exchtime = 0
        self.current_recvtime = 0
        self.current_snapshot: Optional[NormalizedSnapshot] = None
        
        # Diagnostics
        self.diagnostics: Dict[str, Any] = {
            "intervals_processed": 0,
            "orders_submitted": 0,
            "orders_filled": 0,
            "receipts_generated": 0,
        }
    
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
        self.current_exchtime = int(prev.ts_exch)
        self.current_recvtime = self.config.timeline.exchtime_to_recvtime(self.current_exchtime)
        
        # Initial snapshot callback
        orders = self.strategy.on_snapshot(prev, self.oms)
        for order in orders:
            self.oms.submit(order, self.current_recvtime)
            self.diagnostics["orders_submitted"] += 1
        
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
        
        self.diagnostics["intervals_processed"] = interval_count
        
        return {
            "intervals": interval_count,
            "final_exchtime": self.current_exchtime,
            "final_recvtime": self.current_recvtime,
            "diagnostics": self.diagnostics,
        }
    
    def _run_interval(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot) -> None:
        """Run event loop for one interval [prev, curr].
        
        Args:
            prev: Previous snapshot (at T_A)
            curr: Current snapshot (at T_B)
        """
        t_a = int(prev.ts_exch)
        t_b = int(curr.ts_exch)
        
        if t_b <= t_a:
            return
        
        # Convert to recvtime
        recv_a = self.config.timeline.exchtime_to_recvtime(t_a)
        recv_b = self.config.timeline.exchtime_to_recvtime(t_b)
        
        # Build tape for this interval
        tape = self.tape_builder.build(prev, curr)
        
        if not tape:
            # No events, just advance to curr
            self.current_exchtime = t_b
            self.current_recvtime = recv_b
            self.current_snapshot = curr
            orders = self.strategy.on_snapshot(curr, self.oms)
            for order in orders:
                self.oms.submit(order, recv_b)
                self.diagnostics["orders_submitted"] += 1
            return
        
        # Reset exchange for new interval and set tape
        self.exchange.reset()
        
        # Set tape on exchange if it supports it
        if hasattr(self.exchange, 'set_tape'):
            self.exchange.set_tape(tape, t_a, t_b)
        
        # Initialize event queue (uses exchtime)
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
        
        # Get pending orders from OMS and schedule arrivals
        pending_orders = self.oms.get_active_orders()
        
        for order in pending_orders:
            if order.create_time >= recv_a:
                # Convert strategy submit time (recvtime) to exchange arrival time
                recv_arr = order.create_time + self.config.delay_out
                exchtime_arr = self.config.timeline.recvtime_to_exchtime(recv_arr)
                
                if exchtime_arr < t_b:
                    heapq.heappush(event_queue, Event(
                        time=exchtime_arr,
                        event_type=EventType.ORDER_ARRIVAL,
                        data=order,
                    ))
        
        # Event loop
        current_seg_idx = 0
        last_exchtime = t_a
        
        while event_queue:
            event = heapq.heappop(event_queue)
            
            # Advance exchange to event time
            if event.time > last_exchtime and current_seg_idx < len(tape):
                # Process segments up to event time
                while current_seg_idx < len(tape) and tape[current_seg_idx].t_end <= event.time:
                    seg = tape[current_seg_idx]
                    receipts = self.exchange.advance(last_exchtime, seg.t_end, seg)
                    
                    # Schedule receipt delivery to strategy
                    for receipt in receipts:
                        recv_fill = self.config.timeline.exchtime_to_recvtime(receipt.timestamp)
                        recv_recv = recv_fill + self.config.delay_in
                        
                        # Set recv_time on receipt
                        receipt.recv_time = recv_recv
                        
                        if recv_recv <= recv_b:
                            # Convert back to exchtime for event queue
                            exchtime_recv = self.config.timeline.recvtime_to_exchtime(recv_recv)
                            heapq.heappush(event_queue, Event(
                                time=exchtime_recv,
                                event_type=EventType.RECEIPT_TO_STRATEGY,
                                data=receipt,
                            ))
                        
                        self.diagnostics["receipts_generated"] += 1
                    
                    last_exchtime = seg.t_end
                    current_seg_idx += 1
            
            # Process event
            self.current_exchtime = event.time
            self.current_recvtime = self.config.timeline.exchtime_to_recvtime(event.time)
            
            if event.event_type == EventType.ORDER_ARRIVAL:
                order = event.data
                order.arrival_time = event.time
                
                # Get market qty at order price from current snapshot
                market_qty = self._get_market_qty_from_snapshot(order, self.current_snapshot)
                receipt = self.exchange.on_order_arrival(order, event.time, market_qty)
                
                if receipt:
                    # Immediate rejection or IOC result
                    recv_fill = self.config.timeline.exchtime_to_recvtime(receipt.timestamp)
                    recv_recv = recv_fill + self.config.delay_in
                    receipt.recv_time = recv_recv
                    
                    self.oms.on_receipt(receipt)
                    self.diagnostics["receipts_generated"] += 1
                    
                    if receipt.receipt_type in ["FILL", "PARTIAL"]:
                        self.diagnostics["orders_filled"] += 1
                    
            elif event.event_type == EventType.RECEIPT_TO_STRATEGY:
                receipt = event.data
                self.oms.on_receipt(receipt)
                
                if receipt.receipt_type in ["FILL", "PARTIAL"]:
                    self.diagnostics["orders_filled"] += 1
                
                # Strategy callback
                orders = self.strategy.on_receipt(receipt, self.current_snapshot, self.oms)
                for order in orders:
                    self.oms.submit(order, self.current_recvtime)
                    self.diagnostics["orders_submitted"] += 1
                    
                    # Schedule arrival
                    recv_arr = self.current_recvtime + self.config.delay_out
                    exchtime_arr = self.config.timeline.recvtime_to_exchtime(recv_arr)
                    
                    if exchtime_arr < t_b:
                        heapq.heappush(event_queue, Event(
                            time=exchtime_arr,
                            event_type=EventType.ORDER_ARRIVAL,
                            data=order,
                        ))
                        
            elif event.event_type == EventType.INTERVAL_END:
                snapshot = event.data
                
                # Align exchange at boundary
                self.exchange.align_at_boundary(snapshot)
                
                # Strategy callback
                self.current_snapshot = snapshot
                self.current_recvtime = self.config.timeline.exchtime_to_recvtime(event.time)
                
                orders = self.strategy.on_snapshot(snapshot, self.oms)
                for order in orders:
                    self.oms.submit(order, self.current_recvtime)
                    self.diagnostics["orders_submitted"] += 1
    
    def _get_market_qty_from_snapshot(self, order: Order, snapshot: NormalizedSnapshot) -> int:
        """Get market queue depth at order price from snapshot.
        
        Args:
            order: The order
            snapshot: Current market snapshot
            
        Returns:
            Market queue depth at order price
        """
        if snapshot is None:
            return 0
        
        levels = snapshot.bids if order.side.value == "BUY" else snapshot.asks
        
        for level in levels:
            if abs(float(level.price) - float(order.price)) < 1e-9:
                return int(level.qty)
        
        return 0
