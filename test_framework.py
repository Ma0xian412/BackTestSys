"""Comprehensive test suite for the unified EventLoop framework.

Tests validate:
- Tape builder: segment generation, volume allocation, conservation equations
- Exchange simulator: coordinate-axis model, fill time calculation
- Order manager: order lifecycle, receipt processing
- Event loop: two-timeline support, delay handling
- Integration: full backtest flow
"""

from quant_framework.core.types import (
    NormalizedSnapshot, Level, Order, Side, TimeInForce, OrderStatus
)
from quant_framework.tape.builder import UnifiedTapeBuilder, TapeConfig
from quant_framework.exchange.simulator import FIFOExchangeSimulator
from quant_framework.trading.oms import OrderManager, Portfolio
from quant_framework.trading.strategy import SimpleNewStrategy
from quant_framework.runner.event_loop import EventLoopRunner, RunnerConfig, TimelineConfig


def create_test_snapshot(ts: int, bid: float, ask: float, 
                         bid_qty: int = 100, ask_qty: int = 100,
                         last_vol_split=None) -> NormalizedSnapshot:
    """Create a test snapshot."""
    if last_vol_split is None:
        last_vol_split = [(bid, 10), (ask, 10)]
    return NormalizedSnapshot(
        ts_exch=ts,
        bids=[Level(bid, bid_qty)],
        asks=[Level(ask, ask_qty)],
        last_vol_split=last_vol_split,
    )


def create_multi_level_snapshot(ts: int, bids: list, asks: list,
                                 last_vol_split=None) -> NormalizedSnapshot:
    """Create a snapshot with multiple price levels."""
    bid_levels = [Level(p, q) for p, q in bids]
    ask_levels = [Level(p, q) for p, q in asks]
    return NormalizedSnapshot(
        ts_exch=ts,
        bids=bid_levels,
        asks=ask_levels,
        last_vol_split=last_vol_split or [],
    )


def test_tape_builder_basic():
    """Test basic tape builder functionality."""
    print("\n--- Test 1: Tape Builder Basic ---")
    
    config = TapeConfig()
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    
    prev = create_test_snapshot(1000, 100.0, 101.0)
    curr = create_test_snapshot(2000, 100.5, 101.5)
    
    tape = builder.build(prev, curr)
    
    print(f"Generated {len(tape)} segments")
    for seg in tape:
        print(f"  Segment {seg.index}: t=[{seg.t_start}, {seg.t_end}], "
              f"bid={seg.bid_price}, ask={seg.ask_price}")
        print(f"    trades: {dict(seg.trades)}")
        print(f"    cancels: {dict(seg.cancels)}")
        print(f"    activation_bid: {seg.activation_bid}")
    
    assert len(tape) > 0, "Tape should have at least one segment"
    
    # Check activation sets
    for seg in tape:
        assert len(seg.activation_bid) <= 5, "Activation set should have at most 5 prices"
        assert len(seg.activation_ask) <= 5, "Activation set should have at most 5 prices"
    
    print("✓ Tape builder basic test passed")


def test_tape_builder_no_trades():
    """Test tape builder with no trades."""
    print("\n--- Test 2: Tape Builder No Trades ---")
    
    config = TapeConfig()
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    
    prev = create_test_snapshot(1000, 100.0, 101.0, last_vol_split=[])
    curr = create_test_snapshot(2000, 100.5, 101.5, last_vol_split=[])
    
    tape = builder.build(prev, curr)
    
    print(f"Generated {len(tape)} segments (no trades)")
    assert len(tape) == 1, "Should have single segment with no trades"
    
    print("✓ Tape builder no trades test passed")


def test_tape_builder_conservation():
    """Test tape builder conservation equations."""
    print("\n--- Test 3: Tape Builder Conservation ---")
    
    config = TapeConfig(epsilon=1.0, segment_iterations=2)
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    
    # Create snapshots where we can verify conservation
    prev = create_multi_level_snapshot(
        1000,
        bids=[(100.0, 50), (99.0, 30)],
        asks=[(101.0, 40), (102.0, 20)],
        last_vol_split=[(100.0, 20), (101.0, 15)]
    )
    curr = create_multi_level_snapshot(
        2000,
        bids=[(100.0, 40), (99.0, 35)],  # delta_Q_bid @ 100 = -10
        asks=[(101.0, 35), (102.0, 25)],  # delta_Q_ask @ 101 = -5
        last_vol_split=[(100.0, 20), (101.0, 15)]
    )
    
    tape = builder.build(prev, curr)
    
    print(f"Generated {len(tape)} segments")
    
    # Check total trades
    total_bid_trades = sum(
        seg.trades.get((Side.BUY, 100.0), 0) for seg in tape
    )
    total_ask_trades = sum(
        seg.trades.get((Side.SELL, 101.0), 0) for seg in tape
    )
    
    print(f"Total bid trades @ 100.0: {total_bid_trades}")
    print(f"Total ask trades @ 101.0: {total_ask_trades}")
    
    # With symmetric rule: E_bid = E_ask = E
    # So total trades should equal lastvolsplit
    assert total_bid_trades == 20, f"Expected 20 bid trades, got {total_bid_trades}"
    assert total_ask_trades == 15, f"Expected 15 ask trades, got {total_ask_trades}"
    
    print("✓ Tape builder conservation test passed")


def test_exchange_simulator_basic():
    """Test basic exchange simulator functionality."""
    print("\n--- Test 4: Exchange Simulator Basic ---")
    
    exchange = FIFOExchangeSimulator(cancel_front_ratio=0.5)
    
    # Create test order
    order = Order(
        order_id="test-1",
        side=Side.BUY,
        price=100.0,
        qty=10,
    )
    
    # Simulate order arrival
    receipt = exchange.on_order_arrival(order, 1000, market_qty=50)
    
    print(f"Order arrival receipt: {receipt}")
    assert receipt is None, "Order should be accepted (no immediate fill/reject)"
    
    # Check queue depth
    depth = exchange.get_queue_depth(Side.BUY, 100.0)
    print(f"Queue depth: {depth}")
    assert depth >= 10, "Queue should include shadow order"
    
    # Check shadow orders
    shadows = exchange.get_shadow_orders()
    print(f"Shadow orders: {len(shadows)}")
    assert len(shadows) == 1, "Should have one shadow order"
    assert shadows[0].pos >= 0, "Shadow order should have valid position"
    
    print("✓ Exchange simulator basic test passed")


def test_exchange_simulator_ioc():
    """Test exchange simulator with IOC orders."""
    print("\n--- Test 5: Exchange Simulator IOC ---")
    
    exchange = FIFOExchangeSimulator()
    
    # IOC order that can't be filled immediately
    order = Order(
        order_id="ioc-1",
        side=Side.BUY,
        price=100.0,
        qty=10,
        tif=TimeInForce.IOC,
    )
    
    receipt = exchange.on_order_arrival(order, 1000, market_qty=50)
    
    print(f"IOC order receipt: {receipt}")
    assert receipt is not None, "IOC order should get immediate receipt"
    assert receipt.receipt_type == "CANCELED", "IOC should be canceled if no immediate fill"
    
    print("✓ Exchange simulator IOC test passed")


def test_exchange_simulator_coordinate_axis():
    """Test exchange simulator coordinate-axis model."""
    print("\n--- Test 6: Exchange Simulator Coordinate Axis ---")
    
    exchange = FIFOExchangeSimulator(cancel_front_ratio=0.5)
    
    # Create a tape with trades
    config = TapeConfig()
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    
    prev = create_test_snapshot(1000, 100.0, 101.0, bid_qty=30, ask_qty=30)
    curr = create_test_snapshot(2000, 100.0, 101.0, bid_qty=20, ask_qty=20,
                                last_vol_split=[(100.0, 50)])
    
    tape = builder.build(prev, curr)
    
    # Set tape on exchange
    exchange.set_tape(tape, 1000, 2000)
    
    # Submit order 1 - should be at position tail(30) = 30
    order1 = Order(order_id="o1", side=Side.BUY, price=100.0, qty=20)
    exchange.on_order_arrival(order1, 1100, market_qty=30)
    
    # Submit order 2 - should be at position tail + shadow_qty(20) = 50
    order2 = Order(order_id="o2", side=Side.BUY, price=100.0, qty=10)
    exchange.on_order_arrival(order2, 1200, market_qty=30)
    
    shadows = exchange.get_shadow_orders()
    print(f"Shadow order positions:")
    for s in shadows:
        print(f"  {s.order_id}: pos={s.pos}, qty={s.original_qty}")
    
    # Verify FIFO: order1 should be ahead of order2
    o1 = next(s for s in shadows if s.order_id == "o1")
    o2 = next(s for s in shadows if s.order_id == "o2")
    assert o1.pos < o2.pos, "Order 1 should be ahead of Order 2"
    
    print("✓ Exchange simulator coordinate axis test passed")


def test_exchange_simulator_fill():
    """Test exchange simulator fill logic."""
    print("\n--- Test 7: Exchange Simulator Fill ---")
    
    exchange = FIFOExchangeSimulator(cancel_front_ratio=0.5)
    
    # Create tape with enough volume to fill order
    config = TapeConfig()
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    
    prev = create_test_snapshot(1000, 100.0, 101.0, bid_qty=30)
    curr = create_test_snapshot(2000, 100.0, 101.0, bid_qty=10,
                                last_vol_split=[(100.0, 50)])
    
    tape = builder.build(prev, curr)
    exchange.set_tape(tape, 1000, 2000)
    
    # Submit order at position 30 (market queue)
    order = Order(order_id="fill-test", side=Side.BUY, price=100.0, qty=15)
    exchange.on_order_arrival(order, 1050, market_qty=30)
    
    # Advance through segments
    all_receipts = []
    for seg in tape:
        receipts = exchange.advance(seg.t_start, seg.t_end, seg)
        all_receipts.extend(receipts)
        print(f"Segment {seg.index}: generated {len(receipts)} receipts")
    
    print(f"Total receipts: {len(all_receipts)}")
    for r in all_receipts:
        print(f"  {r.order_id}: {r.receipt_type}, fill_qty={r.fill_qty}")
    
    # With 50 trades and market queue 30, order should be filled
    # (trades > market_queue + order_qty)
    filled = any(r.receipt_type in ["FILL", "PARTIAL"] for r in all_receipts)
    assert filled, "Order should have been filled"
    
    print("✓ Exchange simulator fill test passed")


def test_oms():
    """Test order manager."""
    print("\n--- Test 8: Order Manager ---")
    
    portfolio = Portfolio(cash=10000.0)
    oms = OrderManager(portfolio=portfolio)
    
    # Submit order
    order = Order(
        order_id="test-1",
        side=Side.BUY,
        price=100.0,
        qty=10,
    )
    
    oms.submit(order, 1000)
    print(f"Submitted order: {order.order_id}")
    
    # Check active orders
    active = oms.get_active_orders()
    print(f"Active orders: {len(active)}")
    assert len(active) == 1, "Should have one active order"
    
    # Get order by ID
    retrieved = oms.get_order("test-1")
    assert retrieved is not None, "Should retrieve order by ID"
    
    # Process fill receipt
    from quant_framework.core.types import OrderReceipt
    receipt = OrderReceipt(
        order_id="test-1",
        receipt_type="FILL",
        timestamp=1500,
        fill_qty=10,
        fill_price=100.0,
    )
    oms.on_receipt(receipt)
    
    # Check order status
    order = oms.get_order("test-1")
    assert order.status == OrderStatus.FILLED, "Order should be FILLED"
    
    # Check portfolio
    print(f"Portfolio: cash={portfolio.cash}, position={portfolio.position}")
    assert portfolio.position == 10, "Should have position of 10"
    
    print("✓ Order manager test passed")


def test_strategy():
    """Test strategy."""
    print("\n--- Test 9: Strategy ---")
    
    strategy = SimpleNewStrategy(name="TestStrategy")
    oms = OrderManager()
    
    snapshot = create_test_snapshot(1000, 100.0, 101.0)
    
    # Call on_snapshot multiple times (strategy places order every 10 snapshots)
    all_orders = []
    for i in range(15):
        orders = strategy.on_snapshot(snapshot, oms)
        all_orders.extend(orders)
    
    print(f"Strategy generated {len(all_orders)} orders over 15 snapshots")
    assert len(all_orders) == 1, "Strategy should generate 1 order per 10 snapshots"
    
    print("✓ Strategy test passed")


def test_two_timeline():
    """Test two-timeline support."""
    print("\n--- Test 10: Two Timeline ---")
    
    config = TimelineConfig(a=1.0, b=100)  # recvtime = exchtime + 100
    
    # Test conversions
    exchtime = 1000
    recvtime = config.exchtime_to_recvtime(exchtime)
    print(f"exchtime {exchtime} -> recvtime {recvtime}")
    assert recvtime == 1100, f"Expected 1100, got {recvtime}"
    
    # Reverse conversion
    back_to_exch = config.recvtime_to_exchtime(recvtime)
    print(f"recvtime {recvtime} -> exchtime {back_to_exch}")
    assert back_to_exch == exchtime, f"Expected {exchtime}, got {back_to_exch}"
    
    # Test with scaling
    config2 = TimelineConfig(a=2.0, b=50)  # recvtime = 2 * exchtime + 50
    recvtime2 = config2.exchtime_to_recvtime(1000)
    print(f"With a=2.0, b=50: exchtime 1000 -> recvtime {recvtime2}")
    assert recvtime2 == 2050, f"Expected 2050, got {recvtime2}"
    
    print("✓ Two timeline test passed")


def test_integration_basic():
    """Test basic integration of components."""
    print("\n--- Test 11: Integration Basic ---")
    
    # Create components
    config = TapeConfig()
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    exchange = FIFOExchangeSimulator()
    oms = OrderManager()
    strategy = SimpleNewStrategy(name="TestStrategy")
    
    # Create snapshots
    prev = create_test_snapshot(1000, 100.0, 101.0)
    curr = create_test_snapshot(2000, 100.5, 101.5)
    
    # Build tape
    tape = builder.build(prev, curr)
    print(f"Built tape with {len(tape)} segments")
    
    # Set tape on exchange
    exchange.set_tape(tape, 1000, 2000)
    
    # Submit an order via strategy
    orders = strategy.on_snapshot(prev, oms)
    for order in orders:
        oms.submit(order, 1000)
    
    print(f"Submitted {len(orders)} orders")
    
    # Simulate order arrival at exchange
    for order in oms.get_active_orders():
        receipt = exchange.on_order_arrival(order, 1100, market_qty=50)
        if receipt:
            oms.on_receipt(receipt)
    
    # Advance exchange through first segment
    if tape:
        seg = tape[0]
        receipts = exchange.advance(seg.t_start, seg.t_end, seg)
        print(f"Exchange generated {len(receipts)} receipts")
        
        for receipt in receipts:
            oms.on_receipt(receipt)
    
    print("✓ Integration basic test passed")


def test_integration_with_delays():
    """Test integration with delays."""
    print("\n--- Test 12: Integration with Delays ---")
    
    # Create mock feed
    class MockFeed:
        def __init__(self, snapshots):
            self.snapshots = snapshots
            self.idx = 0
        
        def next(self):
            if self.idx < len(self.snapshots):
                snap = self.snapshots[self.idx]
                self.idx += 1
                return snap
            return None
        
        def reset(self):
            self.idx = 0
    
    # Create snapshots
    snapshots = [
        create_test_snapshot(1000, 100.0, 101.0, bid_qty=50, 
                            last_vol_split=[(100.0, 30)]),
        create_test_snapshot(2000, 100.0, 101.0, bid_qty=40,
                            last_vol_split=[(100.0, 40)]),
        create_test_snapshot(3000, 100.0, 101.0, bid_qty=30,
                            last_vol_split=[(100.0, 50)]),
    ]
    
    # Create components
    feed = MockFeed(snapshots)
    tape_config = TapeConfig()
    builder = UnifiedTapeBuilder(config=tape_config, tick_size=1.0)
    exchange = FIFOExchangeSimulator(cancel_front_ratio=0.5)
    oms = OrderManager()
    
    # Strategy that places orders more frequently
    class FrequentStrategy:
        def __init__(self):
            self.count = 0
        
        def on_snapshot(self, snapshot, oms):
            self.count += 1
            if snapshot.bids:
                return [Order(
                    order_id=f"order-{self.count}",
                    side=Side.BUY,
                    price=snapshot.bids[0].price,
                    qty=5,
                )]
            return []
        
        def on_receipt(self, receipt, snapshot, oms):
            return []
    
    strategy = FrequentStrategy()
    
    # Configure with delays
    runner_config = RunnerConfig(
        delay_out=10,  # 10 time units from strategy to exchange
        delay_in=5,    # 5 time units from exchange to strategy
        timeline=TimelineConfig(a=1.0, b=0),
    )
    
    runner = EventLoopRunner(
        feed=feed,
        tape_builder=builder,
        exchange=exchange,
        strategy=strategy,
        oms=oms,
        config=runner_config,
    )
    
    results = runner.run()
    
    print(f"Results: {results}")
    print(f"Orders submitted: {results['diagnostics']['orders_submitted']}")
    print(f"Receipts generated: {results['diagnostics']['receipts_generated']}")
    
    assert results['intervals'] == 2, f"Expected 2 intervals, got {results['intervals']}"
    
    print("✓ Integration with delays test passed")


def test_fill_priority():
    """Test that fills respect time priority (FIFO).
    
    From spec: When order1 arrives with market queue 30, then queue grows to 40,
    then order2 arrives, order1 should fill before order2.
    """
    print("\n--- Test 13: Fill Priority (FIFO) ---")
    
    exchange = FIFOExchangeSimulator(cancel_front_ratio=0.5)
    
    # Create tape where market queue grows then shrinks
    config = TapeConfig()
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    
    # Start with queue 30, end with queue 10
    # With lastvolsplit of 50 trades
    prev = create_test_snapshot(1000, 100.0, 101.0, bid_qty=30)
    curr = create_test_snapshot(2000, 100.0, 101.0, bid_qty=10,
                                last_vol_split=[(100.0, 50)])
    
    tape = builder.build(prev, curr)
    exchange.set_tape(tape, 1000, 2000)
    
    # Order 1: arrives at t=1100 when market_qty=30
    # Position should be at tail = 30
    order1 = Order(order_id="order1", side=Side.BUY, price=100.0, qty=20)
    exchange.on_order_arrival(order1, 1100, market_qty=30)
    
    # Order 2: arrives at t=1200, market_qty still around 30-40
    # Position should be at tail + order1_qty = 30 + 20 = 50
    order2 = Order(order_id="order2", side=Side.BUY, price=100.0, qty=10)
    exchange.on_order_arrival(order2, 1200, market_qty=30)
    
    # Verify positions
    shadows = exchange.get_shadow_orders()
    o1 = next(s for s in shadows if s.order_id == "order1")
    o2 = next(s for s in shadows if s.order_id == "order2")
    
    print(f"Order 1 position: {o1.pos}")
    print(f"Order 2 position: {o2.pos}")
    
    assert o1.pos < o2.pos, "Order 1 should have lower position (earlier in queue)"
    assert o2.pos >= o1.pos + o1.original_qty, "Order 2 should be after Order 1's range"
    
    # Advance and collect fills
    all_receipts = []
    last_t = 1000
    for seg in tape:
        receipts = exchange.advance(last_t, seg.t_end, seg)
        all_receipts.extend(receipts)
        last_t = seg.t_end
    
    # With 50 trades: first 30 consume market queue, then order1 (20), then order2 (10)
    # Order 1 should fill first
    fill_order = [r.order_id for r in all_receipts if r.receipt_type in ["FILL", "PARTIAL"]]
    print(f"Fill order: {fill_order}")
    
    if len(fill_order) >= 2:
        assert fill_order.index("order1") < fill_order.index("order2"), \
            "Order 1 should fill before Order 2"
    
    print("✓ Fill priority test passed")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Testing Unified EventLoop Framework")
    print("="*60)
    
    tests = [
        test_tape_builder_basic,
        test_tape_builder_no_trades,
        test_tape_builder_conservation,
        test_exchange_simulator_basic,
        test_exchange_simulator_ioc,
        test_exchange_simulator_coordinate_axis,
        test_exchange_simulator_fill,
        test_oms,
        test_strategy,
        test_two_timeline,
        test_integration_basic,
        test_integration_with_delays,
        test_fill_priority,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
