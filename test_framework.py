"""Simple test to validate the new EventLoop framework."""

from quant_framework.core.types import NormalizedSnapshot, Level, Order, Side
from quant_framework.tape.builder import UnifiedTapeBuilder, TapeConfig
from quant_framework.exchange.simulator import FIFOExchangeSimulator
from quant_framework.trading.oms import OrderManager, Portfolio
from quant_framework.trading.strategy import SimpleNewStrategy


def create_test_snapshot(ts: int, bid: float, ask: float) -> NormalizedSnapshot:
    """Create a test snapshot."""
    return NormalizedSnapshot(
        ts_exch=ts,
        bids=[Level(bid, 100)],
        asks=[Level(ask, 100)],
        last_vol_split=[(bid, 10), (ask, 10)],
    )


def test_tape_builder():
    """Test tape builder."""
    print("\n--- Test 1: Tape Builder ---")
    
    config = TapeConfig()
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    
    prev = create_test_snapshot(1000, 100.0, 101.0)
    curr = create_test_snapshot(2000, 100.5, 101.5)
    
    tape = builder.build(prev, curr)
    
    print(f"Generated {len(tape)} segments")
    for seg in tape:
        print(f"  Segment {seg.index}: t=[{seg.t_start}, {seg.t_end}], bid={seg.bid_price}, ask={seg.ask_price}")
    
    assert len(tape) > 0, "Tape should have at least one segment"
    print("✓ Tape builder test passed")


def test_exchange_simulator():
    """Test exchange simulator."""
    print("\n--- Test 2: Exchange Simulator ---")
    
    exchange = FIFOExchangeSimulator()
    
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
    assert receipt is None, "Order should be accepted"
    
    # Check queue depth
    depth = exchange.get_queue_depth(Side.BUY, 100.0)
    print(f"Queue depth: {depth}")
    assert depth >= 10, "Queue should include shadow order"
    
    print("✓ Exchange simulator test passed")


def test_oms():
    """Test order manager."""
    print("\n--- Test 3: Order Manager ---")
    
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
    print(f"Retrieved order: {retrieved.order_id}")
    
    print("✓ Order manager test passed")


def test_strategy():
    """Test strategy."""
    print("\n--- Test 4: Strategy ---")
    
    strategy = SimpleNewStrategy(name="TestStrategy")
    oms = OrderManager()
    
    snapshot = create_test_snapshot(1000, 100.0, 101.0)
    
    # Call on_snapshot
    orders = strategy.on_snapshot(snapshot, oms)
    print(f"Strategy generated {len(orders)} orders on snapshot")
    
    print("✓ Strategy test passed")


def test_integration():
    """Test integration of components."""
    print("\n--- Test 5: Integration Test ---")
    
    # Create components
    config = TapeConfig()
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    exchange = FIFOExchangeSimulator()
    oms = OrderManager()
    strategy = SimpleNewStrategy()
    
    # Create snapshots
    prev = create_test_snapshot(1000, 100.0, 101.0)
    curr = create_test_snapshot(2000, 100.5, 101.5)
    
    # Build tape
    tape = builder.build(prev, curr)
    print(f"Built tape with {len(tape)} segments")
    
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
    
    print("✓ Integration test passed")


if __name__ == "__main__":
    print("="*60)
    print("Testing New EventLoop Framework")
    print("="*60)
    
    try:
        test_tape_builder()
        test_exchange_simulator()
        test_oms()
        test_strategy()
        test_integration()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
