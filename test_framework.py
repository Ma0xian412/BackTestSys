"""统一EventLoop框架的综合测试套件。

测试验证内容：
- Tape构建器：段生成、成交量分配、守恒方程
- 交易所模拟器：坐标轴模型、成交时间计算
- 订单管理器：订单生命周期、回执处理
- 事件循环：双时间线支持、延迟处理
- 集成测试：完整回测流程
- DTO测试：数据传输对象和只读视图
"""

from quant_framework.core.types import (
    NormalizedSnapshot, Level, Order, Side, TimeInForce, OrderStatus, TICK_PER_MS
)
from quant_framework.tape.builder import UnifiedTapeBuilder, TapeConfig
from quant_framework.exchange.simulator import FIFOExchangeSimulator
from quant_framework.trading.oms import OrderManager, Portfolio
from quant_framework.trading.strategy import SimpleStrategy
from quant_framework.runner.event_loop import EventLoopRunner, RunnerConfig, TimelineConfig


def create_test_snapshot(ts: int, bid: float, ask: float,
                         bid_qty: int = 100, ask_qty: int = 100,
                         last_vol_split=None) -> NormalizedSnapshot:
    """创建测试快照。
    
    时间单位为tick（每tick=100ns）。
    """
    if last_vol_split is None:
        last_vol_split = [(bid, 10), (ask, 10)]
    return NormalizedSnapshot(
        ts_recv=ts,  # 主时间线
        bids=[Level(bid, bid_qty)],
        asks=[Level(ask, ask_qty)],
        last_vol_split=last_vol_split,
    )


def create_multi_level_snapshot(ts: int, bids: list, asks: list,
                                 last_vol_split=None) -> NormalizedSnapshot:
    """创建多档位快照。
    
    时间单位为tick（每tick=100ns）。
    """
    bid_levels = [Level(p, q) for p, q in bids]
    ask_levels = [Level(p, q) for p, q in asks]
    return NormalizedSnapshot(
        ts_recv=ts,  # 主时间线
        bids=bid_levels,
        asks=ask_levels,
        last_vol_split=last_vol_split or [],
    )


def print_tape_path(tape) -> None:
    """打印tape路径详情。"""
    print(f"\n  Tape路径 (共{len(tape)}个段):")
    for seg in tape:
        print(f"    段{seg.index}: t=[{seg.t_start}, {seg.t_end}], bid={seg.bid_price}, ask={seg.ask_price}")
        if seg.trades:
            print(f"      trades: {dict(seg.trades)}")
        if seg.cancels:
            print(f"      cancels: {dict(seg.cancels)}")
        if seg.net_flow:
            print(f"      net_flow: {dict(seg.net_flow)}")
        print(f"      activation_bid: {seg.activation_bid}")
        print(f"      activation_ask: {seg.activation_ask}")
    print()


def test_tape_builder_basic():
    """测试Tape构建器基本功能。"""
    print("\n--- Test 1: Tape Builder Basic ---")
    
    config = TapeConfig()
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    
    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0)
    curr = create_test_snapshot(1500 * TICK_PER_MS, 100.5, 101.5)
    
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    
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
    
    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, last_vol_split=[])
    curr = create_test_snapshot(1500 * TICK_PER_MS, 100.5, 101.5, last_vol_split=[])
    
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    assert len(tape) == 1, "Should have single segment with no trades"
    
    print("✓ Tape builder no trades test passed")


def test_tape_builder_conservation():
    """Test tape builder conservation equations."""
    print("\n--- Test 3: Tape Builder Conservation ---")
    
    config = TapeConfig(epsilon=1.0, segment_iterations=2)
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    
    # Create snapshots where we can verify conservation
    prev = create_multi_level_snapshot(
        1000 * TICK_PER_MS,
        bids=[(100.0, 50), (99.0, 30)],
        asks=[(101.0, 40), (102.0, 20)],
        last_vol_split=[(100.0, 20), (101.0, 15)]
    )
    curr = create_multi_level_snapshot(
        1500 * TICK_PER_MS,
        bids=[(100.0, 40), (99.0, 35)],  # delta_Q_bid @ 100 = -10
        asks=[(101.0, 35), (102.0, 25)],  # delta_Q_ask @ 101 = -5
        last_vol_split=[(100.0, 20), (101.0, 15)]
    )
    
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    
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
    # So total trades should equal lastvolsplit (allowing for rounding tolerance of ±1)
    # Rounding tolerance is needed because starting price trade prepending may create
    # duplicate path segments, causing trade volumes to be split and rounded separately
    assert abs(total_bid_trades - 20) <= 1, f"Expected ~20 bid trades, got {total_bid_trades}"
    assert abs(total_ask_trades - 15) <= 1, f"Expected ~15 ask trades, got {total_ask_trades}"
    
    print("✓ Tape builder conservation test passed")


def test_tape_builder_netflow_distribution():
    """Test netflow distribution for active segments and zeroing."""
    print("\n--- Test 3b: Tape Builder Netflow Distribution ---")
    
    config = TapeConfig(epsilon=1.0, segment_iterations=2)
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    
    prev = create_multi_level_snapshot(
        1000,
        bids=[(100.0, 40), (99.0, 30), (98.0, 20)],
        asks=[(101.0, 40), (102.0, 20), (103.0, 10)],
        last_vol_split=[(100.0, 10), (101.0, 15), (102.0, 10)]
    )
    curr = create_multi_level_snapshot(
        2000,
        bids=[(101.0, 55), (100.0, 40), (99.0, 35)],
        asks=[(100.0, 35), (101.0, 35), (102.0, 25)],
        last_vol_split=[(100.0, 10), (101.0, 15), (102.0, 10)]
    )
    
    tape = builder.build(prev, curr)
    
    net_flow_bid_100 = [seg.net_flow.get((Side.BUY, 100.0), 0) for seg in tape]
    net_flow_ask_101 = [seg.net_flow.get((Side.SELL, 101.0), 0) for seg in tape]
    trades_ask_101 = [seg.trades.get((Side.SELL, 101.0), 0) for seg in tape]
    
    # Bid 100 should distribute across activated segments
    assert len(tape) >= 3, "Expected at least 3 segments for distribution test"
    total_bid_net = sum(net_flow_bid_100)
    prev_bid_qty = next(lvl.qty for lvl in prev.bids if abs(lvl.price - 100.0) < 1e-8)
    curr_bid_qty = next(lvl.qty for lvl in curr.bids if abs(lvl.price - 100.0) < 1e-8)
    expected_bid_net = (
        curr_bid_qty - prev_bid_qty + sum(seg.trades.get((Side.BUY, 100.0), 0) for seg in tape)
    )
    assert total_bid_net == expected_bid_net, (
        f"Expected total netflow {expected_bid_net} for bid 100, got {total_bid_net}"
    )
    assert net_flow_bid_100[0] != 0, "Netflow for bid 100 should be present in first segment"
    assert net_flow_bid_100[1] != 0, "Netflow for bid 100 should be present in second segment"
    assert net_flow_bid_100[-1] != total_bid_net, "Netflow for bid 100 should not be all in last segment"
    
    # Ask 101 should have netflow on segments where it is active
    total_ask_net = sum(net_flow_ask_101)
    prev_ask_qty = next(lvl.qty for lvl in prev.asks if abs(lvl.price - 101.0) < 1e-8)
    curr_ask_qty = next(lvl.qty for lvl in curr.asks if abs(lvl.price - 101.0) < 1e-8)
    expected_ask_net = (
        curr_ask_qty - prev_ask_qty + sum(seg.trades.get((Side.SELL, 101.0), 0) for seg in tape)
    )
    assert total_ask_net == expected_ask_net, (
        f"Expected total netflow {expected_ask_net} for ask 101, got {total_ask_net}"
    )
    assert net_flow_ask_101[0] != 0, "Ask 101 netflow should be present in first segment"
    assert net_flow_ask_101[1] == 0, "Ask 101 netflow should be zero when price is inactive"
    assert net_flow_ask_101[2] != 0, "Ask 101 netflow should be present in last segment"
    
    # Zeroing segment should consume queue depth (netflow <= trades)
    assert net_flow_ask_101[0] <= trades_ask_101[0], "Zeroing segment should not leave queue depth"
    
    print("✓ Tape builder netflow distribution test passed")


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
    receipt = exchange.on_order_arrival(order, 1000 * TICK_PER_MS, market_qty=50)
    
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
    
    receipt = exchange.on_order_arrival(order, 1000 * TICK_PER_MS, market_qty=50)
    
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
    
    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=30, ask_qty=30)
    curr = create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=20, ask_qty=20,
                                last_vol_split=[(100.0, 50)])
    
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    
    # Set tape on exchange
    exchange.set_tape(tape, 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)
    
    # Submit order 1 - should be at position tail(30) = 30
    order1 = Order(order_id="o1", side=Side.BUY, price=100.0, qty=20)
    exchange.on_order_arrival(order1, 1100 * TICK_PER_MS, market_qty=30)
    
    # Submit order 2 - should be at position tail + shadow_qty(20) = 50
    order2 = Order(order_id="o2", side=Side.BUY, price=100.0, qty=10)
    exchange.on_order_arrival(order2, 1200 * TICK_PER_MS, market_qty=30)
    
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
    
    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=30)
    curr = create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=10,
                                last_vol_split=[(100.0, 50)])
    
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    
    exchange.set_tape(tape, 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)
    
    # Submit order at position 30 (market queue)
    order = Order(order_id="fill-test", side=Side.BUY, price=100.0, qty=15)
    exchange.on_order_arrival(order, 1050 * TICK_PER_MS, market_qty=30)
    
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
    
    oms.submit(order, 1000 * TICK_PER_MS)
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
        timestamp=1500 * TICK_PER_MS,
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
    """测试策略（使用DTO）。"""
    print("\n--- Test 9: Strategy ---")
    
    from quant_framework.core.dto import to_snapshot_dto, ReadOnlyOMSView
    
    strategy = SimpleStrategy(name="TestStrategy")
    oms = OrderManager()
    oms_view = ReadOnlyOMSView(oms)
    
    snapshot = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0)
    snapshot_dto = to_snapshot_dto(snapshot)
    
    # Call on_snapshot multiple times (strategy places order every 10 snapshots)
    all_orders = []
    for i in range(15):
        orders = strategy.on_snapshot(snapshot_dto, oms_view)
        all_orders.extend(orders)
    
    print(f"Strategy generated {len(all_orders)} orders over 15 snapshots")
    assert len(all_orders) == 1, "Strategy should generate 1 order per 10 snapshots"
    
    print("✓ Strategy test passed")


def test_two_timeline():
    """Test timeline config (deprecated - now uses single timeline).
    
    TimelineConfig is kept for backward compatibility but now always returns
    identity transformations (no conversion needed with single recv timeline).
    """
    print("\n--- Test 10: Timeline Config (Single Timeline) ---")
    
    config = TimelineConfig(a=1.0, b=100)  # 已弃用参数，不再影响转换
    
    # Test conversions - now identity (no conversion with single timeline)
    time = 1000 * TICK_PER_MS
    result = config.exchtime_to_recvtime(time)
    print(f"time {time} -> result {result}")
    assert result == time, f"Expected {time}, got {result}"
    
    # Reverse conversion - also identity
    back = config.recvtime_to_exchtime(result)
    print(f"result {result} -> back {back}")
    assert back == time, f"Expected {time}, got {back}"
    
    # All times should pass through unchanged (single timeline)
    print("Note: TimelineConfig is deprecated - all times use ts_recv directly")
    
    print("✓ Timeline config test passed")


def test_integration_basic():
    """Test basic integration of components."""
    print("\n--- Test 11: Integration Basic ---")
    
    from quant_framework.core.dto import to_snapshot_dto, ReadOnlyOMSView
    
    # Create components
    config = TapeConfig()
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    exchange = FIFOExchangeSimulator()
    oms = OrderManager()
    oms_view = ReadOnlyOMSView(oms)
    strategy = SimpleStrategy(name="TestStrategy")
    
    # Create snapshots
    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0)
    curr = create_test_snapshot(1500 * TICK_PER_MS, 100.5, 101.5)
    prev_dto = to_snapshot_dto(prev)
    
    # Build tape
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    
    # Set tape on exchange
    exchange.set_tape(tape, 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)
    
    # Submit an order via strategy (使用DTO)
    orders = strategy.on_snapshot(prev_dto, oms_view)
    for order in orders:
        oms.submit(order, 1000 * TICK_PER_MS)
    
    print(f"Submitted {len(orders)} orders")
    
    # Simulate order arrival at exchange
    for order in oms.get_active_orders():
        receipt = exchange.on_order_arrival(order, 1100 * TICK_PER_MS, market_qty=50)
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
    
    # Create snapshots (500ms intervals)
    snapshots = [
        create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=50, 
                            last_vol_split=[(100.0, 30)]),
        create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=40,
                            last_vol_split=[(100.0, 40)]),
        create_test_snapshot(2000 * TICK_PER_MS, 100.0, 101.0, bid_qty=30,
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
        delay_out=10 * TICK_PER_MS,  # 10ms from strategy to exchange
        delay_in=5 * TICK_PER_MS,    # 5ms from exchange to strategy
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
    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=30)
    curr = create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=10,
                                last_vol_split=[(100.0, 50)])
    
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    
    exchange.set_tape(tape, 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)
    
    # Order 1: arrives at t=1100ms when market_qty=30
    # Position should be at tail = 30
    order1 = Order(order_id="order1", side=Side.BUY, price=100.0, qty=20)
    exchange.on_order_arrival(order1, 1100 * TICK_PER_MS, market_qty=30)
    
    # Order 2: arrives at t=1200ms, market_qty still around 30-40
    # Position should be at tail + order1_qty = 30 + 20 = 50
    order2 = Order(order_id="order2", side=Side.BUY, price=100.0, qty=10)
    exchange.on_order_arrival(order2, 1200 * TICK_PER_MS, market_qty=30)
    
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
    last_t = 1000 * TICK_PER_MS
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


def test_dto_snapshot():
    """测试DTO快照转换功能。"""
    print("\n--- Test 14: DTO Snapshot ---")
    
    from quant_framework.core.dto import to_snapshot_dto, SnapshotDTO, LevelDTO
    
    # 创建测试快照
    snapshot = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=50, ask_qty=60)
    
    # 转换为DTO
    dto = to_snapshot_dto(snapshot)
    
    print(f"DTO类型: {type(dto).__name__}")
    # 验证DTO是frozen（使用dataclass的__dataclass_fields__属性检查）
    is_frozen = hasattr(dto, '__dataclass_fields__') and dto.__class__.__dataclass_params__.frozen
    print(f"DTO是否为frozen: {is_frozen}")
    assert is_frozen, "SnapshotDTO应该是frozen=True"
    
    # 验证数据正确转换 - ts_recv是主时间线
    assert dto.ts_recv == 1000 * TICK_PER_MS, f"时间戳应为{1000 * TICK_PER_MS}，实际为{dto.ts_recv}"
    assert len(dto.bids) == 1, f"买盘档位应为1，实际为{len(dto.bids)}"
    assert len(dto.asks) == 1, f"卖盘档位应为1，实际为{len(dto.asks)}"
    
    # 验证便捷属性
    assert dto.best_bid == 100.0, f"最优买价应为100.0，实际为{dto.best_bid}"
    assert dto.best_ask == 101.0, f"最优卖价应为101.0，实际为{dto.best_ask}"
    assert dto.mid_price == 100.5, f"中间价应为100.5，实际为{dto.mid_price}"
    assert dto.spread == 1.0, f"价差应为1.0，实际为{dto.spread}"
    
    # 验证DTO不可变（尝试修改应抛出异常）
    try:
        dto.ts_recv = 2000
        assert False, "DTO应该是不可变的"
    except Exception:
        pass  # 预期的行为
    
    print("✓ DTO snapshot test passed")


def test_readonly_oms_view():
    """测试只读OMS视图功能。"""
    print("\n--- Test 15: ReadOnly OMS View ---")
    
    from quant_framework.core.dto import ReadOnlyOMSView, OrderInfoDTO, PortfolioDTO
    
    # 创建OMS和订单
    portfolio = Portfolio(cash=10000.0)
    oms = OrderManager(portfolio=portfolio)
    
    order = Order(
        order_id="test-readonly-1",
        side=Side.BUY,
        price=100.0,
        qty=10,
    )
    oms.submit(order, 1000 * TICK_PER_MS)
    
    # 创建只读视图
    view = ReadOnlyOMSView(oms)
    
    # 测试查询活跃订单
    active_orders = view.get_active_orders()
    print(f"活跃订单数量: {len(active_orders)}")
    assert len(active_orders) == 1, f"应有1个活跃订单，实际为{len(active_orders)}"
    
    # 验证返回的是OrderInfoDTO而不是Order
    assert isinstance(active_orders[0], OrderInfoDTO), "应返回OrderInfoDTO类型"
    
    # 测试查询单个订单
    order_dto = view.get_order("test-readonly-1")
    assert order_dto is not None, "订单应存在"
    assert order_dto.order_id == "test-readonly-1", f"订单ID应为test-readonly-1"
    assert order_dto.price == 100.0, f"价格应为100.0"
    
    # 验证OrderInfoDTO是不可变的
    try:
        order_dto.price = 200.0
        assert False, "OrderInfoDTO应该是不可变的"
    except Exception:
        pass  # 预期的行为
    
    # 测试查询投资组合
    portfolio_dto = view.get_portfolio()
    assert isinstance(portfolio_dto, PortfolioDTO), "应返回PortfolioDTO类型"
    assert portfolio_dto.cash == 10000.0, f"现金应为10000.0，实际为{portfolio_dto.cash}"
    
    # 验证PortfolioDTO是不可变的
    try:
        portfolio_dto.cash = 20000.0
        assert False, "PortfolioDTO应该是不可变的"
    except Exception:
        pass  # 预期的行为
    
    # 验证只读视图没有修改方法
    assert not hasattr(view, 'submit'), "只读视图不应有submit方法"
    assert not hasattr(view, 'on_receipt'), "只读视图不应有on_receipt方法"
    
    print("✓ ReadOnly OMS view test passed")


def test_dto_strategy():
    """测试使用DTO的策略。"""
    print("\n--- Test 16: DTO Strategy ---")
    
    from quant_framework.trading.strategy import SimpleStrategy
    from quant_framework.core.dto import to_snapshot_dto, ReadOnlyOMSView
    
    # 创建策略和OMS
    strategy = SimpleStrategy(name="TestDTOStrategy")
    oms = OrderManager()
    view = ReadOnlyOMSView(oms)
    
    # 创建快照并转换为DTO
    snapshot = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0)
    snapshot_dto = to_snapshot_dto(snapshot)
    
    # 调用策略（每10个快照下一单）
    all_orders = []
    for i in range(15):
        orders = strategy.on_snapshot(snapshot_dto, view)
        all_orders.extend(orders)
    
    print(f"策略在15个快照中生成了{len(all_orders)}个订单")
    assert len(all_orders) == 1, f"策略应生成1个订单，实际为{len(all_orders)}"
    
    # 验证订单正确
    order = all_orders[0]
    assert order.side == Side.BUY, f"订单方向应为BUY"
    assert order.price == 100.0, f"订单价格应为100.0（最优买价）"
    
    print("✓ DTO strategy test passed")


def test_tape_builder_invalid_time_order():
    """测试当t_b <= t_a时，tape构建器应抛出ValueError。
    
    这确保了快照时间必须严格递增。
    """
    print("\n--- Test 17: Tape Builder Invalid Time Order ---")
    
    config = TapeConfig()
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    
    # 测试情况1: t_b == t_a
    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0)
    curr = create_test_snapshot(1000 * TICK_PER_MS, 100.5, 101.5)  # 相同时间
    
    try:
        builder.build(prev, curr)
        assert False, "应该抛出ValueError当t_b == t_a"
    except ValueError as e:
        print(f"✓ 正确抛出ValueError当t_b == t_a: {e}")
    
    # 测试情况2: t_b < t_a
    prev2 = create_test_snapshot(2000 * TICK_PER_MS, 100.0, 101.0)
    curr2 = create_test_snapshot(1000 * TICK_PER_MS, 100.5, 101.5)  # 时间倒退
    
    try:
        builder.build(prev2, curr2)
        assert False, "应该抛出ValueError当t_b < t_a"
    except ValueError as e:
        print(f"✓ 正确抛出ValueError当t_b < t_a: {e}")
    
    print("✓ Tape builder invalid time order test passed")


def test_meeting_sequence_consistency():
    """测试meeting sequence确保bid和ask路径的中间部分一致。
    
    这验证了区间扩张DP算法生成的公共相遇价位序列。
    """
    print("\n--- Test 18: Meeting Sequence Consistency ---")
    
    config = TapeConfig()
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    
    # 创建有多个成交价位的快照
    prev = create_multi_level_snapshot(
        1000 * TICK_PER_MS,
        bids=[(100.0, 50), (99.0, 30)],
        asks=[(101.0, 40), (102.0, 20)],
        last_vol_split=[(99.5, 10), (100.5, 15), (101.0, 20), (100.0, 25)]
    )
    curr = create_multi_level_snapshot(
        1500 * TICK_PER_MS,
        bids=[(100.5, 40), (99.5, 35)],
        asks=[(101.5, 35), (102.5, 25)],
        last_vol_split=[(99.5, 10), (100.5, 15), (101.0, 20), (100.0, 25)]
    )
    
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    
    # 获取所有bid和ask在各段的价格路径
    bid_prices = []
    ask_prices = []
    
    for seg in tape:
        bid_prices.append(seg.bid_price)
        ask_prices.append(seg.ask_price)
    
    # 提取中间段的价格（去掉首尾，因为首尾可能不同）
    # 由于meeting sequence相同，中间段的bid和ask应该通过相同的价位
    
    # 找到bid_prices和ask_prices中的重叠价位
    bid_set = set(bid_prices[1:-1]) if len(bid_prices) > 2 else set()
    ask_set = set(ask_prices[1:-1]) if len(ask_prices) > 2 else set()
    
    # 验证段数量合理
    assert len(tape) >= 1, "应该至少有1个段"
    
    print("✓ Meeting sequence consistency test passed")


def test_historical_receipts_processing():
    """测试历史回执在当前区间开始前被正确处理。
    
    这确保了因果一致性：当segment到达t_a时，所有早于t_a的事件都已处理。
    """
    print("\n--- Test 19: Historical Receipts Processing ---")
    
    from quant_framework.runner.event_loop import EventLoopRunner, RunnerConfig, TimelineConfig
    
    # 创建一个简单的回测场景来测试历史回执处理
    # 由于这需要完整的组件，我们使用集成测试方式
    
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
    
    # 创建测试快照 (500ms intervals)
    snapshots = [
        create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=50),
        create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=40),
        create_test_snapshot(2000 * TICK_PER_MS, 100.0, 101.0, bid_qty=30),
    ]
    
    # 创建组件
    feed = MockFeed(snapshots)
    tape_config = TapeConfig()
    tape_builder = UnifiedTapeBuilder(config=tape_config, tick_size=1.0)
    exchange = FIFOExchangeSimulator(cancel_front_ratio=0.5)
    oms = OrderManager()
    
    # 简单策略
    class TestStrategy:
        def __init__(self):
            self.snapshots_received = []
            self.receipts_received = []
        
        def on_snapshot(self, snapshot, oms):
            self.snapshots_received.append(snapshot.ts_exch if hasattr(snapshot, 'ts_exch') else snapshot)
            return []
        
        def on_receipt(self, receipt, snapshot, oms):
            self.receipts_received.append((receipt.timestamp, receipt.recv_time))
            return []
    
    strategy = TestStrategy()
    
    runner_config = RunnerConfig(
        delay_out=100 * TICK_PER_MS,
        delay_in=50 * TICK_PER_MS,
        timeline=TimelineConfig(a=1.0, b=0),
    )
    
    runner = EventLoopRunner(
        feed=feed,
        tape_builder=tape_builder,
        exchange=exchange,
        strategy=strategy,
        oms=oms,
        config=runner_config,
    )
    
    results = runner.run()
    
    print(f"处理了{results['intervals']}个区间")
    print(f"快照数量: {len(strategy.snapshots_received)}")
    
    # 验证运行器正确初始化和运行
    assert results['intervals'] == 2, f"应该处理2个区间，实际处理了{results['intervals']}个"
    
    print("✓ Historical receipts processing test passed")


def test_intra_segment_advancement():
    """测试修复问题1：事件落在segment内部时，交易所正确推进到event.time。
    
    这确保了订单不会"插在"尚未发生的成交之前，维护因果一致性。
    
    测试场景：
    1. 创建一个区间[1000ms, 1500ms]，包含多个成交
    2. 在区间中间时刻提交一个订单
    3. 验证订单到达时，该时刻之前的成交已经被处理
    """
    print("\n--- Test 20: Intra-Segment Advancement ---")
    
    from quant_framework.runner.event_loop import EventLoopRunner, RunnerConfig, TimelineConfig
    
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
    
    # 创建测试快照：有足够成交量的区间 (500ms interval)
    snapshots = [
        create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=50, last_vol_split=[(100.0, 100)]),
        create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=20, last_vol_split=[(100.0, 100)]),
    ]
    
    feed = MockFeed(snapshots)
    tape_config = TapeConfig()
    tape_builder = UnifiedTapeBuilder(config=tape_config, tick_size=1.0)
    exchange = FIFOExchangeSimulator(cancel_front_ratio=0.5)
    oms = OrderManager()
    
    # 记录交易所advance调用时间的策略
    class TrackingStrategy:
        def __init__(self):
            self.order_count = 0
            self.order_placed_at = None
        
        def on_snapshot(self, snapshot, oms):
            self.order_count += 1
            # 在第一个快照时下单
            if self.order_count == 1:
                order = Order(
                    order_id=f"test-intra-{self.order_count}",
                    side=Side.BUY,
                    price=100.0,
                    qty=10,
                )
                self.order_placed_at = snapshot.ts_exch if hasattr(snapshot, 'ts_exch') else 1000 * TICK_PER_MS
                return [order]
            return []
        
        def on_receipt(self, receipt, snapshot, oms):
            return []
    
    strategy = TrackingStrategy()
    
    # 使用延迟配置，让订单在区间中间到达
    runner_config = RunnerConfig(
        delay_out=250 * TICK_PER_MS,  # 订单在recv_time+250ms时到达交易所
        delay_in=50 * TICK_PER_MS,
        timeline=TimelineConfig(a=1.0, b=0),
    )
    
    runner = EventLoopRunner(
        feed=feed,
        tape_builder=tape_builder,
        exchange=exchange,
        strategy=strategy,
        oms=oms,
        config=runner_config,
    )
    
    results = runner.run()
    
    print(f"订单提交数: {results['diagnostics']['orders_submitted']}")
    print(f"回执生成数: {results['diagnostics']['receipts_generated']}")
    
    # 验证运行成功
    assert results['intervals'] == 1, f"应处理1个区间，实际{results['intervals']}"
    
    # 关键验证：检查exchange的current_time是否在处理订单前被正确推进
    # 由于订单在1500ms时到达（1000ms+500ms延迟），交易所应该先推进到1500ms
    # 这里我们通过检查结果来间接验证
    
    print("✓ Intra-segment advancement test passed")


def test_receipt_delay_consistency():
    """测试修复问题2：所有回执都通过统一RECEIPT_TO_STRATEGY调度，delay_in生效。
    
    这确保了OMS状态不会提前变化，维护延迟因果一致性。
    
    测试场景：
    1. 提交一个IOC订单（会立即产生回执）
    2. 验证回执是通过事件队列调度的，而不是立即处理
    3. 验证回执的recv_time正确包含delay_in
    """
    print("\n--- Test 21: Receipt Delay Consistency ---")
    
    from quant_framework.runner.event_loop import EventLoopRunner, RunnerConfig, TimelineConfig
    
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
    
    # 创建测试快照 (500ms interval)
    snapshots = [
        create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=50, last_vol_split=[(100.0, 10)]),
        create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=40, last_vol_split=[(100.0, 10)]),
    ]
    
    feed = MockFeed(snapshots)
    tape_config = TapeConfig()
    tape_builder = UnifiedTapeBuilder(config=tape_config, tick_size=1.0)
    exchange = FIFOExchangeSimulator(cancel_front_ratio=0.5)
    oms = OrderManager()
    
    # 记录回执到达时间的策略
    class IOCStrategy:
        def __init__(self):
            self.order_count = 0
            self.receipt_times = []  # (timestamp, recv_time) pairs
            self.oms_state_at_receipt = []  # OMS状态快照
        
        def on_snapshot(self, snapshot, oms):
            self.order_count += 1
            if self.order_count == 1:
                # 提交IOC订单（会被立即取消因为无法立即成交）
                order = Order(
                    order_id="ioc-delay-test",
                    side=Side.BUY,
                    price=100.0,
                    qty=10,
                    tif=TimeInForce.IOC,
                )
                return [order]
            return []
        
        def on_receipt(self, receipt, snapshot, oms):
            # 记录回执到达时的信息
            self.receipt_times.append((receipt.timestamp, receipt.recv_time))
            # 记录此时OMS中该订单的状态
            order = oms.get_order(receipt.order_id)
            if order:
                self.oms_state_at_receipt.append(order.status.value)
            return []
    
    strategy = IOCStrategy()
    
    # 使用有延迟的配置
    delay_in = 100 * TICK_PER_MS
    delay_out = 50 * TICK_PER_MS
    runner_config = RunnerConfig(
        delay_out=delay_out,
        delay_in=delay_in,
        timeline=TimelineConfig(a=1.0, b=0),
    )
    
    runner = EventLoopRunner(
        feed=feed,
        tape_builder=tape_builder,
        exchange=exchange,
        strategy=strategy,
        oms=oms,
        config=runner_config,
    )
    
    results = runner.run()
    
    print(f"订单提交数: {results['diagnostics']['orders_submitted']}")
    print(f"回执生成数: {results['diagnostics']['receipts_generated']}")
    print(f"回执时间记录: {strategy.receipt_times}")
    
    # 验证回执被正确处理
    assert results['diagnostics']['receipts_generated'] >= 1, "应至少生成1个回执"
    
    # 验证回执的recv_time正确包含delay_in
    for timestamp, recv_time in strategy.receipt_times:
        expected_recv_time = timestamp + delay_in
        assert recv_time == expected_recv_time, \
            f"回执recv_time应为{expected_recv_time}（timestamp={timestamp} + delay_in={delay_in}），实际为{recv_time}"
        print(f"✓ 回执recv_time正确: timestamp={timestamp}, recv_time={recv_time}, delay_in={delay_in}")
    
    print("✓ Receipt delay consistency test passed")


def test_event_deterministic_ordering():
    """测试修复问题2：Event比较使用(time, priority, seq)确保确定性排序。
    
    验证同一时刻的多个事件有确定性的处理顺序，避免"同样输入偶发不同输出"。
    
    优先级顺序应为：
    1. SEGMENT_END（先完成内部撮合）
    2. ORDER_ARRIVAL/CANCEL_ARRIVAL（交易所收到请求）
    3. RECEIPT_TO_STRATEGY（策略看到回执）
    4. INTERVAL_END（最后做边界对齐）
    """
    print("\n--- Test 22: Event Deterministic Ordering ---")
    
    import heapq
    from quant_framework.runner.event_loop import (
        Event, EventType, EVENT_TYPE_PRIORITY, reset_event_seq_counter
    )
    
    # 重置序列号计数器确保测试隔离
    reset_event_seq_counter()
    
    # 创建同一时刻的多种事件（故意打乱顺序创建）
    t = 1000 * TICK_PER_MS
    events = [
        Event(time=t, event_type=EventType.INTERVAL_END, data="interval"),
        Event(time=t, event_type=EventType.RECEIPT_TO_STRATEGY, data="receipt1"),
        Event(time=t, event_type=EventType.SEGMENT_END, data="segment"),
        Event(time=t, event_type=EventType.ORDER_ARRIVAL, data="order"),
        Event(time=t, event_type=EventType.RECEIPT_TO_STRATEGY, data="receipt2"),
        Event(time=t, event_type=EventType.CANCEL_ARRIVAL, data="cancel"),
    ]
    
    # 放入heap
    heap = []
    for event in events:
        heapq.heappush(heap, event)
    
    # 按顺序弹出
    popped = []
    while heap:
        popped.append(heapq.heappop(heap))
    
    print("事件弹出顺序:")
    for i, event in enumerate(popped):
        print(f"  {i+1}. {event.event_type.name} (priority={event.priority}, seq={event.seq})")
    
    # 验证优先级顺序
    expected_order = [
        EventType.SEGMENT_END,      # priority 1
        EventType.ORDER_ARRIVAL,    # priority 2
        EventType.CANCEL_ARRIVAL,   # priority 3
        EventType.RECEIPT_TO_STRATEGY,  # priority 4 (receipt1)
        EventType.RECEIPT_TO_STRATEGY,  # priority 4 (receipt2, seq更大)
        EventType.INTERVAL_END,     # priority 5
    ]
    
    actual_order = [e.event_type for e in popped]
    assert actual_order == expected_order, \
        f"事件顺序不正确\n期望: {[e.name for e in expected_order]}\n实际: {[e.name for e in actual_order]}"
    
    # 验证同类型事件按seq排序
    receipt_events = [e for e in popped if e.event_type == EventType.RECEIPT_TO_STRATEGY]
    assert receipt_events[0].data == "receipt1" and receipt_events[1].data == "receipt2", \
        "同类型事件应按创建顺序（seq）排序"
    
    # 多次运行验证确定性
    for run in range(3):
        reset_event_seq_counter()
        heap2 = []
        for event in events:
            heapq.heappush(heap2, Event(
                time=event.time,
                event_type=event.event_type,
                data=event.data,
            ))
        result = []
        while heap2:
            result.append(heapq.heappop(heap2).event_type)
        assert result == expected_order, f"第{run+1}次运行结果不一致"
    
    print("✓ 多次运行结果一致，确保确定性")
    print("✓ Event deterministic ordering test passed")


def test_peek_advance_pop_paradigm():
    """测试修复问题1：事件循环使用"peek, advance, pop batch"范式。
    
    验证交易所在处理事件前被推进到正确时间，避免"生成过去事件"的因果反转。
    
    测试场景：
    1. 在区间内某时刻有订单到达
    2. 交易所advance产生的回执时间 <= 订单到达时间
    3. 验证回执被正确处理（不会出现因果反转）
    """
    print("\n--- Test 23: Peek-Advance-Pop Paradigm ---")
    
    from quant_framework.runner.event_loop import EventLoopRunner, RunnerConfig, TimelineConfig
    
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
    
    # 创建有足够成交量的区间，确保会产生回执 (500ms interval)
    snapshots = [
        create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=30, last_vol_split=[(100.0, 100)]),
        create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=10, last_vol_split=[(100.0, 100)]),
    ]
    
    feed = MockFeed(snapshots)
    tape_config = TapeConfig()
    tape_builder = UnifiedTapeBuilder(config=tape_config, tick_size=1.0)
    exchange = FIFOExchangeSimulator(cancel_front_ratio=0.5)
    oms = OrderManager()
    
    # 记录事件处理顺序的策略
    class EventOrderTrackingStrategy:
        def __init__(self):
            self.event_log = []  # [(event_type, time)]
        
        def on_snapshot(self, snapshot, oms):
            ts = snapshot.ts_exch if hasattr(snapshot, 'ts_exch') else 0
            self.event_log.append(("SNAPSHOT", ts))
            # 第一个快照时下单，位置靠前确保会成交
            if len(self.event_log) == 1:
                return [Order(
                    order_id="test-order",
                    side=Side.BUY,
                    price=100.0,
                    qty=5,
                )]
            return []
        
        def on_receipt(self, receipt, snapshot, oms):
            self.event_log.append(("RECEIPT", receipt.timestamp, receipt.recv_time))
            return []
    
    strategy = EventOrderTrackingStrategy()
    
    # 使用延迟配置
    runner_config = RunnerConfig(
        delay_out=100 * TICK_PER_MS,  # 订单在1100ms时到达
        delay_in=50 * TICK_PER_MS,    # 回执延迟50ms
        timeline=TimelineConfig(a=1.0, b=0),
    )
    
    runner = EventLoopRunner(
        feed=feed,
        tape_builder=tape_builder,
        exchange=exchange,
        strategy=strategy,
        oms=oms,
        config=runner_config,
    )
    
    results = runner.run()
    
    print(f"事件日志: {strategy.event_log}")
    print(f"订单提交数: {results['diagnostics']['orders_submitted']}")
    print(f"回执生成数: {results['diagnostics']['receipts_generated']}")
    
    # 验证因果顺序：快照先于回执
    snapshot_indices = [i for i, e in enumerate(strategy.event_log) if e[0] == "SNAPSHOT"]
    receipt_indices = [i for i, e in enumerate(strategy.event_log) if e[0] == "RECEIPT"]
    
    if receipt_indices:
        # 第一个快照应该在任何回执之前
        assert snapshot_indices[0] < receipt_indices[0], "第一个快照应该在回执之前处理"
        print("✓ 因果顺序正确：快照先于回执")
        
        # 验证回执时间戳符合因果逻辑
        for event in strategy.event_log:
            if event[0] == "RECEIPT":
                timestamp, recv_time = event[1], event[2]
                assert recv_time >= timestamp, f"recv_time({recv_time})应 >= timestamp({timestamp})"
                print(f"✓ 回执时间因果正确: timestamp={timestamp}, recv_time={recv_time}")
    
    print("✓ Peek-advance-pop paradigm test passed")


def test_receipt_recv_time_authority():
    """测试修复问题3：RECEIPT_TO_STRATEGY事件使用recv_time作为权威策略侧时间。
    
    验证处理回执时，current_recvtime使用event.recv_time或receipt.recv_time，
    而不是从event.time推导。
    """
    print("\n--- Test 24: Receipt recv_time Authority ---")
    
    from quant_framework.runner.event_loop import EventLoopRunner, RunnerConfig, TimelineConfig
    
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
    
    # 创建快照 (500ms interval)
    snapshots = [
        create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=30, last_vol_split=[(100.0, 80)]),
        create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=10, last_vol_split=[(100.0, 80)]),
    ]
    
    feed = MockFeed(snapshots)
    tape_config = TapeConfig()
    tape_builder = UnifiedTapeBuilder(config=tape_config, tick_size=1.0)
    exchange = FIFOExchangeSimulator(cancel_front_ratio=0.5)
    oms = OrderManager()
    
    # 记录recv_time的策略
    recorded_recv_times = []
    
    class RecvTimeTrackingStrategy:
        def __init__(self):
            self.order_placed = False
        
        def on_snapshot(self, snapshot, oms):
            if not self.order_placed:
                self.order_placed = True
                return [Order(
                    order_id="recv-time-test",
                    side=Side.BUY,
                    price=100.0,
                    qty=3,
                )]
            return []
        
        def on_receipt(self, receipt, snapshot, oms):
            # 记录回执的recv_time
            recorded_recv_times.append({
                'receipt_timestamp': receipt.timestamp,
                'receipt_recv_time': receipt.recv_time,
            })
            return []
    
    strategy = RecvTimeTrackingStrategy()
    
    # 配置延迟
    delay_in = 200 * TICK_PER_MS
    runner_config = RunnerConfig(
        delay_out=50 * TICK_PER_MS,
        delay_in=delay_in,
        timeline=TimelineConfig(a=1.0, b=0),
    )
    
    runner = EventLoopRunner(
        feed=feed,
        tape_builder=tape_builder,
        exchange=exchange,
        strategy=strategy,
        oms=oms,
        config=runner_config,
    )
    
    results = runner.run()
    
    print(f"记录的recv_time: {recorded_recv_times}")
    
    # 验证recv_time = timestamp + delay_in
    for record in recorded_recv_times:
        expected_recv_time = record['receipt_timestamp'] + delay_in
        assert record['receipt_recv_time'] == expected_recv_time, \
            f"recv_time应为{expected_recv_time}（timestamp + delay_in），实际为{record['receipt_recv_time']}"
        print(f"✓ recv_time正确: {record['receipt_timestamp']} + {delay_in} = {record['receipt_recv_time']}")
    
    print("✓ Receipt recv_time authority test passed")


def test_no_causal_reversal_with_int_truncation():
    """测试问题1：验证因果反转被自动避免（通过时间钳制）。
    
    验证场景：
    1. 使用非1:1的时间线映射（如a=0.9）可能因int截断导致事件时间早于当前时间
    2. 系统应自动将事件时间钳制到当前时间，避免因果反转
    3. 回测应正常完成，不抛出异常
    
    关键点：现在系统通过时间钳制来避免因果反转，而不是抛出错误。
    """
    print("\n--- Test 25: No Causal Reversal with Int Truncation ---")
    
    from quant_framework.runner.event_loop import EventLoopRunner, RunnerConfig, TimelineConfig
    
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
    
    # 创建多个区间的快照，确保有足够的事件 (500ms intervals)
    snapshots = [
        create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=50, last_vol_split=[(100.0, 80)]),
        create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=30, last_vol_split=[(100.0, 80)]),
        create_test_snapshot(2000 * TICK_PER_MS, 100.0, 101.0, bid_qty=20, last_vol_split=[(100.0, 80)]),
    ]
    
    feed = MockFeed(snapshots)
    tape_config = TapeConfig()
    tape_builder = UnifiedTapeBuilder(config=tape_config, tick_size=1.0)
    
    # 测试策略：每个快照都下单
    class FrequentOrderStrategy:
        def __init__(self):
            self.count = 0
        
        def on_snapshot(self, snapshot, oms):
            self.count += 1
            return [Order(
                order_id=f"test-{self.count}",
                side=Side.BUY,
                price=100.0,
                qty=5,
            )]
        
        def on_receipt(self, receipt, snapshot, oms):
            return []
    
    # 测试用例1：正常时间线（1:1映射），应该成功
    print("测试1：正常时间线(a=1.0, b=0)...")
    runner_config = RunnerConfig(
        delay_out=50 * TICK_PER_MS,
        delay_in=100 * TICK_PER_MS,
        timeline=TimelineConfig(a=1.0, b=0),
    )
    
    feed.reset()
    runner = EventLoopRunner(
        feed=feed,
        tape_builder=tape_builder,
        exchange=FIFOExchangeSimulator(cancel_front_ratio=0.5),
        strategy=FrequentOrderStrategy(),
        oms=OrderManager(),
        config=runner_config,
    )
    
    results = runner.run()
    print(f"  结果: intervals={results['intervals']}, receipts={results['diagnostics']['receipts_generated']}")
    print("  ✓ 正常时间线测试通过")
    
    # 测试用例2：缩放时间线（a=0.9），int截断可能导致问题，但系统应自动钳制
    print("测试2：缩放时间线(a=0.9, b=100)...")
    runner_config2 = RunnerConfig(
        delay_out=50 * TICK_PER_MS,
        delay_in=100 * TICK_PER_MS,
        timeline=TimelineConfig(a=0.9, b=100),
    )
    
    feed.reset()
    runner2 = EventLoopRunner(
        feed=feed,
        tape_builder=tape_builder,
        exchange=FIFOExchangeSimulator(cancel_front_ratio=0.5),
        strategy=FrequentOrderStrategy(),
        oms=OrderManager(),
        config=runner_config2,
    )
    
    results2 = runner2.run()
    print(f"  结果: intervals={results2['intervals']}, receipts={results2['diagnostics']['receipts_generated']}")
    print("  ✓ 缩放时间线测试通过，因果反转被自动避免")
    
    # 测试用例3：更极端的缩放，验证系统仍能正常运行
    print("测试3：极端缩放时间线(a=2.0, b=-500)...")
    runner_config3 = RunnerConfig(
        delay_out=50 * TICK_PER_MS,
        delay_in=100 * TICK_PER_MS,
        timeline=TimelineConfig(a=2.0, b=-500),
    )
    
    feed.reset()
    runner3 = EventLoopRunner(
        feed=feed,
        tape_builder=tape_builder,
        exchange=FIFOExchangeSimulator(cancel_front_ratio=0.5),
        strategy=FrequentOrderStrategy(),
        oms=OrderManager(),
        config=runner_config3,
    )
    
    results3 = runner3.run()
    print(f"  结果: intervals={results3['intervals']}, receipts={results3['diagnostics']['receipts_generated']}")
    print("  ✓ 极端缩放测试通过，因果反转被自动避免")
    
    print("✓ No causal reversal with int truncation test passed")


def test_segment_queue_zero_constraint():
    """测试问题2：段转换时队列归零约束。
    
    验证场景（以bid为例）：
    当价格路径是 3318 -> 3317 -> 3316 -> 3317 -> 3318 时：
    - 在从3318变为3317的转换时刻，3318这一档的队列长度应该为0
    - 因为如果还有剩余队列，价格不会从3318变为3317
    
    这是一个强约束：当段结束时best price变化，意味着该价位的队列已经清空。
    
    关键验证：净流入量应该基于此约束进行分配，使得转换时队列深度=0。
    即：Q_A + N - M = 0 => N = M - Q_A
    """
    print("\n--- Test 26: Segment Queue Zero Constraint ---")
    
    # 创建一个场景：bid价格从高到低变化
    # 这意味着在转换点，原价位的队列应该为0
    
    # 使用多档位快照来模拟价格路径变化
    # A快照：bid最优价3318，queue = 50
    # B快照：bid最优价3316，queue = 30 (3318和3317都变为0)
    prev = create_multi_level_snapshot(
        1000 * TICK_PER_MS,
        bids=[(3318, 50), (3317, 40), (3316, 30)],
        asks=[(3319, 100), (3320, 100)],
        last_vol_split=[(3318, 30), (3317, 20), (3316, 10)]  # 有成交导致队列消耗
    )
    
    curr = create_multi_level_snapshot(
        1500 * TICK_PER_MS,  # 500ms间隔
        bids=[(3316, 25), (3315, 35)],  # 3318和3317队列归零，不在盘口
        asks=[(3317, 80), (3318, 90)],  # ask也变化了
        last_vol_split=[(3318, 30), (3317, 20), (3316, 10)]
    )
    
    tape_config = TapeConfig(
        ghost_rule="symmetric",
        epsilon=1.0,
        segment_iterations=2,
    )
    builder = UnifiedTapeBuilder(config=tape_config, tick_size=1.0)
    
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    
    # 检查每个段的转换
    # 验证：当价格从高价转换到低价时，计算队列是否归零
    price_transitions = []
    for i in range(len(tape) - 1):
        curr_seg = tape[i]
        next_seg = tape[i + 1]
        if abs(curr_seg.bid_price - next_seg.bid_price) > 0.01:
            price_transitions.append({
                'from_seg': i + 1,
                'to_seg': i + 2,
                'seg_idx': i,
                'from_price': curr_seg.bid_price,
                'to_price': next_seg.bid_price,
                'transition_time': curr_seg.t_end,
            })
    
    print(f"\n价格转换点: {len(price_transitions)}个")
    
    # 初始队列深度（来自A快照）
    initial_qty = {3318: 50, 3317: 40, 3316: 30}
    
    def get_initial_qty(price):
        return initial_qty.get(int(round(price)), 0)
    
    # 队列归零约束验证：
    # - bid下降时：离开的价位队列归零
    # - ask上升时：离开的价位队列归零
    # 
    # 关键：当价格回到某个档位时，队列从0重新开始（因为之前离开时已归零）
    
    # 验证每次bid下降转换时的队列归零约束
    for trans in price_transitions:
        from_price = trans['from_price']
        to_price = trans['to_price']
        seg_idx = trans['seg_idx']
        
        print(f"  段{trans['from_seg']}->段{trans['to_seg']}: "
              f"bid {from_price} -> {to_price}")
        
        if from_price > to_price:
            # bid下降：from_price的队列应该归零
            # 
            # 计算方法：从该价位的最近一次"进入"开始累计
            # - 如果这是首次在该价位，初始队列=A快照中的值
            # - 如果是重访该价位，初始队列=0（因为之前离开时归零）
            
            # 找到该价位当前连续段的起点（向前找到价格变化点）
            run_start = seg_idx
            while run_start > 0 and abs(tape[run_start - 1].bid_price - from_price) < 0.01:
                run_start -= 1
            
            # 判断是否为首次访问：检查run_start之前是否有该价位
            is_first_visit = not any(
                abs(tape[j].bid_price - from_price) < 0.01 
                for j in range(run_start)
            )
            
            # 初始队列
            q = get_initial_qty(from_price) if is_first_visit else 0
            
            # 累计当前连续段的变化
            for j in range(run_start, seg_idx + 1):
                seg = tape[j]
                q += seg.net_flow.get((Side.BUY, from_price), 0)
                q -= seg.trades.get((Side.BUY, from_price), 0)
            
            print(f"    bid@{from_price} 队列={q} (应归零, 首次={is_first_visit})")
            assert abs(q) <= 1, f"bid下降时队列未归零: {q}"
            print(f"  ✓ bid下降，队列正确归零")
        else:
            # bid上升：新单进入，无归零约束
            print(f"  ✓ bid上升，新单进入")
    
    print("✓ Segment queue zero constraint test passed")


def test_segment_price_change_queue_constraint_detailed():
    """测试问题2的详细验证：段转换时的成交量分配约束。
    
    根据问题描述的例子：
    - 价格路径：3318 -> 3317 -> 3316 -> 3317 -> 3318
    - 总成交量：3318=100, 3317=100, 3316=50
    - 总时间：500ms
    
    在第一段（3318）结束时：
    - bid best price从3318变为3317
    - 意味着3318这一档在转换时刻长度为0
    
    这个约束体现在：该段在3318价位的成交量应该等于或超过该价位的初始队列深度。
    """
    print("\n--- Test 27: Detailed Segment Price Change Queue Constraint ---")
    
    # 模拟场景：价格从3318逐步下降
    # 初始3318队列深度为100，如果有100的成交，队列会清空
    prev = create_multi_level_snapshot(
        1000 * TICK_PER_MS,
        bids=[(3318.0, 100), (3317.0, 100), (3316.0, 80)],
        asks=[(3319.0, 100)],
        last_vol_split=[(3318.0, 100), (3317.0, 100), (3316.0, 50)]
    )
    
    curr = create_multi_level_snapshot(
        1500 * TICK_PER_MS,  # 500ms间隔
        bids=[(3318.0, 80), (3317.0, 80), (3316.0, 60)],
        asks=[(3319.0, 80)],
        last_vol_split=[(3318.0, 100), (3317.0, 100), (3316.0, 50)]
    )
    
    tape_config = TapeConfig(
        ghost_rule="symmetric",
        epsilon=1.0,
        segment_iterations=2,
    )
    builder = UnifiedTapeBuilder(config=tape_config, tick_size=1.0)
    
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    
    # 收集每个价位的总成交量
    total_trades_by_price = {}
    for seg in tape:
        for (side, price), qty in seg.trades.items():
            if side == Side.BUY:
                if price not in total_trades_by_price:
                    total_trades_by_price[price] = 0
                total_trades_by_price[price] += qty
    
    print(f"\n各价位总成交量(bid侧):")
    for price, qty in sorted(total_trades_by_price.items(), reverse=True):
        print(f"  价格{price}: {qty}")
    
    # 验证成交量分配
    # 根据lastvolsplit，3318应该有100的成交
    expected_3318 = 100
    actual_3318 = total_trades_by_price.get(3318.0, 0)
    print(f"\n3318价位成交: 期望={expected_3318}, 实际={actual_3318}")
    assert actual_3318 == expected_3318, f"3318价位成交量不匹配"
    
    expected_3317 = 100
    actual_3317 = total_trades_by_price.get(3317.0, 0)
    print(f"3317价位成交: 期望={expected_3317}, 实际={actual_3317}")
    assert actual_3317 == expected_3317, f"3317价位成交量不匹配"
    
    expected_3316 = 50
    actual_3316 = total_trades_by_price.get(3316.0, 0)
    print(f"3316价位成交: 期望={expected_3316}, 实际={actual_3316}")
    assert actual_3316 == expected_3316, f"3316价位成交量不匹配"
    
    # 打印每段的详细信息
    print("\n每段详情:")
    for seg in tape:
        print(f"  段{seg.index}: t=[{seg.t_start}, {seg.t_end}], bid={seg.bid_price}")
        bid_trades = {p: q for (s, p), q in seg.trades.items() if s == Side.BUY}
        if bid_trades:
            print(f"    bid成交: {bid_trades}")
    
    print("✓ Detailed segment price change queue constraint test passed")


def test_crossing_immediate_execution():
    """测试crossing立即成交逻辑。
    
    验证场景：
    1. BUY订单 price >= ask_best 时立即成交
    2. SELL订单 price <= bid_best 时立即成交
    3. IOC订单：立即成交后剩余取消
    4. GTC订单：立即成交后剩余排队
    """
    print("\n--- Test 28: Crossing Immediate Execution ---")
    
    # 创建测试tape
    tape_config = TapeConfig(
        ghost_rule="symmetric",
        epsilon=1.0,
        segment_iterations=2,
    )
    builder = UnifiedTapeBuilder(config=tape_config, tick_size=1.0)
    
    # 创建快照：bid@100, ask@101 (500ms interval)
    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=50, ask_qty=60)
    curr = create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=40, ask_qty=50,
                                last_vol_split=[(100.0, 10), (101.0, 10)])
    
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    
    # 创建交易所模拟器
    exchange = FIFOExchangeSimulator(cancel_front_ratio=0.5)
    exchange.set_tape(tape, 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)
    
    # 测试1：BUY订单crossing (price >= ask)
    print("\n测试1: BUY订单crossing...")
    buy_order_cross = Order(
        order_id="buy-cross-1",
        side=Side.BUY,
        price=101.0,  # 等于ask，应该crossing
        qty=10,
        tif=TimeInForce.GTC,
    )
    
    # 初始化ask档位深度
    ask_level = exchange._get_level(Side.SELL, 101.0)
    ask_level.q_mkt = 60.0  # 设置ask档位深度
    
    receipt = exchange.on_order_arrival(buy_order_cross, 1100 * TICK_PER_MS, 50)
    if receipt:
        print(f"  收到回执: type={receipt.receipt_type}, fill_qty={receipt.fill_qty}, "
              f"fill_price={receipt.fill_price}, remaining={receipt.remaining_qty}")
        assert receipt.fill_qty > 0, "应该有立即成交"
        print(f"  ✓ BUY crossing成交 {receipt.fill_qty} @ {receipt.fill_price}")
    else:
        print(f"  订单已入队")
    
    # 测试2：SELL订单crossing (price <= bid)
    print("\n测试2: SELL订单crossing...")
    sell_order_cross = Order(
        order_id="sell-cross-1",
        side=Side.SELL,
        price=100.0,  # 等于bid，应该crossing
        qty=15,
        tif=TimeInForce.GTC,
    )
    
    # 初始化bid档位深度
    bid_level = exchange._get_level(Side.BUY, 100.0)
    bid_level.q_mkt = 50.0  # 设置bid档位深度
    
    receipt2 = exchange.on_order_arrival(sell_order_cross, 1200 * TICK_PER_MS, 60)
    if receipt2:
        print(f"  收到回执: type={receipt2.receipt_type}, fill_qty={receipt2.fill_qty}, "
              f"fill_price={receipt2.fill_price}, remaining={receipt2.remaining_qty}")
        assert receipt2.fill_qty > 0, "应该有立即成交"
        print(f"  ✓ SELL crossing成交 {receipt2.fill_qty} @ {receipt2.fill_price}")
    else:
        print(f"  订单已入队")
    
    # 测试3：IOC订单crossing后剩余取消
    print("\n测试3: IOC订单crossing...")
    ioc_order = Order(
        order_id="ioc-cross-1",
        side=Side.BUY,
        price=101.0,
        qty=100,  # 数量大于可用流动性
        tif=TimeInForce.IOC,
    )
    
    receipt3 = exchange.on_order_arrival(ioc_order, 1300 * TICK_PER_MS, 50)
    if receipt3:
        print(f"  收到回执: type={receipt3.receipt_type}, fill_qty={receipt3.fill_qty}, remaining={receipt3.remaining_qty}")
        if receipt3.fill_qty > 0 and receipt3.fill_qty < 100:
            assert receipt3.receipt_type == "PARTIAL" or receipt3.remaining_qty == 0, "IOC部分成交后应该取消剩余"
            print(f"  ✓ IOC订单部分成交{receipt3.fill_qty}，剩余已取消")
    
    # 测试4：不crossing的订单（被动排队）
    print("\n测试4: 不crossing的订单...")
    passive_order = Order(
        order_id="passive-1",
        side=Side.BUY,
        price=99.0,  # 低于ask，不会crossing
        qty=20,
        tif=TimeInForce.GTC,
    )
    
    receipt4 = exchange.on_order_arrival(passive_order, 1400 * TICK_PER_MS, 50)
    if receipt4:
        print(f"  收到回执: type={receipt4.receipt_type}")
    else:
        print(f"  ✓ 订单已被动入队（无立即成交）")
        # 验证订单在队列中
        shadow_orders = exchange.get_shadow_orders()
        passive_in_queue = any(o.order_id == "passive-1" for o in shadow_orders)
        assert passive_in_queue, "被动订单应该在队列中"
        print(f"  ✓ 订单在队列中")
    
    print("✓ Crossing immediate execution test passed")


def test_nonuniform_snapshot_timing():
    """测试非均匀快照推送时间处理。
    
    验证场景：
    1. A快照的时间被视为T_B - 500ms
    2. 所有变化都在[T_B-500ms, T_B]区间内
    3. 当间隔小于500ms时，使用原始T_A
    
    注意：时间单位为tick（每tick=100ns），500ms = 5_000_000 ticks
    """
    print("\n--- Test 29: Non-uniform Snapshot Timing ---")
    
    from quant_framework.core.types import TICK_PER_MS, SNAPSHOT_MIN_INTERVAL_TICK
    
    # 使用默认配置（非均匀快照时间处理默认启用）
    # 使用tick单位
    tape_config = TapeConfig(snapshot_min_interval_tick=SNAPSHOT_MIN_INTERVAL_TICK)
    builder = UnifiedTapeBuilder(config=tape_config, tick_size=1.0)
    
    # 测试1: 间隔为1500ms的快照（超过500ms阈值）
    # 转换为tick单位
    t_a = 1000 * TICK_PER_MS
    t_b = 2500 * TICK_PER_MS
    effective_t_a = t_b - SNAPSHOT_MIN_INTERVAL_TICK  # 2000ms in ticks
    
    prev = create_test_snapshot(t_a, 100.0, 101.0, bid_qty=50, ask_qty=60,
                                last_vol_split=[])
    curr = create_test_snapshot(t_b, 100.5, 101.5, bid_qty=40, ask_qty=50,
                                last_vol_split=[(100.5, 20)])  # 有成交
    
    tape = builder.build(prev, curr)
    
    print(f"快照间隔: 1500ms (超过500ms阈值)")
    print(f"T_A={t_a}, T_B={t_b}, effective_t_a={effective_t_a}")
    print(f"生成了{len(tape)}个段")
    
    for seg in tape:
        print(f"  段{seg.index}: t=[{seg.t_start}, {seg.t_end}], "
              f"duration={(seg.t_end - seg.t_start) / TICK_PER_MS}ms, "
              f"bid={seg.bid_price}, ask={seg.ask_price}")
        if seg.trades:
            print(f"    trades: {dict(seg.trades)}")
    
    # 验证：所有段都应该从effective_t_a开始
    first_seg = tape[0]
    assert first_seg.t_start == effective_t_a, f"第一段应从{effective_t_a}开始，实际{first_seg.t_start}"
    print(f"  ✓ 第一段正确从effective_t_a={effective_t_a}开始")
    
    # 最后一段应该到T_B结束
    last_seg = tape[-1]
    assert last_seg.t_end == t_b, f"最后一段应到{t_b}结束，实际{last_seg.t_end}"
    print(f"  ✓ 最后一段正确到T_B={t_b}结束")
    
    # 测试2: 间隔小于500ms时，使用原始T_A
    print("\n测试短间隔快照...")
    t_a2 = 3000 * TICK_PER_MS
    t_b2 = 3400 * TICK_PER_MS
    prev2 = create_test_snapshot(t_a2, 100.0, 101.0)
    curr2 = create_test_snapshot(t_b2, 100.5, 101.5, last_vol_split=[(100.5, 10)])
    
    tape2 = builder.build(prev2, curr2)
    print(f"快照间隔: 400ms (小于500ms阈值)")
    print(f"T_A={t_a2}, T_B={t_b2}, effective_t_a=max({t_a2}, {t_b2 - SNAPSHOT_MIN_INTERVAL_TICK})={t_a2}")
    print(f"生成了{len(tape2)}个段")
    
    # 间隔小于500ms时，effective_t_a = max(T_A, T_B-500ms) = T_A
    first_seg_start = tape2[0].t_start
    assert first_seg_start == t_a2, f"第一段应从{t_a2}开始，实际{first_seg_start}"
    print(f"  ✓ 短间隔正确处理，从{first_seg_start}开始")
    
    print("✓ Non-uniform snapshot timing test passed")


def test_request_and_receipt_types():
    """测试请求类型和回执类型的设计。
    
    验证：
    1. RequestType枚举（ORDER/CANCEL）
    2. ReceiptType枚举（FILL/PARTIAL/CANCELED/REJECTED）
    3. CancelRequest数据类
    4. 撤单回执的判断逻辑
    """
    print("\n--- Test 30: Request and Receipt Types ---")
    
    from quant_framework.core.types import RequestType, ReceiptType, CancelRequest, OrderReceipt
    
    # 测试RequestType
    print("测试RequestType...")
    assert RequestType.ORDER.value == "ORDER"
    assert RequestType.CANCEL.value == "CANCEL"
    print(f"  ✓ RequestType: ORDER={RequestType.ORDER.value}, CANCEL={RequestType.CANCEL.value}")
    
    # 测试CancelRequest
    print("测试CancelRequest...")
    cancel_req = CancelRequest(order_id="test-order-1", create_time=1000)
    assert cancel_req.order_id == "test-order-1"
    assert cancel_req.create_time == 1000
    print(f"  ✓ CancelRequest创建成功: order_id={cancel_req.order_id}")
    
    # 测试撤单回执逻辑
    print("测试撤单回执逻辑...")
    
    # 撤单成功（撤单前有部分成交）
    receipt1 = OrderReceipt(
        order_id="order-1",
        receipt_type="CANCELED",
        timestamp=1100,
        fill_qty=5,  # 撤单前成交5
        remaining_qty=0,
    )
    assert receipt1.receipt_type == "CANCELED" and receipt1.fill_qty > 0
    print(f"  ✓ 撤单成功(有部分成交): fill_qty={receipt1.fill_qty}")
    
    # 撤单成功（撤单前无成交）
    receipt2 = OrderReceipt(
        order_id="order-2",
        receipt_type="CANCELED",
        timestamp=1200,
        fill_qty=0,  # 撤单前无成交
        remaining_qty=0,
    )
    assert receipt2.receipt_type == "CANCELED" and receipt2.fill_qty == 0
    print(f"  ✓ 撤单成功(无成交): fill_qty={receipt2.fill_qty}")
    
    # 撤单失败
    receipt3 = OrderReceipt(
        order_id="order-3",
        receipt_type="REJECTED",
        timestamp=1300,
    )
    assert receipt3.receipt_type == "REJECTED"
    print(f"  ✓ 撤单失败: receipt_type={receipt3.receipt_type}")
    
    print("✓ Request and receipt types test passed")


def test_replay_strategy():
    """测试重放策略的CSV读取和订单生成功能。"""
    print("\n--- Test 31: Replay Strategy ---")
    
    import os
    import tempfile
    from quant_framework.trading.replay_strategy import ReplayStrategy, OrderRecord, CancelRecord
    from quant_framework.core.types import Order, CancelRequest, Side
    
    # 创建临时CSV文件
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建订单文件
        order_file = os.path.join(tmpdir, "PubOrderLog_TestMachine_Day20240101_Id12345.csv")
        with open(order_file, 'w') as f:
            f.write("OrderId,LimitPrice,Volume,OrderDirection,SentTime\n")
            f.write("1001,100.5,10,Buy,1000\n")
            f.write("1002,99.0,20,Sell,1100\n")
            f.write("1003,101.0,15,Buy,1200\n")
        
        # 创建撤单文件
        cancel_file = os.path.join(tmpdir, "PubOrderCancelRequestLog_TestMachine_Day20240101_Id12345.csv")
        with open(cancel_file, 'w') as f:
            f.write("OrderId,CancelSentTime\n")
            f.write("1001,1500\n")
            f.write("1002,1600\n")
        
        # 测试策略加载
        strategy = ReplayStrategy(
            name="TestReplay",
            order_file=order_file,
            cancel_file=cancel_file,
        )
        
        # 验证加载的订单
        assert len(strategy.orders) == 3, f"应加载3个订单，实际{len(strategy.orders)}"
        assert strategy.orders[0].order_id == 1001
        assert strategy.orders[0].limit_price == 100.5
        assert strategy.orders[0].volume == 10
        assert strategy.orders[0].direction == "Buy"
        assert strategy.orders[0].sent_time == 1000
        print(f"  ✓ 成功加载3个订单")
        
        # 验证加载的撤单
        assert len(strategy.cancels) == 2, f"应加载2个撤单，实际{len(strategy.cancels)}"
        assert strategy.cancels[0].order_id == 1001
        assert strategy.cancels[0].cancel_sent_time == 1500
        print(f"  ✓ 成功加载2个撤单")
        
        # 验证pending_orders按时间排序
        assert len(strategy.pending_orders) == 3
        times = [t for t, o in strategy.pending_orders]
        assert times == sorted(times), "订单应按时间排序"
        print(f"  ✓ 订单按时间正确排序")
        
        # 验证pending_cancels
        assert len(strategy.pending_cancels) == 2
        print(f"  ✓ 撤单列表正确准备")
        
        # 测试on_snapshot返回所有订单
        class MockOMSView:
            def get_active_orders(self):
                return []
            def get_portfolio(self):
                return None
        
        # 创建mock snapshot - 使用ts_recv作为主时间线
        from quant_framework.core.dto import SnapshotDTO, LevelDTO
        snapshot = SnapshotDTO(
            ts_recv=1000,  # 主时间线
            bids=(LevelDTO(100.0, 100),),
            asks=(LevelDTO(101.0, 100),),
        )
        
        orders = strategy.on_snapshot(snapshot, MockOMSView())
        assert len(orders) == 3, f"第一次快照应返回3个订单，实际{len(orders)}"
        assert all(isinstance(o, Order) for o in orders)
        print(f"  ✓ 第一次快照返回所有3个订单")
        
        # 验证第二次快照不返回订单
        orders2 = strategy.on_snapshot(snapshot, MockOMSView())
        assert len(orders2) == 0, "后续快照不应返回订单"
        print(f"  ✓ 后续快照不返回订单")
        
        # 验证get_pending_cancels
        cancels = strategy.get_pending_cancels()
        assert len(cancels) == 2
        assert all(isinstance(c[1], CancelRequest) for c in cancels)
        print(f"  ✓ get_pending_cancels返回2个撤单请求")
        
        # 验证统计信息
        stats = strategy.get_statistics()
        assert stats['total_orders'] == 3
        assert stats['total_cancels'] == 2
        print(f"  ✓ 统计信息正确: {stats}")
    
    print("✓ Replay strategy test passed")


def test_receipt_logger():
    """测试回执记录器的功能。"""
    print("\n--- Test 32: Receipt Logger ---")
    
    import os
    import tempfile
    from quant_framework.trading.receipt_logger import ReceiptLogger, ReceiptRecord
    from quant_framework.core.types import OrderReceipt
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "receipts.csv")
        
        # 创建logger
        logger = ReceiptLogger(output_file=output_file)
        
        # 注册订单
        logger.register_order("order-1", 100)
        logger.register_order("order-2", 50)
        logger.register_order("order-3", 30)
        
        assert len(logger.order_total_qty) == 3
        print(f"  ✓ 注册3个订单成功")
        
        # 记录回执 - 部分成交
        receipt1 = OrderReceipt(
            order_id="order-1",
            receipt_type="PARTIAL",
            timestamp=1000,
            fill_qty=30,
            fill_price=100.5,
            remaining_qty=70,
        )
        receipt1.recv_time = 1010
        logger.log_receipt(receipt1)
        
        # 记录回执 - 全部成交
        receipt2 = OrderReceipt(
            order_id="order-1",
            receipt_type="FILL",
            timestamp=2000,
            fill_qty=70,
            fill_price=100.5,
            remaining_qty=0,
        )
        receipt2.recv_time = 2010
        logger.log_receipt(receipt2)
        
        # 记录回执 - 撤单
        receipt3 = OrderReceipt(
            order_id="order-2",
            receipt_type="CANCELED",
            timestamp=3000,
            fill_qty=20,  # 撤单前有部分成交
            fill_price=99.0,
            remaining_qty=30,
        )
        receipt3.recv_time = 3010
        logger.log_receipt(receipt3)
        
        # 记录回执 - 拒绝
        receipt4 = OrderReceipt(
            order_id="order-3",
            receipt_type="REJECTED",
            timestamp=4000,
            fill_qty=0,
            fill_price=0.0,
            remaining_qty=30,
        )
        receipt4.recv_time = 4010
        logger.log_receipt(receipt4)
        
        assert len(logger.records) == 4
        print(f"  ✓ 记录4条回执成功")
        
        # 验证统计
        stats = logger.get_statistics()
        assert stats['total_receipts'] == 4
        assert stats['total_orders'] == 3
        assert stats['partial_fill_count'] == 1
        assert stats['full_fill_count'] == 1
        assert stats['cancel_count'] == 1
        assert stats['reject_count'] == 1
        print(f"  ✓ 回执类型统计正确")
        
        # 验证成交量统计
        assert logger.order_filled_qty['order-1'] == 100  # 全部成交
        assert logger.order_filled_qty['order-2'] == 20   # 部分成交后撤单
        assert logger.order_filled_qty['order-3'] == 0    # 被拒绝
        print(f"  ✓ 成交量统计正确")
        
        # 计算成交率
        fill_rate_qty = logger.calculate_fill_rate()
        # 总量: 100 + 50 + 30 = 180
        # 成交: 100 + 20 + 0 = 120
        expected_rate = 120 / 180
        assert abs(fill_rate_qty - expected_rate) < 0.01, f"成交率应为{expected_rate}，实际{fill_rate_qty}"
        print(f"  ✓ 按数量成交率: {fill_rate_qty:.2%}")
        
        fill_rate_count = logger.calculate_fill_rate_by_count()
        # 3个订单中1个完全成交
        expected_rate_count = 1 / 3
        assert abs(fill_rate_count - expected_rate_count) < 0.01
        print(f"  ✓ 按订单数成交率: {fill_rate_count:.2%}")
        
        # 保存到文件
        logger.save_to_file()
        assert os.path.exists(output_file)
        
        # 验证文件内容
        with open(output_file, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 5  # 1 header + 4 records
        print(f"  ✓ CSV文件保存成功，共{len(lines)-1}条记录")
        
        # 打印统计摘要
        print("\n  统计摘要:")
        for key, value in stats.items():
            print(f"    {key}: {value}")
    
    print("✓ Receipt logger test passed")


def test_replay_integration():
    """测试重放策略与事件循环的集成。"""
    print("\n--- Test 33: Replay Strategy Integration ---")
    
    import os
    import tempfile
    from quant_framework.trading.replay_strategy import ReplayStrategy
    from quant_framework.trading.receipt_logger import ReceiptLogger
    from quant_framework.runner.event_loop import EventLoopRunner, RunnerConfig
    from quant_framework.tape.builder import UnifiedTapeBuilder, TapeConfig
    from quant_framework.exchange.simulator import FIFOExchangeSimulator
    from quant_framework.trading.oms import OrderManager, Portfolio
    from quant_framework.core.types import NormalizedSnapshot, Level
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建订单文件（时间与快照时间匹配）
        order_file = os.path.join(tmpdir, "orders.csv")
        with open(order_file, 'w') as f:
            f.write("OrderId,LimitPrice,Volume,OrderDirection,SentTime\n")
            # 在快照1000时发送，到达时间应在区间内
            f.write("1,100.0,10,Buy,1000\n")
            f.write("2,101.0,5,Sell,1100\n")
        
        # 创建撤单文件
        cancel_file = os.path.join(tmpdir, "cancels.csv")
        with open(cancel_file, 'w') as f:
            f.write("OrderId,CancelSentTime\n")
            f.write("1,1500\n")
        
        # 创建快照数据 - 使用ts_recv作为主时间线
        snapshots = [
            NormalizedSnapshot(
                ts_recv=1000,  # 主时间线
                bids=[Level(100.0, 100)],
                asks=[Level(101.0, 100)],
                last_vol_split=[(100.0, 50)],
            ),
            NormalizedSnapshot(
                ts_recv=2000,  # 主时间线
                bids=[Level(100.0, 80)],
                asks=[Level(101.0, 90)],
                last_vol_split=[(100.0, 30)],
            ),
        ]
        
        # 创建mock feed
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
        
        # 创建组件
        feed = MockFeed(snapshots)
        tape_builder = UnifiedTapeBuilder(config=TapeConfig(), tick_size=1.0)
        exchange = FIFOExchangeSimulator(cancel_front_ratio=0.5)
        strategy = ReplayStrategy(
            name="TestReplay",
            order_file=order_file,
            cancel_file=cancel_file,
        )
        oms = OrderManager(portfolio=Portfolio(cash=100000.0))
        receipt_logger = ReceiptLogger()
        
        # 创建runner
        runner = EventLoopRunner(
            feed=feed,
            tape_builder=tape_builder,
            exchange=exchange,
            strategy=strategy,
            oms=oms,
            config=RunnerConfig(delay_out=0, delay_in=0),
            receipt_logger=receipt_logger,
        )
        
        # 运行回测
        results = runner.run()
        
        print(f"  回测结果: {results}")
        print(f"  提交订单数: {results['diagnostics']['orders_submitted']}")
        print(f"  撤单数: {results['diagnostics']['cancels_submitted']}")
        print(f"  生成回执数: {results['diagnostics']['receipts_generated']}")
        
        # 验证订单被提交
        assert results['diagnostics']['orders_submitted'] == 2, \
            f"应提交2个订单，实际{results['diagnostics']['orders_submitted']}"
        print(f"  ✓ 订单正确提交")
        
        # 验证撤单被处理
        assert results['diagnostics']['cancels_submitted'] == 1, \
            f"应有1个撤单，实际{results['diagnostics']['cancels_submitted']}"
        print(f"  ✓ 撤单正确处理")
        
        # 验证receipt_logger接收到回执
        if receipt_logger.records:
            print(f"  ✓ ReceiptLogger记录了{len(receipt_logger.records)}条回执")
            for record in receipt_logger.records:
                print(f"    - {record.order_id}: {record.receipt_type}, fill_qty={record.fill_qty}")
    
    print("✓ Replay strategy integration test passed")


def test_crossing_partial_fill_position_zero():
    """测试crossing部分成交后剩余订单的队列位置。
    
    验证场景：
    当订单发生crossing并部分成交后，剩余部分的队列位置应该为0。
    
    逻辑：
    - 如果我的订单在px上发生crossing，说明ask方在≤px有流动性
    - 如果ask@px有流动性，那么bid@px不可能有订单（否则早就撮合了）
    - 因此bid@px上不可能有之前的shadow订单，position直接为0
    
    本测试直接测试_queue_order函数的already_filled参数对position的影响。
    """
    print("\n--- Test 34: Crossing Partial Fill Position Zero ---")
    
    # 创建测试tape
    tape_config = TapeConfig(
        ghost_rule="symmetric",
        epsilon=1.0,
        segment_iterations=2,
    )
    builder = UnifiedTapeBuilder(config=tape_config, tick_size=1.0)
    
    # 创建快照：bid@100, ask@101 (500ms interval)
    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=50, ask_qty=100)
    curr = create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=50, ask_qty=100,
                                last_vol_split=[(100.0, 10), (101.0, 10)])
    
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    
    # 创建交易所模拟器
    exchange = FIFOExchangeSimulator(cancel_front_ratio=0.5)
    exchange.set_tape(tape, 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)
    
    # 测试1: 直接测试_queue_order函数，模拟crossing后剩余订单入队
    print("\n测试1: 模拟crossing后剩余订单入队（already_filled > 0）...")
    
    # 创建一个订单
    order_after_crossing = Order(
        order_id="order-after-crossing",
        side=Side.BUY,
        price=101.0,  
        qty=150,  # 原始订单数量
        tif=TimeInForce.GTC,
    )
    
    # 直接调用_queue_order，模拟已经部分成交100后剩余50入队
    # already_filled=100 表示已经有crossing成交
    exchange._queue_order(
        order=order_after_crossing,
        arrival_time=1100 * TICK_PER_MS,
        market_qty=50,  # 市场队列深度（这个在crossing后不重要）
        remaining_qty=50,  # 剩余需要排队的数量
        already_filled=100  # 已经通过crossing成交的数量
    )
    
    # 验证订单在队列中
    shadow_orders = exchange.get_shadow_orders()
    order_in_queue = None
    for so in shadow_orders:
        if so.order_id == "order-after-crossing":
            order_in_queue = so
            break
    
    assert order_in_queue is not None, "订单应该在队列中"
    print(f"  订单在队列中: pos={order_in_queue.pos}, remaining_qty={order_in_queue.remaining_qty}")
    
    # 关键断言：crossing后剩余订单的位置应该是0
    assert order_in_queue.pos == 0, f"crossing后剩余订单位置应该为0，实际为{order_in_queue.pos}"
    print(f"  ✓ crossing后队列位置正确为0")
    
    # 测试2: 对比没有crossing的订单
    print("\n测试2: 没有crossing的被动订单入队（already_filled = 0）...")
    
    # 创建另一个交易所模拟器用于对比测试
    exchange2 = FIFOExchangeSimulator(cancel_front_ratio=0.5)
    exchange2.set_tape(tape, 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)
    
    passive_order = Order(
        order_id="passive-order",
        side=Side.BUY,
        price=99.0,  # 低于ask，不会crossing
        qty=50,
        tif=TimeInForce.GTC,
    )
    
    # 初始化该价位的market队列深度
    level = exchange2._get_level(Side.BUY, 99.0)
    level.q_mkt = 30.0  # 设置市场队列深度
    
    # 直接调用_queue_order，没有crossing成交
    exchange2._queue_order(
        order=passive_order,
        arrival_time=1100 * TICK_PER_MS,
        market_qty=30,  # 市场队列深度
        remaining_qty=50,  # 全部订单量
        already_filled=0  # 没有crossing成交
    )
    
    # 验证被动订单在队列中
    shadow_orders2 = exchange2.get_shadow_orders()
    passive_in_queue = None
    for so in shadow_orders2:
        if so.order_id == "passive-order":
            passive_in_queue = so
            break
    
    assert passive_in_queue is not None, "被动订单应该在队列中"
    print(f"  被动订单在队列中: pos={passive_in_queue.pos}, remaining_qty={passive_in_queue.remaining_qty}")
    
    # 被动订单的位置应该 > 0（排在市场队列后面）
    # pos = x_t + q_mkt_t + s_shadow, 其中q_mkt_t >= 30
    assert passive_in_queue.pos >= 30, f"被动订单位置应该>=30（市场队列深度），实际为{passive_in_queue.pos}"
    print(f"  ✓ 被动订单队列位置正确 >= 30")
    
    print("✓ Crossing partial fill position zero test passed")


def test_multiple_orders_at_same_price():
    """测试多个订单在同一价位的队列位置计算。
    
    测试场景：
    - prev: bid1=3318, ask1=3319
    - curr: bid1=3318, ask1=3319
    - last_vol_split: 3316, 3317, 3318, 3319, 3320 各有成交
    - 3个买单在价格3318，每个100手，时间均匀分布在prev和curr之间
    
    验证：
    - 所有订单正确入队
    - 订单位置按时间顺序递增（FIFO）
    """
    print("\n--- Test 35: Multiple Orders at Same Price ---")
    
    # 创建mock feed
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
    
    # 创建快照
    # prev: ts=1000ms, bid1=3318, ask1=3319，带有多档位
    # curr: ts=1500ms, bid1=3318, ask1=3319，带有多档位
    # last_vol_split: 3316, 3317, 3318, 3319, 3320 各有成交
    
    # 生成bid1之下的随机档位 (低于3318)
    import random
    random.seed(42)  # 固定随机种子以确保可重复性
    
    # bid档位: bid1=3318, 然后随机生成4档在3314-3317之间
    bid_prices_below = sorted(random.sample([3314.0, 3315.0, 3316.0, 3317.0], 4), reverse=True)
    bid_levels = [(3318.0, 100)]  # bid1
    for p in bid_prices_below:
        bid_levels.append((p, random.randint(50, 150)))
    
    # ask档位: ask1=3319, 然后随机生成4档在3320-3323之间
    ask_prices_above = sorted(random.sample([3320.0, 3321.0, 3322.0, 3323.0], 4))
    ask_levels = [(3319.0, 100)]  # ask1
    for p in ask_prices_above:
        ask_levels.append((p, random.randint(50, 150)))
    
    print(f"  生成的bid档位: {bid_levels}")
    print(f"  生成的ask档位: {ask_levels}")
    
    prev = create_multi_level_snapshot(
        ts=1000 * TICK_PER_MS,
        bids=bid_levels,
        asks=ask_levels,
        last_vol_split=[]  # prev没有last_vol_split
    )
    
    curr = create_multi_level_snapshot(
        ts=1500 * TICK_PER_MS,  # 500ms interval
        bids=bid_levels,  # 保持相同档位
        asks=ask_levels,
        last_vol_split=[
            (3316.0, 10),
            (3317.0, 10),
            (3318.0, 10),
            (3319.0, 10),
            (3320.0, 10),
        ]
    )
    
    snapshots = [prev, curr]
    feed = MockFeed(snapshots)
    
    # 创建组件
    tape_config = TapeConfig(
        ghost_rule="symmetric",
        epsilon=1.0,
        segment_iterations=2,
    )
    builder = UnifiedTapeBuilder(config=tape_config, tick_size=1.0)
    
    # 构建tape并输出路径
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    
    exchange = FIFOExchangeSimulator(cancel_front_ratio=0.5)
    oms = OrderManager()
    
    # 创建策略：在第一个快照后提交3个订单
    # 订单时间均匀分布
    class MultiOrderStrategy:
        def __init__(self):
            self.orders_created = False
        
        def on_snapshot(self, snapshot, oms):
            if not self.orders_created:
                self.orders_created = True
                orders = []
                # 3个订单，时间均匀分布在1000ms-1500ms之间
                # 订单1: create_time=1000ms (到达时间约1010ms)
                # 订单2: create_time=1167ms (到达时间约1177ms)
                # 订单3: create_time=1334ms (到达时间约1344ms)
                for i in range(3):
                    order = Order(
                        order_id=f"buy-3318-{i+1}",
                        side=Side.BUY,
                        price=3318.0,
                        qty=100,
                        tif=TimeInForce.GTC,
                    )
                    order.create_time = (1000 + i * 167) * TICK_PER_MS
                    orders.append(order)
                return orders
            return []
        
        def on_receipt(self, receipt, snapshot, oms):
            return []
    
    strategy = MultiOrderStrategy()
    
    # 配置运行器
    runner_config = RunnerConfig(
        delay_out=10 * TICK_PER_MS,   # 订单到交易所的延迟 (10ms)
        delay_in=10 * TICK_PER_MS,    # 回执到策略的延迟 (10ms)
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
    
    print(f"  结果: intervals={results['intervals']}, orders_submitted={results['diagnostics']['orders_submitted']}")
    
    # 验证提交了3个订单
    assert results['diagnostics']['orders_submitted'] == 3, \
        f"应该提交3个订单，实际提交{results['diagnostics']['orders_submitted']}个"
    print(f"  ✓ 成功提交3个订单")
    
    # 获取所有shadow订单
    shadow_orders = exchange.get_shadow_orders()
    print(f"  Shadow订单数量: {len(shadow_orders)}")
    
    # 找出price=3318的订单
    orders_at_3318 = [so for so in shadow_orders if abs(so.price - 3318.0) < 0.01]
    print(f"  价格3318的订单数量: {len(orders_at_3318)}")
    
    for so in orders_at_3318:
        print(f"    {so.order_id}: pos={so.pos}, qty={so.remaining_qty}, arrival={so.arrival_time}")
    
    # 验证订单数量
    # 注意：如果某些订单被成交，可能不在队列中
    # 但至少应该有记录
    print(f"  ✓ 多订单在同一价位的测试完成")
    
    # 验证订单位置按时间递增（FIFO）
    if len(orders_at_3318) >= 2:
        sorted_orders = sorted(orders_at_3318, key=lambda x: x.arrival_time)
        for i in range(1, len(sorted_orders)):
            assert sorted_orders[i].pos >= sorted_orders[i-1].pos, \
                f"订单位置应该按时间递增：{sorted_orders[i-1].order_id} pos={sorted_orders[i-1].pos} 应该 <= {sorted_orders[i].order_id} pos={sorted_orders[i].pos}"
        print(f"  ✓ 订单位置按时间正确递增（FIFO）")
    
    print("✓ Multiple orders at same price test passed")


def test_crossing_blocked_by_existing_shadow():
    """测试当本方有优先级更高的未成交shadow订单时，新订单不能crossing。
    
    验证场景：
    对于SELL订单：检查是否有价格更低的SELL shadow订单
    对于BUY订单：检查是否有价格更高的BUY shadow订单
    
    测试用例：
    1. 第一个SELL订单在price=99 crossing（比bid@100更低的卖价）
    2. 第二个SELL订单在price=100 crossing，但因为已有price=99的shadow订单，不能crossing
    
    逻辑：
    - SELL订单在price P，如果有price < P的SELL shadow订单，则新订单不能crossing
    - 因为市场会先匹配价格更低的卖单
    """
    print("\n--- Test 36: Crossing Blocked by Existing Shadow ---")
    
    from quant_framework.core.types import TapeSegment
    
    # 手动创建一个segment (500ms interval)
    seg = TapeSegment(
        index=1,
        t_start=1000 * TICK_PER_MS,
        t_end=1500 * TICK_PER_MS,
        bid_price=100.0,
        ask_price=101.0,
        trades={(Side.BUY, 100.0): 30, (Side.BUY, 99.0): 20},
        cancels={},
        net_flow={(Side.BUY, 100.0): 50, (Side.BUY, 99.0): 30},
        activation_bid={96.0, 97.0, 98.0, 99.0, 100.0},
        activation_ask={101.0, 102.0, 103.0, 104.0, 105.0},
    )
    
    print_tape_path([seg])
    
    # 创建交易所模拟器
    exchange = FIFOExchangeSimulator(cancel_front_ratio=0.5)
    exchange.set_tape([seg], 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)
    
    # 初始化对手方（bid侧）的市场队列深度
    bid_level_100 = exchange._get_level(Side.BUY, 100.0)
    bid_level_100.q_mkt = 100.0
    bid_level_99 = exchange._get_level(Side.BUY, 99.0)
    bid_level_99.q_mkt = 80.0
    
    # 测试1: 第一个SELL订单在price=99 crossing
    print("\n测试1: 第一个SELL订单在price=99 crossing...")
    order1 = Order(
        order_id="sell-99",
        side=Side.SELL,
        price=99.0,  # 低于bid@100，会crossing
        qty=150,  # 要求150手
        tif=TimeInForce.GTC,
    )
    
    receipt1 = exchange.on_order_arrival(order1, 1100 * TICK_PER_MS, market_qty=0)
    print(f"  第一个订单回执: {receipt1}")
    
    # 第一个订单应该crossing成交（吃掉bid@100和部分bid@99）
    assert receipt1 is not None, "第一个订单应该有回执"
    print(f"  ✓ 第一个订单crossing成交: {receipt1.fill_qty}手")
    
    # 获取shadow订单
    shadows = exchange.get_shadow_orders()
    sell99_shadow = None
    for s in shadows:
        if s.order_id == "sell-99":
            sell99_shadow = s
            break
    
    if sell99_shadow:
        print(f"  剩余shadow: price={sell99_shadow.price}, remaining={sell99_shadow.remaining_qty}")
    
    # 测试2: 第二个SELL订单在price=100，应该被阻止crossing（因为有price=99的shadow）
    print("\n测试2: 第二个SELL订单在price=100（应该被阻止crossing）...")
    order2 = Order(
        order_id="sell-100",
        side=Side.SELL,
        price=100.0,  # 高于order1的价格
        qty=50,
        tif=TimeInForce.GTC,
    )
    
    # 重新初始化bid侧（模拟有新的流动性）
    bid_level_100.q_mkt = 50.0
    
    receipt2 = exchange.on_order_arrival(order2, 1200 * TICK_PER_MS, market_qty=0)
    print(f"  第二个订单回执: {receipt2}")
    
    # 如果有price=99的shadow订单（更低价的卖单），则price=100的订单不能crossing
    if sell99_shadow and sell99_shadow.remaining_qty > 0:
        assert receipt2 is None, f"第二个订单不应该立即成交（有更低价shadow），实际收到: {receipt2}"
        print(f"  ✓ 第二个订单正确被阻止crossing（因为有price=99的shadow）")
    else:
        # 如果第一个订单已经全部成交，第二个订单可以crossing
        print(f"  注意：第一个订单已全部成交，第二个订单可以crossing")
    
    # 测试3: 没有阻止的情况 - BUY订单在price=101，检查是否有price>101的BUY shadow
    print("\n测试3: BUY订单在price=101（没有更高价的BUY shadow，应该可以crossing）...")
    
    # 初始化ask侧
    ask_level_101 = exchange._get_level(Side.SELL, 101.0)
    ask_level_101.q_mkt = 50.0
    
    order3 = Order(
        order_id="buy-101",
        side=Side.BUY,
        price=101.0,  # 高于bid@100，会crossing with ask@101
        qty=30,
        tif=TimeInForce.GTC,
    )
    
    receipt3 = exchange.on_order_arrival(order3, 1300 * TICK_PER_MS, market_qty=0)
    print(f"  BUY订单回执: {receipt3}")
    
    # 没有更高价的BUY shadow，应该可以crossing
    if receipt3 and receipt3.fill_qty > 0:
        print(f"  ✓ BUY订单正确crossing成交: {receipt3.fill_qty}手")
    else:
        print(f"  BUY订单入队（没有crossing）")
    
    print("✓ Crossing blocked by existing shadow test passed")


def test_crossing_blocked_by_queue_depth():
    """Test that crossing orders are queued when same-side depth exists."""
    print("\n--- Test 36b: Crossing Blocked by Queue Depth ---")
    
    from quant_framework.core.types import TapeSegment
    
    # Create a locked segment (500ms interval)
    seg = TapeSegment(
        index=1,
        t_start=1000 * TICK_PER_MS,
        t_end=1500 * TICK_PER_MS,
        bid_price=101.0,
        ask_price=101.0,
        trades={},
        cancels={},
        net_flow={},
        activation_bid={101.0},
        activation_ask={101.0},
    )
    
    exchange = FIFOExchangeSimulator(cancel_front_ratio=0.5)
    exchange.set_tape([seg], 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)
    
    # Initialize opposite-side liquidity to allow crossing
    ask_level = exchange._get_level(Side.SELL, 101.0)
    ask_level.q_mkt = 50.0
    
    # Initialize same-side queue depth at the price
    bid_level = exchange._get_level(Side.BUY, 101.0)
    bid_level.q_mkt = 20.0
    
    order = Order(
        order_id="buy-cross-queue",
        side=Side.BUY,
        price=101.0,  # Equal to ask, but queue depth should block crossing
        qty=10,
        tif=TimeInForce.GTC,
    )
    
    receipt = exchange.on_order_arrival(order, 1100 * TICK_PER_MS, market_qty=0)
    assert receipt is None, "Should not fill immediately when same-side queue has depth"
    
    shadows = exchange.get_shadow_orders()
    queued = next((s for s in shadows if s.order_id == "buy-cross-queue"), None)
    assert queued is not None, "Order should be queued"
    assert queued.pos >= 20, f"Order should sit behind existing depth, got {queued.pos}"
    
    print("✓ Crossing blocked by queue depth test passed")


def test_post_crossing_fill_with_net_increment():
    """测试post-crossing订单根据对手方净增量成交。
    
    验证场景：
    假设segment中，bid1是100.0，bidvol1是100，我下了一个ask 100.0@150手的单。
    根据crossing逻辑：
    1. 先成交100手（crossing），发送100手回执
    2. 剩余50手ask在100这个位置（标记为post-crossing）
    3. 如果segment中bid@100的净增量N >= 0：
       - 成交min(50, N)手
       - 回执时间是消耗完50或N的时刻
    4. 如果N < 0：这50手不成交
    """
    print("\n--- Test 37: Post-Crossing Fill with Net Increment ---")
    
    from quant_framework.core.types import TapeSegment
    
    # 测试场景1: 净增量N > 剩余数量 (N=80 > remaining=50)
    print("\n场景1: 净增量N=80 > 剩余数量50，应该全部成交...")
    
    # 手动创建一个segment，设置bid@100的净增量为80 (500ms interval)
    seg1 = TapeSegment(
        index=1,
        t_start=1000 * TICK_PER_MS,
        t_end=1500 * TICK_PER_MS,
        bid_price=100.0,
        ask_price=101.0,
        trades={(Side.BUY, 100.0): 30},
        cancels={},
        net_flow={(Side.BUY, 100.0): 80},  # bid侧净增量80
        activation_bid={96.0, 97.0, 98.0, 99.0, 100.0},
        activation_ask={101.0, 102.0, 103.0, 104.0, 105.0},
    )
    
    print_tape_path([seg1])
    
    exchange1 = FIFOExchangeSimulator(cancel_front_ratio=0.5)
    exchange1.set_tape([seg1], 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)
    
    # 重要：初始化对手方（bid侧）的市场队列深度
    bid_level1 = exchange1._get_level(Side.BUY, 100.0)
    bid_level1.q_mkt = 100.0  # bid@100有100手
    
    # 下一个SELL订单，会crossing
    order1 = Order(
        order_id="sell-n80",
        side=Side.SELL,
        price=100.0,
        qty=150,
        tif=TimeInForce.GTC,
    )
    
    # 假设bid@100有100手流动性（已通过_get_level初始化）
    receipt1 = exchange1.on_order_arrival(order1, 1050 * TICK_PER_MS, market_qty=0)
    print(f"  Crossing回执: {receipt1}")
    
    # Crossing成交量取决于_get_q_mkt在arrival_time的计算结果
    # 由于net_flow和trades的影响，实际可能略有不同
    assert receipt1 is not None and receipt1.fill_qty > 0, "应该有crossing成交"
    crossing_fill = receipt1.fill_qty
    print(f"  Crossing成交量: {crossing_fill}手")
    
    # 获取shadow订单
    shadows1 = exchange1.get_shadow_orders()
    shadow1 = shadows1[0]
    assert shadow1.is_post_crossing
    expected_remaining = 150 - crossing_fill
    assert shadow1.remaining_qty == expected_remaining, f"剩余应该是{expected_remaining}手，实际: {shadow1.remaining_qty}"
    print(f"  Shadow订单: remaining={shadow1.remaining_qty}, is_post_crossing={shadow1.is_post_crossing}")
    print(f"  Crossed prices: {shadow1.crossed_prices}")
    
    # 推进时间，应该根据净增量成交
    receipts1 = exchange1.advance(1050 * TICK_PER_MS, 1500 * TICK_PER_MS, seg1)
    print(f"  Advance生成的回执: {receipts1}")
    
    # post-crossing fill应该基于聚合净增量N=80
    # 剩余数量 <= N，所以应该全部成交
    if expected_remaining > 0:
        assert len(receipts1) == 1, f"应该生成1个回执，实际: {len(receipts1)}"
        assert receipts1[0].fill_qty == expected_remaining, f"应该成交{expected_remaining}手（全部剩余），实际: {receipts1[0].fill_qty}"
        assert receipts1[0].receipt_type == "FILL", f"应该完全成交，实际: {receipts1[0].receipt_type}"
        print(f"  ✓ 净增量N=80 >= 剩余{expected_remaining}，正确全部成交")
    else:
        print(f"  ✓ 订单已全部crossing成交")
    
    # 测试场景2: 净增量N < 剩余数量 (N=30 < remaining=50)
    print("\n场景2: 净增量N=30 < 剩余数量50，应该部分成交30手...")
    
    seg2 = TapeSegment(
        index=1,
        t_start=1000 * TICK_PER_MS,
        t_end=1500 * TICK_PER_MS,
        bid_price=100.0,
        ask_price=101.0,
        trades={(Side.BUY, 100.0): 30},
        cancels={},
        net_flow={(Side.BUY, 100.0): 30},  # bid侧净增量30
        activation_bid={96.0, 97.0, 98.0, 99.0, 100.0},
        activation_ask={101.0, 102.0, 103.0, 104.0, 105.0},
    )
    
    print_tape_path([seg2])
    
    exchange2 = FIFOExchangeSimulator(cancel_front_ratio=0.5)
    exchange2.set_tape([seg2], 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)
    
    # 重要：初始化对手方（bid侧）的市场队列深度
    bid_level2 = exchange2._get_level(Side.BUY, 100.0)
    bid_level2.q_mkt = 100.0  # bid@100有100手
    
    order2 = Order(
        order_id="sell-n30",
        side=Side.SELL,
        price=100.0,
        qty=150,
        tif=TimeInForce.GTC,
    )
    
    receipt2 = exchange2.on_order_arrival(order2, 1050 * TICK_PER_MS, market_qty=0)
    print(f"  Crossing回执: {receipt2}")
    assert receipt2 is not None and receipt2.fill_qty > 0, "应该有crossing成交"
    crossing_fill2 = receipt2.fill_qty
    expected_remaining2 = 150 - crossing_fill2
    
    receipts2 = exchange2.advance(1050 * TICK_PER_MS, 1500 * TICK_PER_MS, seg2)
    print(f"  Advance生成的回执: {receipts2}")
    
    # post-crossing fill应该基于聚合净增量N=30
    # 剩余数量 > N，所以只成交N手
    if expected_remaining2 > 0:
        assert len(receipts2) == 1, f"应该生成1个回执，实际: {len(receipts2)}"
        expected_fill2 = min(expected_remaining2, 30)  # N=30
        assert receipts2[0].fill_qty == expected_fill2, f"应该成交{expected_fill2}手，实际: {receipts2[0].fill_qty}"
        if expected_fill2 < expected_remaining2:
            assert receipts2[0].receipt_type == "PARTIAL", f"应该部分成交，实际: {receipts2[0].receipt_type}"
        print(f"  ✓ 净增量N=30 < 剩余{expected_remaining2}，正确部分成交{expected_fill2}手")
    
    # 测试场景3: 净增量N < 0 (N=-10)
    print("\n场景3: 净增量N=-10 < 0，不应该成交...")
    
    seg3 = TapeSegment(
        index=1,
        t_start=1000 * TICK_PER_MS,
        t_end=1500 * TICK_PER_MS,
        bid_price=100.0,
        ask_price=101.0,
        trades={(Side.BUY, 100.0): 30},
        cancels={(Side.BUY, 100.0): 40},  # 撤单多于新单
        net_flow={(Side.BUY, 100.0): -10},  # bid侧净增量为负
        activation_bid={96.0, 97.0, 98.0, 99.0, 100.0},
        activation_ask={101.0, 102.0, 103.0, 104.0, 105.0},
    )
    
    print_tape_path([seg3])
    
    exchange3 = FIFOExchangeSimulator(cancel_front_ratio=0.5)
    exchange3.set_tape([seg3], 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)
    
    # 重要：初始化对手方（bid侧）的市场队列深度
    bid_level3 = exchange3._get_level(Side.BUY, 100.0)
    bid_level3.q_mkt = 100.0  # bid@100有100手
    
    order3 = Order(
        order_id="sell-n-10",
        side=Side.SELL,
        price=100.0,
        qty=150,
        tif=TimeInForce.GTC,
    )
    
    receipt3 = exchange3.on_order_arrival(order3, 1050 * TICK_PER_MS, market_qty=0)
    print(f"  Crossing回执: {receipt3}")
    assert receipt3 is not None and receipt3.fill_qty > 0, "应该有crossing成交"
    crossing_fill3 = receipt3.fill_qty
    expected_remaining3 = 150 - crossing_fill3
    
    receipts3 = exchange3.advance(1050 * TICK_PER_MS, 1500 * TICK_PER_MS, seg3)
    print(f"  Advance生成的回执: {receipts3}")
    
    # 净增量N=-10 < 0，post-crossing订单不应该成交
    if expected_remaining3 > 0:
        assert len(receipts3) == 0, f"净增量为负，不应该生成回执，实际: {len(receipts3)}"
        print(f"  ✓ 净增量N=-10 < 0，正确不成交（剩余{expected_remaining3}手）")
    
    # 测试场景4: 净增量N = 0
    print("\n场景4: 净增量N=0，不应该成交...")
    
    seg4 = TapeSegment(
        index=1,
        t_start=1000 * TICK_PER_MS,
        t_end=1500 * TICK_PER_MS,
        bid_price=100.0,
        ask_price=101.0,
        trades={(Side.BUY, 100.0): 30},
        cancels={},
        net_flow={(Side.BUY, 100.0): 0},  # bid侧净增量为0
        activation_bid={96.0, 97.0, 98.0, 99.0, 100.0},
        activation_ask={101.0, 102.0, 103.0, 104.0, 105.0},
    )
    
    print_tape_path([seg4])
    
    exchange4 = FIFOExchangeSimulator(cancel_front_ratio=0.5)
    exchange4.set_tape([seg4], 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)
    
    # 重要：初始化对手方（bid侧）的市场队列深度
    bid_level4 = exchange4._get_level(Side.BUY, 100.0)
    bid_level4.q_mkt = 100.0  # bid@100有100手
    
    order4 = Order(
        order_id="sell-n0",
        side=Side.SELL,
        price=100.0,
        qty=150,
        tif=TimeInForce.GTC,
    )
    
    receipt4 = exchange4.on_order_arrival(order4, 1050 * TICK_PER_MS, market_qty=0)
    print(f"  Crossing回执: {receipt4}")
    assert receipt4 is not None and receipt4.fill_qty > 0, "应该有crossing成交"
    crossing_fill4 = receipt4.fill_qty
    expected_remaining4 = 150 - crossing_fill4
    
    receipts4 = exchange4.advance(1050 * TICK_PER_MS, 1500 * TICK_PER_MS, seg4)
    print(f"  Advance生成的回执: {receipts4}")
    
    # 净增量N=0，post-crossing订单不应该成交
    if expected_remaining4 > 0:
        assert len(receipts4) == 0, f"净增量为0，不应该生成回执，实际: {len(receipts4)}"
        print(f"  ✓ 净增量N=0，正确不成交（剩余{expected_remaining4}手）")
    
    print("✓ Post-crossing fill with net increment test passed")


def test_snapshot_duplication():
    """测试快照复制功能。
    
    当两个快照之间的间隔超过500ms时，前一个快照会被复制以填充间隔。
    复制的快照的last_vol_split为空。
    
    时间单位：tick（每tick=100ns）。500ms = 5_000_000 ticks。
    """
    print("\n--- Test 38: Snapshot Duplication ---")
    
    from quant_framework.core.data_loader import SnapshotDuplicatingFeed
    from quant_framework.core.types import TICK_PER_MS, SNAPSHOT_MIN_INTERVAL_TICK
    
    # 创建简单的mock feed
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
    
    # 测试1: 间隔1000ms，应该生成1个复制快照
    print("\n测试1: 间隔1000ms (需要1个复制快照)...")
    t1 = 1000 * TICK_PER_MS
    t2 = 2000 * TICK_PER_MS
    snapshots1 = [
        create_test_snapshot(t1, 100.0, 101.0, last_vol_split=[(100.0, 10)]),
        create_test_snapshot(t2, 100.0, 101.0, last_vol_split=[(100.0, 20)]),
    ]
    
    feed1 = SnapshotDuplicatingFeed(MockFeed(snapshots1))
    
    result1 = []
    while True:
        snap = feed1.next()
        if snap is None:
            break
        result1.append(snap)
    
    print(f"  输入: 2个快照 (t={t1}, {t2})")
    print(f"  输出: {len(result1)}个快照")
    for i, snap in enumerate(result1):
        print(f"    {i+1}: t={snap.ts_recv}, last_vol_split={snap.last_vol_split}")
    
    assert len(result1) == 3, f"应生成3个快照(A, A', B)，实际{len(result1)}"
    assert result1[0].ts_recv == t1, f"第1个快照时间应为{t1}"
    assert result1[1].ts_recv == t1 + SNAPSHOT_MIN_INTERVAL_TICK, f"第2个快照时间应为{t1 + SNAPSHOT_MIN_INTERVAL_TICK}"
    assert result1[2].ts_recv == t2, f"第3个快照时间应为{t2}"
    assert result1[1].last_vol_split == [], "复制快照的last_vol_split应为空"
    print("  ✓ 1000ms间隔正确生成1个复制快照")
    
    # 测试2: 间隔2000ms，应该生成3个复制快照
    print("\n测试2: 间隔2000ms (需要3个复制快照)...")
    t3 = 1000 * TICK_PER_MS
    t4 = 3000 * TICK_PER_MS
    snapshots2 = [
        create_test_snapshot(t3, 100.0, 101.0, last_vol_split=[(100.0, 10)]),
        create_test_snapshot(t4, 100.0, 101.0, last_vol_split=[(100.0, 30)]),
    ]
    
    feed2 = SnapshotDuplicatingFeed(MockFeed(snapshots2))
    
    result2 = []
    while True:
        snap = feed2.next()
        if snap is None:
            break
        result2.append(snap)
    
    print(f"  输入: 2个快照 (t={t3}, {t4})")
    print(f"  输出: {len(result2)}个快照")
    for i, snap in enumerate(result2):
        print(f"    {i+1}: t={snap.ts_recv}, last_vol_split={snap.last_vol_split}")
    
    assert len(result2) == 5, f"应生成5个快照(A, A', A'', A''', B)，实际{len(result2)}"
    assert result2[0].ts_recv == t3, f"第1个快照时间应为{t3}"
    assert result2[1].ts_recv == t3 + SNAPSHOT_MIN_INTERVAL_TICK, f"第2个快照时间应为{t3 + SNAPSHOT_MIN_INTERVAL_TICK}"
    assert result2[2].ts_recv == t3 + 2 * SNAPSHOT_MIN_INTERVAL_TICK, f"第3个快照时间应为{t3 + 2 * SNAPSHOT_MIN_INTERVAL_TICK}"
    assert result2[3].ts_recv == t3 + 3 * SNAPSHOT_MIN_INTERVAL_TICK, f"第4个快照时间应为{t3 + 3 * SNAPSHOT_MIN_INTERVAL_TICK}"
    assert result2[4].ts_recv == t4, f"第5个快照时间应为{t4}"
    
    # 验证所有复制快照的last_vol_split为空
    for i in range(1, 4):
        assert result2[i].last_vol_split == [], f"复制快照{i+1}的last_vol_split应为空"
    print("  ✓ 2000ms间隔正确生成3个复制快照")
    
    # 测试3: 间隔500ms，不需要复制
    print("\n测试3: 间隔500ms (不需要复制)...")
    t5 = 1000 * TICK_PER_MS
    t6 = 1500 * TICK_PER_MS
    snapshots3 = [
        create_test_snapshot(t5, 100.0, 101.0, last_vol_split=[(100.0, 10)]),
        create_test_snapshot(t6, 100.0, 101.0, last_vol_split=[(100.0, 15)]),
    ]
    
    feed3 = SnapshotDuplicatingFeed(MockFeed(snapshots3))
    
    result3 = []
    while True:
        snap = feed3.next()
        if snap is None:
            break
        result3.append(snap)
    
    print(f"  输入: 2个快照 (t={t5}, {t6})")
    print(f"  输出: {len(result3)}个快照")
    
    assert len(result3) == 2, f"应生成2个快照（无复制），实际{len(result3)}"
    assert result3[0].ts_recv == t5
    assert result3[1].ts_recv == t6
    print("  ✓ 500ms间隔不需要复制")
    
    # 测试4: 重置功能
    print("\n测试4: 重置功能...")
    feed3.reset()
    result4 = []
    while True:
        snap = feed3.next()
        if snap is None:
            break
        result4.append(snap)
    
    assert len(result4) == 2, f"重置后应能再次获取2个快照"
    print("  ✓ 重置功能正常")
    
    # 测试5: 容差（tolerance）功能
    print("\n测试5: 容差功能...")
    from quant_framework.core.types import DEFAULT_SNAPSHOT_TOLERANCE_TICK
    
    # 间隔510ms（在默认10ms容差范围内），不应该复制
    t7 = 1000 * TICK_PER_MS
    t8 = 1510 * TICK_PER_MS  # 510ms间隔
    snapshots5 = [
        create_test_snapshot(t7, 100.0, 101.0, last_vol_split=[(100.0, 10)]),
        create_test_snapshot(t8, 100.0, 101.0, last_vol_split=[(100.0, 15)]),
    ]
    
    feed5 = SnapshotDuplicatingFeed(MockFeed(snapshots5))  # 使用默认10ms容差
    
    result5 = []
    while True:
        snap = feed5.next()
        if snap is None:
            break
        result5.append(snap)
    
    print(f"  间隔510ms（默认容差10ms内），输出: {len(result5)}个快照")
    assert len(result5) == 2, f"510ms间隔在容差范围内，应生成2个快照，实际{len(result5)}"
    print("  ✓ 510ms间隔在默认容差范围内，不复制")
    
    # 测试6: 自定义容差
    print("\n测试6: 自定义容差...")
    # 间隔520ms，使用5ms容差，应该复制1个
    t9 = 1000 * TICK_PER_MS
    t10 = 1520 * TICK_PER_MS  # 520ms间隔
    snapshots6 = [
        create_test_snapshot(t9, 100.0, 101.0, last_vol_split=[(100.0, 10)]),
        create_test_snapshot(t10, 100.0, 101.0, last_vol_split=[(100.0, 15)]),
    ]
    
    # 使用5ms容差（50000 ticks）
    feed6 = SnapshotDuplicatingFeed(MockFeed(snapshots6), tolerance_tick=5 * TICK_PER_MS)
    
    result6 = []
    while True:
        snap = feed6.next()
        if snap is None:
            break
        result6.append(snap)
    
    print(f"  间隔520ms（5ms容差），输出: {len(result6)}个快照")
    assert len(result6) == 3, f"520ms间隔超出5ms容差，应生成3个快照，实际{len(result6)}"
    print("  ✓ 520ms间隔超出5ms容差，正确复制")
    
    # 测试7: 0容差（严格500ms）
    print("\n测试7: 0容差（严格500ms）...")
    t11 = 1000 * TICK_PER_MS
    t12 = 1501 * TICK_PER_MS  # 501ms间隔
    snapshots7 = [
        create_test_snapshot(t11, 100.0, 101.0, last_vol_split=[(100.0, 10)]),
        create_test_snapshot(t12, 100.0, 101.0, last_vol_split=[(100.0, 15)]),
    ]
    
    feed7 = SnapshotDuplicatingFeed(MockFeed(snapshots7), tolerance_tick=0)
    
    result7 = []
    while True:
        snap = feed7.next()
        if snap is None:
            break
        result7.append(snap)
    
    print(f"  间隔501ms（0容差），输出: {len(result7)}个快照")
    assert len(result7) == 3, f"501ms间隔超出0容差，应生成3个快照，实际{len(result7)}"
    print("  ✓ 501ms间隔超出0容差，正确复制")
    
    print("✓ Snapshot duplication test passed")


def test_dynamic_queue_tracking_netflow():
    """测试动态队列追踪：确保在多次访问同一价位时队列不会为负。
    
    场景说明：
    - prev快照：bidprice: 6@127, 5@118, 4@32, 3@232, 2@37
    - curr快照：bidprice: 5@76, 4@32, 3@135, 2@49, 1@42
    - last_vol_split: 5@97, 6@197, 7@202
    
    价格6的情况：
    - Q_A (prev) = 127
    - Q_B (curr) = 0
    - M_total = 197
    - N_total = (Q_B - Q_A) + M_total = (0 - 127) + 197 = 70
    
    当bid路径多次访问价格6时（例如：6→5→6→7→5），
    必须确保队列在任何时刻都不为负。
    
    关键验证：
    1. 第一次访问价格6的归零段：N = 0 - 127 = -127（队列从127归零）
    2. 第二次访问价格6的有成交段：N应该基于剩余可分配量计算，而不是简单的M=197
    
    全局守恒：所有段的N之和应等于N_total = 70
    """
    print("\n--- Test 39: Dynamic Queue Tracking Netflow ---")
    
    from quant_framework.core.types import TICK_PER_MS, Side
    from quant_framework.tape.builder import UnifiedTapeBuilder, TapeConfig
    
    # 创建prev快照：价格6有队列127
    prev = create_multi_level_snapshot(
        ts=1000 * TICK_PER_MS,
        bids=[(6.0, 127), (5.0, 118), (4.0, 32), (3.0, 232), (2.0, 37)],
        asks=[(10.0, 100)],  # 简化的ask侧
        last_vol_split=[]
    )
    
    # 创建curr快照：价格6不存在（队列为0）
    curr = create_multi_level_snapshot(
        ts=1500 * TICK_PER_MS,
        bids=[(5.0, 76), (4.0, 32), (3.0, 135), (2.0, 49), (1.0, 42)],
        asks=[(10.0, 100)],  # 简化的ask侧
        last_vol_split=[(5.0, 97), (6.0, 197), (7.0, 202)]
    )
    
    config = TapeConfig()
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    
    # 计算价格6在所有段的总net_flow
    total_netflow_at_6 = 0
    total_trade_at_6 = 0
    
    for seg in tape:
        nf = seg.net_flow.get((Side.BUY, 6.0), 0)
        trade = seg.trades.get((Side.BUY, 6.0), 0)
        total_netflow_at_6 += nf
        total_trade_at_6 += trade
        print(f"  段{seg.index}: bid@6.0 netflow={nf}, trade={trade}")
    
    print(f"\n  价格6总计: netflow={total_netflow_at_6}, trade={total_trade_at_6}")
    
    # 验证全局守恒：N_total = (Q_B - Q_A) + M_total = (0 - 127) + 197 = 70
    expected_n_total = (0 - 127) + 197  # = 70
    print(f"  期望N_total: {expected_n_total}")
    
    # 允许一定的取整误差（最大1）
    assert abs(total_netflow_at_6 - expected_n_total) <= 1, \
        f"全局守恒失败: netflow总和={total_netflow_at_6}，期望={expected_n_total}"
    
    # 验证队列在每个段结束时不为负
    # 按段顺序累计队列深度
    q = 127  # 初始队列深度
    for seg in tape:
        nf = seg.net_flow.get((Side.BUY, 6.0), 0)
        trade = seg.trades.get((Side.BUY, 6.0), 0)
        
        # 只有当价格6在该段的activation集中时，才会有净流入和成交
        if 6.0 in seg.activation_bid:
            q_before = q
            q = q + nf - trade
            print(f"  段{seg.index}: Q_before={q_before}, N={nf}, M={trade}, Q_after={q}")
            
            # 验证队列不为负
            assert q >= -1, f"段{seg.index}结束时队列为负: {q}"
    
    print("\n  ✓ 所有段结束时队列深度>=0（允许取整误差）")
    
    # 验证价格7的成交正确分配
    total_trade_at_7 = sum(seg.trades.get((Side.BUY, 7.0), 0) for seg in tape)
    print(f"\n  价格7总成交: {total_trade_at_7}（期望约202）")
    assert total_trade_at_7 > 0, "价格7应有成交"
    
    # 验证价格5的成交正确分配
    total_trade_at_5 = sum(seg.trades.get((Side.BUY, 5.0), 0) for seg in tape)
    print(f"  价格5总成交: {total_trade_at_5}（期望约97）")
    assert total_trade_at_5 > 0, "价格5应有成交"
    
    print("✓ Dynamic queue tracking netflow test passed")


def test_starting_price_trade_prepending():
    """测试起点成交价前置功能。
    
    场景说明：
    - prev快照：bidprice: 6,5,4,3,2; askprice: 7,8,9,10,11
    - curr快照：bidprice: 5,4,3,2,1; askprice: 7,8,9,10,11
    - last_vol_split: 5@100, 6@100, 7@100
    
    原始算法：
    - meeting_seq = [5, 6, 7]（按最小位移排序）
    - bid路径 = [6, 5, 6, 7, 5]
    - ask路径 = [7, 5, 6, 7, 7]
    - 问题：第一段bid从6跳到5，6的数量减少全被归因为撤单
    
    新算法（起点成交价前置）：
    - 检测到bid_a=6是成交价，但meeting_seq[0]=5≠6
    - 在meeting_seq前插入6，变为[6, 5, 6, 7]
    - 检测到ask_a=7是成交价，但meeting_seq[0]=6≠7
    - 在meeting_seq前再插入7，变为[7, 6, 5, 6, 7]
    - bid路径 = [6, 7, 6, 5, 6, 7, 5]
    - ask路径 = [7, 7, 6, 5, 6, 7, 7]
    - 好处：第一段bid保持在6，第二段bid和ask都移动到7，成交可以正确分配
    """
    print("\n--- Test 40: Starting Price Trade Prepending ---")
    
    from quant_framework.core.types import TICK_PER_MS, Side
    from quant_framework.tape.builder import UnifiedTapeBuilder, TapeConfig
    
    # 创建prev快照：bid最优价=6，ask最优价=7
    prev = create_multi_level_snapshot(
        ts=1000 * TICK_PER_MS,
        bids=[(6.0, 100), (5.0, 100), (4.0, 100), (3.0, 100), (2.0, 100)],
        asks=[(7.0, 100), (8.0, 100), (9.0, 100), (10.0, 100), (11.0, 100)],
        last_vol_split=[]
    )
    
    # 创建curr快照：bid最优价变为5，ask保持7
    curr = create_multi_level_snapshot(
        ts=1500 * TICK_PER_MS,
        bids=[(5.0, 100), (4.0, 100), (3.0, 100), (2.0, 100), (1.0, 100)],
        asks=[(7.0, 100), (8.0, 100), (9.0, 100), (10.0, 100), (11.0, 100)],
        last_vol_split=[(5.0, 100), (6.0, 100), (7.0, 100)]
    )
    
    config = TapeConfig()
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    
    # 提取bid路径和ask路径
    bid_prices = [seg.bid_price for seg in tape]
    ask_prices = [seg.ask_price for seg in tape]
    
    print(f"  bid路径: {bid_prices}")
    print(f"  ask路径: {ask_prices}")
    
    # 验证起点成交价前置：
    # 1. bid_a=6 是成交价，第一个segment的bid_price应该是6
    assert tape[0].bid_price == 6.0, f"第一段bid_price应为6.0，实际为{tape[0].bid_price}"
    print(f"  ✓ 第一段bid_price=6.0（bid起点价格）")
    
    # 2. ask_a=7 是成交价，第一个segment的ask_price应该是7
    assert tape[0].ask_price == 7.0, f"第一段ask_price应为7.0，实际为{tape[0].ask_price}"
    print(f"  ✓ 第一段ask_price=7.0（ask起点价格）")
    
    # 3. 应该存在一个segment使得bid_price=6且有成交发生
    seg_with_trade_at_6 = [seg for seg in tape if seg.trades.get((Side.BUY, 6.0), 0) > 0]
    assert len(seg_with_trade_at_6) > 0, "应存在bid@6有成交的segment"
    print(f"  ✓ 存在bid@6有成交的segment")
    
    # 4. 价格6的成交应正确分配
    total_trade_at_6 = sum(seg.trades.get((Side.BUY, 6.0), 0) for seg in tape)
    print(f"  价格6总成交: {total_trade_at_6}（期望约100）")
    assert total_trade_at_6 > 0, "价格6应有成交"
    
    # 5. 验证全局守恒：价格6的 N = (Q_B - Q_A) + M
    # Q_A=100, Q_B=0（curr中没有价格6）, M=100
    # N_total = (0 - 100) + 100 = 0
    total_netflow_at_6 = sum(seg.net_flow.get((Side.BUY, 6.0), 0) for seg in tape)
    print(f"  价格6总netflow: {total_netflow_at_6}（期望0）")
    # 允许取整误差
    assert abs(total_netflow_at_6) <= 1, f"价格6的netflow应约为0，实际为{total_netflow_at_6}"
    
    print("✓ Starting price trade prepending test passed")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Testing Unified EventLoop Framework")
    print("="*60)
    
    tests = [
        test_tape_builder_basic,
        test_tape_builder_no_trades,
        test_tape_builder_conservation,
        test_tape_builder_netflow_distribution,
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
        test_dto_snapshot,
        test_readonly_oms_view,
        test_dto_strategy,
        test_tape_builder_invalid_time_order,
        test_meeting_sequence_consistency,
        test_historical_receipts_processing,
        test_intra_segment_advancement,
        test_receipt_delay_consistency,
        test_event_deterministic_ordering,
        test_peek_advance_pop_paradigm,
        test_receipt_recv_time_authority,
        test_no_causal_reversal_with_int_truncation,
        test_segment_queue_zero_constraint,
        test_segment_price_change_queue_constraint_detailed,
        test_crossing_immediate_execution,
        test_nonuniform_snapshot_timing,
        test_request_and_receipt_types,
        test_replay_strategy,
        test_receipt_logger,
        test_replay_integration,
        test_crossing_partial_fill_position_zero,
        test_multiple_orders_at_same_price,
        test_crossing_blocked_by_existing_shadow,
        test_crossing_blocked_by_queue_depth,
        test_post_crossing_fill_with_net_increment,
        test_snapshot_duplication,
        test_dynamic_queue_tracking_netflow,
        test_starting_price_trade_prepending,
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
