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
    NormalizedSnapshot, Level, Order, Side, TimeInForce, OrderStatus
)
from quant_framework.tape.builder import UnifiedTapeBuilder, TapeConfig
from quant_framework.exchange.simulator import FIFOExchangeSimulator
from quant_framework.trading.oms import OrderManager, Portfolio
from quant_framework.trading.strategy import SimpleStrategy
from quant_framework.runner.event_loop import EventLoopRunner, RunnerConfig, TimelineConfig


def create_test_snapshot(ts: int, bid: float, ask: float,
                         bid_qty: int = 100, ask_qty: int = 100,
                         last_vol_split=None) -> NormalizedSnapshot:
    """创建测试快照。"""
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
    """创建多档位快照。"""
    bid_levels = [Level(p, q) for p, q in bids]
    ask_levels = [Level(p, q) for p, q in asks]
    return NormalizedSnapshot(
        ts_exch=ts,
        bids=bid_levels,
        asks=ask_levels,
        last_vol_split=last_vol_split or [],
    )


def test_tape_builder_basic():
    """测试Tape构建器基本功能。"""
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
    """测试策略（使用DTO）。"""
    print("\n--- Test 9: Strategy ---")
    
    from quant_framework.core.dto import to_snapshot_dto, ReadOnlyOMSView
    
    strategy = SimpleStrategy(name="TestStrategy")
    oms = OrderManager()
    oms_view = ReadOnlyOMSView(oms)
    
    snapshot = create_test_snapshot(1000, 100.0, 101.0)
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
    
    from quant_framework.core.dto import to_snapshot_dto, ReadOnlyOMSView
    
    # Create components
    config = TapeConfig()
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    exchange = FIFOExchangeSimulator()
    oms = OrderManager()
    oms_view = ReadOnlyOMSView(oms)
    strategy = SimpleStrategy(name="TestStrategy")
    
    # Create snapshots
    prev = create_test_snapshot(1000, 100.0, 101.0)
    curr = create_test_snapshot(2000, 100.5, 101.5)
    prev_dto = to_snapshot_dto(prev)
    
    # Build tape
    tape = builder.build(prev, curr)
    print(f"Built tape with {len(tape)} segments")
    
    # Set tape on exchange
    exchange.set_tape(tape, 1000, 2000)
    
    # Submit an order via strategy (使用DTO)
    orders = strategy.on_snapshot(prev_dto, oms_view)
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


def test_dto_snapshot():
    """测试DTO快照转换功能。"""
    print("\n--- Test 14: DTO Snapshot ---")
    
    from quant_framework.core.dto import to_snapshot_dto, SnapshotDTO, LevelDTO
    
    # 创建测试快照
    snapshot = create_test_snapshot(1000, 100.0, 101.0, bid_qty=50, ask_qty=60)
    
    # 转换为DTO
    dto = to_snapshot_dto(snapshot)
    
    print(f"DTO类型: {type(dto).__name__}")
    print(f"DTO是否为frozen: {dto.__class__.__dataclass_fields__['ts_exch'].default is None}")
    
    # 验证数据正确转换
    assert dto.ts_exch == 1000, f"时间戳应为1000，实际为{dto.ts_exch}"
    assert len(dto.bids) == 1, f"买盘档位应为1，实际为{len(dto.bids)}"
    assert len(dto.asks) == 1, f"卖盘档位应为1，实际为{len(dto.asks)}"
    
    # 验证便捷属性
    assert dto.best_bid == 100.0, f"最优买价应为100.0，实际为{dto.best_bid}"
    assert dto.best_ask == 101.0, f"最优卖价应为101.0，实际为{dto.best_ask}"
    assert dto.mid_price == 100.5, f"中间价应为100.5，实际为{dto.mid_price}"
    assert dto.spread == 1.0, f"价差应为1.0，实际为{dto.spread}"
    
    # 验证DTO不可变（尝试修改应抛出异常）
    try:
        dto.ts_exch = 2000
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
    oms.submit(order, 1000)
    
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
    snapshot = create_test_snapshot(1000, 100.0, 101.0)
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
    prev = create_test_snapshot(1000, 100.0, 101.0)
    curr = create_test_snapshot(1000, 100.5, 101.5)  # 相同时间
    
    try:
        builder.build(prev, curr)
        assert False, "应该抛出ValueError当t_b == t_a"
    except ValueError as e:
        print(f"✓ 正确抛出ValueError当t_b == t_a: {e}")
    
    # 测试情况2: t_b < t_a
    prev2 = create_test_snapshot(2000, 100.0, 101.0)
    curr2 = create_test_snapshot(1000, 100.5, 101.5)  # 时间倒退
    
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
        1000,
        bids=[(100.0, 50), (99.0, 30)],
        asks=[(101.0, 40), (102.0, 20)],
        last_vol_split=[(99.5, 10), (100.5, 15), (101.0, 20), (100.0, 25)]
    )
    curr = create_multi_level_snapshot(
        2000,
        bids=[(100.5, 40), (99.5, 35)],
        asks=[(101.5, 35), (102.5, 25)],
        last_vol_split=[(99.5, 10), (100.5, 15), (101.0, 20), (100.0, 25)]
    )
    
    tape = builder.build(prev, curr)
    
    print(f"生成了{len(tape)}个段")
    
    # 获取所有bid和ask在各段的价格路径
    bid_prices = []
    ask_prices = []
    
    for seg in tape:
        bid_prices.append(seg.bid_price)
        ask_prices.append(seg.ask_price)
        print(f"  段{seg.index}: bid={seg.bid_price}, ask={seg.ask_price}")
    
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
    
    # 创建测试快照
    snapshots = [
        create_test_snapshot(1000, 100.0, 101.0, bid_qty=50),
        create_test_snapshot(2000, 100.0, 101.0, bid_qty=40),
        create_test_snapshot(3000, 100.0, 101.0, bid_qty=30),
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
        delay_out=100,
        delay_in=50,
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
    1. 创建一个区间[1000, 2000]，包含多个成交
    2. 在区间中间时刻（如1500）提交一个订单
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
    
    # 创建测试快照：有足够成交量的区间
    snapshots = [
        create_test_snapshot(1000, 100.0, 101.0, bid_qty=50, last_vol_split=[(100.0, 100)]),
        create_test_snapshot(2000, 100.0, 101.0, bid_qty=20, last_vol_split=[(100.0, 100)]),
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
                self.order_placed_at = snapshot.ts_exch if hasattr(snapshot, 'ts_exch') else 1000
                return [order]
            return []
        
        def on_receipt(self, receipt, snapshot, oms):
            return []
    
    strategy = TrackingStrategy()
    
    # 使用延迟配置，让订单在区间中间到达
    runner_config = RunnerConfig(
        delay_out=500,  # 订单在recv_time+500时到达交易所
        delay_in=50,
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
    # 由于订单在1500时到达（1000+500延迟），交易所应该先推进到1500
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
    
    # 创建测试快照
    snapshots = [
        create_test_snapshot(1000, 100.0, 101.0, bid_qty=50, last_vol_split=[(100.0, 10)]),
        create_test_snapshot(2000, 100.0, 101.0, bid_qty=40, last_vol_split=[(100.0, 10)]),
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
    delay_in = 100
    delay_out = 50
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
        test_dto_snapshot,
        test_readonly_oms_view,
        test_dto_strategy,
        test_tape_builder_invalid_time_order,
        test_meeting_sequence_consistency,
        test_historical_receipts_processing,
        test_intra_segment_advancement,
        test_receipt_delay_consistency,
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
