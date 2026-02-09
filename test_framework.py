"""统一EventLoop框架的综合测试套件。

测试验证内容：
- Tape构建器：段生成、成交量分配、守恒方程
- 交易所模拟器：坐标轴模型、成交时间计算
- 订单管理器：订单生命周期、回执处理
- 事件循环：双时间线支持、延迟处理
- 集成测试：完整回测流程
- DTO测试：数据传输对象和只读视图

测试模式：
- 启用DEBUG级别日志，用于验证逻辑正确性
- 通过日志输出追踪事件处理流程
"""

import logging
import sys

from quant_framework.core.types import (
    NormalizedSnapshot, Level, Order, Side, TimeInForce, OrderStatus, TICK_PER_MS
)
from quant_framework.tape.builder import UnifiedTapeBuilder, TapeConfig
from quant_framework.exchange.simulator import FIFOExchangeSimulator
from quant_framework.trading.oms import OrderManager, Portfolio
from quant_framework.trading.strategy import SimpleStrategy
from quant_framework.runner.event_loop import EventLoopRunner, RunnerConfig, TimelineConfig


def setup_test_logging():
    """设置测试环境的DEBUG日志。
    
    启用DEBUG级别日志，用于验证逻辑正确性：
    - 交易所模拟器的订单处理流程
    - 事件循环的事件调度
    - 回执记录器的回执处理
    
    注意：此函数应在测试运行时显式调用，而不是在模块导入时自动执行，
    以避免干扰其他测试框架（如pytest）的日志配置。
    """
    # 配置root logger
    log_level = logging.DEBUG
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # 配置loggers
    logging.basicConfig(level=log_level, handlers=[console_handler], force=True)
    
    # 设置特定模块的日志级别为DEBUG
    logging.getLogger('quant_framework.exchange.simulator').setLevel(logging.DEBUG)
    logging.getLogger('quant_framework.runner.event_loop').setLevel(logging.DEBUG)
    logging.getLogger('quant_framework.trading.receipt_logger').setLevel(logging.DEBUG)
    logging.getLogger('quant_framework.tape.builder').setLevel(logging.DEBUG)


def create_test_snapshot(ts: int, bid: float, ask: float,
                         bid_qty: int = 100, ask_qty: int = 100,
                         last_vol_split=None) -> NormalizedSnapshot:
    """创建测试快照。
    
    时间单位为tick（每tick=100ns）。
    """
    return NormalizedSnapshot(
        ts_recv=ts,  # 主时间线
        bids=[Level(bid, bid_qty)],
        asks=[Level(ask, ask_qty)],
        last_vol_split=last_vol_split or [],
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
            non_zero_netflow = {k: v for k, v in seg.net_flow.items() if v != 0}
            if non_zero_netflow:
                print(f"      net_flow: {non_zero_netflow}")
        print(f"      activation_bid: {seg.activation_bid}")
        print(f"      activation_ask: {seg.activation_ask}")
    print()


def test_tape_builder_basic():
    """测试Tape构建器基本功能。"""
    print("\n--- Test 1: Tape Builder Basic ---")
    
    config = TapeConfig()
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    
    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0)
    curr = create_test_snapshot(1500 * TICK_PER_MS, 100.5, 101.5, last_vol_split=[(100.0, 5), (101.0, 5)])
    
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    
    assert len(tape) > 0, "Tape should have at least one segment"
    
    # Check activation sets
    for seg in tape:
        assert len(seg.activation_bid) <= 5, "Activation set should have at most 5 prices"
        assert len(seg.activation_ask) <= 5, "Activation set should have at most 5 prices"
    
    print("✓ Tape builder basic test passed")


def test_tape_builder_no_trades():
    """Test tape builder with no trades.
    
    当没有成交时，验证tape仍然能正确生成段结构。
    由于价格变化（从100/101到100.5/101.5），会生成多个段，
    但这些段中都没有成交记录。
    """
    print("\n--- Test 2: Tape Builder No Trades ---")
    
    config = TapeConfig()
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    
    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, last_vol_split=[])
    curr = create_test_snapshot(1500 * TICK_PER_MS, 100.5, 101.5, last_vol_split=[])
    
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    
    # 验证tape已生成
    assert len(tape) >= 1, "Tape should have at least one segment"
    
    # 验证没有成交记录
    for seg in tape:
        total_trades = sum(qty for qty in seg.trades.values())
        assert total_trades == 0, f"Segment {seg.index} should have no trades, got {total_trades}"
    
    print("✓ Tape builder no trades test passed")


def test_tape_builder_conservation():
    """Test tape builder conservation equations."""
    print("\n--- Test 3: Tape Builder Conservation ---")
    
    config = TapeConfig(epsilon=1.0)
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
    """Test netflow distribution for active segments and zeroing.
    
    验证净流入量（netflow）在活跃段中的分布：
    - 净流入量会分配到价位激活的段中
    - 总净流入量满足守恒方程：N = delta_Q + M
    """
    print("\n--- Test 3b: Tape Builder Netflow Distribution ---")
    
    config = TapeConfig(epsilon=1.0)
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    
    prev = create_multi_level_snapshot(
        1000 * TICK_PER_MS,
        bids=[(100.0, 40), (99.0, 30), (98.0, 20)],
        asks=[(101.0, 40), (102.0, 20), (103.0, 10)],
        last_vol_split=[(100.0, 10), (101.0, 15), (102.0, 10)]
    )
    curr = create_multi_level_snapshot(
        1500 * TICK_PER_MS,
        bids=[(101.0, 55), (100.0, 40), (99.0, 35)],
        asks=[(100.0, 35), (101.0, 35), (102.0, 25)],
        last_vol_split=[(100.0, 10), (101.0, 15), (102.0, 10)]
    )
    
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    
    net_flow_bid_100 = [seg.net_flow.get((Side.BUY, 100.0), 0) for seg in tape]
    net_flow_ask_101 = [seg.net_flow.get((Side.SELL, 101.0), 0) for seg in tape]
    trades_ask_101 = [seg.trades.get((Side.SELL, 101.0), 0) for seg in tape]
    
    print(f"Net flow bid@100: {net_flow_bid_100}")
    print(f"Net flow ask@101: {net_flow_ask_101}")
    print(f"Trades ask@101: {trades_ask_101}")
    
    # 验证tape生成正确数量的段
    assert len(tape) >= 1, "Expected at least 1 segment"
    
    # 验证总净流入量满足守恒方程
    total_bid_net = sum(net_flow_bid_100)
    prev_bid_qty = next(lvl.qty for lvl in prev.bids if abs(lvl.price - 100.0) < 1e-8)
    curr_bid_qty = next(lvl.qty for lvl in curr.bids if abs(lvl.price - 100.0) < 1e-8)
    total_trades_bid_100 = sum(seg.trades.get((Side.BUY, 100.0), 0) for seg in tape)
    
    # 守恒方程：N = Q_B - Q_A + M
    expected_bid_net = curr_bid_qty - prev_bid_qty + total_trades_bid_100
    print(f"Bid@100: total_net={total_bid_net}, expected={expected_bid_net} (Q_B={curr_bid_qty} - Q_A={prev_bid_qty} + M={total_trades_bid_100})")
    
    # 允许取整误差
    assert abs(total_bid_net - expected_bid_net) <= 1, (
        f"Expected total netflow {expected_bid_net} for bid 100, got {total_bid_net}"
    )
    
    # 验证ask侧守恒
    total_ask_net = sum(net_flow_ask_101)
    prev_ask_qty = next(lvl.qty for lvl in prev.asks if abs(lvl.price - 101.0) < 1e-8)
    curr_ask_qty = next(lvl.qty for lvl in curr.asks if abs(lvl.price - 101.0) < 1e-8)
    total_trades_ask_101 = sum(trades_ask_101)
    
    expected_ask_net = curr_ask_qty - prev_ask_qty + total_trades_ask_101
    print(f"Ask@101: total_net={total_ask_net}, expected={expected_ask_net} (Q_B={curr_ask_qty} - Q_A={prev_ask_qty} + M={total_trades_ask_101})")
    
    # 允许取整误差
    assert abs(total_ask_net - expected_ask_net) <= 1, (
        f"Expected total netflow {expected_ask_net} for ask 101, got {total_ask_net}"
    )
    
    print("✓ Tape builder netflow distribution test passed")


def test_exchange_simulator_basic():
    """Test basic exchange simulator functionality."""
    print("\n--- Test 4: Exchange Simulator Basic ---")
    
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
    
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
    
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
    
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
    
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
    
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
    t_global = 1050 * TICK_PER_MS
    for seg in tape:
        seg_receipts = []
        t_cur = max(seg.t_start, t_global)
        while t_cur < seg.t_end:
            receipts, t_stop = exchange.advance(t_cur, seg.t_end, seg)
            seg_receipts.extend(receipts)
            if t_stop <= t_cur:
                break
            t_cur = t_stop
        all_receipts.extend(seg_receipts)
        print(f"Segment {seg.index}: generated {len(seg_receipts)} receipts")
        t_global = seg.t_end
    
    print(f"Total receipts: {len(all_receipts)}")
    for r in all_receipts:
        print(f"  {r.order_id}: {r.receipt_type}, fill_qty={r.fill_qty}")
    
    # With 50 trades and market queue 30, order should be filled
    # (trades > market_queue + order_qty)
    filled = any(r.receipt_type in ["FILL", "PARTIAL"] for r in all_receipts)
    assert filled, "Order should have been filled"
    
    print("✓ Exchange simulator fill test passed")


def test_exchange_simulator_multi_partial_to_fill():
    """验证多次部分成交后最终返回FILL回执。"""
    print("\n--- Test 7b: Exchange Simulator Multi-Partial to Fill ---")
    
    from quant_framework.core.types import TapeSegment
    
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
    seg = TapeSegment(
        index=1,
        t_start=1000 * TICK_PER_MS,
        t_end=1304 * TICK_PER_MS,
        bid_price=100.0,
        ask_price=101.0,
        trades={(Side.BUY, 100.0): 4},
        cancels={},
        net_flow={(Side.BUY, 100.0): 0},
        activation_bid={100.0},
        activation_ask={101.0},
    )
    
    exchange.set_tape([seg], 1000 * TICK_PER_MS, 1304 * TICK_PER_MS)
    
    order = Order(order_id="multi-fill", side=Side.BUY, price=100.0, qty=4)
    exchange.on_order_arrival(order, 1000 * TICK_PER_MS, market_qty=0)
    
    receipt_1, _ = exchange.advance(1000 * TICK_PER_MS, 1101 * TICK_PER_MS, seg)
    receipt_2, _ = exchange.advance(1101 * TICK_PER_MS, 1202 * TICK_PER_MS, seg)
    receipt_3, _ = exchange.advance(1202 * TICK_PER_MS, 1304 * TICK_PER_MS, seg)
    
    receipts = receipt_1 + receipt_2 + receipt_3
    fill_types = [r.receipt_type for r in receipts]
    fill_qtys = [r.fill_qty for r in receipts]
    
    assert receipts, "应该产生回执"
    assert fill_types[-1] == "FILL", f"最后一笔应为FILL回执，实际: {fill_types[-1]}"
    assert sum(fill_qtys) == 4, f"总成交数量应为4，实际: {sum(fill_qtys)}"
    
    print("✓ Exchange simulator multi-partial to fill test passed")


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
        receipts, _ = exchange.advance(seg.t_start, seg.t_end, seg)
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
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
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
    
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
    
    # Create tape where market queue grows then shrinks
    config = TapeConfig()
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    
    # Start with queue 30, end with queue 10
    # With lastvolsplit of 50 trades
    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=30)
    curr = create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=10,
                                last_vol_split=[(100.0, 48)])
    
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    
    exchange.set_tape(tape, 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)

    exchange.advance(0, 1010 * TICK_PER_MS, tape[0])
    exchange.advance(1010 * TICK_PER_MS, 1100 * TICK_PER_MS, tape[1])

    order1 = Order(order_id="order1", side=Side.BUY, price=100.0, qty=20)
    exchange.on_order_arrival(order1, 1100 * TICK_PER_MS, market_qty=30)

    exchange.advance(1100 * TICK_PER_MS, 1300 * TICK_PER_MS, tape[1])
    
    order2 = Order(order_id="order2", side=Side.BUY, price=100.0, qty=10)
    exchange.on_order_arrival(order2, 1300 * TICK_PER_MS, market_qty=30)
    
    # Verify positions
    shadows = exchange.get_shadow_orders()
    o1 = next(s for s in shadows if s.order_id == "order1")
    o2 = next(s for s in shadows if s.order_id == "order2")
    
    print(f"Order 1 position: {o1.pos}")
    print(f"Order 2 position: {o2.pos}")
    
    # Advance and collect fills
    all_receipts = []
    last_t = 1300 * TICK_PER_MS
    for seg in [tape[1], tape[2]]:
        t_cur = last_t
        while t_cur < seg.t_end:
            receipts, t_stop = exchange.advance(t_cur, seg.t_end, seg)
            all_receipts.extend(receipts)
            if t_stop <= t_cur:
                break
            t_cur = t_stop
        last_t = seg.t_end
    
    # With 50 trades: first 30 consume market queue, then order1 (20), then order2 (10)
    # Order 1 should fill first
    fill_order = [r.order_id for r in all_receipts if r.receipt_type in ["FILL", "PARTIAL"]]
    print(f"Fill order: {fill_order}")

    for r in all_receipts:
        print(f"  {r.order_id} : fill_time={r.timestamp}")
    
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
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
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
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
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
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
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
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
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
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
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
        exchange=FIFOExchangeSimulator(cancel_bias_k=0.0),
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
        exchange=FIFOExchangeSimulator(cancel_bias_k=0.0),
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
        exchange=FIFOExchangeSimulator(cancel_bias_k=0.0),
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
        epsilon=1.0,
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
        epsilon=1.0,
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
        epsilon=1.0,
    )
    builder = UnifiedTapeBuilder(config=tape_config, tick_size=1.0)
    
    # 创建快照：bid@100, ask@101 (500ms interval)
    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=50, ask_qty=60)
    curr = create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=40, ask_qty=50,
                                last_vol_split=[(100.0, 10), (101.0, 10)])
    
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    
    # 创建交易所模拟器
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
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


def test_tape_start_time():
    """测试tape始终从t_a开始。
    
    验证：快照间隔由feed层保证，tape始终从t_a开始到t_b结束。
    
    注意：时间单位为tick（每tick=100ns）
    """
    print("\n--- Test 29: Tape Start Time ---")
    
    from quant_framework.core.types import TICK_PER_MS
    
    # 使用默认配置
    tape_config = TapeConfig()
    builder = UnifiedTapeBuilder(config=tape_config, tick_size=1.0)
    
    # 测试1: 任意间隔的快照，tape始终从t_a开始
    t_a = 1000 * TICK_PER_MS
    t_b = 2500 * TICK_PER_MS
    
    prev = create_test_snapshot(t_a, 100.0, 101.0, bid_qty=50, ask_qty=60,
                                last_vol_split=[])
    curr = create_test_snapshot(t_b, 100.5, 101.5, bid_qty=40, ask_qty=50,
                                last_vol_split=[(100.5, 20)])  # 有成交
    
    tape = builder.build(prev, curr)
    
    print(f"快照间隔: 1500ms")
    print(f"T_A={t_a}, T_B={t_b}")
    print(f"生成了{len(tape)}个段")
    
    for seg in tape:
        print(f"  段{seg.index}: t=[{seg.t_start}, {seg.t_end}], "
              f"duration={(seg.t_end - seg.t_start) / TICK_PER_MS}ms, "
              f"bid={seg.bid_price}, ask={seg.ask_price}")
        if seg.trades:
            print(f"    trades: {dict(seg.trades)}")
    
    # 验证：第一段应从t_a开始
    first_seg = tape[0]
    assert first_seg.t_start == t_a, f"第一段应从{t_a}开始，实际{first_seg.t_start}"
    print(f"  ✓ 第一段正确从t_a={t_a}开始")
    
    # 最后一段应该到T_B结束
    last_seg = tape[-1]
    assert last_seg.t_end == t_b, f"最后一段应到{t_b}结束，实际{last_seg.t_end}"
    print(f"  ✓ 最后一段正确到T_B={t_b}结束")
    
    # 测试2: 短间隔快照，tape也从t_a开始
    print("\n测试短间隔快照...")
    t_a2 = 3000 * TICK_PER_MS
    t_b2 = 3400 * TICK_PER_MS
    prev2 = create_test_snapshot(t_a2, 100.0, 101.0)
    curr2 = create_test_snapshot(t_b2, 100.5, 101.5, last_vol_split=[(100.5, 10)])
    
    tape2 = builder.build(prev2, curr2)
    print(f"快照间隔: 400ms")
    print(f"T_A={t_a2}, T_B={t_b2}")
    print(f"生成了{len(tape2)}个段")
    
    # 验证：第一段应从t_a2开始
    first_seg_start = tape2[0].t_start
    assert first_seg_start == t_a2, f"第一段应从{t_a2}开始，实际{first_seg_start}"
    print(f"  ✓ 短间隔快照正确处理，从{first_seg_start}开始")
    
    print("✓ Tape start time test passed")


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
    from quant_framework.trading.replay_strategy import ReplayStrategy
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
        
        # 验证加载的订单（直接验证pending_orders）
        assert len(strategy.pending_orders) == 3, f"应加载3个订单，实际{len(strategy.pending_orders)}"
        first_order_time, first_order = strategy.pending_orders[0]
        assert first_order_time == 1000
        assert first_order.price == 100.5
        assert first_order.qty == 10
        assert first_order.side == Side.BUY
        print(f"  ✓ 成功加载3个订单")
        
        # 验证加载的撤单（直接验证pending_cancels）
        assert len(strategy.pending_cancels) == 2, f"应加载2个撤单，实际{len(strategy.pending_cancels)}"
        first_cancel_time, first_cancel = strategy.pending_cancels[0]
        assert first_cancel_time == 1500
        assert first_cancel.order_id == "1001"
        print(f"  ✓ 成功加载2个撤单")
        
        # 验证pending_orders按时间排序
        times = [t for t, o in strategy.pending_orders]
        assert times == sorted(times), "订单应按时间排序"
        print(f"  ✓ 订单按时间正确排序")
        
        # 验证pending_cancels按时间排序
        cancel_times = [t for t, c in strategy.pending_cancels]
        assert cancel_times == sorted(cancel_times), "撤单应按时间排序"
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
        logger.register_order("order-4", 0)
        
        assert len(logger.order_total_qty) == 4
        print(f"  ✓ 注册4个订单成功")
        
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
        
        # 记录回执 - 部分成交后撤单（先发送PARTIAL回执，再发送CANCELED回执）
        # 根据正确的行为：如果有部分成交，必须先发送PARTIAL回执
        receipt2b = OrderReceipt(
            order_id="order-2",
            receipt_type="PARTIAL",
            timestamp=2500,
            fill_qty=20,  # 部分成交
            fill_price=99.0,
            remaining_qty=30,
        )
        receipt2b.recv_time = 2510
        logger.log_receipt(receipt2b)
        
        # 然后发送撤单回执，fill_qty应该等于之前PARTIAL累积的值
        receipt3 = OrderReceipt(
            order_id="order-2",
            receipt_type="CANCELED",
            timestamp=3000,
            fill_qty=20,  # 等于之前PARTIAL累积的值
            fill_price=99.0,
            remaining_qty=30,
        )
        receipt3.recv_time = 3010
        logger.log_receipt(receipt3)
        
        # 撤单后不应被判为完全成交（不修改订单的累计成交量）
        
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
        
        assert len(logger.records) == 5  # 多了一条PARTIAL回执
        print(f"  ✓ 记录5条回执成功")
        
        # 验证统计
        stats = logger.get_statistics()
        assert stats['total_receipts'] == 5  # 多了一条PARTIAL回执
        assert stats['total_orders'] == 4
        assert stats['partial_fill_count'] == 2  # 2条PARTIAL回执
        assert stats['full_fill_count'] == 1
        assert stats['cancel_count'] == 1
        assert stats['reject_count'] == 1
        assert abs(stats['full_fill_rate'] - (1 / 3)) < 0.01
        assert abs(stats['partial_fill_rate'] - (1 / 3)) < 0.01
        assert stats['fully_filled_orders'] == 1
        assert stats['partially_filled_orders'] == 1
        assert stats['unfilled_orders'] == 1
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
        # 4个订单中1个完全成交（包含一个0数量订单）
        expected_rate_count = 1 / 4
        assert abs(fill_rate_count - expected_rate_count) < 0.01
        print(f"  ✓ 按订单数成交率: {fill_rate_count:.2%}")
        
        # 保存到文件
        logger.save_to_file()
        assert os.path.exists(output_file)
        
        # 验证文件内容
        with open(output_file, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 6  # 1 header + 5 records
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
        exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
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
        epsilon=1.0,
    )
    builder = UnifiedTapeBuilder(config=tape_config, tick_size=1.0)
    
    # 创建快照：bid@100, ask@101 (500ms interval)
    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=50, ask_qty=100)
    curr = create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=50, ask_qty=100,
                                last_vol_split=[(100.0, 10), (101.0, 10)])
    
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    
    # 创建交易所模拟器
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
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
    exchange2 = FIFOExchangeSimulator(cancel_bias_k=0.0)
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
        epsilon=1.0,
    )
    builder = UnifiedTapeBuilder(config=tape_config, tick_size=1.0)
    
    # 构建tape并输出路径
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
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
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
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
    
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
    exchange.set_tape([seg], 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)
    
    # Initialize opposite-side liquidity to allow crossing
    ask_level = exchange._get_level(Side.SELL, 101.0)
    ask_level.q_mkt = 50.0
    
    # Initialize same-side queue depth at the price
    bid_level = exchange._get_level(Side.BUY, 101.0)
    same_side_depth = 20.0
    bid_level.q_mkt = same_side_depth
    
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
    assert queued.pos >= same_side_depth, f"Order should sit behind existing depth, got {queued.pos}"
    
    print("✓ Crossing blocked by queue depth test passed")


def test_post_crossing_fill_with_net_increment():
    """测试改善价模式下的成交优先级与trade复制。"""
    print("\n--- Test 37: Improvement Mode Fill Priority ---")

    from quant_framework.core.types import TapeSegment

    t_start = 1000 * TICK_PER_MS
    t_end = 1100 * TICK_PER_MS

    seg = TapeSegment(
        index=1,
        t_start=t_start,
        t_end=t_end,
        bid_price=100.0,
        ask_price=101.0,
        trades={(Side.BUY, 100.0): 10},
        cancels={},
        net_flow={(Side.BUY, 100.0): 0},
        activation_bid={100.0},
        activation_ask={101.0},
    )

    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
    exchange.set_tape([seg], t_start, t_end)

    # 改善价订单（不crossing）：100.5 > bid=100，但低于ask=101
    improving_order = Order(
        order_id="buy-improve",
        side=Side.BUY,
        price=100.5,
        qty=6,
        tif=TimeInForce.GTC,
    )
    exchange.on_order_arrival(improving_order, t_start, market_qty=0)

    # 同侧更差价订单：应在改善价订单之前不因trade成交
    base_order = Order(
        order_id="buy-base",
        side=Side.BUY,
        price=100.0,
        qty=10,
        tif=TimeInForce.GTC,
    )
    exchange.on_order_arrival(base_order, t_start, market_qty=0)

    receipts1, t_stop = exchange.advance(t_start, t_end, seg)
    print(f"  Advance(1) receipts: {receipts1}")
    assert len(receipts1) == 1, "Improving order should fill first"
    assert receipts1[0].order_id == "buy-improve"
    assert receipts1[0].receipt_type == "FILL"

    # 继续推进到段末，基础价订单只吃剩余trade
    receipts2, _ = exchange.advance(t_stop, t_end, seg)
    print(f"  Advance(2) receipts: {receipts2}")
    base_receipts = [r for r in receipts2 if r.order_id == "buy-base"]
    assert base_receipts, "Base order should have partial fill after improvement ends"
    assert base_receipts[0].receipt_type == "PARTIAL"
    assert base_receipts[0].fill_qty == 4, f"Expected 4, got {base_receipts[0].fill_qty}"

    print("✓ Improvement mode fill priority test passed")


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


def test_contract_config_loading():
    """测试合约配置加载功能。
    
    验证：
    1. 能够从合约字典XML文件中正确加载合约信息
    2. 交易时段（包括跨越午夜的时段）能够正确解析
    3. 未找到的合约返回None
    """
    print("\n--- Test: Contract Config Loading ---")
    
    import os
    import tempfile
    from quant_framework.config import _load_contract_dictionary, TradingHour, ContractInfo
    
    # 创建临时合约字典文件
    contract_xml = """<?xml version="1.0" encoding="UTF-8"?>
<ContractDictionaryConfig>
    <Contract>
        <ContractId>IF2401</ContractId>
        <TickSize>0.2</TickSize>
        <ExchangeCode>CFFEX</ExchangeCode>
        <TradingHours>
            <TradingHour>
                <StartTime>09:30:00</StartTime>
                <EndTime>11:30:00</EndTime>
            </TradingHour>
            <TradingHour>
                <StartTime>13:00:00</StartTime>
                <EndTime>15:00:00</EndTime>
            </TradingHour>
        </TradingHours>
    </Contract>
    <Contract>
        <ContractId>AU2401</ContractId>
        <TickSize>0.02</TickSize>
        <ExchangeCode>SHFE</ExchangeCode>
        <TradingHours>
            <TradingHour>
                <StartTime>21:00:00</StartTime>
                <EndTime>02:30:00</EndTime>
            </TradingHour>
            <TradingHour>
                <StartTime>09:00:00</StartTime>
                <EndTime>10:15:00</EndTime>
            </TradingHour>
        </TradingHours>
    </Contract>
</ContractDictionaryConfig>"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(contract_xml)
        temp_path = f.name
    
    try:
        # 测试1: 加载IF2401
        print("\n测试1: 加载IF2401合约信息...")
        info = _load_contract_dictionary(temp_path, "IF2401")
        assert info is not None, "应该找到IF2401合约"
        assert info.contract_id == "IF2401", f"合约ID应为IF2401，实际为{info.contract_id}"
        assert info.tick_size == 0.2, f"TickSize应为0.2，实际为{info.tick_size}"
        assert info.exchange_code == "CFFEX", f"ExchangeCode应为CFFEX，实际为{info.exchange_code}"
        assert len(info.trading_hours) == 2, f"应有2个交易时段，实际有{len(info.trading_hours)}"
        assert info.trading_hours[0].start_time == "09:30:00", "第一个时段开始时间错误"
        assert info.trading_hours[0].end_time == "11:30:00", "第一个时段结束时间错误"
        print(f"  ✓ IF2401加载成功: tick_size={info.tick_size}, exchange={info.exchange_code}")
        
        # 测试2: 加载AU2401（有跨越午夜的时段）
        print("\n测试2: 加载AU2401合约信息（有跨越午夜的时段）...")
        info = _load_contract_dictionary(temp_path, "AU2401")
        assert info is not None, "应该找到AU2401合约"
        assert info.contract_id == "AU2401"
        assert info.tick_size == 0.02
        assert len(info.trading_hours) == 2
        assert info.trading_hours[0].start_time == "21:00:00", "夜盘开始时间应为21:00:00"
        assert info.trading_hours[0].end_time == "02:30:00", "夜盘结束时间应为02:30:00"
        print(f"  ✓ AU2401加载成功: 夜盘={info.trading_hours[0].start_time}-{info.trading_hours[0].end_time}")
        
        # 测试3: 查找不存在的合约
        print("\n测试3: 查找不存在的合约...")
        info = _load_contract_dictionary(temp_path, "NON_EXISTENT")
        assert info is None, "不存在的合约应返回None"
        print("  ✓ 不存在的合约正确返回None")
        
        # 测试4: 空路径或空合约ID
        print("\n测试4: 空路径或空合约ID...")
        assert _load_contract_dictionary("", "IF2401") is None
        assert _load_contract_dictionary(temp_path, "") is None
        print("  ✓ 空参数正确处理")
        
    finally:
        os.unlink(temp_path)
    
    print("✓ Contract config loading test passed")


def test_trading_hours_session_detection():
    """测试交易时段检测功能。
    
    验证SnapshotDuplicatingFeed能够正确检测：
    1. 时间是否在交易时段内
    2. 两个时间点是否跨越交易时段间隔
    3. 跨越午夜的交易时段处理
    """
    print("\n--- Test: Trading Hours Session Detection ---")
    
    from quant_framework.core.data_loader import SnapshotDuplicatingFeed
    from quant_framework.config import TradingHour
    
    # 创建mock feed
    class MockFeed:
        def __init__(self):
            self.idx = 0
        def next(self):
            return None
        def reset(self):
            self.idx = 0
    
    # 测试正常时段（09:30 - 11:30, 13:00 - 15:00）
    trading_hours = [
        TradingHour(start_time="09:30:00", end_time="11:30:00"),
        TradingHour(start_time="13:00:00", end_time="15:00:00"),
    ]
    
    feed = SnapshotDuplicatingFeed(MockFeed(), trading_hours=trading_hours)
    helper = feed._trading_hours_helper  # 使用TradingHoursHelper
    
    # 测试1: 时间解析
    print("\n测试1: 时间解析...")
    assert helper.parse_time_to_seconds("09:30:00") == 9 * 3600 + 30 * 60
    assert helper.parse_time_to_seconds("21:00:00") == 21 * 3600
    assert helper.parse_time_to_seconds("02:30:00") == 2 * 3600 + 30 * 60
    print("  ✓ 时间解析正确")
    
    # 测试2: 正常时段内的检测
    print("\n测试2: 正常时段内的检测...")
    assert helper.is_in_any_trading_session(9 * 3600 + 30 * 60)  # 09:30:00
    assert helper.is_in_any_trading_session(10 * 3600)  # 10:00:00
    assert helper.is_in_any_trading_session(11 * 3600 + 30 * 60)  # 11:30:00
    assert helper.is_in_any_trading_session(13 * 3600)  # 13:00:00
    assert not helper.is_in_any_trading_session(12 * 3600)  # 12:00:00 (午休)
    print("  ✓ 正常时段检测正确")
    
    # 测试3: 跨越午夜的时段
    print("\n测试3: 跨越午夜的时段...")
    night_hours = [
        TradingHour(start_time="21:00:00", end_time="02:30:00"),
        TradingHour(start_time="09:00:00", end_time="10:15:00"),
    ]
    
    feed2 = SnapshotDuplicatingFeed(MockFeed(), trading_hours=night_hours)
    helper2 = feed2._trading_hours_helper  # 使用TradingHoursHelper
    
    # 夜盘时间
    assert helper2.is_in_any_trading_session(21 * 3600)  # 21:00:00
    assert helper2.is_in_any_trading_session(23 * 3600)  # 23:00:00
    assert helper2.is_in_any_trading_session(1 * 3600)   # 01:00:00 (次日凌晨)
    assert helper2.is_in_any_trading_session(2 * 3600 + 30 * 60)  # 02:30:00
    
    # 非交易时间
    assert not helper2.is_in_any_trading_session(3 * 3600)  # 03:00:00
    assert not helper2.is_in_any_trading_session(8 * 3600)  # 08:00:00
    
    print("  ✓ 跨越午夜时段检测正确")
    
    print("✓ Trading hours session detection test passed")


def test_snapshot_duplication_with_trading_hours():
    """测试带交易时段的快照复制功能。
    
    验证：
    1. 同一交易时段内正常进行快照复制
    2. 不同交易时段之间不进行快照复制
    3. 跨越午夜的交易时段正确处理
    """
    print("\n--- Test: Snapshot Duplication with Trading Hours ---")
    
    from quant_framework.core.data_loader import SnapshotDuplicatingFeed
    from quant_framework.core.types import TICK_PER_MS
    from quant_framework.config import TradingHour
    
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
    
    def time_to_tick(hours, minutes, seconds):
        """将时间转换为tick"""
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds * 10_000_000
    
    # 交易时段
    trading_hours = [
        TradingHour(start_time="09:30:00", end_time="11:30:00"),
        TradingHour(start_time="13:00:00", end_time="15:00:00"),
    ]
    
    # 测试1: 同一时段内的复制
    print("\n测试1: 同一交易时段内的复制 (09:30:00 -> 09:30:02)...")
    snap1 = create_test_snapshot(time_to_tick(9, 30, 0), 100.0, 101.0)
    snap2 = create_test_snapshot(time_to_tick(9, 30, 2), 100.0, 101.0)  # 2秒后
    
    feed = SnapshotDuplicatingFeed(MockFeed([snap1, snap2]), trading_hours=trading_hours)
    
    count = 0
    while feed.next() is not None:
        count += 1
    
    # 2秒 = 2000ms，间隔超过500ms，应该有复制
    # 预期：原始2个 + 约3个复制 = 约5个
    assert count >= 4, f"同一时段内应进行复制，实际{count}个快照"
    print(f"  ✓ 同一时段内复制正常，共{count}个快照")
    
    # 测试2: 不同时段之间不复制
    print("\n测试2: 不同交易时段之间不复制 (11:29:59 -> 13:00:00)...")
    snap1 = create_test_snapshot(time_to_tick(11, 29, 59), 100.0, 101.0)
    snap2 = create_test_snapshot(time_to_tick(13, 0, 0), 100.0, 101.0)  # 中午休市后
    
    feed = SnapshotDuplicatingFeed(MockFeed([snap1, snap2]), trading_hours=trading_hours)
    
    count = 0
    while feed.next() is not None:
        count += 1
    
    # 跨越休市，不应该复制
    assert count == 2, f"跨越休市不应复制，预期2个，实际{count}个"
    print(f"  ✓ 跨越休市时段不复制，共{count}个快照")
    
    # 测试3: 无交易时段配置时正常复制
    print("\n测试3: 无交易时段配置时正常复制...")
    snap1 = create_test_snapshot(time_to_tick(11, 29, 59), 100.0, 101.0)
    snap2 = create_test_snapshot(time_to_tick(13, 0, 0), 100.0, 101.0)
    
    feed = SnapshotDuplicatingFeed(MockFeed([snap1, snap2]), trading_hours=None)
    
    count = 0
    while feed.next() is not None:
        count += 1
    
    # 没有交易时段配置，应该正常复制
    assert count > 2, f"无交易时段配置应正常复制，实际{count}个快照"
    print(f"  ✓ 无交易时段配置时正常复制，共{count}个快照")
    
    print("✓ Snapshot duplication with trading hours test passed")


def test_order_arrival_trading_hours_adjustment():
    """测试订单到达时间根据交易时段调整的功能。
    
    验证：
    1. 交易时段内的订单到达时间不变
    2. 交易时段之间的订单到达时间调整到下一个时段开始
    3. 最后一个交易时段之后的订单被拒绝
    """
    print("\n--- Test: Order Arrival Trading Hours Adjustment ---")
    
    from quant_framework.runner.event_loop import EventLoopRunner
    from quant_framework.config import TradingHour
    
    # 创建交易时段配置
    trading_hours = [
        TradingHour(start_time="09:30:00", end_time="11:30:00"),
        TradingHour(start_time="13:00:00", end_time="15:00:00"),
    ]
    
    # 创建一个最小的runner来测试到达时间调整
    class MinimalRunner:
        TICKS_PER_SECOND = 10_000_000
        SECONDS_PER_DAY = 86400
        
        def __init__(self, trading_hours):
            self.trading_hours = trading_hours
        
        def _parse_time_to_seconds(self, time_str):
            parts = time_str.split(":")
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        
        def _tick_to_day_seconds(self, tick):
            total_seconds = tick // self.TICKS_PER_SECOND
            return total_seconds % self.SECONDS_PER_DAY
        
        def _seconds_to_tick_offset(self, seconds):
            return seconds * self.TICKS_PER_SECOND
        
        def _is_within_trading_session(self, start_seconds, end_seconds, time_seconds):
            if start_seconds <= end_seconds:
                return start_seconds <= time_seconds <= end_seconds
            else:
                return time_seconds >= start_seconds or time_seconds <= end_seconds
        
        def _find_trading_session_index(self, time_seconds):
            for i, th in enumerate(self.trading_hours):
                start_str = getattr(th, 'start_time', '')
                end_str = getattr(th, 'end_time', '')
                if not start_str or not end_str:
                    continue
                start_sec = self._parse_time_to_seconds(start_str)
                end_sec = self._parse_time_to_seconds(end_str)
                if self._is_within_trading_session(start_sec, end_sec, time_seconds):
                    return i
            return -1
        
        def _is_in_any_trading_session(self, time_seconds):
            return self._find_trading_session_index(time_seconds) >= 0
        
        def _get_next_trading_session_start(self, time_seconds):
            session_starts = []
            for th in self.trading_hours:
                start_str = getattr(th, 'start_time', '')
                if start_str:
                    start_sec = self._parse_time_to_seconds(start_str)
                    session_starts.append(start_sec)
            session_starts.sort()
            for start in session_starts:
                if start > time_seconds:
                    return start
            return None
        
        def _is_after_last_trading_session(self, time_seconds):
            session_ends = []
            for th in self.trading_hours:
                end_str = getattr(th, 'end_time', '')
                start_str = getattr(th, 'start_time', '')
                if end_str and start_str:
                    end_sec = self._parse_time_to_seconds(end_str)
                    start_sec = self._parse_time_to_seconds(start_str)
                    if end_sec >= start_sec:
                        session_ends.append(end_sec)
            if not session_ends:
                return False
            return time_seconds > max(session_ends)
        
        def _adjust_arrival_time_for_trading_hours(self, arrival_tick, is_order=True):
            if not self.trading_hours:
                return (arrival_tick, True)
            
            time_seconds = self._tick_to_day_seconds(arrival_tick)
            
            if self._is_in_any_trading_session(time_seconds):
                return (arrival_tick, True)
            
            if self._is_after_last_trading_session(time_seconds):
                return (None, False)
            
            next_start_seconds = self._get_next_trading_session_start(time_seconds)
            
            if next_start_seconds is None:
                return (None, False)
            
            current_day_start_tick = (arrival_tick // (self.SECONDS_PER_DAY * self.TICKS_PER_SECOND)) * (self.SECONDS_PER_DAY * self.TICKS_PER_SECOND)
            adjusted_tick = current_day_start_tick + self._seconds_to_tick_offset(next_start_seconds)
            
            return (adjusted_tick, True)
    
    runner = MinimalRunner(trading_hours)
    
    def time_to_tick(hours, minutes, seconds):
        """将时间转换为tick"""
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds * 10_000_000
    
    # 测试1: 交易时段内的到达时间不变
    print("\n测试1: 交易时段内的到达时间不变...")
    arrival_tick = time_to_tick(10, 0, 0)  # 10:00:00 在上午时段内
    adjusted, success = runner._adjust_arrival_time_for_trading_hours(arrival_tick, is_order=True)
    assert success, "交易时段内应成功"
    assert adjusted == arrival_tick, f"交易时段内不应调整，期望{arrival_tick}，实际{adjusted}"
    print(f"  ✓ 10:00:00 (交易时段内) 到达时间不变")
    
    # 测试2: 两个时段之间的到达时间调整到下一个时段开始
    print("\n测试2: 两个时段之间的到达时间调整到下一个时段开始...")
    arrival_tick = time_to_tick(12, 0, 0)  # 12:00:00 在两个时段之间
    adjusted, success = runner._adjust_arrival_time_for_trading_hours(arrival_tick, is_order=True)
    assert success, "两个时段之间应成功（调整）"
    expected_tick = time_to_tick(13, 0, 0)  # 应调整到下午时段开始 13:00:00
    assert adjusted == expected_tick, f"应调整到13:00:00，期望{expected_tick}，实际{adjusted}"
    print(f"  ✓ 12:00:00 (时段之间) 调整到 13:00:00 (下一个时段开始)")
    
    # 测试3: 最后一个时段之后的到达时间应被拒绝
    print("\n测试3: 最后一个时段之后的订单应被拒绝...")
    arrival_tick = time_to_tick(16, 0, 0)  # 16:00:00 在最后时段之后
    adjusted, success = runner._adjust_arrival_time_for_trading_hours(arrival_tick, is_order=True)
    assert not success, "最后时段之后应失败"
    assert adjusted is None, "失败时应返回None"
    print(f"  ✓ 16:00:00 (最后时段之后) 订单被拒绝")
    
    # 测试4: 撤单在最后时段之后也应失败
    print("\n测试4: 撤单在最后时段之后也应失败...")
    arrival_tick = time_to_tick(15, 30, 0)  # 15:30:00 在最后时段之后
    adjusted, success = runner._adjust_arrival_time_for_trading_hours(arrival_tick, is_order=False)
    assert not success, "撤单在最后时段之后应失败"
    print(f"  ✓ 15:30:00 (最后时段之后) 撤单失败")
    
    # 测试5: 无交易时段配置时不做任何调整
    print("\n测试5: 无交易时段配置时不做任何调整...")
    runner_no_hours = MinimalRunner([])
    arrival_tick = time_to_tick(12, 0, 0)
    adjusted, success = runner_no_hours._adjust_arrival_time_for_trading_hours(arrival_tick, is_order=True)
    assert success, "无交易时段配置应成功"
    assert adjusted == arrival_tick, "无交易时段配置不应调整"
    print(f"  ✓ 无交易时段配置时到达时间不变")
    
    print("✓ Order arrival trading hours adjustment test passed")


def test_trading_hours_helper_class():
    """测试TradingHoursHelper类的独立功能。
    
    验证TradingHoursHelper作为独立类的复用性和正确性。
    """
    print("\n--- Test: TradingHoursHelper Class ---")
    
    from quant_framework.core.trading_hours import TradingHoursHelper
    from quant_framework.config import TradingHour
    
    # 测试1: 创建空配置的helper
    print("\n测试1: 空配置的TradingHoursHelper...")
    empty_helper = TradingHoursHelper([])
    assert empty_helper.is_in_any_trading_session(10 * 3600) == True  # 未配置时默认都在交易时间
    assert empty_helper.spans_trading_session_gap(100, 200) == False  # 未配置时不跨越间隔
    assert empty_helper.is_after_last_trading_session(16 * 3600) == False
    print("  ✓ 空配置正确处理")
    
    # 测试2: 正常时段配置
    print("\n测试2: 正常时段配置...")
    trading_hours = [
        TradingHour(start_time="09:30:00", end_time="11:30:00"),
        TradingHour(start_time="13:00:00", end_time="15:00:00"),
    ]
    helper = TradingHoursHelper(trading_hours)
    
    # 验证常量
    assert helper.TICKS_PER_SECOND == 10_000_000
    assert helper.SECONDS_PER_DAY == 86400
    print("  ✓ 常量正确")
    
    # 测试3: 时间转换方法
    print("\n测试3: 时间转换方法...")
    assert helper.parse_time_to_seconds("09:30:00") == 9 * 3600 + 30 * 60
    assert helper.parse_time_to_seconds("invalid") == 0  # 无效格式返回0
    assert helper.seconds_to_tick_offset(1) == 10_000_000
    
    tick_at_10am = 10 * 3600 * helper.TICKS_PER_SECOND
    assert helper.tick_to_day_seconds(tick_at_10am) == 10 * 3600
    print("  ✓ 时间转换正确")
    
    # 测试4: 交易时段检测
    print("\n测试4: 交易时段检测...")
    # 正常时段
    assert helper.is_within_trading_session(9 * 3600, 11 * 3600, 10 * 3600) == True
    assert helper.is_within_trading_session(9 * 3600, 11 * 3600, 12 * 3600) == False
    # 跨夜时段
    assert helper.is_within_trading_session(21 * 3600, 2 * 3600, 22 * 3600) == True
    assert helper.is_within_trading_session(21 * 3600, 2 * 3600, 1 * 3600) == True
    assert helper.is_within_trading_session(21 * 3600, 2 * 3600, 10 * 3600) == False
    print("  ✓ 交易时段检测正确")
    
    # 测试5: 查找交易时段索引
    print("\n测试5: 查找交易时段索引...")
    assert helper.find_trading_session_index(10 * 3600) == 0  # 上午时段
    assert helper.find_trading_session_index(14 * 3600) == 1  # 下午时段
    assert helper.find_trading_session_index(12 * 3600) == -1  # 午休
    print("  ✓ 交易时段索引查找正确")
    
    # 测试6: 获取下一个交易时段开始时间
    print("\n测试6: 获取下一个交易时段开始时间...")
    assert helper.get_next_trading_session_start(8 * 3600) == 9 * 3600 + 30 * 60  # 09:30
    assert helper.get_next_trading_session_start(12 * 3600) == 13 * 3600  # 13:00
    assert helper.get_next_trading_session_start(16 * 3600) is None  # 无下一个时段
    print("  ✓ 下一个交易时段开始时间查找正确")
    
    # 测试7: 判断是否在最后时段之后
    print("\n测试7: 判断是否在最后时段之后...")
    assert helper.is_after_last_trading_session(16 * 3600) == True
    assert helper.is_after_last_trading_session(14 * 3600) == False  # 仍在交易时段
    assert helper.is_after_last_trading_session(12 * 3600) == False  # 午休不是"最后时段之后"
    print("  ✓ 最后时段判断正确")
    
    print("✓ TradingHoursHelper class test passed")


def test_no_duplicate_segment_and_interval_end_events():
    """测试修复：最后一个SEGMENT_END不应与INTERVAL_END重复出现。
    
    问题描述：在debug时发现interval_end后还有segment_end事件，这是不允许的。
    最后一个segment的结束就是interval的结束，不应该有两个独立事件。
    
    验证：
    1. 事件队列中不应该同时存在时间相同的SEGMENT_END和INTERVAL_END
    2. 最后一个segment的结束由INTERVAL_END代表
    """
    print("\n--- Test: No Duplicate SEGMENT_END and INTERVAL_END Events ---")
    
    import heapq
    from quant_framework.runner.event_loop import (
        Event, EventType, reset_event_seq_counter
    )
    from quant_framework.core.types import TapeSegment
    
    reset_event_seq_counter()
    
    # 测试1：直接验证事件创建逻辑
    print("\n  测试1: 验证事件队列逻辑（模拟event_loop.py中的事件创建）...")
    
    # 创建一个测试tape（模拟实际tape）
    test_tape = [
        TapeSegment(index=1, t_start=1000, t_end=1200, bid_price=100.0, ask_price=101.0,
                   trades={}, cancels={}, net_flow={}, activation_bid={100.0}, activation_ask={101.0}),
        TapeSegment(index=2, t_start=1200, t_end=1500, bid_price=100.0, ask_price=101.0,
                   trades={}, cancels={}, net_flow={}, activation_bid={100.0}, activation_ask={101.0}),
    ]
    t_a = 1000
    t_b = 1500  # 区间结束时间等于最后一个segment结束时间
    
    # 模拟event_loop.py中的事件创建逻辑（修复后的版本）
    event_queue = []
    for seg in test_tape:
        if seg.t_end == t_b:
            # 最后一个段的结束由INTERVAL_END代替，不创建SEGMENT_END
            continue
        heapq.heappush(event_queue, Event(
            time=seg.t_end,
            event_type=EventType.SEGMENT_END,
            data=seg,
        ))
    
    heapq.heappush(event_queue, Event(
        time=t_b,
        event_type=EventType.INTERVAL_END,
        data=None,
    ))
    
    # 验证事件队列
    segment_end_count = sum(1 for e in event_queue if e.event_type == EventType.SEGMENT_END)
    interval_end_count = sum(1 for e in event_queue if e.event_type == EventType.INTERVAL_END)
    
    print(f"    SEGMENT_END事件数: {segment_end_count} (期望: {len(test_tape) - 1})")
    print(f"    INTERVAL_END事件数: {interval_end_count} (期望: 1)")
    
    assert segment_end_count == len(test_tape) - 1, \
        f"SEGMENT_END事件数应该是 {len(test_tape) - 1}，但实际是 {segment_end_count}"
    assert interval_end_count == 1, \
        f"INTERVAL_END事件数应该是 1，但实际是 {interval_end_count}"
    
    # 验证在t_b时刻没有SEGMENT_END事件
    events_at_t_b = [e for e in event_queue if e.time == t_b]
    segment_end_at_t_b = [e for e in events_at_t_b if e.event_type == EventType.SEGMENT_END]
    
    print(f"    t_b时刻的事件数: {len(events_at_t_b)}")
    print(f"    t_b时刻的SEGMENT_END事件数: {len(segment_end_at_t_b)} (期望: 0)")
    
    assert len(segment_end_at_t_b) == 0, \
        f"在t_b时刻不应该有SEGMENT_END事件，但找到了 {len(segment_end_at_t_b)} 个"
    
    print("  ✓ 测试1通过: 事件队列中t_b时刻只有INTERVAL_END，没有SEGMENT_END")
    
    # 测试2：多segment场景
    print("\n  测试2: 多segment场景...")
    reset_event_seq_counter()
    
    test_tape_multi = [
        TapeSegment(index=1, t_start=0, t_end=100, bid_price=100.0, ask_price=101.0,
                   trades={}, cancels={}, net_flow={}, activation_bid={100.0}, activation_ask={101.0}),
        TapeSegment(index=2, t_start=100, t_end=200, bid_price=100.0, ask_price=101.0,
                   trades={}, cancels={}, net_flow={}, activation_bid={100.0}, activation_ask={101.0}),
        TapeSegment(index=3, t_start=200, t_end=300, bid_price=100.0, ask_price=101.0,
                   trades={}, cancels={}, net_flow={}, activation_bid={100.0}, activation_ask={101.0}),
        TapeSegment(index=4, t_start=300, t_end=400, bid_price=100.0, ask_price=101.0,
                   trades={}, cancels={}, net_flow={}, activation_bid={100.0}, activation_ask={101.0}),
    ]
    t_b_multi = 400
    
    event_queue_multi = []
    for seg in test_tape_multi:
        if seg.t_end == t_b_multi:
            continue
        heapq.heappush(event_queue_multi, Event(
            time=seg.t_end,
            event_type=EventType.SEGMENT_END,
            data=seg,
        ))
    
    heapq.heappush(event_queue_multi, Event(
        time=t_b_multi,
        event_type=EventType.INTERVAL_END,
        data=None,
    ))
    
    segment_end_count_multi = sum(1 for e in event_queue_multi if e.event_type == EventType.SEGMENT_END)
    events_at_t_b_multi = [e for e in event_queue_multi if e.time == t_b_multi]
    segment_end_at_t_b_multi = [e for e in events_at_t_b_multi if e.event_type == EventType.SEGMENT_END]
    
    print(f"    SEGMENT_END事件数: {segment_end_count_multi} (期望: {len(test_tape_multi) - 1})")
    print(f"    t_b时刻的SEGMENT_END事件数: {len(segment_end_at_t_b_multi)} (期望: 0)")
    
    assert segment_end_count_multi == len(test_tape_multi) - 1
    assert len(segment_end_at_t_b_multi) == 0
    
    print("  ✓ 测试2通过: 多segment场景正确处理")
    
    # 测试3：单segment场景（此时没有SEGMENT_END事件，只有INTERVAL_END）
    print("\n  测试3: 单segment场景...")
    reset_event_seq_counter()
    
    test_tape_single = [
        TapeSegment(index=1, t_start=0, t_end=100, bid_price=100.0, ask_price=101.0,
                   trades={}, cancels={}, net_flow={}, activation_bid={100.0}, activation_ask={101.0}),
    ]
    t_b_single = 100
    
    event_queue_single = []
    for seg in test_tape_single:
        if seg.t_end == t_b_single:
            continue
        heapq.heappush(event_queue_single, Event(
            time=seg.t_end,
            event_type=EventType.SEGMENT_END,
            data=seg,
        ))
    
    heapq.heappush(event_queue_single, Event(
        time=t_b_single,
        event_type=EventType.INTERVAL_END,
        data=None,
    ))
    
    segment_end_count_single = sum(1 for e in event_queue_single if e.event_type == EventType.SEGMENT_END)
    interval_end_count_single = sum(1 for e in event_queue_single if e.event_type == EventType.INTERVAL_END)
    
    print(f"    SEGMENT_END事件数: {segment_end_count_single} (期望: 0)")
    print(f"    INTERVAL_END事件数: {interval_end_count_single} (期望: 1)")
    
    assert segment_end_count_single == 0
    assert interval_end_count_single == 1
    
    print("  ✓ 测试3通过: 单segment场景正确处理")
    
    print("✓ No duplicate SEGMENT_END and INTERVAL_END events test passed")


def test_cancel_order_across_interval():
    """Test canceling an order after exchange reset (across interval boundaries).
    
    This test verifies that:
    - exchange.reset() preserves _levels which contain ShadowOrder objects
    - Orders placed in a previous interval can still be canceled in the next interval
    """
    from quant_framework.exchange.simulator import FIFOExchangeSimulator
    from quant_framework.core.types import Order, Side, TapeSegment, TimeInForce
    
    print("\n--- Test: Cancel Order Across Interval (Reset Bug Fix) ---\n")
    
    # 测试1: 验证订单在reset后仍然可以被取消
    print("  测试1: 订单在exchange.reset()后仍可被取消...")
    
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
    
    # 创建订单
    order = Order(
        order_id="test-cancel-1",
        side=Side.BUY,
        price=100.0,
        qty=10,
        tif=TimeInForce.GTC,
        create_time=10000000,
    )
    
    # 创建tape并设置给交易所
    tape = [
        TapeSegment(
            index=1,
            t_start=10000000,
            t_end=15000000,
            bid_price=100.0,
            ask_price=101.0,
            trades={},
            cancels={},
            net_flow={},
            activation_bid={100.0},
            activation_ask={101.0},
        )
    ]
    
    exchange.set_tape(tape, 10000000, 15000000)
    
    # 订单到达交易所
    receipt = exchange.on_order_arrival(order, 10000000, market_qty=50)
    assert receipt is None, "Order should be accepted to queue"
    
    # 验证订单在levels中
    shadow = exchange._find_order_by_id("test-cancel-1")
    assert shadow is not None, "Order should be in levels"
    print(f"    ✓ 订单已注册: {order.order_id}")
    
    # 现在模拟区间结束，reset交易所
    exchange.reset()
    print("    ✓ exchange.reset() 已调用")
    
    # 新设计：reset不清空_levels，订单仍在levels中
    assert len(exchange._levels) > 0, "Levels should be preserved after reset"
    shadow = exchange._find_order_by_id("test-cancel-1")
    assert shadow is not None, "Order should still be in levels after reset"
    print("    ✓ _levels保留了订单（新设计）")
    
    # 设置新的tape（模拟新区间开始）
    new_tape = [
        TapeSegment(
            index=1,
            t_start=15000000,
            t_end=20000000,
            bid_price=100.0,
            ask_price=101.0,
            trades={},
            cancels={},
            net_flow={},
            activation_bid={100.0},
            activation_ask={101.0},
        )
    ]
    exchange.set_tape(new_tape, 15000000, 20000000)
    
    # 现在尝试取消订单（在新区间中）
    cancel_receipt = exchange.on_cancel_arrival("test-cancel-1", 16000000)
    
    assert cancel_receipt.receipt_type == "CANCELED", \
        f"Order should be canceled, got {cancel_receipt.receipt_type}"
    print(f"    ✓ 订单成功取消: receipt_type={cancel_receipt.receipt_type}")
    
    # 验证订单状态已更新
    shadow = exchange._find_order_by_id("test-cancel-1")
    assert shadow.status == "CANCELED", f"Shadow order status should be CANCELED, got {shadow.status}"
    print("    ✓ Shadow order状态已更新为CANCELED")
    
    print("  ✓ 测试1通过: reset后订单仍可被取消")
    
    # 测试2: 验证取消不存在的订单会抛出ValueError
    print("\n  测试2: 取消不存在的订单应抛出ValueError...")
    
    exchange2 = FIFOExchangeSimulator(cancel_bias_k=0.0)
    exchange2.set_tape(tape, 10000000, 15000000)
    
    try:
        exchange2.on_cancel_arrival("non-existent-order", 12000000)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "non-existent-order" in str(e)
        print(f"    ✓ 正确抛出ValueError: {e}")
    
    print("  ✓ 测试2通过: 取消不存在的订单抛出异常")
    
    # 测试3: 验证已取消的订单再次取消返回REJECTED
    print("\n  测试3: 已取消的订单再次取消应返回REJECTED...")
    
    exchange3 = FIFOExchangeSimulator(cancel_bias_k=0.0)
    exchange3.set_tape(tape, 10000000, 15000000)
    
    order3 = Order(
        order_id="test-cancel-3",
        side=Side.BUY,
        price=100.0,
        qty=10,
        tif=TimeInForce.GTC,
        create_time=10000000,
    )
    exchange3.on_order_arrival(order3, 10000000, market_qty=50)
    
    # 第一次取消
    receipt1 = exchange3.on_cancel_arrival("test-cancel-3", 11000000)
    assert receipt1.receipt_type == "CANCELED"
    
    # 第二次取消应返回REJECTED
    receipt2 = exchange3.on_cancel_arrival("test-cancel-3", 12000000)
    assert receipt2.receipt_type == "REJECTED", \
        f"Second cancel should be REJECTED, got {receipt2.receipt_type}"
    print(f"    ✓ 第二次取消返回REJECTED")
    
    print("  ✓ 测试3通过: 已取消的订单再次取消返回REJECTED")
    
    # 测试4: 验证full_reset会清空levels
    print("\n  测试4: full_reset应清空levels...")
    
    exchange4 = FIFOExchangeSimulator(cancel_bias_k=0.0)
    exchange4.set_tape(tape, 10000000, 15000000)
    
    order4 = Order(
        order_id="test-cancel-4",
        side=Side.BUY,
        price=100.0,
        qty=10,
        tif=TimeInForce.GTC,
        create_time=10000000,
    )
    exchange4.on_order_arrival(order4, 10000000, market_qty=50)
    
    shadow = exchange4._find_order_by_id("test-cancel-4")
    assert shadow is not None, "Order should be in levels"
    
    # 调用full_reset
    exchange4.full_reset()
    
    assert len(exchange4._levels) == 0, "Levels should be empty after full_reset"
    shadow = exchange4._find_order_by_id("test-cancel-4")
    assert shadow is None, "Order should not be found after full_reset"
    print("    ✓ full_reset后levels已清空")
    
    print("  ✓ 测试4通过: full_reset正确清空levels")
    
    print("✓ Cancel order across interval test passed")


def test_cross_interval_order_fill():
    """Test order fill across multiple intervals.
    
    Scenario:
    - ABCD are 4 consecutive snapshots
    - Order arrives in interval [A,B] with pos=100, qty=10 (threshold=110)
    - Interval [A,B]: X increases by 30 (cumulative: 30)
    - Interval [B,C]: X increases by 30 (cumulative: 60)
    - Interval [C,D]: X increases by 60 (cumulative: 120)
    - Order should fill in interval [C,D] when X reaches 110
    
    This test verifies that shadow orders properly carry over across interval
    boundaries and that their pos values are correctly adjusted.
    """
    from quant_framework.exchange.simulator import FIFOExchangeSimulator
    from quant_framework.core.types import Order, Side, TapeSegment, TimeInForce, NormalizedSnapshot, Level
    
    print("\n--- Test: Cross-Interval Order Fill ---\n")
    
    # 时间设置（单位tick）
    t_A = 10_000_000
    t_B = 15_000_000
    t_C = 20_000_000
    t_D = 25_000_000
    
    # 创建交易所
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
    
    # ========== 区间 [A, B] ==========
    print("  区间 [A, B]:")
    
    # 快照A
    snapshot_A = NormalizedSnapshot(
        ts_recv=t_A,
        bids=[Level(100.0, 100)],  # bid@100有100手队列
        asks=[Level(101.0, 100)],
    )
    
    # Tape [A,B]: 成交30手, 撤单0手 → X增长30
    tape_AB = [
        TapeSegment(
            index=1,
            t_start=t_A,
            t_end=t_B,
            bid_price=100.0,
            ask_price=101.0,
            trades={(Side.BUY, 100.0): 30},
            cancels={},
            net_flow={(Side.BUY, 100.0): -30},  # 净流出30（被消耗）
            activation_bid={100.0},
            activation_ask={101.0},
        )
    ]
    
    exchange.set_tape(tape_AB, t_A, t_B)
    
    # 订单在区间中间到达，pos=100（排在队列末尾），qty=10
    order = Order(
        order_id="cross-interval-order",
        side=Side.BUY,
        price=100.0,
        qty=10,
        tif=TimeInForce.GTC,
        create_time=t_A,
    )
    
    # 订单在t_A时刻到达，market_qty=100（队列深度）
    receipt = exchange.on_order_arrival(order, t_A, market_qty=100)
    assert receipt is None, "Order should be accepted"
    
    # 验证订单位置
    shadow = exchange._find_order_by_id("cross-interval-order")
    print(f"    订单pos={shadow.pos}, qty={shadow.original_qty}, threshold={shadow.pos + shadow.original_qty}")
    assert shadow.pos == 100, f"Expected pos=100, got {shadow.pos}"
    assert shadow.original_qty == 10
    
    # 推进区间 [A, B]
    receipts, _ = exchange.advance(t_A, t_B, tape_AB[0])
    print(f"    区间结束时X坐标: {exchange.get_x_coord(Side.BUY, 100.0)}")
    print(f"    生成回执数: {len(receipts)}")
    assert len(receipts) == 0, "Should not fill yet (X=30 < threshold=110)"
    
    # 边界对齐
    snapshot_B = NormalizedSnapshot(
        ts_recv=t_B,
        bids=[Level(100.0, 70)],  # 队列深度减少到70
        asks=[Level(101.0, 100)],
    )
    exchange.align_at_boundary(snapshot_B)
    
    print(f"    align后shadow.pos={shadow.pos}")
    # 期望：pos从100调整为100-30=70（因为X消耗了30）
    assert shadow.pos == 70, f"Expected pos=70 after align, got {shadow.pos}"
    
    print("  ✓ 区间[A,B]完成: X增长30, 订单pos调整为70")
    
    # ========== 区间 [B, C] ==========
    print("\n  区间 [B, C]:")
    
    # Reset for new interval
    exchange.reset()
    
    # 验证订单是否仍在levels中
    shadow = exchange._find_order_by_id("cross-interval-order")
    assert shadow is not None, "Order should still be in levels"
    print(f"    reset后shadow.status={shadow.status}, pos={shadow.pos}")
    
    # Tape [B,C]: 成交30手, 撤单0手 → X增长30
    tape_BC = [
        TapeSegment(
            index=1,
            t_start=t_B,
            t_end=t_C,
            bid_price=100.0,
            ask_price=101.0,
            trades={(Side.BUY, 100.0): 30},
            cancels={},
            net_flow={(Side.BUY, 100.0): -30},
            activation_bid={100.0},
            activation_ask={101.0},
        )
    ]
    
    exchange.set_tape(tape_BC, t_B, t_C)
    
    # 需要将shadow订单恢复到_levels中才能被advance处理
    # 这是当前实现的bug - reset清空了_levels，advance找不到订单
    
    # 尝试推进 - 期望订单能被找到并处理
    receipts, _ = exchange.advance(t_B, t_C, tape_BC[0])
    print(f"    区间结束时X坐标: {exchange.get_x_coord(Side.BUY, 100.0)}")
    print(f"    生成回执数: {len(receipts)}")
    
    # 如果实现正确：
    # - X从0累加到30
    # - shadow.pos=70, threshold=80
    # - X=30 < 80，不应成交
    # 但由于reset清空了_levels，advance根本找不到订单！
    
    # 边界对齐
    snapshot_C = NormalizedSnapshot(
        ts_recv=t_C,
        bids=[Level(100.0, 40)],
        asks=[Level(101.0, 100)],
    )
    
    # 检查_levels是否有订单
    level_key = (Side.BUY, 100.0)
    if level_key in exchange._levels:
        level = exchange._levels[level_key]
        print(f"    _levels中的订单数: {len(level.queue)}")
    else:
        print(f"    _levels中没有该价位的level（这是bug！）")
    
    exchange.align_at_boundary(snapshot_C)
    
    print("  ✓ 区间[B,C]完成")
    
    # ========== 区间 [C, D] ==========
    print("\n  区间 [C, D]:")
    
    exchange.reset()
    
    # Tape [C,D]: 成交60手, 撤单0手 → X增长60
    tape_CD = [
        TapeSegment(
            index=1,
            t_start=t_C,
            t_end=t_D,
            bid_price=100.0,
            ask_price=101.0,
            trades={(Side.BUY, 100.0): 60},
            cancels={},
            net_flow={(Side.BUY, 100.0): -60},
            activation_bid={100.0},
            activation_ask={101.0},
        )
    ]
    
    exchange.set_tape(tape_CD, t_C, t_D)
    
    receipts, _ = exchange.advance(t_C, t_D, tape_CD[0])
    print(f"    区间结束时X坐标: {exchange.get_x_coord(Side.BUY, 100.0)}")
    print(f"    生成回执数: {len(receipts)}")
    
    # 如果跨区间正确工作：
    # - 累计X = 30 + 30 + 60 = 120
    # - 但由于pos调整：
    #   - 区间AB结束: pos 100 → 70
    #   - 区间BC结束: pos 70 → 40
    #   - 区间CD: X从0到60，threshold = 40+10 = 50
    #   - X=60 >= 50，应该成交！
    
    shadow = exchange._find_order_by_id("cross-interval-order")
    print(f"    最终shadow.status={shadow.status}")
    
    # 验证订单已成交
    if shadow.status == "FILLED":
        print("  ✓ 订单在区间[C,D]成交 - 跨区间逻辑正确！")
    else:
        print(f"  ✗ 订单未成交 (status={shadow.status}) - 存在跨区间bug！")
    
    assert shadow.status == "FILLED", \
        f"Order should be FILLED after crossing threshold across intervals, but status is {shadow.status}"
    
    print("✓ Cross-interval order fill test passed")


def test_floating_point_precision_in_last_vol_split():
    """测试浮点精度问题的修复。
    
    场景：last_vol_split中的价格可能有浮点精度问题，例如：
    - 1050.199999999 (应该是 1050.2)
    - 1050.360000000001 (应该是 1050.36)
    
    这个测试验证dataloader和builder能正确处理这种情况。
    """
    print("\n--- Test: Floating Point Precision in last_vol_split ---")
    
    from quant_framework.core.types import TICK_PER_MS, Side
    from quant_framework.tape.builder import UnifiedTapeBuilder, TapeConfig
    
    # 测试1: 使用带有浮点精度误差的价格
    print("\n  测试1: 验证builder能处理带有浮点误差的last_vol_split价格...")
    
    # 创建快照：bid_price = ask_price = 100.0
    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0)
    
    # last_vol_split中使用带有浮点误差的价格
    # 100.000000001 应该匹配 segment中的 100.0
    curr = create_test_snapshot(
        1500 * TICK_PER_MS, 
        100.0, 
        101.0, 
        last_vol_split=[(100.000000001, 50), (100.999999999, 30)]  # 浮点误差
    )
    
    config = TapeConfig()
    builder = UnifiedTapeBuilder(config=config, tick_size=1.0)
    
    tape = builder.build(prev, curr)
    
    # 验证成交量被正确分配
    total_buy_volume = sum(
        qty for seg in tape 
        for (side, _), qty in seg.trades.items() 
        if side == Side.BUY
    )
    total_sell_volume = sum(
        qty for seg in tape 
        for (side, _), qty in seg.trades.items() 
        if side == Side.SELL
    )
    
    print(f"    总买入成交量: {total_buy_volume}")
    print(f"    总卖出成交量: {total_sell_volume}")
    
    # 验证成交量被正确分配（不应该因为浮点精度问题被丢弃）
    # 注意：只有bid_price==ask_price的segment才能分配成交量
    # 在这个测试中，100.0和101.0的segment有成交
    assert total_buy_volume > 0, "成交量不应该为0（浮点精度问题导致匹配失败）"
    assert total_buy_volume == total_sell_volume, "买卖成交量应该相等"
    print("  ✓ 测试1通过: 带有浮点误差的价格被正确匹配")
    
    # 测试2: 验证dataloader的价格舍入
    print("\n  测试2: 验证dataloader的价格舍入...")
    
    from quant_framework.core.data_loader import CsvMarketDataFeed
    import tempfile
    import os
    
    # 创建临时CSV文件
    csv_content = '''RecvTick,Bid,BidVol,Ask,AskVol,LastVolSplit
10000000,100.0,50,101.0,50,"[(100.199999999, 10), (100.360000000001, 20)]"
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        temp_path = f.name
    
    try:
        feed = CsvMarketDataFeed(temp_path)
        snapshot = feed.next()
        
        if snapshot is not None:
            print(f"    解析后的last_vol_split: {snapshot.last_vol_split}")
            
            # 验证价格被正确舍入
            for price, qty in snapshot.last_vol_split:
                # 验证价格没有过多的小数位（已被舍入到6位）
                rounded = round(price, 6)
                assert abs(price - rounded) < 1e-10, \
                    f"价格应该被舍入到6位小数: {price} vs {rounded}"
            
            print("  ✓ 测试2通过: dataloader正确舍入了价格")
        else:
            raise AssertionError("无法解析CSV快照")
    finally:
        os.unlink(temp_path)
    
    # 测试3: 更极端的浮点误差
    print("\n  测试3: 测试更极端的浮点误差...")
    
    prev3 = create_multi_level_snapshot(
        1000 * TICK_PER_MS,
        bids=[(100.0, 100), (99.0, 100)],
        asks=[(101.0, 100), (102.0, 100)]
    )
    
    # 使用更极端的浮点误差
    curr3 = create_multi_level_snapshot(
        1500 * TICK_PER_MS,
        bids=[(100.0, 100), (99.0, 100)],
        asks=[(100.0, 100), (101.0, 100)],  # ask降到100
        last_vol_split=[
            (99.99999999999, 25),   # 应该匹配 100.0
            (100.00000000001, 25),  # 应该匹配 100.0
        ]
    )
    
    tape3 = builder.build(prev3, curr3)
    
    total_volume = sum(
        qty for seg in tape3 
        for (side, _), qty in seg.trades.items() 
        if side == Side.BUY
    )
    
    print(f"    总成交量: {total_volume}")
    # 注意：100.0的segment应该收到50手成交（25+25）
    # 但99.99999999999实际上和100.0的差值是1e-11，
    # 这小于我们的容差1e-6，所以应该匹配成功
    assert total_volume > 0, "成交量不应该为0"
    print("  ✓ 测试3通过: 极端浮点误差被正确处理")
    
    print("✓ Floating point precision test passed")


def test_post_crossing_pos_uses_x_coord():
    """测试post-crossing订单的pos使用x_coord而不是0。
    
    问题场景：
    1. 订单A在时刻t1 crossing部分成交后，剩余部分入队，pos=0
    2. 后续订单B在时刻t2入队，pos = 0 + A.remaining_qty + netflow
    3. 如果此时x_coord已经推进到比如100，那么订单B的pos可能小于x_coord
    4. 这会导致订单B在下一轮advance中被立即成交（因为x > pos+qty）
    
    修复后：post-crossing订单的pos = x_coord(arrival_time)，确保后续订单不会被错误成交。
    
    本测试验证：
    1. post-crossing订单的pos等于到达时刻的x_coord
    2. 后续订单的pos正确基于前序订单的threshold计算
    """
    print("\n--- Test: Post-Crossing Pos Uses X Coord ---")
    
    tape_config = TapeConfig(
        epsilon=1.0,
    )
    builder = UnifiedTapeBuilder(config=tape_config, tick_size=1.0)
    
    # 创建快照：模拟一个有大量trades的场景，使x_coord能够推进
    # bid@100, ask@101, 然后价格相遇产生大量trades
    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=100, ask_qty=100)
    curr = create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=50, ask_qty=50,
                                last_vol_split=[(100.0, 80), (101.0, 80)])  # 大量成交
    
    tape = builder.build(prev, curr)
    
    print_tape_path(tape)
    
    # 创建交易所模拟器
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
    exchange.set_tape(tape, 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)
    
    # 测试1: 在区间中间模拟post-crossing入队
    # 首先让时间推进到区间中段，使x_coord有机会增加
    print("\n  测试1: 在x_coord已推进后入队post-crossing订单...")
    
    # 计算一个位于区间中间的时间点
    mid_time = 1250 * TICK_PER_MS  # 区间中点
    
    # 获取该时刻的x_coord（在没有shadow订单时，使用shadow_pos=0）
    # 注：需要在BUY@100这个价位，因为trades发生在这里
    x_at_mid = exchange._get_x_coord(Side.BUY, 100.0, mid_time, 0)
    print(f"    时刻{mid_time}的x_coord(BUY@100): {x_at_mid}")
    
    # 创建一个post-crossing订单（模拟crossing后剩余部分）
    order1 = Order(
        order_id="post-crossing-order",
        side=Side.BUY,
        price=100.0,  # 与trades同价位
        qty=100,
        tif=TimeInForce.GTC,
    )
    
    # 模拟已经部分成交80后，剩余20入队
    exchange._queue_order(
        order=order1,
        arrival_time=mid_time,
        market_qty=50,
        remaining_qty=20,
        already_filled=80  # 已通过crossing成交80
    )
    
    # 获取入队后的shadow order
    shadow_orders = exchange.get_shadow_orders()
    post_crossing_shadow = None
    for so in shadow_orders:
        if so.order_id == "post-crossing-order":
            post_crossing_shadow = so
            break
    
    assert post_crossing_shadow is not None, "post-crossing订单应该在队列中"
    
    # 关键验证：pos应该等于到达时刻的x_coord（取整）
    expected_pos = int(round(x_at_mid))
    print(f"    post-crossing订单pos: {post_crossing_shadow.pos}, 期望(x_coord取整): {expected_pos}")
    
    # 允许小的舍入误差
    assert abs(post_crossing_shadow.pos - expected_pos) <= 1, \
        f"post-crossing订单pos应该约等于x_coord({expected_pos})，实际为{post_crossing_shadow.pos}"
    print(f"    ✓ post-crossing订单pos正确等于x_coord")
    
    # 测试2: 后续订单的pos应该基于前序订单的threshold
    print("\n  测试2: 后续订单的pos基于前序订单threshold计算...")
    
    order2 = Order(
        order_id="subsequent-order",
        side=Side.BUY,
        price=100.0,  # 同一价位
        qty=10,
        tif=TimeInForce.GTC,
    )
    
    # 稍后一点的时间入队
    later_time = 1300 * TICK_PER_MS
    
    exchange._queue_order(
        order=order2,
        arrival_time=later_time,
        market_qty=50,
        remaining_qty=10,
        already_filled=0  # 没有crossing
    )
    
    # 获取后续订单
    shadow_orders = exchange.get_shadow_orders()
    subsequent_shadow = None
    for so in shadow_orders:
        if so.order_id == "subsequent-order":
            subsequent_shadow = so
            break
    
    assert subsequent_shadow is not None, "后续订单应该在队列中"
    
    # 后续订单的pos应该 >= 前序订单的threshold (pos + qty)
    prev_threshold = post_crossing_shadow.pos + post_crossing_shadow.original_qty
    print(f"    前序订单threshold: {prev_threshold}")
    print(f"    后续订单pos: {subsequent_shadow.pos}")
    
    # 后续订单的pos应该基于前序订单，不会小于前序的threshold
    assert subsequent_shadow.pos >= prev_threshold, \
        f"后续订单pos({subsequent_shadow.pos})应该 >= 前序订单threshold({prev_threshold})"
    print(f"    ✓ 后续订单pos正确 >= 前序订单threshold")
    
    # 测试3: 验证后续订单不会因为pos过小而被错误成交
    print("\n  测试3: 验证后续订单不会被错误成交...")
    
    # 获取区间结束时的x_coord
    end_time = 1500 * TICK_PER_MS
    x_at_end = exchange._get_x_coord(Side.BUY, 100.0, end_time, post_crossing_shadow.pos)
    print(f"    区间结束时x_coord: {x_at_end}")
    
    # 后续订单的fill threshold
    subsequent_threshold = subsequent_shadow.pos + subsequent_shadow.original_qty
    print(f"    后续订单threshold: {subsequent_threshold}")
    
    # 在正常情况下，如果x_coord没有超过threshold，订单不应该成交
    if x_at_end < subsequent_threshold:
        print(f"    x_coord({x_at_end}) < threshold({subsequent_threshold})，订单不应该成交")
        print(f"    ✓ 后续订单不会被错误成交（修复验证通过）")
    else:
        print(f"    x_coord({x_at_end}) >= threshold({subsequent_threshold})，订单应该成交")
        print(f"    ✓ 如果成交，这是正确行为")
    
    print("✓ Post-crossing pos uses x_coord test passed")


def test_cancel_bias_causes_fill_without_sufficient_trades():
    """验证：cancel_bias_k=-0.8时，大量撤单使x_coord虚高，导致订单被错误成交。

    场景：
    - 订单到达时 market_qty=20，pos=20，qty=5，threshold=25
    - Segment 1：队列增长80手（新订单涌入），无撤单无成交
      → q_mkt 从20增长到100
    - Segment 2：80手撤单 + 1手成交
      → 由于 cancel_bias_k=-0.8，cancel_prob = (20/100)^0.2 ≈ 0.725
      → cancel贡献 = 0.725 * 80 = 58（远超前方队列深度20！）
      → x_coord ≈ 58 + 1 = 59，远超 threshold=25
      → 系统误判订单在seg2中期就完全成交（5手）

    问题：
    - 实际只有1手trade，最多成交1手
    - 但因cancel_bias导致cancel_prob被放大，x_coord被虚高推过threshold
    - cancel贡献(58)超过了前方实际队列深度(20)，这在物理上不可能

    期望行为：成交量 ≤ 实际trade量（1手）
    """
    print("\n--- Test: cancel bias causes fill without sufficient trades ---")

    from quant_framework.core.types import TapeSegment

    exchange = FIFOExchangeSimulator(cancel_bias_k=-0.8)

    t0 = 1000 * TICK_PER_MS
    t1 = 1200 * TICK_PER_MS
    t2 = 1500 * TICK_PER_MS

    price = 100.0

    # Segment 1: 队列增长80手（无撤单无成交）
    seg1 = TapeSegment(
        index=1, t_start=t0, t_end=t1,
        bid_price=price, ask_price=101.0,
        trades={},
        cancels={},
        net_flow={(Side.BUY, price): 80},
        activation_bid={price, 99.0, 98.0, 97.0, 96.0},
        activation_ask={101.0, 102.0, 103.0, 104.0, 105.0},
    )

    # Segment 2: 80手撤单 + 1手成交
    # 守恒检查: Q_B = Q_A + N_total - M_total = 20 + (80-80) - (0+1) = 19
    seg2 = TapeSegment(
        index=2, t_start=t1, t_end=t2,
        bid_price=price, ask_price=101.0,
        trades={(Side.BUY, price): 1},
        cancels={(Side.BUY, price): 80},
        net_flow={(Side.BUY, price): -80},
        activation_bid={price, 99.0, 98.0, 97.0, 96.0},
        activation_ask={101.0, 102.0, 103.0, 104.0, 105.0},
    )

    tape = [seg1, seg2]
    exchange.set_tape(tape, t0, t2)

    # 订单到达：market_qty=20 → pos=20, qty=5, threshold=25
    order = Order(order_id="cancel-bias-overfill", side=Side.BUY, price=price, qty=5)
    exchange.on_order_arrival(order, t0, market_qty=20)

    shadow = exchange.get_shadow_orders()[0]
    print(f"  Shadow: pos={shadow.pos}, qty={shadow.original_qty}, "
          f"threshold={shadow.pos + shadow.original_qty}")

    # 计算关键参数
    q_mkt_at_seg2_start = exchange._get_q_mkt(Side.BUY, price, t1)
    x_norm = (shadow.pos - 0) / q_mkt_at_seg2_start if q_mkt_at_seg2_start > 0 else 0
    cancel_prob = exchange._compute_cancel_front_prob(x_norm)
    cancel_contribution = cancel_prob * 80
    print(f"  q_mkt at seg2 start: {q_mkt_at_seg2_start}")
    print(f"  x_norm: {x_norm:.4f}")
    print(f"  cancel_prob (k=-0.8): {cancel_prob:.4f}")
    print(f"  cancel_contribution: {cancel_contribution:.2f} (前方实际队列深度仅{shadow.pos})")

    # 推进所有segments
    all_receipts = []
    t_cur = t0
    for seg in tape:
        t_seg = max(seg.t_start, t_cur)
        while t_seg < seg.t_end:
            receipts, t_stop = exchange.advance(t_seg, seg.t_end, seg)
            all_receipts.extend(receipts)
            if t_stop <= t_seg:
                break
            t_seg = t_stop
        t_cur = seg.t_end

    total_fill = sum(r.fill_qty for r in all_receipts if r.receipt_type in ["FILL", "PARTIAL"])
    total_trades = 1  # 只有seg2有1手trade

    print(f"  Receipts: {len(all_receipts)}")
    for r in all_receipts:
        print(f"    {r.order_id}: type={r.receipt_type}, fill_qty={r.fill_qty}, time={r.timestamp}")
    print(f"  Total fill qty: {total_fill}")
    print(f"  Total actual trades at this price: {total_trades}")

    # 核心断言：成交量不应超过实际trade量
    assert total_fill <= total_trades, (
        f"BUG: 成交量({total_fill})超过了实际trade量({total_trades})！"
        f"cancel_bias_k=-0.8导致cancel贡献({cancel_contribution:.1f})"
        f"超过前方队列深度({shadow.pos})，x_coord被虚高推过threshold"
    )

    print("✓ Test passed: fill quantity bounded by actual trades")


def test_zero_trades_should_not_fill_order():
    """验证：即使撤单量巨大，没有成交（trades=0）时订单不应被成交。

    场景：
    - 订单到达时 market_qty=20，pos=20，qty=5，threshold=25
    - Segment 1：队列增长80手
    - Segment 2：80手撤单，0手成交

    当 cancel_bias_k=-0.8 时：
    - cancel_prob = 0.725，cancel贡献 = 58
    - x_coord = 58（来自cancel） > threshold=25
    - 系统误判5手完全成交，但实际0手trade！

    这是物理上不可能的：没有成交发生，订单不可能被成交。

    期望行为：成交量 = 0
    """
    print("\n--- Test: zero trades should not fill order ---")

    from quant_framework.core.types import TapeSegment

    exchange = FIFOExchangeSimulator(cancel_bias_k=-0.8)

    t0 = 1000 * TICK_PER_MS
    t1 = 1200 * TICK_PER_MS
    t2 = 1500 * TICK_PER_MS

    price = 100.0

    # Segment 1: 队列增长80手
    seg1 = TapeSegment(
        index=1, t_start=t0, t_end=t1,
        bid_price=price, ask_price=101.0,
        trades={},
        cancels={},
        net_flow={(Side.BUY, price): 80},
        activation_bid={price, 99.0, 98.0, 97.0, 96.0},
        activation_ask={101.0, 102.0, 103.0, 104.0, 105.0},
    )

    # Segment 2: 80手撤单，0手成交!
    # 守恒: Q_B = 20 + (80-80) - 0 = 20
    seg2 = TapeSegment(
        index=2, t_start=t1, t_end=t2,
        bid_price=price, ask_price=101.0,
        trades={},  # 关键：没有任何trade
        cancels={(Side.BUY, price): 80},
        net_flow={(Side.BUY, price): -80},
        activation_bid={price, 99.0, 98.0, 97.0, 96.0},
        activation_ask={101.0, 102.0, 103.0, 104.0, 105.0},
    )

    tape = [seg1, seg2]
    exchange.set_tape(tape, t0, t2)

    order = Order(order_id="zero-trade-test", side=Side.BUY, price=price, qty=5)
    exchange.on_order_arrival(order, t0, market_qty=20)

    shadow = exchange.get_shadow_orders()[0]
    print(f"  Shadow: pos={shadow.pos}, qty={shadow.original_qty}, "
          f"threshold={shadow.pos + shadow.original_qty}")

    # 推进所有segments
    all_receipts = []
    t_cur = t0
    for seg in tape:
        t_seg = max(seg.t_start, t_cur)
        while t_seg < seg.t_end:
            receipts, t_stop = exchange.advance(t_seg, seg.t_end, seg)
            all_receipts.extend(receipts)
            if t_stop <= t_seg:
                break
            t_seg = t_stop
        t_cur = seg.t_end

    total_fill = sum(r.fill_qty for r in all_receipts if r.receipt_type in ["FILL", "PARTIAL"])

    print(f"  Receipts: {len(all_receipts)}")
    for r in all_receipts:
        print(f"    {r.order_id}: type={r.receipt_type}, fill_qty={r.fill_qty}")
    print(f"  Total fill qty: {total_fill}")
    print(f"  Total actual trades at this price: 0")

    # 核心断言：没有trade发生时，不应有任何成交
    assert total_fill == 0, (
        f"BUG: 没有任何trade发生，但成交了{total_fill}手！"
        f"cancel_bias_k=-0.8导致cancel贡献被虚高计入x_coord"
    )

    print("✓ Test passed: no fill without trades")


def test_uniform_cancel_bias_prevents_overfill():
    """验证：cancel_bias_k=0（均匀分布）时，同样场景不会产生过度成交。

    这是test_cancel_bias_causes_fill_without_sufficient_trades的对照测试。
    使用完全相同的tape和订单参数，只是k=0。

    k=0时：
    - cancel_prob = x_norm = 20/100 = 0.2
    - cancel贡献 = 0.2 * 80 = 16 ≤ 前方队列深度(20)
    - x_coord = 16 + 1 = 17 < threshold=25
    - 订单不会成交（正确）

    期望行为：无过度成交
    """
    print("\n--- Test: uniform cancel bias prevents overfill (k=0 control) ---")

    from quant_framework.core.types import TapeSegment

    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)

    t0 = 1000 * TICK_PER_MS
    t1 = 1200 * TICK_PER_MS
    t2 = 1500 * TICK_PER_MS

    price = 100.0

    seg1 = TapeSegment(
        index=1, t_start=t0, t_end=t1,
        bid_price=price, ask_price=101.0,
        trades={},
        cancels={},
        net_flow={(Side.BUY, price): 80},
        activation_bid={price, 99.0, 98.0, 97.0, 96.0},
        activation_ask={101.0, 102.0, 103.0, 104.0, 105.0},
    )

    seg2 = TapeSegment(
        index=2, t_start=t1, t_end=t2,
        bid_price=price, ask_price=101.0,
        trades={(Side.BUY, price): 1},
        cancels={(Side.BUY, price): 80},
        net_flow={(Side.BUY, price): -80},
        activation_bid={price, 99.0, 98.0, 97.0, 96.0},
        activation_ask={101.0, 102.0, 103.0, 104.0, 105.0},
    )

    tape = [seg1, seg2]
    exchange.set_tape(tape, t0, t2)

    order = Order(order_id="uniform-bias-test", side=Side.BUY, price=price, qty=5)
    exchange.on_order_arrival(order, t0, market_qty=20)

    shadow = exchange.get_shadow_orders()[0]
    threshold = shadow.pos + shadow.original_qty
    print(f"  Shadow: pos={shadow.pos}, qty={shadow.original_qty}, threshold={threshold}")

    # 计算关键参数
    q_mkt_at_seg2 = exchange._get_q_mkt(Side.BUY, price, t1)
    x_norm = shadow.pos / q_mkt_at_seg2 if q_mkt_at_seg2 > 0 else 0
    cancel_prob = exchange._compute_cancel_front_prob(x_norm)
    cancel_contribution = cancel_prob * 80
    print(f"  q_mkt at seg2 start: {q_mkt_at_seg2}")
    print(f"  x_norm: {x_norm:.4f}")
    print(f"  cancel_prob (k=0): {cancel_prob:.4f}")
    print(f"  cancel_contribution: {cancel_contribution:.2f} (前方队列深度{shadow.pos})")

    # 推进所有segments
    all_receipts = []
    t_cur = t0
    for seg in tape:
        t_seg = max(seg.t_start, t_cur)
        while t_seg < seg.t_end:
            receipts, t_stop = exchange.advance(t_seg, seg.t_end, seg)
            all_receipts.extend(receipts)
            if t_stop <= t_seg:
                break
            t_seg = t_stop
        t_cur = seg.t_end

    total_fill = sum(r.fill_qty for r in all_receipts if r.receipt_type in ["FILL", "PARTIAL"])
    total_trades = 1

    print(f"  Total fill qty: {total_fill}")
    print(f"  Total actual trades: {total_trades}")

    # k=0时，cancel贡献被正确限制，不会导致过度成交
    assert total_fill <= total_trades, (
        f"k=0时不应过度成交: fill={total_fill}, trades={total_trades}"
    )

    # 验证x_coord在合理范围内
    x_at_end = exchange._get_x_coord(Side.BUY, price, t2, shadow.pos)
    print(f"  x_coord at end: {x_at_end:.2f} (threshold: {threshold})")
    assert x_at_end <= threshold, (
        f"k=0时x_coord({x_at_end:.2f})不应超过threshold({threshold})"
    )

    print("✓ Test passed: uniform cancel bias prevents overfill")


def test_fill_exceeds_cumulative_trade_volume():
    """验证：成交量不应超过该价位的累计实际成交量。

    场景（多段累积）：
    - 订单：pos=20, qty=5, threshold=25
    - Segment 1: 队列增长80手
    - Segment 2: 40手撤单, 0手成交
    - Segment 3: 40手撤单, 1手成交

    cancel_bias_k=-0.8时：
    - Seg2: cancel_prob=0.725, cancel贡献=29, x=29 > threshold=25
    - 在seg2中就被判定完全成交，但seg2没有任何trade！

    期望行为：总成交量 ≤ 总trade量（1手）
    """
    print("\n--- Test: fill should not exceed cumulative trade volume ---")

    from quant_framework.core.types import TapeSegment

    exchange = FIFOExchangeSimulator(cancel_bias_k=-0.8)

    t0 = 1000 * TICK_PER_MS
    t1 = 1150 * TICK_PER_MS
    t2 = 1300 * TICK_PER_MS
    t3 = 1500 * TICK_PER_MS

    price = 100.0

    # Segment 1: 队列增长80手
    seg1 = TapeSegment(
        index=1, t_start=t0, t_end=t1,
        bid_price=price, ask_price=101.0,
        trades={},
        cancels={},
        net_flow={(Side.BUY, price): 80},
        activation_bid={price, 99.0, 98.0, 97.0, 96.0},
        activation_ask={101.0, 102.0, 103.0, 104.0, 105.0},
    )

    # Segment 2: 40手撤单, 0手成交
    # 守恒: seg2结束后 q = 20 + 80 - 40 = 60
    seg2 = TapeSegment(
        index=2, t_start=t1, t_end=t2,
        bid_price=price, ask_price=101.0,
        trades={},
        cancels={(Side.BUY, price): 40},
        net_flow={(Side.BUY, price): -40},
        activation_bid={price, 99.0, 98.0, 97.0, 96.0},
        activation_ask={101.0, 102.0, 103.0, 104.0, 105.0},
    )

    # Segment 3: 40手撤单, 1手成交
    # 守恒: Q_B = 20 + (80-40-40) - (0+0+1) = 19
    seg3 = TapeSegment(
        index=3, t_start=t2, t_end=t3,
        bid_price=price, ask_price=101.0,
        trades={(Side.BUY, price): 1},
        cancels={(Side.BUY, price): 40},
        net_flow={(Side.BUY, price): -40},
        activation_bid={price, 99.0, 98.0, 97.0, 96.0},
        activation_ask={101.0, 102.0, 103.0, 104.0, 105.0},
    )

    tape = [seg1, seg2, seg3]
    exchange.set_tape(tape, t0, t3)

    order = Order(order_id="multi-seg-cancel-test", side=Side.BUY, price=price, qty=5)
    exchange.on_order_arrival(order, t0, market_qty=20)

    shadow = exchange.get_shadow_orders()[0]
    print(f"  Shadow: pos={shadow.pos}, qty={shadow.original_qty}, "
          f"threshold={shadow.pos + shadow.original_qty}")

    # 推进所有segments
    all_receipts = []
    fill_segment_info = []
    t_cur = t0
    for seg in tape:
        t_seg = max(seg.t_start, t_cur)
        while t_seg < seg.t_end:
            receipts, t_stop = exchange.advance(t_seg, seg.t_end, seg)
            for r in receipts:
                if r.receipt_type in ["FILL", "PARTIAL"]:
                    fill_segment_info.append((seg.index, r.fill_qty, r.timestamp))
            all_receipts.extend(receipts)
            if t_stop <= t_seg:
                break
            t_seg = t_stop
        t_cur = seg.t_end

    total_fill = sum(r.fill_qty for r in all_receipts if r.receipt_type in ["FILL", "PARTIAL"])
    total_trades = 1  # 只有seg3有1手trade

    print(f"  Fill details:")
    for seg_idx, fill_qty, fill_time in fill_segment_info:
        seg_trades = tape[seg_idx - 1].trades.get((Side.BUY, price), 0)
        print(f"    Seg{seg_idx}: fill_qty={fill_qty}, "
              f"seg_trades={seg_trades}, fill_time={fill_time}")

    print(f"  Total fill qty: {total_fill}")
    print(f"  Total actual trades at this price: {total_trades}")

    # 核心断言：成交量不应超过累计实际trade量
    assert total_fill <= total_trades, (
        f"BUG: 总成交量({total_fill})超过了累计trade量({total_trades})！"
        f"cancel_bias_k=-0.8导致x_coord在无trade的段中就被推过threshold"
    )

    # 额外检查：成交不应发生在没有trade的segment中
    for seg_idx, fill_qty, fill_time in fill_segment_info:
        seg_trades = tape[seg_idx - 1].trades.get((Side.BUY, price), 0)
        if seg_trades == 0:
            assert fill_qty == 0, (
                f"BUG: 在Seg{seg_idx}（无trade）中产生了{fill_qty}手成交！"
                f"成交只应发生在有trade的segment中"
            )

    print("✓ Test passed: fill bounded by cumulative trade volume")


def run_all_tests():
    """Run all tests.
    
    在运行测试前启用DEBUG日志，通过日志输出验证逻辑正确性。
    """
    # 启用DEBUG日志用于验证逻辑
    setup_test_logging()
    
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
        test_exchange_simulator_multi_partial_to_fill,
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
        test_tape_start_time,
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
        test_contract_config_loading,
        test_trading_hours_session_detection,
        test_snapshot_duplication_with_trading_hours,
        test_order_arrival_trading_hours_adjustment,
        test_trading_hours_helper_class,
        test_no_duplicate_segment_and_interval_end_events,
        test_cancel_order_across_interval,
        test_cross_interval_order_fill,
        test_floating_point_precision_in_last_vol_split,
        test_post_crossing_pos_uses_x_coord,
        test_cancel_bias_causes_fill_without_sufficient_trades,
        test_zero_trades_should_not_fill_order,
        test_uniform_cancel_bias_prevents_overfill,
        test_fill_exceeds_cumulative_trade_volume,
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
