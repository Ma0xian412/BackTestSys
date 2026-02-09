"""交易所模拟器（FIFOExchangeSimulator）单元测试。

验证内容：
- 订单到达与队列注册
- IOC 订单立即取消
- 坐标轴模型与 FIFO 排序
- 成交逻辑（单次 / 多次部分成交）
- 成交优先级
- 多订单同价位队列
- 改善价模式
"""

from quant_framework.core.types import (
    Order, Side, TimeInForce, TapeSegment, TICK_PER_MS,
)
from quant_framework.tape.builder import UnifiedTapeBuilder, TapeConfig
from quant_framework.exchange.simulator import FIFOExchangeSimulator

from tests.conftest import create_test_snapshot, create_multi_level_snapshot, print_tape_path, MockFeed

from quant_framework.trading.oms import OrderManager
from quant_framework.runner.event_loop import EventLoopRunner, RunnerConfig, TimelineConfig


# ---------------------------------------------------------------------------
# 基本功能
# ---------------------------------------------------------------------------

def test_basic_order_arrival():
    """订单到达：GTC 订单被接受入队，队列深度和 shadow 记录正确。"""
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)

    order = Order(order_id="test-1", side=Side.BUY, price=100.0, qty=10)
    receipt = exchange.on_order_arrival(order, 1000 * TICK_PER_MS, market_qty=50)

    assert receipt is None, "GTC 订单应被接受（无立即成交/拒绝）"
    assert exchange.get_queue_depth(Side.BUY, 100.0) >= 10, "队列应包含 shadow 订单"

    shadows = exchange.get_shadow_orders()
    assert len(shadows) == 1, "应有一个 shadow 订单"
    assert shadows[0].pos >= 0, "shadow 订单位置应 ≥ 0"


def test_ioc_order():
    """IOC 订单：无法立即成交时应收到 CANCELED 回执。"""
    exchange = FIFOExchangeSimulator()

    order = Order(
        order_id="ioc-1", side=Side.BUY, price=100.0, qty=10,
        tif=TimeInForce.IOC,
    )
    receipt = exchange.on_order_arrival(order, 1000 * TICK_PER_MS, market_qty=50)

    assert receipt is not None, "IOC 订单应收到立即回执"
    assert receipt.receipt_type == "CANCELED", "IOC 应被取消（无法立即成交）"


# ---------------------------------------------------------------------------
# 坐标轴模型
# ---------------------------------------------------------------------------

def test_coordinate_axis():
    """坐标轴模型：先到订单的 pos 小于后到订单（FIFO）。"""
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
    builder = UnifiedTapeBuilder(config=TapeConfig(), tick_size=1.0)

    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=30, ask_qty=30)
    curr = create_test_snapshot(
        1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=20, ask_qty=20,
        last_vol_split=[(100.0, 50)],
    )
    tape = builder.build(prev, curr)
    print_tape_path(tape)

    exchange.set_tape(tape, 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)

    order1 = Order(order_id="o1", side=Side.BUY, price=100.0, qty=20)
    exchange.on_order_arrival(order1, 1100 * TICK_PER_MS, market_qty=30)

    order2 = Order(order_id="o2", side=Side.BUY, price=100.0, qty=10)
    exchange.on_order_arrival(order2, 1200 * TICK_PER_MS, market_qty=30)

    shadows = exchange.get_shadow_orders()
    o1 = next(s for s in shadows if s.order_id == "o1")
    o2 = next(s for s in shadows if s.order_id == "o2")

    assert o1.pos < o2.pos, "先到订单 pos 应小于后到订单"


# ---------------------------------------------------------------------------
# 成交逻辑
# ---------------------------------------------------------------------------

def test_fill():
    """成交逻辑：当 trades > market_queue + order_qty 时订单应被成交。"""
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
    builder = UnifiedTapeBuilder(config=TapeConfig(), tick_size=1.0)

    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=30)
    curr = create_test_snapshot(
        1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=10,
        last_vol_split=[(100.0, 50)],
    )
    tape = builder.build(prev, curr)
    print_tape_path(tape)

    exchange.set_tape(tape, 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)

    order = Order(order_id="fill-test", side=Side.BUY, price=100.0, qty=15)
    exchange.on_order_arrival(order, 1050 * TICK_PER_MS, market_qty=30)

    all_receipts = []
    t_global = 1050 * TICK_PER_MS
    for seg in tape:
        t_cur = max(seg.t_start, t_global)
        while t_cur < seg.t_end:
            receipts, t_stop = exchange.advance(t_cur, seg.t_end, seg)
            all_receipts.extend(receipts)
            if t_stop <= t_cur:
                break
            t_cur = t_stop
        t_global = seg.t_end

    filled = any(r.receipt_type in ["FILL", "PARTIAL"] for r in all_receipts)
    assert filled, "订单应被成交"


def test_multi_partial_to_fill():
    """多次部分成交：最终应返回 FILL 回执，总成交量等于订单量。"""
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)

    seg = TapeSegment(
        index=1,
        t_start=1000 * TICK_PER_MS, t_end=1304 * TICK_PER_MS,
        bid_price=100.0, ask_price=101.0,
        trades={(Side.BUY, 100.0): 4},
        cancels={}, net_flow={(Side.BUY, 100.0): 0},
        activation_bid={100.0}, activation_ask={101.0},
    )
    exchange.set_tape([seg], 1000 * TICK_PER_MS, 1304 * TICK_PER_MS)

    order = Order(order_id="multi-fill", side=Side.BUY, price=100.0, qty=4)
    exchange.on_order_arrival(order, 1000 * TICK_PER_MS, market_qty=0)

    r1, _ = exchange.advance(1000 * TICK_PER_MS, 1101 * TICK_PER_MS, seg)
    r2, _ = exchange.advance(1101 * TICK_PER_MS, 1202 * TICK_PER_MS, seg)
    r3, _ = exchange.advance(1202 * TICK_PER_MS, 1304 * TICK_PER_MS, seg)

    receipts = r1 + r2 + r3
    assert receipts, "应产生回执"
    assert receipts[-1].receipt_type == "FILL", f"最后应为 FILL，实际 {receipts[-1].receipt_type}"
    assert sum(r.fill_qty for r in receipts) == 4, "总成交量应为 4"


# ---------------------------------------------------------------------------
# 成交优先级
# ---------------------------------------------------------------------------

def test_fill_priority_fifo():
    """FIFO 优先级：先到订单应先成交。"""
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
    builder = UnifiedTapeBuilder(config=TapeConfig(), tick_size=1.0)

    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=30)
    curr = create_test_snapshot(
        1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=10,
        last_vol_split=[(100.0, 48)],
    )
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

    fill_order = [r.order_id for r in all_receipts if r.receipt_type in ["FILL", "PARTIAL"]]
    if len(fill_order) >= 2:
        assert fill_order.index("order1") < fill_order.index("order2"), (
            "order1 应先于 order2 成交"
        )


def test_multiple_orders_same_price():
    """多订单同价位：订单位置按到达时间递增（FIFO）。"""
    import random
    random.seed(42)

    bid_levels = [(3318.0, 100)]
    for p in sorted(random.sample([3314.0, 3315.0, 3316.0, 3317.0], 4), reverse=True):
        bid_levels.append((p, random.randint(50, 150)))
    ask_levels = [(3319.0, 100)]
    for p in sorted(random.sample([3320.0, 3321.0, 3322.0, 3323.0], 4)):
        ask_levels.append((p, random.randint(50, 150)))

    prev = create_multi_level_snapshot(
        1000 * TICK_PER_MS, bids=bid_levels, asks=ask_levels, last_vol_split=[],
    )
    curr = create_multi_level_snapshot(
        1500 * TICK_PER_MS, bids=bid_levels, asks=ask_levels,
        last_vol_split=[(3316.0, 10), (3317.0, 10), (3318.0, 10), (3319.0, 10), (3320.0, 10)],
    )

    builder = UnifiedTapeBuilder(config=TapeConfig(epsilon=1.0), tick_size=1.0)
    tape = builder.build(prev, curr)
    print_tape_path(tape)

    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
    oms = OrderManager()

    class MultiOrderStrategy:
        def __init__(self):
            self.created = False

        def on_snapshot(self, snapshot, oms_view):
            if not self.created:
                self.created = True
                orders = []
                for i in range(3):
                    o = Order(order_id=f"buy-3318-{i+1}", side=Side.BUY,
                              price=3318.0, qty=100, tif=TimeInForce.GTC)
                    o.create_time = (1000 + i * 167) * TICK_PER_MS
                    orders.append(o)
                return orders
            return []

        def on_receipt(self, receipt, snapshot, oms_view):
            return []

    runner = EventLoopRunner(
        feed=MockFeed([prev, curr]),
        tape_builder=builder, exchange=exchange,
        strategy=MultiOrderStrategy(), oms=oms,
        config=RunnerConfig(
            delay_out=10 * TICK_PER_MS, delay_in=10 * TICK_PER_MS,
            timeline=TimelineConfig(a=1.0, b=0),
        ),
    )
    results = runner.run()

    assert results['diagnostics']['orders_submitted'] == 3, "应提交 3 个订单"

    orders_at_3318 = [s for s in exchange.get_shadow_orders() if abs(s.price - 3318.0) < 0.01]
    if len(orders_at_3318) >= 2:
        sorted_orders = sorted(orders_at_3318, key=lambda x: x.arrival_time)
        for i in range(1, len(sorted_orders)):
            assert sorted_orders[i].pos >= sorted_orders[i - 1].pos, (
                "订单位置应按到达时间递增"
            )


# ---------------------------------------------------------------------------
# 改善价模式
# ---------------------------------------------------------------------------

def test_improvement_mode_fill():
    """改善价模式：改善价订单应先于同侧更差价订单成交。"""
    seg = TapeSegment(
        index=1,
        t_start=1000 * TICK_PER_MS, t_end=1100 * TICK_PER_MS,
        bid_price=100.0, ask_price=101.0,
        trades={(Side.BUY, 100.0): 10},
        cancels={}, net_flow={(Side.BUY, 100.0): 0},
        activation_bid={100.0}, activation_ask={101.0},
    )

    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
    exchange.set_tape([seg], seg.t_start, seg.t_end)

    # 改善价订单 100.5 > bid=100 但 < ask=101
    improving = Order(order_id="buy-improve", side=Side.BUY, price=100.5, qty=6, tif=TimeInForce.GTC)
    exchange.on_order_arrival(improving, seg.t_start, market_qty=0)

    base = Order(order_id="buy-base", side=Side.BUY, price=100.0, qty=10, tif=TimeInForce.GTC)
    exchange.on_order_arrival(base, seg.t_start, market_qty=0)

    r1, t_stop = exchange.advance(seg.t_start, seg.t_end, seg)
    assert len(r1) == 1, "改善价订单应先成交"
    assert r1[0].order_id == "buy-improve"
    assert r1[0].receipt_type == "FILL"

    r2, _ = exchange.advance(t_stop, seg.t_end, seg)
    base_receipts = [r for r in r2 if r.order_id == "buy-base"]
    assert base_receipts, "基础价订单应有部分成交"
    assert base_receipts[0].receipt_type == "PARTIAL"
    assert base_receipts[0].fill_qty == 4, f"期望 4，实际 {base_receipts[0].fill_qty}"
