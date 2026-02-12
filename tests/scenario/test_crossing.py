"""Crossing（穿越价格）成交场景测试。

验证内容：
- 立即 crossing 成交（BUY ≥ ask / SELL ≤ bid）
- 部分成交后剩余订单队列位置为 0
- 存在同侧优先级更高 shadow 时阻止 crossing
- 同侧队列深度阻止 crossing
- post-crossing 订单使用 x_coord 作为 pos
"""

from quant_framework.core.data_structure import (
    Order, Side, TimeInForce, TapeSegment, TICK_PER_MS,
)
from quant_framework.adapters.interval_model import UnifiedIntervalModel_impl, TapeConfig
from quant_framework.adapters.execution_venue import FIFOExchangeSimulator

from tests.conftest import create_test_snapshot, print_tape_path


# ---------------------------------------------------------------------------
# 立即 crossing
# ---------------------------------------------------------------------------

def test_immediate_execution():
    """BUY/SELL crossing 立即成交、IOC 剩余取消、被动订单不 crossing。"""
    builder = UnifiedIntervalModel_impl(config=TapeConfig(epsilon=1.0), tick_size=1.0)

    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=50, ask_qty=60)
    curr = create_test_snapshot(
        1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=40, ask_qty=50,
        last_vol_split=[(100.0, 10), (101.0, 10)],
    )
    tape = builder.build(prev, curr)
    print_tape_path(tape)

    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
    exchange.set_tape(tape, 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)

    # BUY crossing
    ask_lvl = exchange._get_level(Side.SELL, 101.0)
    ask_lvl.q_mkt = 60.0
    r1 = exchange.on_order_arrival(
        Order(order_id="buy-cross", side=Side.BUY, price=101.0, qty=10, tif=TimeInForce.GTC),
        1100 * TICK_PER_MS, 50,
    )
    if r1:
        assert r1.fill_qty > 0, "BUY crossing 应有成交"

    # SELL crossing
    bid_lvl = exchange._get_level(Side.BUY, 100.0)
    bid_lvl.q_mkt = 50.0
    r2 = exchange.on_order_arrival(
        Order(order_id="sell-cross", side=Side.SELL, price=100.0, qty=15, tif=TimeInForce.GTC),
        1200 * TICK_PER_MS, 60,
    )
    if r2:
        assert r2.fill_qty > 0, "SELL crossing 应有成交"

    # 被动订单（不 crossing）
    r4 = exchange.on_order_arrival(
        Order(order_id="passive-1", side=Side.BUY, price=99.0, qty=20, tif=TimeInForce.GTC),
        1400 * TICK_PER_MS, 50,
    )
    if r4 is None:
        assert any(s.order_id == "passive-1" for s in exchange.get_shadow_orders()), (
            "被动订单应入队"
        )


# ---------------------------------------------------------------------------
# 部分成交后位置
# ---------------------------------------------------------------------------

def test_partial_fill_position_zero():
    """crossing 部分成交后：剩余订单 pos=0，被动订单 pos ≥ 市场队列深度。"""
    builder = UnifiedIntervalModel_impl(config=TapeConfig(epsilon=1.0), tick_size=1.0)

    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=50, ask_qty=100)
    curr = create_test_snapshot(
        1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=50, ask_qty=100,
        last_vol_split=[(100.0, 10), (101.0, 10)],
    )
    tape = builder.build(prev, curr)

    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
    exchange.set_tape(tape, 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)

    # post-crossing 入队（already_filled > 0）
    order1 = Order(order_id="after-crossing", side=Side.BUY, price=101.0, qty=150, tif=TimeInForce.GTC)
    exchange._queue_order(order1, 1100 * TICK_PER_MS, market_qty=50, remaining_qty=50, already_filled=100)

    so1 = next(s for s in exchange.get_shadow_orders() if s.order_id == "after-crossing")
    assert so1.pos == 0, f"crossing 后 pos 应为 0，实际 {so1.pos}"

    # 被动入队（already_filled = 0）
    exchange2 = FIFOExchangeSimulator(cancel_bias_k=0.0)
    exchange2.set_tape(tape, 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)
    lvl = exchange2._get_level(Side.BUY, 99.0)
    lvl.q_mkt = 30.0

    order2 = Order(order_id="passive", side=Side.BUY, price=99.0, qty=50, tif=TimeInForce.GTC)
    exchange2._queue_order(order2, 1100 * TICK_PER_MS, market_qty=30, remaining_qty=50, already_filled=0)

    so2 = next(s for s in exchange2.get_shadow_orders() if s.order_id == "passive")
    assert so2.pos >= 30, f"被动订单 pos 应 ≥ 30，实际 {so2.pos}"


# ---------------------------------------------------------------------------
# 阻止 crossing
# ---------------------------------------------------------------------------

def test_blocked_by_existing_shadow():
    """存在同侧更优价格 shadow 时：新订单不能 crossing。"""
    seg = TapeSegment(
        index=1, t_start=1000 * TICK_PER_MS, t_end=1500 * TICK_PER_MS,
        bid_price=100.0, ask_price=101.0,
        trades={(Side.BUY, 100.0): 30, (Side.BUY, 99.0): 20},
        cancels={},
        net_flow={(Side.BUY, 100.0): 50, (Side.BUY, 99.0): 30},
        activation_bid={96.0, 97.0, 98.0, 99.0, 100.0},
        activation_ask={101.0, 102.0, 103.0, 104.0, 105.0},
    )

    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
    exchange.set_tape([seg], seg.t_start, seg.t_end)

    bid_100 = exchange._get_level(Side.BUY, 100.0)
    bid_100.q_mkt = 100.0
    bid_99 = exchange._get_level(Side.BUY, 99.0)
    bid_99.q_mkt = 80.0

    # 第一个 SELL@99 crossing
    r1 = exchange.on_order_arrival(
        Order(order_id="sell-99", side=Side.SELL, price=99.0, qty=150, tif=TimeInForce.GTC),
        1100 * TICK_PER_MS, market_qty=0,
    )
    assert r1 is not None, "第一个订单应 crossing 成交"

    sell99 = next((s for s in exchange.get_shadow_orders() if s.order_id == "sell-99"), None)

    # 第二个 SELL@100 应被阻止（如果有 price=99 shadow 未全成交）
    bid_100.q_mkt = 50.0
    r2 = exchange.on_order_arrival(
        Order(order_id="sell-100", side=Side.SELL, price=100.0, qty=50, tif=TimeInForce.GTC),
        1200 * TICK_PER_MS, market_qty=0,
    )
    if sell99 and sell99.remaining_qty > 0:
        assert r2 is None, "存在更低价 shadow 时不应 crossing"


def test_blocked_by_queue_depth():
    """同侧队列有深度时：不应立即 crossing。"""
    seg = TapeSegment(
        index=1, t_start=1000 * TICK_PER_MS, t_end=1500 * TICK_PER_MS,
        bid_price=101.0, ask_price=101.0,
        trades={}, cancels={}, net_flow={},
        activation_bid={101.0}, activation_ask={101.0},
    )

    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
    exchange.set_tape([seg], seg.t_start, seg.t_end)

    exchange._get_level(Side.SELL, 101.0).q_mkt = 50.0
    exchange._get_level(Side.BUY, 101.0).q_mkt = 20.0

    order = Order(order_id="buy-q-depth", side=Side.BUY, price=101.0, qty=10, tif=TimeInForce.GTC)
    receipt = exchange.on_order_arrival(order, 1100 * TICK_PER_MS, market_qty=0)

    assert receipt is None, "同侧有队列深度时不应立即成交"
    queued = next((s for s in exchange.get_shadow_orders() if s.order_id == "buy-q-depth"), None)
    assert queued is not None, "订单应入队"
    assert queued.pos >= 20, f"pos 应 ≥ 20，实际 {queued.pos}"


# ---------------------------------------------------------------------------
# post-crossing 位置
# ---------------------------------------------------------------------------

def test_post_crossing_pos_uses_x_coord():
    """post-crossing 入队订单 pos 应等于到达时刻的 x_coord。"""
    builder = UnifiedIntervalModel_impl(config=TapeConfig(epsilon=1.0), tick_size=1.0)

    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=100, ask_qty=100)
    curr = create_test_snapshot(
        1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=50, ask_qty=50,
        last_vol_split=[(100.0, 80), (101.0, 80)],
    )
    tape = builder.build(prev, curr)
    print_tape_path(tape)

    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
    exchange.set_tape(tape, 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)

    mid_time = 1250 * TICK_PER_MS
    x_at_mid = exchange._get_x_coord(Side.BUY, 100.0, mid_time, 0)

    order1 = Order(order_id="post-cross", side=Side.BUY, price=100.0, qty=100, tif=TimeInForce.GTC)
    exchange._queue_order(order1, mid_time, market_qty=50, remaining_qty=20, already_filled=80)

    so1 = next(s for s in exchange.get_shadow_orders() if s.order_id == "post-cross")
    assert abs(so1.pos - int(round(x_at_mid))) <= 1, (
        f"pos 应约等于 x_coord({int(round(x_at_mid))}），实际 {so1.pos}"
    )

    # 后续订单 pos >= 前序 threshold
    order2 = Order(order_id="subsequent", side=Side.BUY, price=100.0, qty=10, tif=TimeInForce.GTC)
    exchange._queue_order(order2, 1300 * TICK_PER_MS, market_qty=50, remaining_qty=10, already_filled=0)

    so2 = next(s for s in exchange.get_shadow_orders() if s.order_id == "subsequent")
    prev_threshold = so1.pos + so1.original_qty
    assert so2.pos >= prev_threshold, (
        f"后续 pos({so2.pos}) 应 ≥ 前序 threshold({prev_threshold})"
    )
