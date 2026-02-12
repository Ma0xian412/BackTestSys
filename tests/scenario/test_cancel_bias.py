"""Cancel bias 场景测试。

验证 cancel_bias_k 参数对成交判定的影响：
- k < 0（前方偏重撤单）：成交量不应超过实际 trade 量
- k = 0（均匀分布）：对照测试
- k > 0（后方偏重撤单）：验证行为合理
- 零 trade 场景：无成交时订单不应被成交
- 多段累积场景：跨段累计后成交量仍不超过累计 trade 量
"""

from quant_framework.core.data_structure import Order, Side, TapeSegment, TICK_PER_MS
from quant_framework.adapters.execution_venue import FIFOExchangeSimulator


def _advance_all(exchange, tape, t_start):
    """推进交易所通过所有段，返回全部回执。"""
    all_receipts = []
    t_cur = t_start
    for seg in tape:
        t_seg = max(seg.t_start, t_cur)
        while t_seg < seg.t_end:
            receipts, t_stop = exchange.advance(t_seg, seg.t_end, seg)
            all_receipts.extend(receipts)
            if t_stop <= t_seg:
                break
            t_seg = t_stop
        t_cur = seg.t_end
    return all_receipts


def _build_two_segment_tape(price, t0, t1, t2, netflow_seg1, cancels_seg2, trades_seg2):
    """构建两段 tape：seg1 队列增长，seg2 撤单+成交。"""
    seg1 = TapeSegment(
        index=1, t_start=t0, t_end=t1,
        bid_price=price, ask_price=price + 1.0,
        trades={}, cancels={},
        net_flow={(Side.BUY, price): netflow_seg1},
        activation_bid={price, price - 1, price - 2, price - 3, price - 4},
        activation_ask={price + 1, price + 2, price + 3, price + 4, price + 5},
    )
    seg2 = TapeSegment(
        index=2, t_start=t1, t_end=t2,
        bid_price=price, ask_price=price + 1.0,
        trades={(Side.BUY, price): trades_seg2} if trades_seg2 > 0 else {},
        cancels={(Side.BUY, price): cancels_seg2} if cancels_seg2 > 0 else {},
        net_flow={(Side.BUY, price): -cancels_seg2},
        activation_bid={price, price - 1, price - 2, price - 3, price - 4},
        activation_ask={price + 1, price + 2, price + 3, price + 4, price + 5},
    )
    return [seg1, seg2]


# ---------------------------------------------------------------------------
# k < 0（前方偏重撤单）
# ---------------------------------------------------------------------------

def test_negative_bias_overfill():
    """k=-0.8：cancel 贡献被放大，但成交量不应超过实际 trade 量。

    场景：pos=20, qty=5, threshold=25。seg2 有 80 手撤单 + 1 手成交。
    """
    price = 100.0
    t0, t1, t2 = 1000 * TICK_PER_MS, 1200 * TICK_PER_MS, 1500 * TICK_PER_MS

    tape = _build_two_segment_tape(price, t0, t1, t2, netflow_seg1=80, cancels_seg2=80, trades_seg2=1)

    exchange = FIFOExchangeSimulator(cancel_bias_k=-0.8)
    exchange.set_tape(tape, t0, t2)

    order = Order(order_id="neg-bias", side=Side.BUY, price=price, qty=5)
    exchange.on_order_arrival(order, t0, market_qty=20)

    receipts = _advance_all(exchange, tape, t0)
    total_fill = sum(r.fill_qty for r in receipts if r.receipt_type in ["FILL", "PARTIAL"])

    assert total_fill <= 1, (
        f"成交量({total_fill}) 不应超过实际 trade 量(1)"
    )


def test_zero_trades_no_fill():
    """k=-0.8 + 零 trade：即使撤单量巨大，也不应有任何成交。"""
    price = 100.0
    t0, t1, t2 = 1000 * TICK_PER_MS, 1200 * TICK_PER_MS, 1500 * TICK_PER_MS

    tape = _build_two_segment_tape(price, t0, t1, t2, netflow_seg1=80, cancels_seg2=80, trades_seg2=0)

    exchange = FIFOExchangeSimulator(cancel_bias_k=-0.8)
    exchange.set_tape(tape, t0, t2)

    order = Order(order_id="zero-trade", side=Side.BUY, price=price, qty=5)
    exchange.on_order_arrival(order, t0, market_qty=20)

    receipts = _advance_all(exchange, tape, t0)
    total_fill = sum(r.fill_qty for r in receipts if r.receipt_type in ["FILL", "PARTIAL"])

    assert total_fill == 0, f"无 trade 时不应有成交，实际 {total_fill}"


# ---------------------------------------------------------------------------
# k = 0（均匀分布对照）
# ---------------------------------------------------------------------------

def test_uniform_bias_control():
    """k=0（均匀分布）：与 k=-0.8 同场景，不应过度成交。"""
    price = 100.0
    t0, t1, t2 = 1000 * TICK_PER_MS, 1200 * TICK_PER_MS, 1500 * TICK_PER_MS

    tape = _build_two_segment_tape(price, t0, t1, t2, netflow_seg1=80, cancels_seg2=80, trades_seg2=1)

    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)
    exchange.set_tape(tape, t0, t2)

    order = Order(order_id="uniform-bias", side=Side.BUY, price=price, qty=5)
    exchange.on_order_arrival(order, t0, market_qty=20)

    shadow = exchange.get_shadow_orders()[0]
    threshold = shadow.pos + shadow.original_qty

    receipts = _advance_all(exchange, tape, t0)
    total_fill = sum(r.fill_qty for r in receipts if r.receipt_type in ["FILL", "PARTIAL"])

    assert total_fill <= 1, f"k=0 不应过度成交: fill={total_fill}, trades=1"

    x_at_end = exchange._get_x_coord(Side.BUY, price, t2, shadow.pos)
    assert x_at_end <= threshold, (
        f"k=0 x_coord({x_at_end:.2f}) 不应超过 threshold({threshold})"
    )


# ---------------------------------------------------------------------------
# k > 0（后方偏重撤单）
# ---------------------------------------------------------------------------

def test_positive_bias_behavior():
    """k=0.8（后方偏重撤单）：前方撤单概率更低，订单更难被成交。

    使用同样的场景：pos=20, qty=5, threshold=25, seg2 有 80 手撤单 + 1 手成交。
    k > 0 时 cancel_prob < x_norm，cancel 对前方的贡献更小，x_coord 推进更慢。
    """
    price = 100.0
    t0, t1, t2 = 1000 * TICK_PER_MS, 1200 * TICK_PER_MS, 1500 * TICK_PER_MS

    tape = _build_two_segment_tape(price, t0, t1, t2, netflow_seg1=80, cancels_seg2=80, trades_seg2=1)

    exchange_pos = FIFOExchangeSimulator(cancel_bias_k=0.8)
    exchange_pos.set_tape(tape, t0, t2)

    order = Order(order_id="pos-bias", side=Side.BUY, price=price, qty=5)
    exchange_pos.on_order_arrival(order, t0, market_qty=20)

    shadow_pos = exchange_pos.get_shadow_orders()[0]

    receipts_pos = _advance_all(exchange_pos, tape, t0)
    total_fill_pos = sum(r.fill_qty for r in receipts_pos if r.receipt_type in ["FILL", "PARTIAL"])

    # k > 0 时成交量同样不应超过实际 trade 量
    assert total_fill_pos <= 1, (
        f"k=0.8 成交量({total_fill_pos}) 不应超过实际 trade 量(1)"
    )

    # 与 k=0 对比：k > 0 的 cancel_prob 应更小
    x_norm = shadow_pos.pos / 100.0 if 100.0 > 0 else 0  # q_mkt ≈ 100 at seg2 start
    cancel_prob_pos = exchange_pos._compute_cancel_front_prob(x_norm)
    # k=0 时 cancel_prob = x_norm
    assert cancel_prob_pos <= x_norm + 0.01, (
        f"k=0.8 的 cancel_prob({cancel_prob_pos:.4f}) 应 ≤ x_norm({x_norm:.4f})"
    )

    # 验证 k > 0 也能正常运行零 trade 场景
    tape_zero = _build_two_segment_tape(price, t0, t1, t2, netflow_seg1=80, cancels_seg2=80, trades_seg2=0)
    exchange_pos2 = FIFOExchangeSimulator(cancel_bias_k=0.8)
    exchange_pos2.set_tape(tape_zero, t0, t2)

    order2 = Order(order_id="pos-bias-zero", side=Side.BUY, price=price, qty=5)
    exchange_pos2.on_order_arrival(order2, t0, market_qty=20)

    receipts_pos2 = _advance_all(exchange_pos2, tape_zero, t0)
    total_fill_pos2 = sum(r.fill_qty for r in receipts_pos2 if r.receipt_type in ["FILL", "PARTIAL"])
    assert total_fill_pos2 == 0, f"k=0.8 零 trade 不应有成交，实际 {total_fill_pos2}"


# ---------------------------------------------------------------------------
# 多段累积
# ---------------------------------------------------------------------------

def test_multi_segment_cumulative():
    """多段累积：跨 3 段时成交量不应超过累计 trade 量。

    seg1: +80 netflow, seg2: 40 cancel + 0 trade, seg3: 40 cancel + 1 trade。
    """
    price = 100.0
    t0 = 1000 * TICK_PER_MS
    t1, t2, t3 = 1150 * TICK_PER_MS, 1300 * TICK_PER_MS, 1500 * TICK_PER_MS

    seg1 = TapeSegment(
        index=1, t_start=t0, t_end=t1,
        bid_price=price, ask_price=101.0,
        trades={}, cancels={},
        net_flow={(Side.BUY, price): 80},
        activation_bid={price, 99.0, 98.0, 97.0, 96.0},
        activation_ask={101.0, 102.0, 103.0, 104.0, 105.0},
    )
    seg2 = TapeSegment(
        index=2, t_start=t1, t_end=t2,
        bid_price=price, ask_price=101.0,
        trades={}, cancels={(Side.BUY, price): 40},
        net_flow={(Side.BUY, price): -40},
        activation_bid={price, 99.0, 98.0, 97.0, 96.0},
        activation_ask={101.0, 102.0, 103.0, 104.0, 105.0},
    )
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
    exchange = FIFOExchangeSimulator(cancel_bias_k=-0.8)
    exchange.set_tape(tape, t0, t3)

    order = Order(order_id="multi-seg", side=Side.BUY, price=price, qty=5)
    exchange.on_order_arrival(order, t0, market_qty=20)

    receipts = _advance_all(exchange, tape, t0)
    fill_info = [(r.fill_qty, r.timestamp) for r in receipts if r.receipt_type in ["FILL", "PARTIAL"]]
    total_fill = sum(fq for fq, _ in fill_info)

    assert total_fill <= 1, (
        f"总成交量({total_fill}) 不应超过累计 trade 量(1)"
    )

    # 成交不应发生在无 trade 的段
    for r in receipts:
        if r.receipt_type in ["FILL", "PARTIAL"] and r.fill_qty > 0:
            for seg in tape:
                if seg.t_start <= r.timestamp < seg.t_end:
                    seg_trades = seg.trades.get((Side.BUY, price), 0)
                    if seg_trades == 0:
                        assert r.fill_qty == 0, (
                            f"段 {seg.index}（无 trade）不应有成交"
                        )
