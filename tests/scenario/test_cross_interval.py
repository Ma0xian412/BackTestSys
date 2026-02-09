"""跨区间场景测试。

验证内容：
- 跨区间撤单（exchange.reset() 后仍可取消）
- 跨区间成交（pos 在 align_at_boundary 后正确调整）
"""

from quant_framework.core.types import (
    Order, Side, TimeInForce, TapeSegment, NormalizedSnapshot, Level,
)
from quant_framework.exchange.simulator import FIFOExchangeSimulator


# ---------------------------------------------------------------------------
# 跨区间撤单
# ---------------------------------------------------------------------------

def test_cancel_across_interval():
    """exchange.reset() 后：订单仍在 _levels 中，可被取消。"""
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)

    tape = [TapeSegment(
        index=1, t_start=10_000_000, t_end=15_000_000,
        bid_price=100.0, ask_price=101.0,
        trades={}, cancels={}, net_flow={},
        activation_bid={100.0}, activation_ask={101.0},
    )]
    exchange.set_tape(tape, 10_000_000, 15_000_000)

    order = Order(order_id="cancel-1", side=Side.BUY, price=100.0, qty=10,
                  tif=TimeInForce.GTC, create_time=10_000_000)
    exchange.on_order_arrival(order, 10_000_000, market_qty=50)
    assert exchange._find_order_by_id("cancel-1") is not None

    # reset 保留 _levels
    exchange.reset()
    assert len(exchange._levels) > 0, "reset 应保留 _levels"
    assert exchange._find_order_by_id("cancel-1") is not None

    # 新区间中取消
    new_tape = [TapeSegment(
        index=1, t_start=15_000_000, t_end=20_000_000,
        bid_price=100.0, ask_price=101.0,
        trades={}, cancels={}, net_flow={},
        activation_bid={100.0}, activation_ask={101.0},
    )]
    exchange.set_tape(new_tape, 15_000_000, 20_000_000)

    r = exchange.on_cancel_arrival("cancel-1", 16_000_000)
    assert r.receipt_type == "CANCELED"
    assert exchange._find_order_by_id("cancel-1").status == "CANCELED"

    # 取消不存在的订单
    exchange2 = FIFOExchangeSimulator(cancel_bias_k=0.0)
    exchange2.set_tape(tape, 10_000_000, 15_000_000)
    try:
        exchange2.on_cancel_arrival("non-existent", 12_000_000)
        assert False, "应抛出 ValueError"
    except ValueError:
        pass

    # 重复取消返回 REJECTED
    exchange3 = FIFOExchangeSimulator(cancel_bias_k=0.0)
    exchange3.set_tape(tape, 10_000_000, 15_000_000)
    o3 = Order(order_id="cancel-3", side=Side.BUY, price=100.0, qty=10,
               tif=TimeInForce.GTC, create_time=10_000_000)
    exchange3.on_order_arrival(o3, 10_000_000, market_qty=50)
    exchange3.on_cancel_arrival("cancel-3", 11_000_000)
    r3 = exchange3.on_cancel_arrival("cancel-3", 12_000_000)
    assert r3.receipt_type == "REJECTED"

    # full_reset 清空 _levels
    exchange4 = FIFOExchangeSimulator(cancel_bias_k=0.0)
    exchange4.set_tape(tape, 10_000_000, 15_000_000)
    o4 = Order(order_id="cancel-4", side=Side.BUY, price=100.0, qty=10,
               tif=TimeInForce.GTC, create_time=10_000_000)
    exchange4.on_order_arrival(o4, 10_000_000, market_qty=50)
    exchange4.full_reset()
    assert len(exchange4._levels) == 0
    assert exchange4._find_order_by_id("cancel-4") is None


# ---------------------------------------------------------------------------
# 跨区间成交
# ---------------------------------------------------------------------------

def test_fill_across_interval():
    """跨区间成交：pos 在 align_at_boundary 后正确调整，最终在第三个区间成交。

    区间 AB: X+30, pos 100→70
    区间 BC: X+30, pos 70→40
    区间 CD: X+60, threshold=50 → 成交
    """
    exchange = FIFOExchangeSimulator(cancel_bias_k=0.0)

    t_A, t_B, t_C, t_D = 10_000_000, 15_000_000, 20_000_000, 25_000_000

    # === 区间 AB ===
    tape_AB = [TapeSegment(
        index=1, t_start=t_A, t_end=t_B,
        bid_price=100.0, ask_price=101.0,
        trades={(Side.BUY, 100.0): 30}, cancels={},
        net_flow={(Side.BUY, 100.0): -30},
        activation_bid={100.0}, activation_ask={101.0},
    )]
    exchange.set_tape(tape_AB, t_A, t_B)

    order = Order(order_id="cross-fill", side=Side.BUY, price=100.0, qty=10,
                  tif=TimeInForce.GTC, create_time=t_A)
    exchange.on_order_arrival(order, t_A, market_qty=100)

    shadow = exchange._find_order_by_id("cross-fill")
    assert shadow.pos == 100

    receipts, _ = exchange.advance(t_A, t_B, tape_AB[0])
    assert len(receipts) == 0, "AB 区间不应成交"

    exchange.align_at_boundary(NormalizedSnapshot(
        ts_recv=t_B, bids=[Level(100.0, 70)], asks=[Level(101.0, 100)],
    ))
    assert shadow.pos == 70

    # === 区间 BC ===
    exchange.reset()
    tape_BC = [TapeSegment(
        index=1, t_start=t_B, t_end=t_C,
        bid_price=100.0, ask_price=101.0,
        trades={(Side.BUY, 100.0): 30}, cancels={},
        net_flow={(Side.BUY, 100.0): -30},
        activation_bid={100.0}, activation_ask={101.0},
    )]
    exchange.set_tape(tape_BC, t_B, t_C)

    receipts, _ = exchange.advance(t_B, t_C, tape_BC[0])

    exchange.align_at_boundary(NormalizedSnapshot(
        ts_recv=t_C, bids=[Level(100.0, 40)], asks=[Level(101.0, 100)],
    ))

    # === 区间 CD ===
    exchange.reset()
    tape_CD = [TapeSegment(
        index=1, t_start=t_C, t_end=t_D,
        bid_price=100.0, ask_price=101.0,
        trades={(Side.BUY, 100.0): 60}, cancels={},
        net_flow={(Side.BUY, 100.0): -60},
        activation_bid={100.0}, activation_ask={101.0},
    )]
    exchange.set_tape(tape_CD, t_C, t_D)

    receipts, _ = exchange.advance(t_C, t_D, tape_CD[0])

    shadow = exchange._find_order_by_id("cross-fill")
    assert shadow.status == "FILLED", (
        f"跨区间后应成交，实际 status={shadow.status}"
    )
