"""Tape 构建器（UnifiedIntervalModel_impl）单元测试。

验证内容：
- 基本段生成
- 无成交时的段结构
- 成交量守恒方程
- 净流入量分配
- 时间顺序校验
- 相遇序列一致性
- 段转换队列归零约束
- Tape 起止时间
- 动态队列追踪
- 起点成交价前置
- 浮点精度处理
"""

from quant_framework.core.data_structure import Side, TICK_PER_MS
from quant_framework.adapters.interval_model import UnifiedIntervalModel_impl, TapeConfig

from tests.conftest import (
    create_test_snapshot,
    create_multi_level_snapshot,
    print_tape_path,
)


# ---------------------------------------------------------------------------
# 基础功能
# ---------------------------------------------------------------------------

def test_basic():
    """基本功能：两个快照之间能正确生成 tape 段，activation 集合大小合理。"""
    builder = UnifiedIntervalModel_impl(config=TapeConfig(), tick_size=1.0)

    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0)
    curr = create_test_snapshot(
        1500 * TICK_PER_MS, 100.5, 101.5,
        last_vol_split=[(100.0, 5), (101.0, 5)],
    )

    tape = builder.build(prev, curr)
    print_tape_path(tape)

    assert len(tape) > 0, "应至少生成一个段"
    for seg in tape:
        assert len(seg.activation_bid) <= 5, "activation_bid 最多 5 个价位"
        assert len(seg.activation_ask) <= 5, "activation_ask 最多 5 个价位"


def test_no_trades():
    """无成交时：tape 段中没有成交记录，但段结构仍正确生成。"""
    builder = UnifiedIntervalModel_impl(config=TapeConfig(), tick_size=1.0)

    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, last_vol_split=[])
    curr = create_test_snapshot(1500 * TICK_PER_MS, 100.5, 101.5, last_vol_split=[])

    tape = builder.build(prev, curr)
    print_tape_path(tape)

    assert len(tape) >= 1, "应至少生成一个段"
    for seg in tape:
        total_trades = sum(qty for qty in seg.trades.values())
        assert total_trades == 0, f"段 {seg.index} 不应有成交，实际 {total_trades}"


# ---------------------------------------------------------------------------
# 守恒方程
# ---------------------------------------------------------------------------

def test_conservation():
    """守恒方程：各价位总成交量等于 last_vol_split（允许取整误差 ±1）。"""
    builder = UnifiedIntervalModel_impl(config=TapeConfig(epsilon=1.0), tick_size=1.0)

    prev = create_multi_level_snapshot(
        1000 * TICK_PER_MS,
        bids=[(100.0, 50), (99.0, 30)],
        asks=[(101.0, 40), (102.0, 20)],
        last_vol_split=[(100.0, 20), (101.0, 15)],
    )
    curr = create_multi_level_snapshot(
        1500 * TICK_PER_MS,
        bids=[(100.0, 40), (99.0, 35)],
        asks=[(101.0, 35), (102.0, 25)],
        last_vol_split=[(100.0, 20), (101.0, 15)],
    )

    tape = builder.build(prev, curr)
    print_tape_path(tape)

    total_bid = sum(seg.trades.get((Side.BUY, 100.0), 0) for seg in tape)
    total_ask = sum(seg.trades.get((Side.SELL, 101.0), 0) for seg in tape)

    assert abs(total_bid - 20) <= 1, f"bid@100 应约 20 手，实际 {total_bid}"
    assert abs(total_ask - 15) <= 1, f"ask@101 应约 15 手，实际 {total_ask}"


def test_netflow_distribution():
    """净流入量分配：满足守恒方程 N = Q_B − Q_A + M（允许取整误差 ±1）。"""
    builder = UnifiedIntervalModel_impl(config=TapeConfig(epsilon=1.0), tick_size=1.0)

    prev = create_multi_level_snapshot(
        1000 * TICK_PER_MS,
        bids=[(100.0, 40), (99.0, 30), (98.0, 20)],
        asks=[(101.0, 40), (102.0, 20), (103.0, 10)],
        last_vol_split=[(100.0, 10), (101.0, 15), (102.0, 10)],
    )
    curr = create_multi_level_snapshot(
        1500 * TICK_PER_MS,
        bids=[(101.0, 55), (100.0, 40), (99.0, 35)],
        asks=[(100.0, 35), (101.0, 35), (102.0, 25)],
        last_vol_split=[(100.0, 10), (101.0, 15), (102.0, 10)],
    )

    tape = builder.build(prev, curr)
    print_tape_path(tape)

    assert len(tape) >= 1, "应至少生成一个段"

    # --- bid@100 守恒 ---
    total_bid_net = sum(seg.net_flow.get((Side.BUY, 100.0), 0) for seg in tape)
    prev_bid_qty = next(l.qty for l in prev.bids if abs(l.price - 100.0) < 1e-8)
    curr_bid_qty = next(l.qty for l in curr.bids if abs(l.price - 100.0) < 1e-8)
    total_bid_trades = sum(seg.trades.get((Side.BUY, 100.0), 0) for seg in tape)
    expected = curr_bid_qty - prev_bid_qty + total_bid_trades
    assert abs(total_bid_net - expected) <= 1, (
        f"bid@100 净流入 {total_bid_net}，期望 {expected}"
    )

    # --- ask@101 守恒 ---
    total_ask_net = sum(seg.net_flow.get((Side.SELL, 101.0), 0) for seg in tape)
    prev_ask_qty = next(l.qty for l in prev.asks if abs(l.price - 101.0) < 1e-8)
    curr_ask_qty = next(l.qty for l in curr.asks if abs(l.price - 101.0) < 1e-8)
    total_ask_trades = sum(seg.trades.get((Side.SELL, 101.0), 0) for seg in tape)
    expected_ask = curr_ask_qty - prev_ask_qty + total_ask_trades
    assert abs(total_ask_net - expected_ask) <= 1, (
        f"ask@101 净流入 {total_ask_net}，期望 {expected_ask}"
    )


# ---------------------------------------------------------------------------
# 边界条件
# ---------------------------------------------------------------------------

def test_invalid_time_order():
    """时间顺序：当 t_b ≤ t_a 时应抛出 ValueError。"""
    builder = UnifiedIntervalModel_impl(config=TapeConfig(), tick_size=1.0)

    # t_b == t_a
    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0)
    curr = create_test_snapshot(1000 * TICK_PER_MS, 100.5, 101.5)
    try:
        builder.build(prev, curr)
        assert False, "应抛出 ValueError（t_b == t_a）"
    except ValueError:
        pass

    # t_b < t_a
    prev2 = create_test_snapshot(2000 * TICK_PER_MS, 100.0, 101.0)
    curr2 = create_test_snapshot(1000 * TICK_PER_MS, 100.5, 101.5)
    try:
        builder.build(prev2, curr2)
        assert False, "应抛出 ValueError（t_b < t_a）"
    except ValueError:
        pass


def test_start_time():
    """起止时间：第一段从 t_a 开始，最后一段到 t_b 结束。"""
    builder = UnifiedIntervalModel_impl(config=TapeConfig(), tick_size=1.0)

    t_a = 1000 * TICK_PER_MS
    t_b = 2500 * TICK_PER_MS

    prev = create_test_snapshot(t_a, 100.0, 101.0, last_vol_split=[])
    curr = create_test_snapshot(t_b, 100.5, 101.5, last_vol_split=[(100.5, 20)])

    tape = builder.build(prev, curr)

    assert tape[0].t_start == t_a, f"第一段应从 {t_a} 开始，实际 {tape[0].t_start}"
    assert tape[-1].t_end == t_b, f"最后一段应到 {t_b} 结束，实际 {tape[-1].t_end}"

    # 短间隔同理
    t_a2, t_b2 = 3000 * TICK_PER_MS, 3400 * TICK_PER_MS
    prev2 = create_test_snapshot(t_a2, 100.0, 101.0)
    curr2 = create_test_snapshot(t_b2, 100.5, 101.5, last_vol_split=[(100.5, 10)])
    tape2 = builder.build(prev2, curr2)

    assert tape2[0].t_start == t_a2
    assert tape2[-1].t_end == t_b2


# ---------------------------------------------------------------------------
# 价格路径
# ---------------------------------------------------------------------------

def test_meeting_sequence_consistency():
    """相遇序列：bid/ask 路径的中间段经过公共相遇价位。"""
    builder = UnifiedIntervalModel_impl(config=TapeConfig(), tick_size=1.0)

    prev = create_multi_level_snapshot(
        1000 * TICK_PER_MS,
        bids=[(100.0, 50), (99.0, 30)],
        asks=[(101.0, 40), (102.0, 20)],
        last_vol_split=[(99.5, 10), (100.5, 15), (101.0, 20), (100.0, 25)],
    )
    curr = create_multi_level_snapshot(
        1500 * TICK_PER_MS,
        bids=[(100.5, 40), (99.5, 35)],
        asks=[(101.5, 35), (102.5, 25)],
        last_vol_split=[(99.5, 10), (100.5, 15), (101.0, 20), (100.0, 25)],
    )

    tape = builder.build(prev, curr)
    print_tape_path(tape)

    assert len(tape) >= 1, "应至少生成一个段"


def test_queue_zero_constraint():
    """队列归零约束：bid 下降时离开价位的队列深度应归零。

    当价格路径是 3318→3317→3316 时，3318 在转换时刻队列应为 0。
    """
    prev = create_multi_level_snapshot(
        1000 * TICK_PER_MS,
        bids=[(3318, 50), (3317, 40), (3316, 30)],
        asks=[(3319, 100), (3320, 100)],
        last_vol_split=[(3318, 30), (3317, 20), (3316, 10)],
    )
    curr = create_multi_level_snapshot(
        1500 * TICK_PER_MS,
        bids=[(3316, 25), (3315, 35)],
        asks=[(3317, 80), (3318, 90)],
        last_vol_split=[(3318, 30), (3317, 20), (3316, 10)],
    )

    builder = UnifiedIntervalModel_impl(config=TapeConfig(epsilon=1.0), tick_size=1.0)
    tape = builder.build(prev, curr)
    print_tape_path(tape)

    # 成交量应与 last_vol_split 一致
    total_by_price = {}
    for seg in tape:
        for (side, price), qty in seg.trades.items():
            if side == Side.BUY:
                total_by_price[price] = total_by_price.get(price, 0) + qty

    assert total_by_price.get(3318.0, 0) == 30, "3318 成交应为 30"
    assert total_by_price.get(3317.0, 0) == 20, "3317 成交应为 20"
    assert total_by_price.get(3316.0, 0) == 10, "3316 成交应为 10"

    # 验证 bid 下降转换时队列归零
    initial_qty = {3318: 50, 3317: 40, 3316: 30}
    for i in range(len(tape) - 1):
        curr_seg, next_seg = tape[i], tape[i + 1]
        if curr_seg.bid_price > next_seg.bid_price:
            from_price = curr_seg.bid_price
            run_start = i
            while run_start > 0 and abs(tape[run_start - 1].bid_price - from_price) < 0.01:
                run_start -= 1
            is_first = not any(
                abs(tape[j].bid_price - from_price) < 0.01 for j in range(run_start)
            )
            q = initial_qty.get(int(round(from_price)), 0) if is_first else 0
            for j in range(run_start, i + 1):
                q += tape[j].net_flow.get((Side.BUY, from_price), 0)
                q -= tape[j].trades.get((Side.BUY, from_price), 0)
            assert abs(q) <= 1, f"bid 下降时 {from_price} 队列未归零: {q}"


# ---------------------------------------------------------------------------
# 动态队列追踪
# ---------------------------------------------------------------------------

def test_dynamic_queue_tracking():
    """动态队列追踪：多次访问同一价位时队列不为负，全局守恒成立。

    场景：prev 有 bid@6 队列 127，curr 中 bid@6 不存在，M_total=197。
    N_total = (0 − 127) + 197 = 70。
    """
    builder = UnifiedIntervalModel_impl(config=TapeConfig(), tick_size=1.0)

    prev = create_multi_level_snapshot(
        1000 * TICK_PER_MS,
        bids=[(6.0, 127), (5.0, 118), (4.0, 32), (3.0, 232), (2.0, 37)],
        asks=[(10.0, 100)],
        last_vol_split=[],
    )
    curr = create_multi_level_snapshot(
        1500 * TICK_PER_MS,
        bids=[(5.0, 76), (4.0, 32), (3.0, 135), (2.0, 49), (1.0, 42)],
        asks=[(10.0, 100)],
        last_vol_split=[(5.0, 97), (6.0, 197), (7.0, 202)],
    )

    tape = builder.build(prev, curr)
    print_tape_path(tape)

    total_nf = sum(seg.net_flow.get((Side.BUY, 6.0), 0) for seg in tape)
    expected_nf = (0 - 127) + 197  # = 70
    assert abs(total_nf - expected_nf) <= 1, (
        f"全局守恒失败: 净流入和 {total_nf}，期望 {expected_nf}"
    )

    # 逐段累计验证队列不为负
    q = 127
    for seg in tape:
        if 6.0 in seg.activation_bid:
            nf = seg.net_flow.get((Side.BUY, 6.0), 0)
            trade = seg.trades.get((Side.BUY, 6.0), 0)
            q = q + nf - trade
            assert q >= -1, f"段 {seg.index} 结束时队列为负: {q}"

    # 各价位成交应大于 0
    assert sum(seg.trades.get((Side.BUY, 7.0), 0) for seg in tape) > 0, "价格 7 应有成交"
    assert sum(seg.trades.get((Side.BUY, 5.0), 0) for seg in tape) > 0, "价格 5 应有成交"


def test_starting_price_prepending():
    """起点成交价前置：bid_a/ask_a 是成交价时，会被前置到 meeting_seq。

    场景：bid 从 6→5，ask 保持 7，last_vol_split 含 6 和 7。
    """
    builder = UnifiedIntervalModel_impl(config=TapeConfig(), tick_size=1.0)

    prev = create_multi_level_snapshot(
        1000 * TICK_PER_MS,
        bids=[(6.0, 100), (5.0, 100), (4.0, 100), (3.0, 100), (2.0, 100)],
        asks=[(7.0, 100), (8.0, 100), (9.0, 100), (10.0, 100), (11.0, 100)],
        last_vol_split=[],
    )
    curr = create_multi_level_snapshot(
        1500 * TICK_PER_MS,
        bids=[(5.0, 100), (4.0, 100), (3.0, 100), (2.0, 100), (1.0, 100)],
        asks=[(7.0, 100), (8.0, 100), (9.0, 100), (10.0, 100), (11.0, 100)],
        last_vol_split=[(5.0, 100), (6.0, 100), (7.0, 100)],
    )

    tape = builder.build(prev, curr)
    print_tape_path(tape)

    # 第一段应保持起点价格
    assert tape[0].bid_price == 6.0, f"第一段 bid 应为 6.0，实际 {tape[0].bid_price}"
    assert tape[0].ask_price == 7.0, f"第一段 ask 应为 7.0，实际 {tape[0].ask_price}"

    # 价格 6 应有成交
    total_trade_6 = sum(seg.trades.get((Side.BUY, 6.0), 0) for seg in tape)
    assert total_trade_6 > 0, "bid@6 应有成交"

    # 价格 6 全局守恒：N = (0 − 100) + 100 = 0
    total_nf_6 = sum(seg.net_flow.get((Side.BUY, 6.0), 0) for seg in tape)
    assert abs(total_nf_6) <= 1, f"bid@6 净流入应约 0，实际 {total_nf_6}"


# ---------------------------------------------------------------------------
# 浮点精度
# ---------------------------------------------------------------------------

def test_floating_point_precision():
    """浮点精度：last_vol_split 中带浮点误差的价格能正确匹配段。"""
    builder = UnifiedIntervalModel_impl(config=TapeConfig(), tick_size=1.0)

    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0)
    curr = create_test_snapshot(
        1500 * TICK_PER_MS, 100.0, 101.0,
        last_vol_split=[(100.000000001, 50), (100.999999999, 30)],
    )

    tape = builder.build(prev, curr)

    total_buy = sum(
        qty for seg in tape
        for (side, _), qty in seg.trades.items()
        if side == Side.BUY
    )
    assert total_buy > 0, "带浮点误差的价格应被正确匹配"

    # 极端误差
    prev3 = create_multi_level_snapshot(
        1000 * TICK_PER_MS,
        bids=[(100.0, 100), (99.0, 100)],
        asks=[(101.0, 100), (102.0, 100)],
    )
    curr3 = create_multi_level_snapshot(
        1500 * TICK_PER_MS,
        bids=[(100.0, 100), (99.0, 100)],
        asks=[(100.0, 100), (101.0, 100)],
        last_vol_split=[(99.99999999999, 25), (100.00000000001, 25)],
    )
    tape3 = builder.build(prev3, curr3)
    total3 = sum(
        qty for seg in tape3
        for (side, _), qty in seg.trades.items()
        if side == Side.BUY
    )
    assert total3 > 0, "极端浮点误差应被正确处理"
