"""测试公共 fixtures 和辅助函数。

提供所有测试模块共享的：
- 日志配置 fixture（自动启用 DEBUG 日志）
- 数据创建辅助函数（快照、多档位快照）
- MockFeed 模拟数据源
- Tape 打印工具
"""

import logging
import sys

import pytest

from quant_framework.core.types import Level, NormalizedSnapshot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _setup_debug_logging():
    """自动启用 DEBUG 级别日志，用于验证逻辑正确性。"""
    log_level = logging.DEBUG
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
    ))
    logging.basicConfig(level=log_level, handlers=[handler], force=True)
    for name in (
        'quant_framework.exchange.simulator',
        'quant_framework.core.kernel',
        'quant_framework.core.handlers',
        'quant_framework.trading.receipt_logger',
        'quant_framework.tape.builder',
    ):
        logging.getLogger(name).setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def create_test_snapshot(
    ts: int,
    bid: float,
    ask: float,
    bid_qty: int = 100,
    ask_qty: int = 100,
    last_vol_split=None,
) -> NormalizedSnapshot:
    """创建单档位测试快照。时间单位为 tick（每 tick = 100 ns）。"""
    return NormalizedSnapshot(
        ts_recv=ts,
        bids=[Level(bid, bid_qty)],
        asks=[Level(ask, ask_qty)],
        last_vol_split=last_vol_split or [],
    )


def create_multi_level_snapshot(
    ts: int,
    bids: list,
    asks: list,
    last_vol_split=None,
) -> NormalizedSnapshot:
    """创建多档位测试快照。时间单位为 tick（每 tick = 100 ns）。"""
    return NormalizedSnapshot(
        ts_recv=ts,
        bids=[Level(p, q) for p, q in bids],
        asks=[Level(p, q) for p, q in asks],
        last_vol_split=last_vol_split or [],
    )


def print_tape_path(tape) -> None:
    """打印 tape 路径详情（调试用）。"""
    print(f"\n  Tape 路径（共 {len(tape)} 个段）:")
    for seg in tape:
        print(f"    段{seg.index}: t=[{seg.t_start}, {seg.t_end}], "
              f"bid={seg.bid_price}, ask={seg.ask_price}")
        if seg.trades:
            print(f"      trades: {dict(seg.trades)}")
        if seg.cancels:
            print(f"      cancels: {dict(seg.cancels)}")
        if seg.net_flow:
            non_zero = {k: v for k, v in seg.net_flow.items() if v != 0}
            if non_zero:
                print(f"      net_flow: {non_zero}")
        print(f"      activation_bid: {seg.activation_bid}")
        print(f"      activation_ask: {seg.activation_ask}")
    print()


# ---------------------------------------------------------------------------
# MockFeed（共享模拟数据源）
# ---------------------------------------------------------------------------

class MockFeed:
    """模拟快照数据源，依次返回预设快照列表。"""

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
