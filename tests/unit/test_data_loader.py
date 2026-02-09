"""数据加载 / 快照复制单元测试。

验证内容：
- SnapshotDuplicatingFeed 在超过 500ms 间隔时正确复制
- 容差（tolerance）功能
- 交易时段感知的复制逻辑
"""

from quant_framework.core.data_loader import SnapshotDuplicatingFeed
from quant_framework.core.types import TICK_PER_MS, SNAPSHOT_MIN_INTERVAL_TICK, DEFAULT_SNAPSHOT_TOLERANCE_TICK
from quant_framework.config import TradingHour

from tests.conftest import create_test_snapshot, MockFeed


def test_snapshot_duplication():
    """快照复制：间隔 > 500ms 时插入复制快照，≤ 500ms 时不复制。"""
    # 1000ms 间隔 → 1 个复制
    t1, t2 = 1000 * TICK_PER_MS, 2000 * TICK_PER_MS
    feed = SnapshotDuplicatingFeed(MockFeed([
        create_test_snapshot(t1, 100.0, 101.0, last_vol_split=[(100.0, 10)]),
        create_test_snapshot(t2, 100.0, 101.0, last_vol_split=[(100.0, 20)]),
    ]))

    result = []
    while (s := feed.next()) is not None:
        result.append(s)

    assert len(result) == 3, f"应 3 个快照（A, A', B），实际 {len(result)}"
    assert result[0].ts_recv == t1
    assert result[1].ts_recv == t1 + SNAPSHOT_MIN_INTERVAL_TICK
    assert result[2].ts_recv == t2
    assert result[1].last_vol_split == [], "复制快照的 last_vol_split 应为空"

    # 2000ms 间隔 → 3 个复制
    t3, t4 = 1000 * TICK_PER_MS, 3000 * TICK_PER_MS
    feed2 = SnapshotDuplicatingFeed(MockFeed([
        create_test_snapshot(t3, 100.0, 101.0, last_vol_split=[(100.0, 10)]),
        create_test_snapshot(t4, 100.0, 101.0, last_vol_split=[(100.0, 30)]),
    ]))
    result2 = []
    while (s := feed2.next()) is not None:
        result2.append(s)
    assert len(result2) == 5, f"应 5 个快照，实际 {len(result2)}"
    for i in range(1, 4):
        assert result2[i].last_vol_split == []

    # 500ms 间隔 → 无复制
    t5, t6 = 1000 * TICK_PER_MS, 1500 * TICK_PER_MS
    feed3 = SnapshotDuplicatingFeed(MockFeed([
        create_test_snapshot(t5, 100.0, 101.0, last_vol_split=[(100.0, 10)]),
        create_test_snapshot(t6, 100.0, 101.0, last_vol_split=[(100.0, 15)]),
    ]))
    result3 = []
    while (s := feed3.next()) is not None:
        result3.append(s)
    assert len(result3) == 2

    # 重置功能
    feed3.reset()
    result4 = []
    while (s := feed3.next()) is not None:
        result4.append(s)
    assert len(result4) == 2

    # 容差: 510ms 在默认 10ms 容差内 → 不复制
    feed5 = SnapshotDuplicatingFeed(MockFeed([
        create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0),
        create_test_snapshot(1510 * TICK_PER_MS, 100.0, 101.0),
    ]))
    r5 = []
    while (s := feed5.next()) is not None:
        r5.append(s)
    assert len(r5) == 2, "510ms 在容差内不应复制"

    # 自定义 5ms 容差: 520ms → 复制
    feed6 = SnapshotDuplicatingFeed(MockFeed([
        create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0),
        create_test_snapshot(1520 * TICK_PER_MS, 100.0, 101.0),
    ]), tolerance_tick=5 * TICK_PER_MS)
    r6 = []
    while (s := feed6.next()) is not None:
        r6.append(s)
    assert len(r6) == 3, "520ms 超出 5ms 容差应复制"

    # 0 容差: 501ms → 复制
    feed7 = SnapshotDuplicatingFeed(MockFeed([
        create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0),
        create_test_snapshot(1501 * TICK_PER_MS, 100.0, 101.0),
    ]), tolerance_tick=0)
    r7 = []
    while (s := feed7.next()) is not None:
        r7.append(s)
    assert len(r7) == 3, "501ms 超出 0 容差应复制"


def test_duplication_with_trading_hours():
    """交易时段感知：同一时段内正常复制，跨时段不复制。"""
    trading_hours = [
        TradingHour(start_time="09:30:00", end_time="11:30:00"),
        TradingHour(start_time="13:00:00", end_time="15:00:00"),
    ]

    def _time_to_tick(h, m, s):
        return (h * 3600 + m * 60 + s) * 10_000_000

    # 同一时段内 2 秒间隔 → 复制
    feed1 = SnapshotDuplicatingFeed(MockFeed([
        create_test_snapshot(_time_to_tick(9, 30, 0), 100.0, 101.0),
        create_test_snapshot(_time_to_tick(9, 30, 2), 100.0, 101.0),
    ]), trading_hours=trading_hours)

    count1 = 0
    while feed1.next() is not None:
        count1 += 1
    assert count1 >= 4, f"同一时段内应复制，实际 {count1} 个快照"

    # 跨休市 → 不复制
    feed2 = SnapshotDuplicatingFeed(MockFeed([
        create_test_snapshot(_time_to_tick(11, 29, 59), 100.0, 101.0),
        create_test_snapshot(_time_to_tick(13, 0, 0), 100.0, 101.0),
    ]), trading_hours=trading_hours)

    count2 = 0
    while feed2.next() is not None:
        count2 += 1
    assert count2 == 2, f"跨休市不应复制，实际 {count2} 个快照"

    # 无交易时段配置 → 正常复制
    feed3 = SnapshotDuplicatingFeed(MockFeed([
        create_test_snapshot(_time_to_tick(11, 29, 59), 100.0, 101.0),
        create_test_snapshot(_time_to_tick(13, 0, 0), 100.0, 101.0),
    ]), trading_hours=None)

    count3 = 0
    while feed3.next() is not None:
        count3 += 1
    assert count3 > 2, f"无配置应正常复制，实际 {count3} 个快照"
