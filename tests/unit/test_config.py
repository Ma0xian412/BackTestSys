"""配置与交易时段单元测试。

验证内容：
- 合约配置 XML 加载
- TradingHoursHelper 独立功能
- 交易时段检测（含跨越午夜）
"""

import os
import tempfile

from quant_framework.config import _load_contract_dictionary, TradingHour
from quant_framework.utils.trading_hours import TradingHoursHelper
from quant_framework.adapters.market_data_feed import SnapshotDuplicatingFeed_Impl

from tests.conftest import MockFeed


# ---------------------------------------------------------------------------
# 合约配置
# ---------------------------------------------------------------------------

_CONTRACT_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<ContractDictionaryConfig>
    <Contract>
        <ContractId>IF2401</ContractId>
        <TickSize>0.2</TickSize>
        <ExchangeCode>CFFEX</ExchangeCode>
        <TradingHours>
            <TradingHour><StartTime>09:30:00</StartTime><EndTime>11:30:00</EndTime></TradingHour>
            <TradingHour><StartTime>13:00:00</StartTime><EndTime>15:00:00</EndTime></TradingHour>
        </TradingHours>
    </Contract>
    <Contract>
        <ContractId>AU2401</ContractId>
        <TickSize>0.02</TickSize>
        <ExchangeCode>SHFE</ExchangeCode>
        <TradingHours>
            <TradingHour><StartTime>21:00:00</StartTime><EndTime>02:30:00</EndTime></TradingHour>
            <TradingHour><StartTime>09:00:00</StartTime><EndTime>10:15:00</EndTime></TradingHour>
        </TradingHours>
    </Contract>
</ContractDictionaryConfig>"""


def test_contract_config_loading():
    """合约配置加载：正确解析 XML、不存在时返回 None。"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(_CONTRACT_XML)
        path = f.name

    try:
        # IF2401
        info = _load_contract_dictionary(path, "IF2401")
        assert info is not None
        assert info.contract_id == "IF2401"
        assert info.tick_size == 0.2
        assert info.exchange_code == "CFFEX"
        assert len(info.trading_hours) == 2
        assert info.trading_hours[0].start_time == "09:30:00"

        # AU2401（跨午夜）
        info2 = _load_contract_dictionary(path, "AU2401")
        assert info2 is not None
        assert info2.tick_size == 0.02
        assert info2.trading_hours[0].start_time == "21:00:00"
        assert info2.trading_hours[0].end_time == "02:30:00"

        # 不存在
        assert _load_contract_dictionary(path, "NON_EXISTENT") is None

        # 空参数
        assert _load_contract_dictionary("", "IF2401") is None
        assert _load_contract_dictionary(path, "") is None
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# TradingHoursHelper
# ---------------------------------------------------------------------------

def test_trading_hours_helper():
    """TradingHoursHelper：时间解析、时段检测、索引查找、下一时段。"""
    # 空配置
    empty = TradingHoursHelper([])
    assert empty.is_in_any_trading_session(10 * 3600) is True
    assert empty.spans_trading_session_gap(100, 200) is False
    assert empty.is_after_last_trading_session(16 * 3600) is False

    # 正常时段
    hours = [
        TradingHour(start_time="09:30:00", end_time="11:30:00"),
        TradingHour(start_time="13:00:00", end_time="15:00:00"),
    ]
    h = TradingHoursHelper(hours)

    assert h.TICKS_PER_SECOND == 10_000_000
    assert h.SECONDS_PER_DAY == 86400

    # 时间解析
    assert h.parse_time_to_seconds("09:30:00") == 9 * 3600 + 30 * 60
    assert h.parse_time_to_seconds("invalid") == 0
    assert h.seconds_to_tick_offset(1) == 10_000_000

    tick_10am = 10 * 3600 * h.TICKS_PER_SECOND
    assert h.tick_to_day_seconds(tick_10am) == 10 * 3600

    # 时段内/外
    assert h.is_within_trading_session(9 * 3600, 11 * 3600, 10 * 3600) is True
    assert h.is_within_trading_session(9 * 3600, 11 * 3600, 12 * 3600) is False
    assert h.is_within_trading_session(21 * 3600, 2 * 3600, 22 * 3600) is True
    assert h.is_within_trading_session(21 * 3600, 2 * 3600, 1 * 3600) is True
    assert h.is_within_trading_session(21 * 3600, 2 * 3600, 10 * 3600) is False

    # 索引
    assert h.find_trading_session_index(10 * 3600) == 0
    assert h.find_trading_session_index(14 * 3600) == 1
    assert h.find_trading_session_index(12 * 3600) == -1

    # 下一时段
    assert h.get_next_trading_session_start(8 * 3600) == 9 * 3600 + 30 * 60
    assert h.get_next_trading_session_start(12 * 3600) == 13 * 3600
    assert h.get_next_trading_session_start(16 * 3600) is None

    # 最后时段之后
    assert h.is_after_last_trading_session(16 * 3600) is True
    assert h.is_after_last_trading_session(14 * 3600) is False
    assert h.is_after_last_trading_session(12 * 3600) is False


def test_session_detection():
    """交易时段检测：正常时段与跨午夜时段。"""
    # 正常时段
    hours = [
        TradingHour(start_time="09:30:00", end_time="11:30:00"),
        TradingHour(start_time="13:00:00", end_time="15:00:00"),
    ]
    feed = SnapshotDuplicatingFeed_Impl(MockFeed([]), trading_hours=hours)
    helper = feed._trading_hours_helper

    assert helper.is_in_any_trading_session(9 * 3600 + 30 * 60)
    assert helper.is_in_any_trading_session(10 * 3600)
    assert helper.is_in_any_trading_session(13 * 3600)
    assert not helper.is_in_any_trading_session(12 * 3600)

    # 跨午夜
    night_hours = [
        TradingHour(start_time="21:00:00", end_time="02:30:00"),
        TradingHour(start_time="09:00:00", end_time="10:15:00"),
    ]
    feed2 = SnapshotDuplicatingFeed_Impl(MockFeed([]), trading_hours=night_hours)
    helper2 = feed2._trading_hours_helper

    assert helper2.is_in_any_trading_session(21 * 3600)
    assert helper2.is_in_any_trading_session(23 * 3600)
    assert helper2.is_in_any_trading_session(1 * 3600)
    assert helper2.is_in_any_trading_session(2 * 3600 + 30 * 60)
    assert not helper2.is_in_any_trading_session(3 * 3600)
    assert not helper2.is_in_any_trading_session(8 * 3600)
