"""DTO 和只读视图单元测试。

验证内容：
- NormalizedSnapshot 不可变与便捷属性
- ReadOnlyOMSView 查询与不可变性
"""

from quant_framework.core.types import NormalizedSnapshot, Order, Side, TICK_PER_MS
from quant_framework.core.read_only_view import (
    ReadOnlyOMSView, OrderSnapshot, PortfolioSnapshot,
)
from quant_framework.trading.oms import OMSImpl, Portfolio

from tests.conftest import create_test_snapshot


def test_normalized_snapshot_is_frozen():
    """NormalizedSnapshot：frozen 属性、便捷属性、不可变。"""
    snap = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=50, ask_qty=60)

    # frozen
    is_frozen = (
        hasattr(snap, '__dataclass_fields__')
        and snap.__class__.__dataclass_params__.frozen
    )
    assert is_frozen, "NormalizedSnapshot 应为 frozen"
    assert isinstance(snap, NormalizedSnapshot)

    # 数据
    assert snap.ts_recv == 1000 * TICK_PER_MS
    assert len(snap.bids) == 1
    assert len(snap.asks) == 1
    assert isinstance(snap.bids, tuple)
    assert isinstance(snap.asks, tuple)

    # 便捷属性
    assert snap.best_bid == 100.0
    assert snap.best_ask == 101.0
    assert snap.mid_price == 100.5
    assert snap.spread == 1.0

    # 不可变
    try:
        snap.ts_recv = 2000
        assert False, "快照应不可变"
    except Exception:
        pass


def test_readonly_oms_view():
    """ReadOnlyOMSView：返回 DTO、不可变、无修改方法。"""
    portfolio = Portfolio(cash=10000.0)
    oms = OMSImpl(portfolio=portfolio)

    order = Order(order_id="ro-1", side=Side.BUY, price=100.0, qty=10)
    oms.submit_order(order, 1000 * TICK_PER_MS)

    view = ReadOnlyOMSView(oms)

    # 活跃订单
    active = view.get_active_orders()
    assert len(active) == 1
    assert isinstance(active[0], OrderSnapshot)

    # 单个订单
    dto = view.get_order("ro-1")
    assert dto is not None
    assert dto.order_id == "ro-1"
    assert dto.price == 100.0

    # 不可变
    try:
        dto.price = 200.0
        assert False, "OrderSnapshot 应不可变"
    except Exception:
        pass

    # 投资组合
    p_dto = view.get_portfolio()
    assert isinstance(p_dto, PortfolioSnapshot)
    assert p_dto.cash == 10000.0
    try:
        p_dto.cash = 20000.0
        assert False, "PortfolioSnapshot 应不可变"
    except Exception:
        pass

    # 无修改方法
    assert not hasattr(view, 'submit'), "只读视图不应有 submit"
    assert not hasattr(view, 'on_receipt'), "只读视图不应有 on_receipt"
