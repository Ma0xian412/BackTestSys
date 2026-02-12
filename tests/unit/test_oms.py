"""订单管理器（OMS_Impl）单元测试。

验证内容：
- 订单提交与查询
- 回执处理（成交后状态更新）
- 投资组合仓位更新
"""

from quant_framework.core.data_structure import (
    Order, Side, OrderStatus, OrderReceipt, TICK_PER_MS,
)
from quant_framework.adapters.IOMS.oms import OMS_Impl, Portfolio


def test_order_lifecycle():
    """订单生命周期：提交 → 查询 → 成交 → 状态更新 → 仓位变化。"""
    portfolio = Portfolio(cash=10000.0)
    oms = OMS_Impl(portfolio=portfolio)

    # 提交
    order = Order(order_id="test-1", side=Side.BUY, price=100.0, qty=10)
    oms.submit_order(order, 1000 * TICK_PER_MS)

    active = oms.get_active_orders()
    assert len(active) == 1, f"应有 1 个活跃订单，实际 {len(active)}"

    retrieved = oms.get_order("test-1")
    assert retrieved is not None, "应能通过 ID 查询订单"

    # 成交
    receipt = OrderReceipt(
        order_id="test-1", receipt_type="FILL",
        timestamp=1500 * TICK_PER_MS, fill_qty=10, fill_price=100.0,
    )
    oms.apply_receipt(receipt)

    assert oms.get_order("test-1").status == OrderStatus.FILLED, "订单应为 FILLED"
    assert portfolio.position == 10, f"仓位应为 10，实际 {portfolio.position}"
