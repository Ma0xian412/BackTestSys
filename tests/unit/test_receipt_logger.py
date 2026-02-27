"""回执记录器（ReceiptLogger）单元测试。

验证内容：
- 回执记录（通过 on_receipt_delivered 接口）
- 统计指标（从 OMS 查询）
- CSV 文件保存
"""

import os
import tempfile

from quant_framework.core.data_structure import Order, OrderReceipt, OrderStatus, Side
from quant_framework.adapters.observability.ReceiptLogger_Impl import ReceiptLogger_Impl
from quant_framework.adapters.IOMS.oms import OMS_Impl, Portfolio


def test_receipt_logger():
    """完整流程：OMS 登记订单 + 应用回执，Obs 记录 + 统计。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "receipts.csv")
        obs = ReceiptLogger_Impl(output_file=output_file)
        oms = OMS_Impl(portfolio=Portfolio())

        obs.set_oms(oms)
        oms.subscribe_new(obs.on_order_submitted)
        oms.subscribe_receipt(obs.on_receipt_delivered)

        o1 = Order(order_id="order-1", side=Side.BUY, price=100.5, qty=100)
        o2 = Order(order_id="order-2", side=Side.BUY, price=99.0, qty=50)
        o3 = Order(order_id="order-3", side=Side.SELL, price=101.0, qty=30)
        o4 = Order(order_id="order-4", side=Side.BUY, price=100.0, qty=0)

        oms.submit_order(o1, 100)
        oms.submit_order(o2, 200)
        oms.submit_order(o3, 300)
        oms.submit_order(o4, 400)

        assert obs._diagnostics["orders_submitted"] == 4

        r1a = OrderReceipt(order_id="order-1", receipt_type="PARTIAL",
                           timestamp=1000, fill_qty=30, fill_price=100.5, remaining_qty=70)
        r1a.recv_time = 1010
        oms.apply_receipt(r1a)

        r1b = OrderReceipt(order_id="order-1", receipt_type="FILL",
                           timestamp=2000, fill_qty=70, fill_price=100.5, remaining_qty=0)
        r1b.recv_time = 2010
        oms.apply_receipt(r1b)

        r2a = OrderReceipt(order_id="order-2", receipt_type="PARTIAL",
                           timestamp=2500, fill_qty=20, fill_price=99.0, remaining_qty=30)
        r2a.recv_time = 2510
        oms.apply_receipt(r2a)

        r2b = OrderReceipt(order_id="order-2", receipt_type="CANCELED",
                           timestamp=3000, fill_qty=20, fill_price=99.0, remaining_qty=30)
        r2b.recv_time = 3010
        oms.apply_receipt(r2b)

        r3 = OrderReceipt(order_id="order-3", receipt_type="REJECTED",
                          timestamp=4000, fill_qty=0, fill_price=0.0, remaining_qty=30)
        r3.recv_time = 4010
        oms.apply_receipt(r3)

        assert len(obs.records) == 5

        stats = obs.get_statistics()
        assert stats["total_receipts"] == 5
        assert stats["total_orders"] == 4
        assert stats["partial_fill_count"] == 2
        assert stats["full_fill_count"] == 1
        assert stats["cancel_count"] == 1
        assert stats["reject_count"] == 1
        assert abs(stats["full_fill_rate"] - 1 / 3) < 0.01
        assert stats["fully_filled_orders"] == 1
        assert stats["partially_filled_orders"] == 1
        assert stats["unfilled_orders"] == 1

        assert oms.orders["order-1"].filled_qty == 100
        assert oms.orders["order-2"].filled_qty == 20
        assert oms.orders["order-3"].filled_qty == 0

        assert abs(obs.calculate_fill_rate() - 120 / 180) < 0.01
        assert abs(obs.calculate_fill_rate_by_count() - 1 / 4) < 0.01

        obs.save_to_file()
        assert os.path.exists(output_file)
        with open(output_file) as f:
            lines = f.readlines()
        assert len(lines) == 6, "1 行表头 + 5 行记录"
