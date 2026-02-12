"""回执记录器（ReceiptLogger）单元测试。

验证内容：
- 订单注册
- 各类回执记录（PARTIAL / FILL / CANCELED / REJECTED）
- 统计指标（成交率、成交量）
- CSV 文件保存
"""

import os
import tempfile

from quant_framework.core.data_structure import OrderReceipt
from quant_framework.adapters.observability.ReceiptLogger_Impl import ReceiptLogger_Impl


def test_receipt_logger():
    """完整流程：注册订单、记录回执、统计指标、保存 CSV。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "receipts.csv")
        logger = ReceiptLogger_Impl(output_file=output_file)

        # 注册 4 个订单
        logger.register_order("order-1", 100)
        logger.register_order("order-2", 50)
        logger.register_order("order-3", 30)
        logger.register_order("order-4", 0)
        assert len(logger.order_total_qty) == 4

        # order-1: 部分 → 全部成交
        r1a = OrderReceipt(order_id="order-1", receipt_type="PARTIAL",
                           timestamp=1000, fill_qty=30, fill_price=100.5, remaining_qty=70)
        r1a.recv_time = 1010
        logger.log_receipt(r1a)

        r1b = OrderReceipt(order_id="order-1", receipt_type="FILL",
                           timestamp=2000, fill_qty=70, fill_price=100.5, remaining_qty=0)
        r1b.recv_time = 2010
        logger.log_receipt(r1b)

        # order-2: 部分成交 → 撤单
        r2a = OrderReceipt(order_id="order-2", receipt_type="PARTIAL",
                           timestamp=2500, fill_qty=20, fill_price=99.0, remaining_qty=30)
        r2a.recv_time = 2510
        logger.log_receipt(r2a)

        r2b = OrderReceipt(order_id="order-2", receipt_type="CANCELED",
                           timestamp=3000, fill_qty=20, fill_price=99.0, remaining_qty=30)
        r2b.recv_time = 3010
        logger.log_receipt(r2b)

        # order-3: 拒绝
        r3 = OrderReceipt(order_id="order-3", receipt_type="REJECTED",
                          timestamp=4000, fill_qty=0, fill_price=0.0, remaining_qty=30)
        r3.recv_time = 4010
        logger.log_receipt(r3)

        # 验证记录数
        assert len(logger.records) == 5

        # 统计
        stats = logger.get_statistics()
        assert stats['total_receipts'] == 5
        assert stats['total_orders'] == 4
        assert stats['partial_fill_count'] == 2
        assert stats['full_fill_count'] == 1
        assert stats['cancel_count'] == 1
        assert stats['reject_count'] == 1
        assert abs(stats['full_fill_rate'] - 1 / 3) < 0.01
        assert stats['fully_filled_orders'] == 1
        assert stats['partially_filled_orders'] == 1
        assert stats['unfilled_orders'] == 1

        # 成交量统计
        assert logger.order_filled_qty['order-1'] == 100
        assert logger.order_filled_qty['order-2'] == 20
        assert logger.order_filled_qty['order-3'] == 0

        # 按数量成交率 = 120 / 180
        assert abs(logger.calculate_fill_rate() - 120 / 180) < 0.01
        # 按订单数成交率 = 1 / 4
        assert abs(logger.calculate_fill_rate_by_count() - 1 / 4) < 0.01

        # CSV 保存
        logger.save_to_file()
        assert os.path.exists(output_file)
        with open(output_file) as f:
            lines = f.readlines()
        assert len(lines) == 6, "1 行表头 + 5 行记录"
