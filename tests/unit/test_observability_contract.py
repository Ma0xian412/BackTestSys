"""可观测性端口契约测试。"""

from quant_framework.adapters.observability.ReceiptLogger_Impl import ReceiptLogger_Impl


def test_observability_sink_excludes_oms_interface():
    """IObservabilitySinks 实现不应暴露 OMS 状态机/查询接口。"""
    sink = ReceiptLogger_Impl()

    assert hasattr(sink, "on_order_submitted")
    assert hasattr(sink, "on_receipt_delivered")
    assert hasattr(sink, "get_diagnostics")
    assert hasattr(sink, "get_run_result")

    assert not hasattr(sink, "submit_order")
    assert not hasattr(sink, "submit_cancel")
    assert not hasattr(sink, "apply_receipt")
    assert not hasattr(sink, "view")
    assert not hasattr(sink, "get_order")
    assert not hasattr(sink, "get_active_orders")
