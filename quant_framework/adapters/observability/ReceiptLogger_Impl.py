"""回执记录器 - 直接实现 IObservabilitySinks 接口。

本模块实现订单回执的记录和分析功能，同时作为可观测性端口的实现：
- ReceiptLogger_Impl: 直接继承 IObservabilitySinks，记录所有回执到文件
- 计算成交率（fill rate）
- 诊断计数
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable, Set

from ...core.port import IObservabilitySinks
from ...core.data_structure import CancelRequest, Order, OrderReceipt, ReceiptType


# 设置模块级logger
logger = logging.getLogger(__name__)


@dataclass
class ReceiptRecord:
    """回执记录。
    
    Attributes:
        order_id: 订单ID
        exch_time: 交易所时间（回执产生的交易所时间戳）
        recv_time: 本地接收时间
        receipt_type: 回执类型（PARTIAL/FILL/CANCELED/REJECTED）
        fill_qty: 成交数量
        fill_price: 成交价格
        remaining_qty: 剩余数量
    """
    order_id: str
    exch_time: int
    recv_time: int
    receipt_type: str
    fill_qty: int
    fill_price: float
    remaining_qty: int


# 回执回调类型定义
ReceiptCallback = Callable[[OrderReceipt], None]


class ReceiptLogger_Impl(IObservabilitySinks):
    """回执记录器 - 直接实现 IObservabilitySinks。
    
    功能：
    1. 实现 IObservabilitySinks 接口（诊断计数 + 运行结果）
    2. 记录所有回执到内存
    3. 保存回执到CSV文件
    4. 计算成交率（fill rate）
    5. 支持实时打印回执（verbose模式）
    6. 支持自定义回执回调
    
    回执类型说明：
    - PARTIAL: 部分成交
    - FILL: 全部成交
    - CANCELED: 已撤单
    - REJECTED: 已拒绝
    
    Attributes:
        records: 回执记录列表
        output_file: 输出文件路径
        order_total_qty: 订单总数量统计 {order_id: total_qty}
        order_filled_qty: 订单已成交数量统计 {order_id: filled_qty}
        verbose: 是否实时打印回执
        callback: 自定义回执回调函数
    """
    
    def __init__(
        self, 
        output_file: Optional[str] = None,
        verbose: bool = False,
        callback: Optional[ReceiptCallback] = None
    ):
        """初始化回执记录器。
        
        Args:
            output_file: 输出文件路径（可选，如果不提供则只保存在内存中）
            verbose: 是否实时打印回执到控制台
            callback: 自定义回执回调函数，签名为 (receipt: OrderReceipt) -> None
                      回调函数抛出的异常将被记录但不会中断回测
        """
        self.output_file = output_file
        self.verbose = verbose
        self.callback = callback
        self.records: List[ReceiptRecord] = []
        
        # 诊断计数（来自原 ObservabilityImpl）
        self._final_time = 0
        self._error: str | None = None
        self._diagnostics = {
            "intervals_processed": 0,
            "orders_submitted": 0,
            "orders_filled": 0,
            "receipts_generated": 0,
            "cancels_submitted": 0,
        }
        
        # 统计数据
        self.order_total_qty: Dict[str, int] = {}  # order_id -> 原始订单数量
        self.order_filled_qty: Dict[str, int] = {}  # order_id -> 已成交数量
        self._partial_fill_counts: Dict[str, int] = {}  # 部分成交次数
        self._full_fill_counts: Dict[str, int] = {}  # 全部成交次数
        self._cancel_counts: Dict[str, int] = {}  # 撤单次数
        self._reject_counts: Dict[str, int] = {}  # 拒绝次数
        self._canceled_orders: Set[str] = set()  # Track canceled order IDs to exclude from full-fill stats

    # -----------------------------------------------------------------------
    # IObservabilitySinks 接口实现
    # -----------------------------------------------------------------------

    def on_order_submitted(self, order: Order) -> None:
        self._diagnostics["orders_submitted"] += 1
        self.register_order(order.order_id, order.qty)

    def on_cancel_submitted(self, request: CancelRequest) -> None:
        self._diagnostics["cancels_submitted"] += 1

    def on_receipt_generated(self, receipt: OrderReceipt) -> None:
        self._diagnostics["receipts_generated"] += 1

    def on_receipt_delivered(self, receipt: OrderReceipt) -> None:
        self.log_receipt(receipt)
        if receipt.receipt_type in {"FILL", "PARTIAL"}:
            self._diagnostics["orders_filled"] += 1

    def on_interval_end(self, stats: object) -> None:
        self._diagnostics["intervals_processed"] += 1

    def on_run_end(self, final_time: int, error: str | None = None) -> None:
        self._final_time = int(final_time)
        self._error = error

    def get_diagnostics(self) -> dict:
        return dict(self._diagnostics)

    def get_run_result(self) -> dict:
        result = {
            "intervals": self._diagnostics["intervals_processed"],
            "final_time": self._final_time,
            "diagnostics": self.get_diagnostics(),
        }
        if self._error is not None:
            result["error"] = self._error
        return result

    # -----------------------------------------------------------------------
    # 自身属性（兼容 main.py 中 receipt_logger 属性访问）
    # -----------------------------------------------------------------------

    @property
    def receipt_logger(self) -> "ReceiptLogger_Impl":
        """兼容属性：返回自身，因为本类已直接实现全部功能。"""
        return self

    # -----------------------------------------------------------------------
    # 回执记录功能
    # -----------------------------------------------------------------------
    
    def register_order(self, order_id: str, qty: int) -> None:
        """注册新订单，用于计算成交率。
        
        Args:
            order_id: 订单ID
            qty: 订单数量
        """
        self.order_total_qty[order_id] = qty
        self.order_filled_qty[order_id] = 0
        logger.debug(f"[ReceiptLogger] Order registered: order_id={order_id}, qty={qty}")
    
    def log_receipt(self, receipt: OrderReceipt) -> None:
        """记录一条回执。
        
        Args:
            receipt: 订单回执
        """
        record = ReceiptRecord(
            order_id=receipt.order_id,
            exch_time=receipt.timestamp,
            recv_time=receipt.recv_time if receipt.recv_time else 0,
            receipt_type=receipt.receipt_type,
            fill_qty=receipt.fill_qty,
            fill_price=receipt.fill_price,
            remaining_qty=receipt.remaining_qty,
        )
        self.records.append(record)
        
        # 更新统计
        self._update_statistics(receipt)
        
        # 实时打印回执（verbose模式）
        if self.verbose:
            self._print_receipt(receipt)
        
        # 调用自定义回调（带异常保护）
        if self.callback:
            try:
                self.callback(receipt)
            except Exception as e:
                logger.warning(f"Receipt callback raised exception: {e}")
    
    def _print_receipt(self, receipt: OrderReceipt) -> None:
        """打印单条回执信息。
        
        Args:
            receipt: 订单回执
        """
        print(
            f"[Receipt] {receipt.receipt_type:8s} | "
            f"order_id={receipt.order_id} | "
            f"fill_qty={receipt.fill_qty} | "
            f"fill_price={receipt.fill_price:.2f} | "
            f"remaining={receipt.remaining_qty} | "
            f"timestamp={receipt.timestamp}"
        )
    
    def _update_statistics(self, receipt: OrderReceipt) -> None:
        """更新统计数据。
        
        Args:
            receipt: 订单回执
        """
        order_id = receipt.order_id
        
        if receipt.receipt_type == "PARTIAL":
            self._partial_fill_counts[order_id] = self._partial_fill_counts.get(order_id, 0) + 1
            # 部分成交累加已成交数量
            if order_id in self.order_filled_qty:
                self.order_filled_qty[order_id] += receipt.fill_qty
                
        elif receipt.receipt_type == "FILL":
            self._full_fill_counts[order_id] = self._full_fill_counts.get(order_id, 0) + 1
            # 全部成交设置为订单总量
            if order_id in self.order_total_qty:
                self.order_filled_qty[order_id] = self.order_total_qty[order_id]
                
        elif receipt.receipt_type == "CANCELED":
            self._cancel_counts[order_id] = self._cancel_counts.get(order_id, 0) + 1
            self._canceled_orders.add(order_id)
            # 撤单回执的fill_qty应该等于之前PARTIAL回执累积的成交量
            # 这里不更新order_filled_qty，因为它已经通过PARTIAL回执正确累积
            # 一致性检查：如果不匹配则记录警告，帮助发现潜在的bug
            if order_id in self.order_filled_qty:
                expected_filled = self.order_filled_qty[order_id]
                if receipt.fill_qty != expected_filled:
                    logger.warning(
                        f"[ReceiptLogger] Cancel fill_qty mismatch for {order_id}: "
                        f"receipt.fill_qty={receipt.fill_qty}, accumulated={expected_filled}"
                    )
                
        elif receipt.receipt_type == "REJECTED":
            self._reject_counts[order_id] = self._reject_counts.get(order_id, 0) + 1
    
    def save_to_file(self, filepath: Optional[str] = None) -> None:
        """保存回执记录到CSV文件。
        
        Args:
            filepath: 输出文件路径（可选，如果不提供则使用初始化时的路径）
        """
        output_path = filepath or self.output_file
        if not output_path:
            raise ValueError("No output file specified")
        
        # 确保目录存在
        dir_path = os.path.dirname(output_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'order_id', 'exch_time', 'recv_time', 'receipt_type',
                'fill_qty', 'fill_price', 'remaining_qty'
            ])
            writer.writeheader()
            
            for record in self.records:
                writer.writerow({
                    'order_id': record.order_id,
                    'exch_time': record.exch_time,
                    'recv_time': record.recv_time,
                    'receipt_type': record.receipt_type,
                    'fill_qty': record.fill_qty,
                    'fill_price': record.fill_price,
                    'remaining_qty': record.remaining_qty,
                })
    
    def calculate_fill_rate(self) -> float:
        """计算总成交率。
        
        成交率 = 总成交数量 / 总订单数量
        
        Returns:
            成交率（0.0 - 1.0）
        """
        total_qty = sum(self.order_total_qty.values())
        if total_qty == 0:
            return 0.0
        
        filled_qty = sum(self.order_filled_qty.values())
        return filled_qty / total_qty
    
    def calculate_fill_rate_by_count(self) -> float:
        """按订单数量计算成交率。
        
        成交率 = 完全成交订单数 / 总订单数
        
        Returns:
            成交率（0.0 - 1.0）
        """
        total_orders = len(self.order_total_qty)
        if total_orders == 0:
            return 0.0
        
        fully_filled = len(self._full_fill_counts)
        return fully_filled / total_orders
    
    def get_statistics(self) -> dict:
        """获取统计信息。
        
        Returns:
            统计信息字典
        """
        total_orders = len(self.order_total_qty)
        total_qty = sum(self.order_total_qty.values())
        filled_qty = sum(self.order_filled_qty.values())
        
        # 直接使用 _full_fill_counts 获取全部成交订单数
        fully_filled_orders = len(self._full_fill_counts)
        
        partially_filled_orders = 0
        unfilled_orders = 0
        for oid, order_qty in self.order_total_qty.items():
            if order_qty <= 0:
                # Skip invalid/non-tradable quantities (0 or negative; defensive guard)
                continue
            # 跳过已全部成交的订单（已通过 _full_fill_counts 统计）
            if oid in self._full_fill_counts:
                continue
            order_filled_qty = self.order_filled_qty.get(oid, 0)
            if oid in self._canceled_orders:
                if order_filled_qty > 0:
                    partially_filled_orders += 1
                else:
                    unfilled_orders += 1
                continue
            # 非撤单、非全部成交的订单
            if order_filled_qty > 0:
                partially_filled_orders += 1
            else:
                unfilled_orders += 1
        countable_orders = fully_filled_orders + partially_filled_orders + unfilled_orders
        full_fill_rate = fully_filled_orders / countable_orders if countable_orders else 0.0
        partial_fill_rate = partially_filled_orders / countable_orders if countable_orders else 0.0
        
        return {
            'total_receipts': len(self.records),
            'total_orders': total_orders,
            'total_order_qty': total_qty,
            'total_filled_qty': filled_qty,
            'fill_rate_by_qty': self.calculate_fill_rate(),
            'fill_rate_by_count': self.calculate_fill_rate_by_count(),
            'full_fill_rate': full_fill_rate,
            'partial_fill_rate': partial_fill_rate,
            'partial_fill_count': sum(self._partial_fill_counts.values()),
            'full_fill_count': sum(self._full_fill_counts.values()),
            'cancel_count': sum(self._cancel_counts.values()),
            'reject_count': sum(self._reject_counts.values()),
            'fully_filled_orders': fully_filled_orders,
            'partially_filled_orders': partially_filled_orders,
            'unfilled_orders': unfilled_orders,
        }
    
    def print_summary(self) -> None:
        """打印回执统计摘要。"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("回执记录器摘要 (Receipt Logger Summary)")
        print("=" * 60)
        print(f"Total Receipts (总回执数): {stats['total_receipts']}")
        print(f"Total Orders (总订单数): {stats['total_orders']}")
        print(f"Total Order Qty (总下单数量): {stats['total_order_qty']}")
        print(f"Total Filled Qty (总成交数量): {stats['total_filled_qty']}")
        print()
        print("回执类型分布 (Receipt Type Distribution):")
        print(f"  - Partial Fill (部分成交回执数): {stats['partial_fill_count']}")
        print(f"  - Full Fill (全部成交回执数): {stats['full_fill_count']}")
        if stats['full_fill_count'] != stats['fully_filled_orders']:
            print(f"    * 注: 部分成交后撤单/未满量的订单只会产生Partial Fill回执，不会产生Full Fill回执")
        print(f"  - Canceled (已撤单): {stats['cancel_count']}")
        print(f"  - Rejected (已拒绝): {stats['reject_count']}")
        # 说明部分成交回执数和部分成交订单数的区别
        if stats['partial_fill_count'] != stats['partially_filled_orders']:
            print(f"    * 注: 部分成交回执数({stats['partial_fill_count']}) != 部分成交订单数({stats['partially_filled_orders']})")
            print(f"      原因: 一个订单可能产生多个PARTIAL回执（多次部分成交）")
        print()
        print("订单最终状态分布 (Order Final Status Distribution):")
        print(f"  - Fully Filled (完全成交): {stats['fully_filled_orders']}")
        print(f"  - Partially Filled (仅部分成交): {stats['partially_filled_orders']}")
        print(f"  - Unfilled (未成交): {stats['unfilled_orders']}")
        print()
        print("最终成交率 (Final Fill Rate):")
        print(f"  - Full Fill Rate (完全成交率): {stats['full_fill_rate']:.2%}")
        print(f"  - Partial Fill Rate (部分成交率): {stats['partial_fill_rate']:.2%}")
        print()
        print("成交率 (Fill Rate):")
        print(f"  - By Quantity (按数量): {stats['fill_rate_by_qty']:.2%}")
        print(f"  - By Order Count (按订单数): {stats['fill_rate_by_count']:.2%}")
        print("=" * 60)
    
    def get_records_as_dicts(self) -> List[Dict]:
        """获取所有回执记录作为字典列表。
        
        Returns:
            回执记录的字典列表，适合转换为DataFrame或JSON
        """
        return [
            {
                'order_id': r.order_id,
                'exch_time': r.exch_time,
                'recv_time': r.recv_time,
                'receipt_type': r.receipt_type,
                'fill_qty': r.fill_qty,
                'fill_price': r.fill_price,
                'remaining_qty': r.remaining_qty,
            }
            for r in self.records
        ]
    
    def print_all_receipts(self) -> None:
        """打印所有已记录的回执。"""
        if not self.records:
            print("No receipts recorded.")
            return
        
        print("\n" + "=" * 80)
        print("All Receipts")
        print("=" * 80)
        print(f"{'Type':10s} | {'Order ID':20s} | {'Fill Qty':>10s} | {'Price':>12s} | {'Remaining':>10s} | {'Exch Time':>15s}")
        print("-" * 80)
        for r in self.records:
            print(f"{r.receipt_type:10s} | {r.order_id:20s} | {r.fill_qty:>10d} | {r.fill_price:>12.2f} | {r.remaining_qty:>10d} | {r.exch_time:>15d}")
        print("=" * 80)
    
    def clear(self) -> None:
        """清空所有记录和统计数据。"""
        self.records.clear()
        self.order_total_qty.clear()
        self.order_filled_qty.clear()
        self._partial_fill_counts.clear()
        self._full_fill_counts.clear()
        self._cancel_counts.clear()
        self._reject_counts.clear()
        self._canceled_orders.clear()


class NullObservability_Impl(ReceiptLogger_Impl):
    """无输出观测实现（但保留诊断计数）。"""

    def __init__(self) -> None:
        super().__init__(output_file=None, verbose=False, callback=None)
