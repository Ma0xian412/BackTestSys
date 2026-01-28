"""回执记录模块。

本模块实现订单回执的记录和分析功能：
- ReceiptLogger: 记录所有回执到文件
- 计算成交率（fill rate）
"""

import csv
import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from datetime import datetime

from ..core.types import OrderReceipt, ReceiptType


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


class ReceiptLogger:
    """回执记录器。
    
    功能：
    1. 记录所有回执到内存
    2. 保存回执到CSV文件
    3. 计算成交率（fill rate）
    4. 支持实时打印回执（verbose模式）
    5. 支持自定义回执回调
    
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
        
        # 统计数据
        self.order_total_qty: Dict[str, int] = {}  # order_id -> 原始订单数量
        self.order_filled_qty: Dict[str, int] = {}  # order_id -> 已成交数量
        self._partial_fill_counts: Dict[str, int] = {}  # 部分成交次数
        self._full_fill_counts: Dict[str, int] = {}  # 全部成交次数
        self._cancel_counts: Dict[str, int] = {}  # 撤单次数
        self._reject_counts: Dict[str, int] = {}  # 拒绝次数
    
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
            # 撤单时，如果有部分成交，fill_qty表示撤单前的已成交量
            if order_id in self.order_filled_qty and receipt.fill_qty > 0:
                # 撤单回执的fill_qty是累计已成交量
                self.order_filled_qty[order_id] = receipt.fill_qty
                
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
        
        return {
            'total_receipts': len(self.records),
            'total_orders': total_orders,
            'total_order_qty': total_qty,
            'total_filled_qty': filled_qty,
            'fill_rate_by_qty': self.calculate_fill_rate(),
            'fill_rate_by_count': self.calculate_fill_rate_by_count(),
            'partial_fill_count': sum(self._partial_fill_counts.values()),
            'full_fill_count': sum(self._full_fill_counts.values()),
            'cancel_count': sum(self._cancel_counts.values()),
            'reject_count': sum(self._reject_counts.values()),
            'fully_filled_orders': len(self._full_fill_counts),
            'partially_filled_orders': len([
                oid for oid in self.order_filled_qty 
                if 0 < self.order_filled_qty[oid] < self.order_total_qty.get(oid, 0)
            ]),
            'unfilled_orders': len([
                oid for oid in self.order_total_qty 
                if self.order_filled_qty.get(oid, 0) == 0
            ]),
        }
    
    def print_summary(self) -> None:
        """打印回执统计摘要。"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("Receipt Logger Summary")
        print("=" * 60)
        print(f"Total Receipts: {stats['total_receipts']}")
        print(f"Total Orders: {stats['total_orders']}")
        print(f"Total Order Qty: {stats['total_order_qty']}")
        print(f"Total Filled Qty: {stats['total_filled_qty']}")
        print()
        print("Receipt Type Distribution:")
        print(f"  - Partial Fill: {stats['partial_fill_count']}")
        print(f"  - Full Fill: {stats['full_fill_count']}")
        print(f"  - Canceled: {stats['cancel_count']}")
        print(f"  - Rejected: {stats['reject_count']}")
        print()
        print("Order Status:")
        print(f"  - Fully Filled: {stats['fully_filled_orders']}")
        print(f"  - Partially Filled: {stats['partially_filled_orders']}")
        print(f"  - Unfilled: {stats['unfilled_orders']}")
        print()
        print("Fill Rate:")
        print(f"  - By Quantity: {stats['fill_rate_by_qty']:.2%}")
        print(f"  - By Order Count: {stats['fill_rate_by_count']:.2%}")
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
