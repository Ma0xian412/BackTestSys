"""重放策略模块。

本模块实现从CSV文件读取历史订单和撤单记录，并在回测中重放。

用于验证框架有效性：将实盘下单/撤单记录作为策略输入，
比较回测结果与实盘结果。

CSV文件格式：
- 下单文件: PubOrderLog_{machineName}_Day{yyyymmdd}_Id{contractId}.csv
  字段: OrderId, LimitPrice, Volume, OrderDirection, SentTime
- 撤单文件: PubOrderCancelRequestLog_{machineName}_Day{yyyymmdd}_Id{contractId}.csv
  字段: OrderId, CancelSentTime
"""

import csv
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

from ..core.interfaces import IStrategy
from ..core.types import Order, CancelRequest, Side, OrderReceipt
from ..core.dto import SnapshotDTO, ReadOnlyOMSView


@dataclass
class OrderRecord:
    """下单记录。
    
    Attributes:
        order_id: 订单ID
        limit_price: 限价
        volume: 数量
        direction: 方向 (Buy/Sell)
        sent_time: 发送时间（对应RecvTime）
    """
    order_id: int
    limit_price: float
    volume: int
    direction: str
    sent_time: int


@dataclass
class CancelRecord:
    """撤单记录。
    
    Attributes:
        order_id: 要撤销的订单ID
        cancel_sent_time: 撤单发送时间
    """
    order_id: int
    cancel_sent_time: int


class ReplayStrategy(IStrategy):
    """重放策略 - 从CSV文件读取历史订单和撤单记录并重放。
    
    策略在收到第一张快照时，将所有订单和撤单按时间顺序排入事件队列。
    
    使用方法：
    1. 实例化时提供订单文件和撤单文件路径
    2. 策略会自动读取CSV文件
    3. 在第一张快照到达时，返回所有应发出的订单
    4. 后续通过on_receipt处理回执
    
    Attributes:
        name: 策略名称
        order_file: 下单文件路径
        cancel_file: 撤单文件路径
        orders: 解析后的订单记录列表
        cancels: 解析后的撤单记录列表
        is_first_snapshot: 是否是第一张快照
        pending_orders: 待发送的订单（按sent_time排序）
        pending_cancels: 待发送的撤单（按cancel_sent_time排序）
        current_time: 当前策略时间
    """
    
    def __init__(
        self,
        name: str = "ReplayStrategy",
        order_file: Optional[str] = None,
        cancel_file: Optional[str] = None,
    ):
        """初始化重放策略。
        
        Args:
            name: 策略名称
            order_file: 下单文件路径（可选）
            cancel_file: 撤单文件路径（可选）
        """
        self.name = name
        self.order_file = order_file
        self.cancel_file = cancel_file
        
        # 解析后的记录
        self.orders: List[OrderRecord] = []
        self.cancels: List[CancelRecord] = []
        
        # 状态
        self.is_first_snapshot = True
        self.pending_orders: List[Tuple[int, Order]] = []  # (sent_time, order)
        self.pending_cancels: List[Tuple[int, CancelRequest]] = []  # (cancel_sent_time, cancel_request)
        self.current_time = 0
        
        # 订单ID映射：原始order_id -> 内部order_id字符串
        self._order_id_map: dict = {}
        
        # 加载文件
        if order_file and os.path.exists(order_file):
            self._load_orders(order_file)
        if cancel_file and os.path.exists(cancel_file):
            self._load_cancels(cancel_file)
        
        # 按时间排序
        self._prepare_pending_events()
    
    def _load_orders(self, filepath: str) -> None:
        """从CSV文件加载下单记录。
        
        Args:
            filepath: CSV文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    record = OrderRecord(
                        order_id=int(row['OrderId']),
                        limit_price=float(row['LimitPrice']),
                        volume=int(row['Volume']),
                        direction=row['OrderDirection'],
                        sent_time=int(row['SentTime']),
                    )
                    self.orders.append(record)
                except (KeyError, ValueError) as e:
                    # 跳过无效行
                    print(f"Warning: Skipping invalid order row: {row}, error: {e}")
    
    def _load_cancels(self, filepath: str) -> None:
        """从CSV文件加载撤单记录。
        
        Args:
            filepath: CSV文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    record = CancelRecord(
                        order_id=int(row['OrderId']),
                        cancel_sent_time=int(row['CancelSentTime']),
                    )
                    self.cancels.append(record)
                except (KeyError, ValueError) as e:
                    # 跳过无效行
                    print(f"Warning: Skipping invalid cancel row: {row}, error: {e}")
    
    def _prepare_pending_events(self) -> None:
        """准备待处理的订单和撤单事件，按时间排序。"""
        # 转换订单记录为Order对象
        for record in self.orders:
            order_id_str = f"{self.name}-{record.order_id}"
            self._order_id_map[record.order_id] = order_id_str
            
            side = Side.BUY if record.direction.upper() == "BUY" else Side.SELL
            order = Order(
                order_id=order_id_str,
                side=side,
                price=record.limit_price,
                qty=record.volume,
            )
            self.pending_orders.append((record.sent_time, order))
        
        # 转换撤单记录为CancelRequest对象
        for record in self.cancels:
            # 撤单时需要使用映射后的订单ID
            order_id_str = self._order_id_map.get(
                record.order_id, 
                f"{self.name}-{record.order_id}"
            )
            cancel_request = CancelRequest(
                order_id=order_id_str,
                create_time=record.cancel_sent_time,
            )
            self.pending_cancels.append((record.cancel_sent_time, cancel_request))
        
        # 按时间排序
        self.pending_orders.sort(key=lambda x: x[0])
        self.pending_cancels.sort(key=lambda x: x[0])
    
    def on_snapshot(self, snapshot: SnapshotDTO, oms_view: ReadOnlyOMSView) -> List[Order]:
        """快照到达时回调。
        
        在第一张快照到达时，返回所有应发送的订单。
        后续快照不再发送新订单。
        
        Args:
            snapshot: 行情快照DTO（不可变）
            oms_view: OMS只读视图
            
        Returns:
            要提交的新订单列表
        """
        if self.is_first_snapshot:
            self.is_first_snapshot = False
            # 返回所有待发送的订单，订单的create_time会在submit时设置
            orders_to_send = []
            for sent_time, order in self.pending_orders:
                # 设置订单的预期发送时间到order对象上，以便后续处理
                order.create_time = sent_time
                orders_to_send.append(order)
            return orders_to_send
        
        return []
    
    def on_receipt(self, receipt: OrderReceipt, snapshot: SnapshotDTO, oms_view: ReadOnlyOMSView) -> List[Order]:
        """订单回执到达时回调。
        
        重放策略不根据回执生成新订单，只记录状态。
        
        Args:
            receipt: 订单回执
            snapshot: 当前行情快照DTO
            oms_view: OMS只读视图
            
        Returns:
            空列表（不生成新订单）
        """
        # 重放策略不生成新订单
        return []
    
    def get_pending_cancels(self) -> List[Tuple[int, CancelRequest]]:
        """获取所有待发送的撤单请求。
        
        Returns:
            (发送时间, 撤单请求) 元组列表
        """
        return self.pending_cancels.copy()
    
    def get_statistics(self) -> dict:
        """获取策略统计信息。
        
        Returns:
            统计信息字典
        """
        return {
            'total_orders': len(self.orders),
            'total_cancels': len(self.cancels),
            'pending_orders': len(self.pending_orders),
            'pending_cancels': len(self.pending_cancels),
        }
