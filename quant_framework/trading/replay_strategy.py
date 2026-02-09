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
import logging
import os
from typing import List, Tuple, Optional

from ..core.interfaces import IStrategy
from ..core.types import Order, CancelRequest, Side, OrderReceipt
from ..core.dto import SnapshotDTO, ReadOnlyOMSView


logger = logging.getLogger(__name__)


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
        
        # 状态
        self.is_first_snapshot = True
        self.pending_orders: List[Tuple[int, Order]] = []  # (sent_time, order)
        self.pending_cancels: List[Tuple[int, CancelRequest]] = []  # (cancel_sent_time, cancel_request)
        self.current_time = 0
        
        # 统计计数器
        self._total_orders_loaded = 0
        self._total_cancels_loaded = 0
        
        # 订单ID映射：原始order_id -> 内部order_id字符串（用于撤单关联）
        self._order_id_map: dict = {}
        
        # 加载文件，直接存入pending_orders/pending_cancels
        if order_file and os.path.exists(order_file):
            self._load_orders(order_file)
        if cancel_file and os.path.exists(cancel_file):
            self._load_cancels(cancel_file)
        
        # 一次性排序
        self.pending_orders.sort(key=lambda x: x[0])
        self.pending_cancels.sort(key=lambda x: x[0])
        
        # 记录初始加载总数
        self._total_orders_loaded = len(self.pending_orders)
        self._total_cancels_loaded = len(self.pending_cancels)
    
    def _load_orders(self, filepath: str) -> None:
        """从CSV文件加载下单记录，直接转换为Order对象存入pending_orders。
        
        Args:
            filepath: CSV文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    order_id = int(row['OrderId'])
                    sent_time = int(row['SentTime'])
                    limit_price = float(row['LimitPrice'])
                    volume = int(row['Volume'])
                    direction = row['OrderDirection']
                    
                    order_id_str = str(order_id)
                    self._order_id_map[order_id] = order_id_str
                    
                    side = Side.BUY if direction.upper() == "BUY" else Side.SELL
                    order = Order(
                        order_id=order_id_str,
                        side=side,
                        price=limit_price,
                        qty=volume,
                    )
                    self.pending_orders.append((sent_time, order))
                except (KeyError, ValueError) as e:
                    # 跳过无效行
                    logger.warning(f"Skipping invalid order row: {row}, error: {e}")
    
    def _load_cancels(self, filepath: str) -> None:
        """从CSV文件加载撤单记录，直接转换为CancelRequest对象存入pending_cancels。
        
        Args:
            filepath: CSV文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    order_id = int(row['OrderId'])
                    cancel_sent_time = int(row['CancelSentTime'])
                    
                    # 撤单时需要使用映射后的订单ID
                    order_id_str = self._order_id_map.get(
                        order_id, 
                        str(order_id)
                    )
                    cancel_request = CancelRequest(
                        order_id=order_id_str,
                        create_time=cancel_sent_time,
                    )
                    self.pending_cancels.append((cancel_sent_time, cancel_request))
                except (KeyError, ValueError) as e:
                    # 跳过无效行
                    logger.warning(f"Skipping invalid cancel row: {row}, error: {e}")
    
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
            'total_orders': self._total_orders_loaded,
            'total_cancels': self._total_cancels_loaded,
            'pending_orders': len(self.pending_orders),
            'pending_cancels': len(self.pending_cancels),
        }
