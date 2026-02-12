"""重放策略：读取 CSV 后以 on_event 统一回放动作。"""

import csv
import logging
import os
from typing import List, Tuple, Optional

from ...core.data_structure import (
    EVENT_KIND_RECEIPT_DELIVERY,
    EVENT_KIND_SNAPSHOT_ARRIVAL,
    Action,
    ActionType,
    StrategyContext,
)
from ...core.port import IStrategy
from ...core.data_structure import CancelRequest, Order, Side


logger = logging.getLogger(__name__)


class ReplayStrategy_Impl(IStrategy):
    """重放策略 - 首张快照时一次性发出订单与撤单动作。"""
    
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
    
    def on_event(self, e, ctx: StrategyContext) -> List:
        if e.kind == EVENT_KIND_SNAPSHOT_ARRIVAL and self.is_first_snapshot:
            self.is_first_snapshot = False
            actions = []
            for sent_time, order in self.pending_orders:
                order.create_time = sent_time
                actions.append(Action(action_type=ActionType.PLACE_ORDER, create_time=sent_time, payload=order))
            for sent_time, cancel in self.pending_cancels:
                cancel.create_time = sent_time
                actions.append(Action(action_type=ActionType.CANCEL_ORDER, create_time=sent_time, payload=cancel))
            return actions

        if e.kind == EVENT_KIND_RECEIPT_DELIVERY:
            return []

        return []
    
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
