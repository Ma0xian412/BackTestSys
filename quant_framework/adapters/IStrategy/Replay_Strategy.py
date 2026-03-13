"""重放策略：读取 CSV 后以 on_event 统一回放动作。"""

import csv
import logging
import os
import re
from datetime import datetime
from typing import Any, List, Tuple, Optional

from ...core.data_structure import (
    EVENT_KIND_RECEIPT_DELIVERY,
    EVENT_KIND_MDARRIVE,
    Action,
    ActionType,
    StrategyContext,
)
from ...core.port import IStrategy
from ...core.data_structure import CancelRequest, Order, Side


logger = logging.getLogger(__name__)
_CONTRACT_ID_COLUMNS = ("ContractId", "contract_id", "ResultContractId")
_PARTITION_DAY_COLUMNS = ("PartitionDay", "partition_day", "TradingDay", "TradeDate", "ActionDay")
_MACHINE_NAME_COLUMNS = ("MachineName", "machine_name")
_TIME_COLUMNS = ("SentTime", "CancelSentTime", "RecvTick", "ExchTick")
_FILENAME_CONTRACT_ID_PATTERN = re.compile(r"id(\d+)", re.IGNORECASE)


def _safe_int(value: object) -> Optional[int]:
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except (TypeError, ValueError):
            return None


def _is_valid_partition_day(day: int) -> bool:
    if day < 19000101 or day > 21001231:
        return False
    try:
        datetime.strptime(str(day), "%Y%m%d")
        return True
    except ValueError:
        return False


def _extract_partition_day(value: object) -> Optional[int]:
    text = str(value).strip()
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) < 8:
        return None
    for i in range(len(digits) - 7):
        candidate = int(digits[i:i + 8])
        if _is_valid_partition_day(candidate):
            return candidate
    return None


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
        self._inferred_contract_id: Optional[int] = None
        self._inferred_partition_day: Optional[int] = None
        self._inferred_machine_name: str = ""
        
        # 加载文件，直接存入pending_orders/pending_cancels
        if order_file and os.path.exists(order_file):
            self._load_orders(order_file)
        if cancel_file and os.path.exists(cancel_file):
            self._load_cancels(cancel_file)
        self._infer_contract_id_from_filename(order_file)
        self._infer_contract_id_from_filename(cancel_file)
        
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
                self._infer_metadata_from_row(row)
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
                self._infer_metadata_from_row(row)
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

    def _infer_metadata_from_row(self, row: dict[str, Any]) -> None:
        contract_id = self._pick_first_int(row, _CONTRACT_ID_COLUMNS)
        partition_day = self._pick_first_partition_day(row)
        machine_name = self._pick_first_text(row, _MACHINE_NAME_COLUMNS)
        self._set_inferred_contract_id(contract_id)
        self._set_inferred_partition_day(partition_day)
        self._set_inferred_machine_name(machine_name)

    def _pick_first_int(self, row: dict[str, Any], columns: tuple[str, ...]) -> Optional[int]:
        for key in columns:
            value = _safe_int(row.get(key))
            if value is not None and value > 0:
                return value
        return None

    def _pick_first_partition_day(self, row: dict[str, Any]) -> Optional[int]:
        for key in _PARTITION_DAY_COLUMNS:
            value = _extract_partition_day(row.get(key))
            if value is not None:
                return value
        for key in _TIME_COLUMNS:
            value = _extract_partition_day(row.get(key))
            if value is not None:
                return value
        return None

    @staticmethod
    def _pick_first_text(row: dict[str, Any], columns: tuple[str, ...]) -> str:
        for key in columns:
            value = str(row.get(key, "")).strip()
            if value:
                return value
        return ""

    def _set_inferred_contract_id(self, value: Optional[int]) -> None:
        if value is None:
            return
        if self._inferred_contract_id is None:
            self._inferred_contract_id = value
            return
        if self._inferred_contract_id != value:
            logger.warning(
                "ReplayStrategy inferred conflicting ContractId values: %s vs %s",
                self._inferred_contract_id,
                value,
            )

    def _set_inferred_partition_day(self, value: Optional[int]) -> None:
        if value is None:
            return
        if self._inferred_partition_day is None:
            self._inferred_partition_day = value
            return
        if self._inferred_partition_day != value:
            logger.warning(
                "ReplayStrategy inferred conflicting PartitionDay values: %s vs %s",
                self._inferred_partition_day,
                value,
            )

    def _set_inferred_machine_name(self, value: str) -> None:
        if not value:
            return
        if not self._inferred_machine_name:
            self._inferred_machine_name = value
            return
        if self._inferred_machine_name != value:
            logger.warning(
                "ReplayStrategy inferred conflicting MachineName values: %s vs %s",
                self._inferred_machine_name,
                value,
            )

    def _infer_contract_id_from_filename(self, filepath: Optional[str]) -> None:
        if not filepath:
            return
        filename = os.path.basename(filepath)
        match = _FILENAME_CONTRACT_ID_PATTERN.search(filename)
        if not match:
            return
        self._set_inferred_contract_id(_safe_int(match.group(1)))

    def get_inferred_result_metadata(self) -> dict[str, object]:
        """返回从 replay 数据中推断的结果元数据。"""
        out: dict[str, object] = {}
        if self._inferred_contract_id and self._inferred_contract_id > 0:
            out["contract_id"] = self._inferred_contract_id
        if self._inferred_partition_day and self._inferred_partition_day > 0:
            out["partition_day"] = self._inferred_partition_day
        if self._inferred_machine_name:
            out["machine_name"] = self._inferred_machine_name
        return out
    
    def on_event(self, e, ctx: StrategyContext) -> List:
        if e.kind == EVENT_KIND_MDARRIVE and self.is_first_snapshot:
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
